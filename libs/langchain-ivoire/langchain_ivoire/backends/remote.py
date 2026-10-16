"""HTTP client for the Abyshire Baoulé API (``https://api.abyshire.ci``) or a self-hosted server.

Endpoints used:

=========================  ===================================================================
``POST /v1/translate``     ``{"texts", "source", "target", "style"}`` ->
                           ``{"texts", "pivot", "model", "timings_ms", "usage", "request_id"}``
``GET  /v1/languages``     languages and what each supports (ASR, TTS, MT directions, styles)
``GET  /v1/models``        model names and versions
``GET  /health``           status
``POST /listen_bci``       multipart ``audio`` -> ``{bci, fra, eng, asr_confidence, ...}``
``POST /tts_bci``          ``{"bci": text}`` -> ``{audio_url}``; the WAV is then fetched from
                           ``base_url + audio_url``
``POST /speak_bci``        ``{"text": English, "use_phrasebook": bool}`` -> ``{fra, bci, audio_url}``
``POST /listen_eng``       multipart ``audio`` -> English ASR (self-hosted servers only)
``POST /agent_text``       ``{"fra": ...}`` or ``{"eng": ...}`` -> server-side LLM reply (self-hosted
                           servers only)
=========================  ===================================================================

``base_url`` may or may not end in ``/v1``: the ``/v1/...`` endpoints are found either way, and the
older endpoints are served under both prefixes.

* **Authentication:** ``Authorization: Bearer <key>``. The key comes from ``api_key=`` or
  ``IVOIRE_API_KEY`` and is never logged, printed or included in exceptions. A LAN server needs none.
* **Errors:** RFC 9457 ``application/problem+json`` answers become typed exceptions
  (see :mod:`langchain_ivoire.exceptions`) carrying the problem ``code`` and the ``X-Request-Id``.
* **Retries:** 429, 502, 503 and 504 answers and connection failures are retried with exponential
  backoff and jitter; ``Retry-After`` (seconds or an HTTP date) is honoured, capped at
  ``max_backoff``. Every POST carries an ``Idempotency-Key`` that stays the same across the retries
  of one call, so a retried request is never applied twice.
* **Batching:** :meth:`RemoteBackend.translate` sends at most 32 texts per request and splits texts
  longer than 1000 characters at sentence boundaries, then joins the pieces again.
"""

from __future__ import annotations

import asyncio
import email.utils
import os
import random
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import httpx

from langchain_ivoire._audio import sniff_media_type, wav_info
from langchain_ivoire._settings import SettingsLike, load_settings, section
from langchain_ivoire._version import __version__
from langchain_ivoire.backends.base import IvoireBackend, pivot_path
from langchain_ivoire.exceptions import (
    IvoireAPIError,
    IvoireAuthenticationError,
    IvoireConfigError,
    IvoireConnectionError,
    IvoireIdempotencyConflictError,
    IvoireInvalidRequestError,
    IvoireNotFoundError,
    IvoirePermissionError,
    IvoireRateLimitError,
    IvoireServerError,
    IvoireTooLargeError,
    IvoireUnsupportedError,
    IvoireUnsupportedLanguageError,
    IvoireUpstreamUnavailableError,
)
from langchain_ivoire.types import Speech, Transcription, Translation

__all__ = ["DEFAULT_BASE_URL", "MAX_BATCH", "MAX_TEXT_CHARS", "RemoteBackend"]

DEFAULT_BASE_URL = "https://api.abyshire.ci"
MAX_BATCH = 32
"""Most texts the server accepts in one ``/v1/translate`` request."""
MAX_TEXT_CHARS = 1000
"""Longest text the server accepts; longer texts are split at sentence boundaries."""

_RETRY_STATUS = frozenset({429, 502, 503, 504})

_BY_CODE: dict[str, type[IvoireAPIError]] = {
    "invalid_request": IvoireInvalidRequestError,
    "too_large": IvoireTooLargeError,
    "unsupported_language": IvoireUnsupportedLanguageError,
    "unauthorized": IvoireAuthenticationError,
    "forbidden": IvoirePermissionError,
    "not_found": IvoireNotFoundError,
    "model_not_found": IvoireNotFoundError,
    "method_not_allowed": IvoireNotFoundError,
    "length_required": IvoireInvalidRequestError,
    "idempotency_conflict": IvoireIdempotencyConflictError,
    "rate_limited": IvoireRateLimitError,
    "upstream_unavailable": IvoireUpstreamUnavailableError,
    "internal": IvoireServerError,
}


def _by_status(code: int) -> type[IvoireAPIError]:
    if code in (400, 411, 422):
        return IvoireInvalidRequestError
    if code == 413:
        return IvoireTooLargeError
    if code == 401:
        return IvoireAuthenticationError
    if code == 403:
        return IvoirePermissionError
    if code in (404, 405):
        return IvoireNotFoundError
    if code == 409:
        return IvoireIdempotencyConflictError
    if code == 429:
        return IvoireRateLimitError
    if code == 503:
        return IvoireUpstreamUnavailableError
    if code >= 500:
        return IvoireServerError
    return IvoireAPIError


def _json_body(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return None


def _problem(resp: httpx.Response) -> dict[str, Any] | None:
    """The RFC 9457 problem details of an error answer, if it has them."""
    body = _json_body(resp)
    if not isinstance(body, dict):
        return None
    ctype = resp.headers.get("content-type", "")
    if ctype.startswith("application/problem+json") or ("code" in body and "title" in body):
        return {k: body[k] for k in ("type", "title", "status", "detail", "code") if k in body}
    return None


def _error_message(resp: httpx.Response) -> str:
    body = _json_body(resp)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):  # OpenAI-style {"error": {"message": ..., "code": ...}}
            err = err.get("message")
        msg = body.get("detail") or body.get("title") or err or body.get("message")
        if msg:
            return str(msg)[:300]
    return (resp.text or resp.reason_phrase or "error")[:300]


def _request_id(resp: httpx.Response) -> str | None:
    rid = resp.headers.get("x-request-id")
    if rid:
        return str(rid)
    body = _json_body(resp)
    if isinstance(body, dict) and isinstance(body.get("request_id"), str):
        return str(body["request_id"])
    return None


def _retry_after(resp: httpx.Response) -> float | None:
    """``Retry-After`` in seconds (delta-seconds or an HTTP date)."""
    v = (resp.headers.get("retry-after") or "").strip()
    if not v:
        return None
    if v.replace(".", "", 1).isdigit():
        return float(v)
    try:
        when = email.utils.parsedate_to_datetime(v)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


_KEY_HINT = "set IVOIRE_API_KEY or pass api_key=; keys are issued by Abyshire"


def _raise_for_status(resp: httpx.Response, key_hint: str = _KEY_HINT) -> None:
    status = resp.status_code
    if status < 400:
        return
    problem = _problem(resp)
    code = (problem or {}).get("code")
    cls = _BY_CODE.get(str(code)) if code else None
    cls = cls or _by_status(status)
    rid = _request_id(resp)
    msg = f"{status}{f' {code}' if code else ''} from {resp.request.method} {resp.request.url.path}"
    msg += f": {_error_message(resp)}"
    if rid:
        msg += f" (request id {rid})"
    if cls is IvoireAuthenticationError:
        msg += f" ({key_hint})"
    kw: dict[str, Any] = {"code": code, "problem": problem, "request_id": rid}
    if issubclass(cls, IvoireRateLimitError):
        raise cls(msg, status, _retry_after(resp), **kw)
    raise cls(msg, status, **kw)


def _split_long(text: str, limit: int = MAX_TEXT_CHARS) -> list[str]:
    """Split ``text`` into pieces of at most ``limit`` characters, at sentence ends if possible."""
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    cur = ""
    for sent in re.split(r"(?<=[.!?…])\s+", text.strip()):
        words = sent.split() if len(sent) > limit else [sent]
        for w in words:
            while len(w) > limit:  # one very long token
                if cur:
                    pieces.append(cur)
                    cur = ""
                pieces.append(w[:limit])
                w = w[limit:]
            if not cur:
                cur = w
            elif len(cur) + 1 + len(w) <= limit:
                cur = f"{cur} {w}"
            else:
                pieces.append(cur)
                cur = w
    if cur:
        pieces.append(cur)
    return pieces


class _TranslatePlan:
    """Texts -> request batches, and the answers back -> :class:`Translation` objects."""

    def __init__(
        self, texts: list[str], source: str, target: str, style: str, wire_style: str | None = None
    ) -> None:
        self.texts, self.source, self.target, self.style = texts, source, target, style
        self.wire_style = wire_style or style
        """The style name sent to the server (see ``RemoteBackend.style_aliases``)."""
        self.owner: list[int] = []
        self.pieces: list[str] = []
        for i, t in enumerate(texts):
            for p in _split_long(t):
                self.pieces.append(p)
                self.owner.append(i)
        self.batches = [
            self.pieces[i : i + MAX_BATCH] for i in range(0, len(self.pieces), MAX_BATCH)
        ]
        self.outs: list[str] = []
        self.pivots: list[str | None] = []
        self.timings: dict[str, int] = {}
        self.request_ids: list[str] = []
        self.models: dict[str, Any] = {}
        self.usage: dict[str, int] = {}

    def payload(self, batch: list[str]) -> dict[str, Any]:
        return {
            "texts": batch,
            "source": self.source,
            "target": self.target,
            "style": self.wire_style,
        }

    def add(self, batch: list[str], d: dict[str, Any], rid: str | None) -> None:
        outs = [str(x) for x in d.get("texts") or []]
        if len(outs) != len(batch):
            raise IvoireServerError(
                f"/v1/translate returned {len(outs)} texts for {len(batch)}", 200, request_id=rid
            )
        self.outs += outs
        pivot = (d.get("pivot") or {}).get("fra")
        self.pivots += [str(p) for p in pivot] if pivot else [None] * len(batch)
        for k, v in (d.get("timings_ms") or {}).items():
            if isinstance(v, (int, float)):
                self.timings[k] = self.timings.get(k, 0) + int(v)
        for k, v in (d.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                self.usage[k] = self.usage.get(k, 0) + int(v)
        if d.get("model"):
            self.models = dict(d["model"])
        rid = rid or d.get("request_id")
        if rid:
            self.request_ids.append(str(rid))

    def results(self) -> list[Translation]:
        n = len(self.texts)
        outs: list[list[str]] = [[] for _ in range(n)]
        pivs: list[list[str]] = [[] for _ in range(n)]
        has_pivot = [False] * n
        for i, o, p in zip(self.owner, self.outs, self.pivots, strict=True):
            outs[i].append(o)
            if p is not None:
                pivs[i].append(p)
                has_pivot[i] = True
        meta: dict[str, Any] = {"model": self.models, "usage": self.usage}
        if self.request_ids:
            meta["request_id"] = self.request_ids[-1]
            if len(self.request_ids) > 1:
                meta["request_ids"] = list(self.request_ids)
        res = []
        for i, src in enumerate(self.texts):
            text = " ".join(outs[i])
            via: tuple[tuple[str, str], ...] = ((self.target, text),)
            if has_pivot[i] and "fra" not in (self.source, self.target):
                via = (("fra", " ".join(pivs[i])), *via)
            res.append(
                Translation(
                    text=text,
                    source=self.source,
                    target=self.target,
                    source_text=src,
                    style=self.style,
                    via=via,
                    timings_ms=dict(self.timings),
                    metadata=dict(meta),
                )
            )
        return res


def _endpoint_missing(e: IvoireAPIError) -> bool:
    """A 404/405 without problem details: the server predates this endpoint."""
    return e.status_code in (404, 405) and e.code is None


class RemoteBackend(IvoireBackend):
    """Calls the hosted Abyshire API (default) or a self-hosted server.

    Args:
        api_key: API key; defaults to the ``IVOIRE_API_KEY`` environment variable. Not needed on a LAN.
        base_url: defaults to ``IVOIRE_BASE_URL`` or ``https://api.abyshire.ci``.
            Any server that speaks this API works: the hosted API's gateway (the default), another
            Abyshire gateway URL you were given, or a self-hosted server (``http://<host>:8090``).
        timeout: seconds per request.
        max_retries: retries for 429/502/503/504 answers and connection errors.
        backoff_factor: first retry delay in seconds (doubles each retry, plus jitter).
        max_backoff: cap on one delay (also caps ``Retry-After``).
        use_phrasebook: let the server answer English->Baoulé from its checked phrasebook
            (only used by the fallback for servers without ``/v1/translate``).
        client: bring your own ``httpx`` clients (proxies, mTLS, testing).
        async_client: See ``client``.
        style_aliases: style -> name sent to the server, for a server that names a style
            differently (default: the ``"remote"`` section of the settings file, else none).
        settings: settings file path, or the settings as a mapping (default: looked up; see
            :mod:`langchain_ivoire._settings`).
    """

    name = "remote"
    _key_hint = _KEY_HINT

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        max_backoff: float = 20.0,
        use_phrasebook: bool = False,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
        style_aliases: Mapping[str, str] | None = None,
        settings: SettingsLike = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("IVOIRE_BASE_URL") or DEFAULT_BASE_URL).rstrip(
            "/"
        )
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        key = api_key if api_key is not None else os.environ.get("IVOIRE_API_KEY")
        self._api_key: str | None = (key or "").strip() or None
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
        self.use_phrasebook = use_phrasebook
        self._client = client
        self._aclient = async_client
        self._own_aclient = async_client is None
        self._aclient_loop: asyncio.AbstractEventLoop | None = None
        if style_aliases is None:
            style_aliases = section(load_settings(settings), "remote").get("style_aliases")
        if style_aliases is not None and not (
            isinstance(style_aliases, Mapping)
            and all(isinstance(k, str) and isinstance(v, str) for k, v in style_aliases.items())
        ):
            raise IvoireConfigError("style_aliases must map strings to strings")
        self.style_aliases: dict[str, str] = dict(style_aliases or {})
        self._has_translate: bool | None = None  # learned from the server (404 -> False)
        self.last_request_id: str | None = None
        """``X-Request-Id`` of the latest response (for support requests)."""

    def __repr__(self) -> str:  # never show the key
        return (
            f"RemoteBackend(base_url={self.base_url!r}, api_key={'***' if self._api_key else None})"
        )

    @property
    def has_api_key(self) -> bool:
        return self._api_key is not None

    # ---- HTTP plumbing -----------------------------------------------------------------------
    def _headers(self, idempotency_key: str | None = None) -> dict[str, str]:
        h = {"User-Agent": f"langchain-ivoire/{__version__}", "Accept": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        if idempotency_key:
            h["Idempotency-Key"] = idempotency_key
        return h

    def _url(self, path: str) -> str:
        return self.base_url + "/" + path.lstrip("/")

    def _v1(self, path: str) -> str:
        """Path of a ``/v1`` endpoint, whether or not ``base_url`` already ends in ``/v1``."""
        return path if self.base_url.endswith("/v1") else "/v1" + path

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout, follow_redirects=False)
        return self._client

    @property
    def async_client(self) -> httpx.AsyncClient:
        """The async client; an owned one is rebuilt when used from a new event loop (an
        ``httpx.AsyncClient`` cannot outlive the loop it was first used in, e.g. across
        ``asyncio.run`` calls)."""
        try:
            loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        stale = self._own_aclient and loop is not None and self._aclient_loop is not loop
        if self._aclient is None or stale:
            self._aclient = httpx.AsyncClient(timeout=self.timeout, follow_redirects=False)
            self._own_aclient = True
            self._aclient_loop = loop
        return self._aclient

    def _delay(self, attempt: int, resp: httpx.Response | None) -> float:
        ra = _retry_after(resp) if resp is not None else None
        if ra is not None:
            return float(min(ra, self.max_backoff))
        base = self.backoff_factor * (2.0**attempt)
        return min(base + random.uniform(0, self.backoff_factor), self.max_backoff)

    def _finish(self, resp: httpx.Response) -> httpx.Response:
        self.last_request_id = _request_id(resp) or self.last_request_id
        _raise_for_status(resp, self._key_hint)
        return resp

    def _request(self, method: str, path: str, **kw: Any) -> httpx.Response:
        idem = str(uuid.uuid4()) if method == "POST" else None
        for attempt in range(self.max_retries + 1):
            resp: httpx.Response | None = None
            try:
                resp = self.client.request(
                    method, self._url(path), headers=self._headers(idem), **kw
                )
            except httpx.TransportError as e:
                if attempt >= self.max_retries:
                    raise IvoireConnectionError(
                        f"cannot reach {self.base_url} ({type(e).__name__})"
                    ) from None
            else:
                if resp.status_code not in _RETRY_STATUS or attempt >= self.max_retries:
                    return self._finish(resp)
            time.sleep(self._delay(attempt, resp))
        raise AssertionError("unreachable")  # pragma: no cover

    async def _arequest(self, method: str, path: str, **kw: Any) -> httpx.Response:
        idem = str(uuid.uuid4()) if method == "POST" else None
        for attempt in range(self.max_retries + 1):
            resp: httpx.Response | None = None
            try:
                resp = await self.async_client.request(
                    method, self._url(path), headers=self._headers(idem), **kw
                )
            except httpx.TransportError as e:
                if attempt >= self.max_retries:
                    raise IvoireConnectionError(
                        f"cannot reach {self.base_url} ({type(e).__name__})"
                    ) from None
            else:
                if resp.status_code not in _RETRY_STATUS or attempt >= self.max_retries:
                    return self._finish(resp)
            await asyncio.sleep(self._delay(attempt, resp))
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _check_lang(lang: str) -> None:
        if lang != "bci":
            raise IvoireUnsupportedError(f"the remote API only serves Baoulé ('bci'), not {lang!r}")

    # ---- service info ------------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        r: dict[str, Any] = self._request("GET", "/health").json()
        return r

    async def ahealth(self) -> dict[str, Any]:
        r: dict[str, Any] = (await self._arequest("GET", "/health")).json()
        return r

    def languages(self) -> list[dict[str, Any]]:
        """``GET /v1/languages``: each language with ``asr``, ``tts``, ``mt`` and ``styles``."""
        d = self._request("GET", self._v1("/languages")).json()
        return list(d.get("languages") or [])

    async def alanguages(self) -> list[dict[str, Any]]:
        d = (await self._arequest("GET", self._v1("/languages"))).json()
        return list(d.get("languages") or [])

    def models(self) -> dict[str, Any]:
        """``GET /v1/models``: the model names and versions the server runs."""
        r: dict[str, Any] = self._request("GET", self._v1("/models")).json()
        return r

    async def amodels(self) -> dict[str, Any]:
        r: dict[str, Any] = (await self._arequest("GET", self._v1("/models"))).json()
        return r

    # ---- speech ------------------------------------------------------------------------------
    @staticmethod
    def _files(audio: bytes) -> dict[str, tuple[str, bytes, str]]:
        fname, mime = sniff_media_type(audio)
        return {"audio": (fname, audio, mime)}

    @staticmethod
    def _transcription(d: dict[str, Any], audio: bytes, lang: str) -> Transcription:
        heard = d.get("heard_bci") or d.get("bci") or ""
        info = wav_info(audio)
        meta: dict[str, Any] = {
            "server_route": d.get("route"),
            "timings_ms": d.get("timings_ms"),
            "translations": {"fra": d.get("fra", ""), "eng": d.get("eng", "")},
        }
        if d.get("usage"):
            meta["usage"] = d["usage"]
        if d.get("phrase"):  # the server snapped the utterance to a phrasebook entry
            meta["server_phrase"] = d["phrase"]
            meta["server_text"] = d.get("bci", "")
        conf = d.get("asr_confidence")
        return Transcription(
            text=heard,
            lang=lang,
            confidence=float(conf) if conf is not None else None,
            segments=(heard,) if heard else (),
            audio_s=d.get("audio_s")
            if d.get("audio_s") is not None
            else (info[1] if info else None),
            low_confidence=bool(d.get("low_confidence")),
            metadata=meta,
        )

    def transcribe(self, audio: bytes, *, lang: str) -> Transcription:
        self._check_lang(lang)
        d = self._request("POST", "/listen_bci", files=self._files(audio)).json()
        return self._transcription(d, audio, lang)

    async def atranscribe(self, audio: bytes, *, lang: str) -> Transcription:
        self._check_lang(lang)
        d = (await self._arequest("POST", "/listen_bci", files=self._files(audio))).json()
        return self._transcription(d, audio, lang)

    def _speech(self, d: dict[str, Any], wav: bytes, text: str, lang: str) -> Speech:
        info = wav_info(wav)
        return Speech(
            audio=wav,
            text=d.get("bci", text),
            lang=lang,
            sample_rate=info[0] if info else None,
            audio_s=info[1] if info else d.get("audio_s"),
            metadata={k: d[k] for k in ("timings_ms", "phrase", "fra", "eng", "usage") if k in d},
        )

    def synthesize(self, text: str, *, lang: str) -> Speech:
        self._check_lang(lang)
        d = self._request("POST", "/tts_bci", json={"bci": text}).json()
        wav = self._request("GET", d["audio_url"]).content
        return self._speech(d, wav, text, lang)

    async def asynthesize(self, text: str, *, lang: str) -> Speech:
        self._check_lang(lang)
        d = (await self._arequest("POST", "/tts_bci", json={"bci": text})).json()
        wav = (await self._arequest("GET", d["audio_url"])).content
        return self._speech(d, wav, text, lang)

    def speak_english(self, text: str) -> dict[str, Any]:
        """``POST /speak_bci``: English text -> French, Baoulé text and an ``audio_url``."""
        r: dict[str, Any] = self._request(
            "POST", "/speak_bci", json={"text": text, "use_phrasebook": self.use_phrasebook}
        ).json()
        return r

    def listen_english(self, audio: bytes, translate: bool = True) -> dict[str, Any]:
        """``POST /listen_eng``: English speech -> text (and Baoulé, if ``translate``)."""
        r: dict[str, Any] = self._request(
            "POST", "/listen_eng", files=self._files(audio), data={"translate": int(translate)}
        ).json()
        return r

    def agent_text(
        self, *, fra: str | None = None, eng: str | None = None, conv: str | None = None
    ) -> dict[str, Any]:
        """``POST /agent_text``: the server's own LLM agent (French or English in; Baoulé reply)."""
        if not (fra or eng):
            raise ValueError("pass fra= or eng=")
        payload: dict[str, Any] = {"fra": fra} if fra else {"eng": eng}
        if conv:
            payload["conv"] = conv
        r: dict[str, Any] = self._request("POST", "/agent_text", json=payload).json()
        return r

    def fetch_audio(self, audio_url: str) -> bytes:
        """Download a generated clip (``audio_url`` values are relative to the base URL)."""
        return self._request("GET", audio_url).content

    # ---- translation -------------------------------------------------------------------------
    def _fallback(
        self, texts: list[str], source: str, target: str, style: str
    ) -> list[Translation]:
        """Servers without ``/v1/translate``: English -> French/Baoulé via ``/speak_bci`` (everyday)."""
        if source == "eng" and target in ("fra", "bci") and style == "everyday":
            res = []
            for t in texts:
                d = self.speak_english(t)
                via: list[tuple[str, str]] = [("fra", d.get("fra", ""))]
                if target == "bci":
                    via.append(("bci", d.get("bci", "")))
                res.append(
                    Translation(
                        text=via[-1][1],
                        source=source,
                        target=target,
                        source_text=t,
                        style=style,
                        via=tuple(via),
                        timings_ms=dict(d.get("timings_ms") or {}),
                    )
                )
            return res
        raise IvoireUnsupportedError(
            f"the server at {self.base_url} has no POST /v1/translate endpoint, so "
            f"{source}->{target} text translation is not available remotely (only "
            "English->French/Baoulé via /speak_bci). Use a server version with /v1/translate "
            "(the hosted API has it)."
        )

    def translate(
        self, texts: Sequence[str], *, source: str, target: str, style: str = "everyday"
    ) -> list[Translation]:
        """Translate ``texts`` with ``POST /v1/translate`` (batched, long texts split)."""
        texts = list(texts)
        pivot_path(source, target)  # validates source != target
        if not texts:
            return []
        if self._has_translate is not False:
            plan = _TranslatePlan(texts, source, target, style, self.style_aliases.get(style))
            try:
                for batch in plan.batches:
                    r = self._request("POST", self._v1("/translate"), json=plan.payload(batch))
                    plan.add(batch, r.json(), _request_id(r))
            except IvoireAPIError as e:
                if not _endpoint_missing(e):
                    raise
                self._has_translate = False
            else:
                self._has_translate = True
                return plan.results()
        return self._fallback(texts, source, target, style)

    async def atranslate(
        self, texts: Sequence[str], *, source: str, target: str, style: str = "everyday"
    ) -> list[Translation]:
        texts = list(texts)
        pivot_path(source, target)
        if not texts:
            return []
        if self._has_translate is not False:
            plan = _TranslatePlan(texts, source, target, style, self.style_aliases.get(style))
            try:
                for batch in plan.batches:
                    r = await self._arequest(
                        "POST", self._v1("/translate"), json=plan.payload(batch)
                    )
                    plan.add(batch, r.json(), _request_id(r))
            except IvoireAPIError as e:
                if not _endpoint_missing(e):
                    raise
                self._has_translate = False
            else:
                self._has_translate = True
                return plan.results()
        return await asyncio.to_thread(self._fallback, texts, source, target, style)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        self.close()
        if self._aclient is not None:
            await self._aclient.aclose()
            self._aclient = None
