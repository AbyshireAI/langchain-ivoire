"""OpenAI-compatible client for the Baoulé models: through OpenRouter, or the hosted API's own gateway.

The hosted API also speaks the OpenAI wire format, so the same four services can be reached through
OpenRouter (one OpenRouter key for every model you use there) or directly at Abyshire's gateway.

=================================  ==========================  ==================================
Endpoint                           Component                   Default model id
=================================  ==========================  ==================================
``POST /chat/completions``         :class:`IvoireTranslator`   ``abyshire/translate-baoule``
``POST /audio/transcriptions``     :class:`IvoireTranscriber`  ``abyshire/listen-baoule``
``POST /audio/speech``             :class:`IvoireSpeaker`      ``abyshire/speak-baoule``
``POST /embeddings``               :class:`IvoireEmbeddings`   ``abyshire/embed-baoule``
=================================  ==========================  ==================================

**Translation convention.** A chat completion translates the text of its single ``user`` message.
The direction and style travel twice, so they survive any router in between:

* a ``system`` message ``"ivoire-translate source=<code> target=<code> style=<style>"``, and
* a top-level ``"translation": {"source": ..., "target": ..., "style": ...}`` request field.

The server reads the ``translation`` field first and the system message otherwise. Codes are
``"bci"``, ``"fra"`` and ``"eng"``; styles are ``"everyday"``, ``"tale"`` and ``"formal"``. The
reply's ``choices[0].message.content`` is the translation, and nothing else.

**Transcription and speech.** ``language`` is the ISO 639-3 code (``"bci"``); speech is requested
as ``response_format="wav"``. Embedding requests carry ``input_type`` (``"query"`` for questions,
``"document"`` for texts you index).

**Access.** Translation is the public product: with OpenRouter's base URL only the translator works
out of the box. Speech recognition, the voice and embeddings are available **by approval**: through
the Abyshire API with an approved key (``base_url=`` the hosted API URL, ``IVOIRE_API_KEY``), or
through OpenRouter only for organisations that were granted the model, by passing its model id
explicitly (``listen_model=``, ``speak_model=``, ``embed_model=``). Otherwise those calls raise
:class:`~langchain_ivoire.IvoireApprovalRequiredError` before anything is sent.

**Keys.** With OpenRouter's base URL the key comes from ``api_key=`` or ``OPENROUTER_API_KEY``. With
any other base URL (the hosted gateway, or your own) it comes from ``api_key=`` or
``IVOIRE_API_KEY``, so an OpenRouter key is never sent anywhere but OpenRouter. Keys are never
logged, printed or included in exceptions.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, NoReturn
from urllib.parse import urlsplit

import httpx

from langchain_ivoire._audio import sniff_media_type, wav_info
from langchain_ivoire.backends.base import STYLES, pivot_path
from langchain_ivoire.backends.remote import RemoteBackend, _request_id, _split_long
from langchain_ivoire.exceptions import (
    IvoireApprovalRequiredError,
    IvoireServerError,
    IvoireUnsupportedError,
)
from langchain_ivoire.types import Speech, Transcription, Translation

__all__ = [
    "EMBED_MODEL",
    "LISTEN_MODEL",
    "OPENROUTER_BASE_URL",
    "REQUEST_ACCESS",
    "SPEAK_MODEL",
    "TRANSLATE_MODEL",
    "OpenRouterBackend",
    "translation_instruction",
]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TRANSLATE_MODEL = "abyshire/translate-baoule"
LISTEN_MODEL = "abyshire/listen-baoule"
SPEAK_MODEL = "abyshire/speak-baoule"
EMBED_MODEL = "abyshire/embed-baoule"
EMBED_BATCH = 64
"""Most texts sent in one ``/embeddings`` request."""
EMBED_KINDS = ("document", "query")
_OPENROUTER_HOST = "openrouter.ai"
REQUEST_ACCESS = "request access from Abyshire (api@abyshire.co.uk)"
_APPROVAL_ONLY = {
    "listen_model": "Baoulé speech recognition (abyshire/listen-baoule)",
    "speak_model": "the Baoulé voice (abyshire/speak-baoule)",
    "embed_model": "Baoulé search embeddings (abyshire/embed-baoule)",
}


def translation_instruction(source: str, target: str, style: str) -> str:
    """The ``system`` message that tells the server the direction and style."""
    return f"ivoire-translate source={source} target={target} style={style}"


def _is_openrouter(base_url: str) -> bool:
    host = (urlsplit(base_url).hostname or "").lower()
    return host == _OPENROUTER_HOST or host.endswith("." + _OPENROUTER_HOST)


def _add_usage(total: dict[str, int], usage: Any) -> None:
    if isinstance(usage, dict):
        for k, v in usage.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                total[k] = total.get(k, 0) + int(v)


class OpenRouterBackend(RemoteBackend):
    """Calls the Baoulé models over the OpenAI-compatible API, through OpenRouter by default.

    Example:
        .. code-block:: python

            from langchain_ivoire import IvoireTranslator

            # OPENROUTER_API_KEY is set in the environment
            to_bci = IvoireTranslator(source="fra", target="bci", backend="openrouter")

    Args:
        api_key: defaults to ``OPENROUTER_API_KEY`` when ``base_url`` is OpenRouter's, else to
            ``IVOIRE_API_KEY``.
        base_url: defaults to ``https://openrouter.ai/api/v1``. Point it at the hosted API's
            gateway (or your own OpenAI-compatible deployment) to call it directly.
        translate_model: model id for translation (``abyshire/translate-baoule``).
        listen_model: model id for speech recognition (``abyshire/listen-baoule``). Approval
            only: through OpenRouter it must be given explicitly (a model id your organisation
            was granted); see *Access* above.
        speak_model: model id for speech synthesis (``abyshire/speak-baoule``). Approval only,
            as ``listen_model``.
        embed_model: model id for search embeddings (``abyshire/embed-baoule``). Approval only,
            as ``listen_model``.
        voice: the ``voice`` sent with speech requests.
        app_url: optional ``HTTP-Referer`` header (OpenRouter uses it to attribute your app).
        app_title: optional ``X-Title`` header (the same, by name).
        **kwargs: the remaining :class:`RemoteBackend` options (``timeout``, ``max_retries``,
            ``backoff_factor``, ``max_backoff``, ``client``, ``async_client``).
    """

    name = "openrouter"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        translate_model: str = TRANSLATE_MODEL,
        listen_model: str | None = None,
        speak_model: str | None = None,
        embed_model: str | None = None,
        voice: str = "default",
        app_url: str | None = None,
        app_title: str | None = None,
        **kwargs: Any,
    ) -> None:
        url = (base_url or OPENROUTER_BASE_URL).rstrip("/")
        env = "OPENROUTER_API_KEY" if _is_openrouter(url) else "IVOIRE_API_KEY"
        key = api_key if api_key is not None else os.environ.get(env)
        kwargs.setdefault("style_aliases", {})  # the OpenAI-compatible API uses the public names
        super().__init__(api_key=key or "", base_url=url, **kwargs)
        self._key_hint = f"set {env} or pass api_key="
        self.via_openrouter = _is_openrouter(url)
        """True when ``base_url`` is OpenRouter's (approval-only services then need a model id)."""
        given = {
            "listen_model": listen_model,
            "speak_model": speak_model,
            "embed_model": embed_model,
        }
        self._explicit = {k for k, v in given.items() if v is not None}
        for label, model in (("translate_model", translate_model), *given.items()):
            if model is not None and (not isinstance(model, str) or not model.strip()):
                raise ValueError(f"{label} must be a non-empty model id")
        self.translate_model = translate_model
        self.listen_model = listen_model or LISTEN_MODEL
        self.speak_model = speak_model or SPEAK_MODEL
        self.embed_model = embed_model or EMBED_MODEL
        self.voice = voice
        self.app_url = app_url
        self.app_title = app_title

    def __repr__(self) -> str:  # never show the key
        return (
            f"OpenRouterBackend(base_url={self.base_url!r}, "
            f"api_key={'***' if self._api_key else None})"
        )

    def _headers(self, idempotency_key: str | None = None) -> dict[str, str]:
        h = super()._headers(idempotency_key)
        h["Accept"] = "application/json, audio/*;q=0.9"
        if self.app_url:
            h["HTTP-Referer"] = self.app_url
        if self.app_title:
            h["X-Title"] = self.app_title
        return h

    # ---- service info ------------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        """Local status only (an OpenAI-compatible API has no health route)."""
        return {"ok": True, "backend": self.name, "base_url": self.base_url}

    async def ahealth(self) -> dict[str, Any]:
        return self.health()

    def languages(self) -> list[dict[str, Any]]:
        return super(RemoteBackend, self).languages()

    async def alanguages(self) -> list[dict[str, Any]]:
        return self.languages()

    def _check_access(self, option: str) -> None:
        """Approval-only services: through OpenRouter, only with an explicitly configured model id."""
        if self.via_openrouter and option not in self._explicit:
            raise IvoireApprovalRequiredError(
                f"{_APPROVAL_ONLY[option]} is available by approval through the Abyshire API, "
                f"not publicly on OpenRouter; {REQUEST_ACCESS}. With an approved key, use "
                "backend='remote' or OpenRouterBackend(base_url=<the Abyshire API URL>). If your "
                f"organisation was granted the model on OpenRouter, pass {option}='<model id>'."
            )

    def _legacy(self, what: str) -> NoReturn:
        raise IvoireUnsupportedError(
            f"{what} is a native-API endpoint; the OpenAI-compatible API does not serve it "
            "(use RemoteBackend)"
        )

    def speak_english(self, text: str) -> dict[str, Any]:
        self._legacy("/speak_bci")

    def listen_english(self, audio: bytes, translate: bool = True) -> dict[str, Any]:
        self._legacy("/listen_eng")

    def agent_text(
        self, *, fra: str | None = None, eng: str | None = None, conv: str | None = None
    ) -> dict[str, Any]:
        self._legacy("/agent_text")

    def fetch_audio(self, audio_url: str) -> bytes:
        self._legacy("audio_url downloads")

    # ---- translation: POST /chat/completions -------------------------------------------------
    def _chat_payload(self, text: str, source: str, target: str, style: str) -> dict[str, Any]:
        return {
            "model": self.translate_model,
            "messages": [
                {"role": "system", "content": translation_instruction(source, target, style)},
                {"role": "user", "content": text},
            ],
            "translation": {"source": source, "target": target, "style": style},
            "stream": False,
        }

    @staticmethod
    def _chat_text(resp: httpx.Response) -> tuple[str, dict[str, Any]]:
        d = resp.json()
        try:
            content = d["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = None
        if not isinstance(content, str):
            raise IvoireServerError(
                "chat completion without a text reply",
                resp.status_code,
                request_id=_request_id(resp),
            )
        return content.strip(), d

    def _check_translate(self, source: str, target: str, style: str) -> None:
        pivot_path(source, target)  # validates source != target
        if style not in STYLES:
            raise ValueError(f"unknown style {style!r}; choose one of {', '.join(STYLES)}")

    def _translation(
        self,
        src: str,
        outs: list[str],
        ds: list[dict[str, Any]],
        rids: list[str],
        *,
        source: str,
        target: str,
        style: str,
    ) -> Translation:
        usage: dict[str, int] = {}
        for d in ds:
            _add_usage(usage, d.get("usage"))
        meta: dict[str, Any] = {"model": ds[-1].get("model") or self.translate_model}
        meta["usage"] = usage
        if rids:
            meta["request_id"] = rids[-1]
        text = " ".join(outs)
        return Translation(
            text=text,
            source=source,
            target=target,
            source_text=src,
            style=style,
            via=((target, text),),
            metadata=meta,
        )

    def translate(
        self, texts: Sequence[str], *, source: str, target: str, style: str = "everyday"
    ) -> list[Translation]:
        """One chat completion per text (long texts are split at sentence ends, then joined)."""
        texts = list(texts)
        self._check_translate(source, target, style)
        res = []
        for t in texts:
            outs: list[str] = []
            ds: list[dict[str, Any]] = []
            rids: list[str] = []
            for piece in _split_long(t):
                r = self._request(
                    "POST",
                    self._v1("/chat/completions"),
                    json=self._chat_payload(piece, source, target, style),
                )
                out, d = self._chat_text(r)
                outs.append(out)
                ds.append(d)
                rid = _request_id(r) or d.get("id")
                if rid:
                    rids.append(str(rid))
            res.append(
                self._translation(t, outs, ds, rids, source=source, target=target, style=style)
            )
        return res

    async def atranslate(
        self, texts: Sequence[str], *, source: str, target: str, style: str = "everyday"
    ) -> list[Translation]:
        texts = list(texts)
        self._check_translate(source, target, style)
        res = []
        for t in texts:
            outs: list[str] = []
            ds: list[dict[str, Any]] = []
            rids: list[str] = []
            for piece in _split_long(t):
                r = await self._arequest(
                    "POST",
                    self._v1("/chat/completions"),
                    json=self._chat_payload(piece, source, target, style),
                )
                out, d = self._chat_text(r)
                outs.append(out)
                ds.append(d)
                rid = _request_id(r) or d.get("id")
                if rid:
                    rids.append(str(rid))
            res.append(
                self._translation(t, outs, ds, rids, source=source, target=target, style=style)
            )
        return res

    # ---- speech -> text: POST /audio/transcriptions ------------------------------------------
    def _transcribe_form(self, audio: bytes, lang: str) -> dict[str, Any]:
        fname, mime = sniff_media_type(audio)
        return {
            "files": {"file": (fname, audio, mime)},
            "data": {"model": self.listen_model, "language": lang, "response_format": "json"},
        }

    def _to_transcription(self, resp: httpx.Response, audio: bytes, lang: str) -> Transcription:
        d = resp.json()
        text = str(d.get("text") or "").strip()
        info = wav_info(audio)
        dur = d.get("duration")
        conf = d.get("confidence")
        meta: dict[str, Any] = {"model": d.get("model") or self.listen_model}
        if d.get("usage"):
            meta["usage"] = d["usage"]
        rid = _request_id(resp)
        if rid:
            meta["request_id"] = rid
        return Transcription(
            text=text,
            lang=lang,
            confidence=float(conf) if isinstance(conf, (int, float)) else None,
            segments=(text,) if text else (),
            audio_s=float(dur) if isinstance(dur, (int, float)) else (info[1] if info else None),
            low_confidence=bool(d.get("low_confidence")),
            metadata=meta,
        )

    def transcribe(self, audio: bytes, *, lang: str) -> Transcription:
        self._check_lang(lang)
        self._check_access("listen_model")
        r = self._request(
            "POST", self._v1("/audio/transcriptions"), **self._transcribe_form(audio, lang)
        )
        return self._to_transcription(r, audio, lang)

    async def atranscribe(self, audio: bytes, *, lang: str) -> Transcription:
        self._check_lang(lang)
        self._check_access("listen_model")
        r = await self._arequest(
            "POST", self._v1("/audio/transcriptions"), **self._transcribe_form(audio, lang)
        )
        return self._to_transcription(r, audio, lang)

    # ---- text -> speech: POST /audio/speech --------------------------------------------------
    def _speech_payload(self, text: str) -> dict[str, Any]:
        return {
            "model": self.speak_model,
            "input": text,
            "voice": self.voice,
            "response_format": "wav",
        }

    def _to_speech(self, resp: httpx.Response, text: str, lang: str) -> Speech:
        ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype in ("application/json", "application/problem+json"):
            raise IvoireServerError(
                "speech request answered with JSON instead of audio",
                resp.status_code,
                request_id=_request_id(resp),
            )
        audio = resp.content
        info = wav_info(audio)
        meta: dict[str, Any] = {"model": self.speak_model}
        rid = _request_id(resp)
        if rid:
            meta["request_id"] = rid
        return Speech(
            audio=audio,
            text=text,
            lang=lang,
            sample_rate=info[0] if info else None,
            audio_s=info[1] if info else None,
            media_type=ctype if ctype.startswith("audio/") else "audio/wav",
            metadata=meta,
        )

    def synthesize(self, text: str, *, lang: str) -> Speech:
        self._check_lang(lang)
        self._check_access("speak_model")
        r = self._request("POST", self._v1("/audio/speech"), json=self._speech_payload(text))
        return self._to_speech(r, text, lang)

    async def asynthesize(self, text: str, *, lang: str) -> Speech:
        self._check_lang(lang)
        self._check_access("speak_model")
        r = await self._arequest("POST", self._v1("/audio/speech"), json=self._speech_payload(text))
        return self._to_speech(r, text, lang)

    # ---- text -> vectors: POST /embeddings ---------------------------------------------------
    def _embed_batches(self, texts: Sequence[str], kind: str) -> list[dict[str, Any]]:
        self._check_access("embed_model")
        if kind not in EMBED_KINDS:
            raise ValueError(f"kind must be one of {', '.join(EMBED_KINDS)}, not {kind!r}")
        texts = list(texts)
        return [
            {
                "model": self.embed_model,
                "input": texts[i : i + EMBED_BATCH],
                "encoding_format": "float",
                "input_type": kind,
            }
            for i in range(0, len(texts), EMBED_BATCH)
        ]

    @staticmethod
    def _vectors(resp: httpx.Response, n: int) -> list[list[float]]:
        rows = resp.json().get("data") or []
        try:
            ordered = sorted(rows, key=lambda r: int(r.get("index", 0)))
            vecs = [[float(x) for x in r["embedding"]] for r in ordered]
        except (KeyError, TypeError, ValueError):
            vecs = []
        if len(vecs) != n:
            raise IvoireServerError(
                f"/embeddings returned {len(vecs)} vectors for {n} texts",
                resp.status_code,
                request_id=_request_id(resp),
            )
        return vecs

    def embed(self, texts: Sequence[str], *, kind: str = "document") -> list[list[float]]:
        out: list[list[float]] = []
        for payload in self._embed_batches(texts, kind):
            r = self._request("POST", self._v1("/embeddings"), json=payload)
            out += self._vectors(r, len(payload["input"]))
        return out

    async def aembed(self, texts: Sequence[str], *, kind: str = "document") -> list[list[float]]:
        out: list[list[float]] = []
        for payload in self._embed_batches(texts, kind):
            r = await self._arequest("POST", self._v1("/embeddings"), json=payload)
            out += self._vectors(r, len(payload["input"]))
        return out
