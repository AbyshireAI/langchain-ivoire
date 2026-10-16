"""Backend interface: where the models actually run (the hosted API, or a fake for tests)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import itertools
import time
from collections.abc import Coroutine, Sequence
from typing import Any, TypeVar

from langchain_ivoire._languages import LANGUAGES, PIVOT_LANGUAGES
from langchain_ivoire.exceptions import IvoireUnsupportedError
from langchain_ivoire.types import Speech, Transcription, Translation

__all__ = ["STYLES", "IvoireBackend", "pivot_path", "run_sync"]

STYLES: tuple[str, ...] = ("everyday", "tale", "formal")
"""Translation registers: everyday speech (default), storytelling, and a formal register."""

T = TypeVar("T")


def pivot_path(source: str, target: str) -> list[str]:
    """Hops for a direction: one hop if French is an endpoint, else ``[source, "fra", target]``."""
    if source == target:
        raise ValueError(f"source and target are the same language ({source!r})")
    return [source, target] if "fra" in (source, target) else [source, "fra", target]


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine to completion from sync code (in a helper thread if a loop is running)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


def _ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


class IvoireBackend:
    """Runs speech recognition, translation and speech synthesis for the components.

    Async-first: a subclass implements, for each capability, **either** the async method
    (:meth:`atranscribe`, :meth:`asynthesize`, :meth:`atranslate_hop`) **or** the sync one
    (:meth:`transcribe`, :meth:`synthesize`, :meth:`translate_hop`); the other is derived (async
    from sync in a worker thread, sync from async with :func:`run_sync`). A backend that can
    translate a whole direction at once may override :meth:`translate`/:meth:`atranslate` instead
    of the per-hop methods (English <-> Ivorian language otherwise pivots through French).
    """

    name: str = "base"

    def _overrides(self, method: str) -> bool:
        return getattr(type(self), method) is not getattr(IvoireBackend, method)

    def _missing(self, what: str) -> IvoireUnsupportedError:
        return IvoireUnsupportedError(f"{type(self).__name__} does not implement {what}")

    # ---- speech -> text ----------------------------------------------------------------------
    def transcribe(self, audio: bytes, *, lang: str) -> Transcription:
        """Recognise speech in ``lang`` from encoded audio bytes (wav, ogg, webm, m4a, ...)."""
        if self._overrides("atranscribe"):
            return run_sync(self.atranscribe(audio, lang=lang))
        raise self._missing("transcribe()")

    async def atranscribe(self, audio: bytes, *, lang: str) -> Transcription:
        if self._overrides("transcribe"):
            return await asyncio.to_thread(self.transcribe, audio, lang=lang)
        raise self._missing("atranscribe()")

    # ---- text -> speech ----------------------------------------------------------------------
    def synthesize(self, text: str, *, lang: str) -> Speech:
        """Speak ``text`` (in ``lang``); returns WAV bytes."""
        if self._overrides("asynthesize"):
            return run_sync(self.asynthesize(text, lang=lang))
        raise self._missing("synthesize()")

    async def asynthesize(self, text: str, *, lang: str) -> Speech:
        if self._overrides("synthesize"):
            return await asyncio.to_thread(self.synthesize, text, lang=lang)
        raise self._missing("asynthesize()")

    # ---- text -> text ------------------------------------------------------------------------
    def translate_hop(
        self, texts: Sequence[str], *, source: str, target: str, style: str
    ) -> list[str]:
        """Translate one hop (``"fra"`` is ``source`` or ``target``)."""
        if self._overrides("atranslate_hop"):
            return run_sync(self.atranslate_hop(texts, source=source, target=target, style=style))
        raise self._missing("translate_hop()")

    async def atranslate_hop(
        self, texts: Sequence[str], *, source: str, target: str, style: str
    ) -> list[str]:
        if self._overrides("translate_hop"):
            return await asyncio.to_thread(
                self.translate_hop, list(texts), source=source, target=target, style=style
            )
        raise self._missing("atranslate_hop()")

    @staticmethod
    def _assemble(
        texts: list[str],
        outs: list[str],
        vias: list[list[tuple[str, str]]],
        timings: dict[str, int],
        source: str,
        target: str,
        style: str,
    ) -> list[Translation]:
        return [
            Translation(
                text=out,
                source=source,
                target=target,
                source_text=src,
                style=style,
                via=tuple(v),
                timings_ms=dict(timings),
            )
            for src, out, v in zip(texts, outs, vias, strict=True)
        ]

    def translate(
        self, texts: Sequence[str], *, source: str, target: str, style: str = "everyday"
    ) -> list[Translation]:
        """Translate ``texts``; English <-> Ivorian language pivots through French."""
        texts = list(texts)
        if not texts:
            return []
        if self._overrides("atranslate") and not self._overrides("translate_hop"):
            return run_sync(self.atranslate(texts, source=source, target=target, style=style))
        cur = texts
        vias: list[list[tuple[str, str]]] = [[] for _ in texts]
        timings: dict[str, int] = {}
        for a, b in itertools.pairwise(pivot_path(source, target)):
            t0 = time.perf_counter()
            cur = self.translate_hop(cur, source=a, target=b, style=style)
            timings[f"mt_{a}_{b}"] = _ms(t0)
            for v, t in zip(vias, cur, strict=True):
                v.append((b, t))
        return self._assemble(texts, cur, vias, timings, source, target, style)

    async def atranslate(
        self, texts: Sequence[str], *, source: str, target: str, style: str = "everyday"
    ) -> list[Translation]:
        texts = list(texts)
        if not texts:
            return []
        if self._overrides("translate") and not self._overrides("atranslate_hop"):
            return await asyncio.to_thread(
                self.translate, texts, source=source, target=target, style=style
            )
        cur = texts
        vias: list[list[tuple[str, str]]] = [[] for _ in texts]
        timings: dict[str, int] = {}
        for a, b in itertools.pairwise(pivot_path(source, target)):
            t0 = time.perf_counter()
            cur = await self.atranslate_hop(cur, source=a, target=b, style=style)
            timings[f"mt_{a}_{b}"] = _ms(t0)
            for v, t in zip(vias, cur, strict=True):
                v.append((b, t))
        return self._assemble(texts, cur, vias, timings, source, target, style)

    # ---- text -> vectors (optional) ----------------------------------------------------------
    def embed(self, texts: Sequence[str], *, kind: str = "document") -> list[list[float]]:
        """Embed ``texts`` (Baoulé or French) into one shared vector space.

        ``kind`` is ``"document"`` for texts you index and ``"query"`` for search questions.
        Optional: only backends that serve search embeddings implement it.
        """
        if self._overrides("aembed"):
            return run_sync(self.aembed(texts, kind=kind))
        raise self._missing("embed()")

    async def aembed(self, texts: Sequence[str], *, kind: str = "document") -> list[list[float]]:
        if self._overrides("embed"):
            return await asyncio.to_thread(self.embed, list(texts), kind=kind)
        raise self._missing("aembed()")

    # ---- service info ------------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        """Backend status (never includes secrets)."""
        return {"ok": True, "backend": self.name}

    def languages(self) -> list[dict[str, Any]]:
        """Languages this backend serves, in the ``GET /v1/languages`` shape."""
        return [
            {
                "code": lg.code,
                "name": lg.name,
                "asr": True,
                "tts": True,
                "mt": list(PIVOT_LANGUAGES),
                "styles": list(STYLES),
            }
            for lg in LANGUAGES.values()
            if lg.status == "available"
        ]

    def models(self) -> dict[str, Any]:
        """Models behind this backend (``GET /v1/models`` for the remote one)."""
        return {"backend": self.name}

    def close(self) -> None:
        """Release network clients / models."""

    async def ahealth(self) -> dict[str, Any]:
        return await asyncio.to_thread(self.health)

    async def alanguages(self) -> list[dict[str, Any]]:
        return self.languages()

    async def amodels(self) -> dict[str, Any]:
        return self.models()

    async def aclose(self) -> None:
        self.close()
