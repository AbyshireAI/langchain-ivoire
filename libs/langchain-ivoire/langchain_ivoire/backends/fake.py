"""Deterministic in-memory backend for tests, examples and offline development (no models, no network)."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_ivoire._audio import silent_wav, wav_info
from langchain_ivoire.backends.base import IvoireBackend
from langchain_ivoire.types import Speech, Transcription

__all__ = ["FakeBackend"]


class FakeBackend(IvoireBackend):
    """Scripted backend.

    Args:
        transcripts: what :meth:`transcribe` returns. A sequence is returned in order (cycling), a
            mapping is looked up by the audio bytes (or their sha256 hex digest).
        translations: ``{(source, target, text): translation}`` per hop. Unknown texts come back
            as ``"[source>target] text"`` so tests can see every hop.
        confidence: ASR confidence to report (a sequence is used in order, cycling).

    Every call is recorded in :attr:`calls` as ``(method, args)``.
    """

    name = "fake"

    def __init__(
        self,
        transcripts: Sequence[str] | Mapping[Any, str] | None = None,
        translations: Mapping[tuple[str, str, str], str] | None = None,
        confidence: float | Sequence[float] = 0.95,
        sample_rate: int = 16000,
    ) -> None:
        self.transcripts = transcripts if transcripts is not None else ["bla di like"]
        self.translations = dict(translations or {})
        self.confidence = confidence
        self.sample_rate = sample_rate
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._n_asr = 0

    def transcribe(self, audio: bytes, *, lang: str) -> Transcription:
        self.calls.append(("transcribe", {"bytes": len(audio), "lang": lang}))
        if isinstance(self.transcripts, Mapping):
            key = hashlib.sha256(audio).hexdigest()
            text = self.transcripts.get(audio, self.transcripts.get(key, ""))
        else:
            text = self.transcripts[self._n_asr % len(self.transcripts)] if self.transcripts else ""
        conf = self.confidence
        c = conf if isinstance(conf, (int, float)) else conf[self._n_asr % len(conf)]
        self._n_asr += 1
        info = wav_info(audio)
        return Transcription(
            text=text,
            lang=lang,
            confidence=float(c) if text else 0.0,
            segments=(text,) if text else (),
            audio_s=info[1] if info else None,
        )

    def translate_hop(
        self, texts: Sequence[str], *, source: str, target: str, style: str
    ) -> list[str]:
        self.calls.append(
            (
                "translate",
                {"texts": list(texts), "source": source, "target": target, "style": style},
            )
        )
        return [
            self.translations.get((source, target, t), f"[{source}>{target}] {t}") for t in texts
        ]

    def synthesize(self, text: str, *, lang: str) -> Speech:
        self.calls.append(("synthesize", {"text": text, "lang": lang}))
        seconds = round(0.2 + 0.06 * len(text), 2)
        return Speech(
            audio=silent_wav(seconds, self.sample_rate),
            text=text,
            lang=lang,
            sample_rate=self.sample_rate,
            audio_s=seconds,
        )
