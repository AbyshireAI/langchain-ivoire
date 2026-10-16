"""Speech -> text in an Ivorian language."""

from __future__ import annotations

import dataclasses
from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig

from langchain_ivoire._audio import AudioInput, read_audio
from langchain_ivoire._base import BackendComponent
from langchain_ivoire.backends import BackendLike
from langchain_ivoire.types import Transcription

__all__ = ["IvoireTranscriber"]


class IvoireTranscriber(BackendComponent, Runnable[AudioInput, Transcription]):
    """Transcribe speech (``bytes``, a file path, or ``{"audio": ...}``) into text.

    Example:
        .. code-block:: python

            from langchain_ivoire import IvoireTranscriber

            asr = IvoireTranscriber(lang="bci")          # remote API, key from IVOIRE_API_KEY
            t = asr.invoke("come_and_eat.wav")
            t.text, t.confidence, t.low_confidence

    Args:
        lang: ISO 639-3 code (only ``"bci"`` is implemented; others raise ``NotImplementedError``).
        backend: ``"remote"`` (default), ``"openrouter"``, ``"fake"`` or an
            :class:`IvoireBackend`.
        api_key: remote backend options (default: ``IVOIRE_API_KEY`` / ``IVOIRE_BASE_URL``).
        base_url: See ``api_key``.
        low_confidence_threshold: below this ASR confidence ``low_confidence`` is set (the demo
            asks the speaker to repeat).
    """

    def __init__(
        self,
        lang: str = "bci",
        *,
        backend: BackendLike = None,
        api_key: str | None = None,
        base_url: str | None = None,
        low_confidence_threshold: float = 0.80,
    ) -> None:
        self._setup(lang, backend, api_key, base_url)
        self.low_confidence_threshold = low_confidence_threshold
        self.name = "ivoire_transcriber"

    def _finish(self, t: Transcription, source: str | None) -> Transcription:
        low = t.low_confidence or (
            bool(t.text)
            and t.confidence is not None
            and t.confidence < self.low_confidence_threshold
        )
        meta = {**t.metadata, "source": source} if source else t.metadata
        return dataclasses.replace(t, low_confidence=low, metadata=meta)

    def _transcribe(self, input: AudioInput) -> Transcription:
        data, source = read_audio(input)
        return self._finish(self.backend.transcribe(data, lang=self.lang), source)

    async def _atranscribe(self, input: AudioInput) -> Transcription:
        data, source = read_audio(input)
        return self._finish(await self.backend.atranscribe(data, lang=self.lang), source)

    def invoke(
        self, input: AudioInput, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Transcription:
        return self._call_with_config(self._transcribe, input, config)

    async def ainvoke(
        self, input: AudioInput, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Transcription:
        return await self._acall_with_config(self._atranscribe, input, config)
