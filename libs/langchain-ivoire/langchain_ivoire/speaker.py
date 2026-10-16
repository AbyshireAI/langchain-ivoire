"""Text -> speech (WAV bytes) in an Ivorian language."""

from __future__ import annotations

from typing import Any, Union

from langchain_core.runnables import Runnable, RunnableConfig

from langchain_ivoire._base import BackendComponent
from langchain_ivoire.backends import BackendLike
from langchain_ivoire.types import Speech, Translation

__all__ = ["IvoireSpeaker", "SpeakerInput"]

SpeakerInput = Union[str, Translation, "dict[str, Any]"]
"""Text in ``lang``, a :class:`Translation` into ``lang``, or ``{"text": ...}``."""


class IvoireSpeaker(BackendComponent, Runnable[SpeakerInput, Speech]):
    """Speak text in ``lang``; returns a :class:`Speech` with WAV bytes.

    Example:
        .. code-block:: python

            from langchain_ivoire import IvoireSpeaker, IvoireTranslator

            say = IvoireTranslator(lang="bci", source="fra", target="bci") | IvoireSpeaker(lang="bci")
            say.invoke("Venez manger.").save("venez.wav")
    """

    def __init__(
        self,
        lang: str = "bci",
        *,
        backend: BackendLike = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._setup(lang, backend, api_key, base_url)
        self.name = "ivoire_speaker"

    def _text(self, input: SpeakerInput) -> str:
        if isinstance(input, str):
            return input
        if isinstance(input, Translation):
            if input.target != self.lang:
                raise ValueError(
                    f"this speaker speaks {self.lang!r} but the translation is into {input.target!r}"
                )
            return input.text
        if isinstance(input, dict) and "text" in input:
            return str(input["text"])
        raise TypeError(f"unsupported speaker input: {type(input).__name__}")

    def _speak(self, input: SpeakerInput) -> Speech:
        text = self._text(input).strip()
        if not text:
            raise ValueError("nothing to say (empty text)")
        return self.backend.synthesize(text, lang=self.lang)

    async def _aspeak(self, input: SpeakerInput) -> Speech:
        text = self._text(input).strip()
        if not text:
            raise ValueError("nothing to say (empty text)")
        return await self.backend.asynthesize(text, lang=self.lang)

    def invoke(
        self, input: SpeakerInput, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Speech:
        return self._call_with_config(self._speak, input, config)

    async def ainvoke(
        self, input: SpeakerInput, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Speech:
        return await self._acall_with_config(self._aspeak, input, config)
