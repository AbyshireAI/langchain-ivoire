"""Text translation between an Ivorian language, French and English (English pivots via French)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Union

from langchain_core.runnables import Runnable, RunnableConfig

from langchain_ivoire._base import BackendComponent
from langchain_ivoire._languages import resolve_text_language
from langchain_ivoire.backends import STYLES, BackendLike
from langchain_ivoire.types import Transcription, Translation

__all__ = ["IvoireTranslator", "TranslatorInput"]

TranslatorInput = Union[str, Transcription, Translation, "dict[str, Any]"]
"""Text, a :class:`Transcription`/:class:`Translation` (its ``.text``), or
``{"text": ..., "source"?: ..., "target"?: ..., "style"?: ...}``."""


def _check_style(style: str) -> str:
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; choose one of {', '.join(STYLES)}")
    return style


class IvoireTranslator(BackendComponent, Runnable[TranslatorInput, Translation]):
    """Translate between ``lang`` and French/English.

    Example:
        .. code-block:: python

            from langchain_ivoire import IvoireTranslator

            to_fr = IvoireTranslator(lang="bci")                       # bci -> fra
            to_bci = IvoireTranslator(lang="bci", source="eng", target="bci")
            to_fr.invoke("Bla di like.").text   # 'Venez manger.'

    Args:
        lang: the Ivorian language (``"bci"``).
        source: direction; default ``lang`` -> ``"fra"``. Codes: ``lang``, ``"fra"``, ``"eng"``.
        target: See ``source``.
        style: ``"everyday"`` (default), ``"tale"`` (story register) or ``"formal"``.
        backend: see :class:`~langchain_ivoire.IvoireTranscriber`.
        api_key: See ``backend``.
        base_url: See ``backend``.
    """

    def __init__(
        self,
        lang: str = "bci",
        *,
        source: str | None = None,
        target: str | None = None,
        style: str = "everyday",
        backend: BackendLike = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._setup(lang, backend, api_key, base_url)
        self.source = resolve_text_language(source or self.lang)
        self.target = resolve_text_language(
            target or ("fra" if self.source != "fra" else self.lang)
        )
        if self.source == self.target:
            raise ValueError(f"source and target are both {self.source!r}")
        self.style = _check_style(style)
        self.name = "ivoire_translator"

    def __repr__(self) -> str:
        return (
            f"IvoireTranslator(lang={self.lang!r}, source={self.source!r}, target={self.target!r}, "
            f"style={self.style!r}, backend={self.backend!r})"
        )

    def _parse(self, input: TranslatorInput) -> tuple[str, str, str, str]:
        if isinstance(input, str):
            return input, self.source, self.target, self.style
        if isinstance(input, (Transcription, Translation)):
            src = input.lang if isinstance(input, Transcription) else input.target
            return input.text, resolve_text_language(src), self.target, self.style
        if isinstance(input, dict):
            if "text" not in input:
                raise ValueError("dict input needs a 'text' key")
            return (
                str(input["text"]),
                resolve_text_language(input.get("source") or self.source),
                resolve_text_language(input.get("target") or self.target),
                _check_style(input.get("style") or self.style),
            )
        raise TypeError(f"unsupported translator input: {type(input).__name__}")

    # ---- batch helpers used by with_language / the voice agent -------------------------------
    def translate_texts(
        self,
        texts: Sequence[str],
        *,
        source: str | None = None,
        target: str | None = None,
        style: str | None = None,
    ) -> list[Translation]:
        """Translate several texts in one backend call (no callbacks)."""
        return self.backend.translate(
            list(texts),
            source=resolve_text_language(source or self.source),
            target=resolve_text_language(target or self.target),
            style=_check_style(style or self.style),
        )

    async def atranslate_texts(
        self,
        texts: Sequence[str],
        *,
        source: str | None = None,
        target: str | None = None,
        style: str | None = None,
    ) -> list[Translation]:
        return await self.backend.atranslate(
            list(texts),
            source=resolve_text_language(source or self.source),
            target=resolve_text_language(target or self.target),
            style=_check_style(style or self.style),
        )

    # ---- Runnable ----------------------------------------------------------------------------
    def _translate(self, input: TranslatorInput) -> Translation:
        text, src, tgt, style = self._parse(input)
        if not text.strip():
            return Translation(text="", source=src, target=tgt, source_text=text, style=style)
        return self.backend.translate([text], source=src, target=tgt, style=style)[0]

    async def _atranslate(self, input: TranslatorInput) -> Translation:
        text, src, tgt, style = self._parse(input)
        if not text.strip():
            return Translation(text="", source=src, target=tgt, source_text=text, style=style)
        return (await self.backend.atranslate([text], source=src, target=tgt, style=style))[0]

    def invoke(
        self, input: TranslatorInput, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Translation:
        return self._call_with_config(self._translate, input, config)

    async def ainvoke(
        self, input: TranslatorInput, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Translation:
        return await self._acall_with_config(self._atranslate, input, config)
