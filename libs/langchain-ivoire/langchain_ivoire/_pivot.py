"""The French pivot shared by :class:`IvoireLanguageMiddleware` and :class:`IvoireLanguageChat`.

User (Ivorian language) --lang->fra--> model (French in, short simple French out) --fra->lang--> user.

Messages keep the user's language in ``content``; the French is stored next to it in
``additional_kwargs["ivoire"]["fra"]`` so later turns are not translated twice and the model only
ever sees French (the "French view", built by :func:`french_view`).
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

from langchain_ivoire._languages import LANGUAGES
from langchain_ivoire.translator import IvoireTranslator
from langchain_ivoire.types import Translation

__all__ = [
    "DEFAULT_SYSTEM_HINT",
    "IVOIRE_KEY",
    "LanguagePivot",
    "build_system_hint",
    "clean_reply",
    "french_view",
    "message_text",
    "stored_french",
    "with_text",
]

IVOIRE_KEY = "ivoire"
"""Key under ``additional_kwargs`` (and ``response_metadata`` for the chat wrapper)."""

DEFAULT_SYSTEM_HINT = (
    "Tu parles avec une personne qui s'exprime en {language}. Ses messages te sont traduits "
    "automatiquement en français et ta réponse sera traduite automatiquement en {language}. "
    "La traduction peut être imparfaite : si un message est étrange, réponds au sens le plus "
    "probable ou pose une question simple. Réponds uniquement en français simple de tous les jours, "
    "{length}, avec des mots courants. Pas de listes, pas de markdown, "
    "pas d'émojis, pas de parenthèses."
)
"""French system hint (``{language}``: French name of the language; ``{length}``: sentence cap)."""

_NUMBERS = {1: "une", 2: "deux", 3: "trois", 4: "quatre", 5: "cinq"}


def _length_phrase(max_sentences: int | None) -> str:
    if not max_sentences:
        return "en une ou deux phrases courtes"
    if max_sentences == 1:
        return "en une seule phrase courte"
    n = _NUMBERS.get(max_sentences, str(max_sentences))
    return f"en {n} phrases courtes au maximum"


def build_system_hint(
    lang: str, system_hint: str | bool | None, max_sentences: int | None = None
) -> str | None:
    """``True`` -> :data:`DEFAULT_SYSTEM_HINT` for ``lang``; a string -> itself; else ``None``."""
    if system_hint is True:
        return DEFAULT_SYSTEM_HINT.format(
            language=LANGUAGES[lang].french_name, length=_length_phrase(max_sentences)
        )
    if isinstance(system_hint, str):
        return system_hint
    return None


def message_text(msg: BaseMessage) -> str:
    """Plain text of a message (string content, or the text blocks of list content)."""
    c = msg.content
    if isinstance(c, str):
        return c
    parts: list[str] = []
    for b in c:
        if isinstance(b, str):
            parts.append(b)
        elif isinstance(b, dict) and b.get("type") == "text":
            parts.append(str(b.get("text", "")))
    return "".join(parts)


def with_text(msg: BaseMessage, text: str, **update: Any) -> BaseMessage:
    """Copy of ``msg`` whose text is ``text`` (non-text content blocks are kept)."""
    if isinstance(msg.content, str):
        return msg.model_copy(update={"content": text, **update})
    others = [
        b
        for b in msg.content
        if not (isinstance(b, str) or (isinstance(b, dict) and b.get("type") == "text"))
    ]
    return msg.model_copy(update={"content": [{"type": "text", "text": text}, *others], **update})


def clean_reply(text: str, max_sentences: int | None = None) -> str:
    """Strip markdown, list marks, emoji, quotes and brackets (translators prefer plain sentences)."""
    t = re.sub(r"[*_#`>\[\]|~]", " ", text or "")
    t = re.sub(r"^\s*[-•\d]+[.)]?\s+", "", t, flags=re.M)
    t = re.sub(r"[\U0001F000-\U0001FAFF☀-➿️]", "", t)
    t = re.sub(r"[«»\"“”()]", "", t)
    t = " ".join(t.split())
    if max_sentences:
        sents = re.findall(r"[^.!?]+[.!?]?", t)
        t = " ".join(s.strip() for s in sents[:max_sentences] if s.strip()) or t
    return t


def stored_french(msg: BaseMessage) -> str | None:
    """The French already known for ``msg`` (from this package's metadata), else ``None``."""
    meta = (msg.additional_kwargs or {}).get(IVOIRE_KEY)
    if isinstance(meta, dict) and isinstance(meta.get("fra"), str):
        return str(meta["fra"])
    if isinstance(msg, AIMessage):
        rm = (msg.response_metadata or {}).get(IVOIRE_KEY)
        if isinstance(rm, dict) and isinstance(rm.get("output_fra"), str):
            return str(rm["output_fra"])
    return None


def _strip_meta(msg: BaseMessage) -> dict[str, Any]:
    return {k: v for k, v in (msg.additional_kwargs or {}).items() if k != IVOIRE_KEY}


def french_view(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """The messages as the model should see them: stored French swapped in, metadata removed."""
    out: list[BaseMessage] = []
    for m in messages:
        fra = stored_french(m)
        if fra is None:
            out.append(m)
        else:
            out.append(with_text(m, fra, additional_kwargs=_strip_meta(m)))
    return out


def _ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


class LanguagePivot:
    """Translation in and out of French, the system hint and the reply cleaning.

    Args:
        translator: an :class:`IvoireTranslator` for ``lang``.
        style: translation register.
        system_hint: French hint for the model (``None``: none).
        clean: strip markdown/emoji from the French reply before translating it.
        max_sentences: keep at most this many sentences of the French reply.
    """

    def __init__(
        self,
        translator: IvoireTranslator,
        *,
        style: str = "everyday",
        system_hint: str | None = None,
        clean: bool = True,
        max_sentences: int | None = None,
    ) -> None:
        self.translator = translator
        self.lang = translator.lang
        self.style = style
        self.system_hint = system_hint
        self.clean = clean
        self.max_sentences = max_sentences

    # ---- system hint -------------------------------------------------------------------------
    def merge_hint(self, system: SystemMessage | None) -> SystemMessage | None:
        """``system`` with the hint appended (or a new system message holding just the hint)."""
        if not self.system_hint:
            return system
        if system is None:
            return SystemMessage(content=self.system_hint)
        if isinstance(system.content, str):
            text = f"{system.content}\n\n{self.system_hint}" if system.content else self.system_hint
            return system.model_copy(update={"content": text})
        blocks = [*system.content, {"type": "text", "text": self.system_hint}]
        return system.model_copy(update={"content": blocks})

    # ---- translation -------------------------------------------------------------------------
    def to_french(self, texts: Sequence[str]) -> list[Translation]:
        return self.translator.translate_texts(
            list(texts), source=self.lang, target="fra", style=self.style
        )

    async def ato_french(self, texts: Sequence[str]) -> list[Translation]:
        return await self.translator.atranslate_texts(
            list(texts), source=self.lang, target="fra", style=self.style
        )

    def finish_french(self, raw: str) -> str:
        """The model's raw French reply, cleaned and capped, ready to translate."""
        return clean_reply(raw, self.max_sentences) if self.clean else raw

    def from_french(self, fra: str) -> str:
        if not fra.strip():
            return ""
        return self.translator.translate_texts(
            [fra], source="fra", target=self.lang, style=self.style
        )[0].text

    async def afrom_french(self, fra: str) -> str:
        if not fra.strip():
            return ""
        out = await self.translator.atranslate_texts(
            [fra], source="fra", target=self.lang, style=self.style
        )
        return out[0].text

    def meta(self, fra: str, **extra: Any) -> dict[str, Any]:
        return {"lang": self.lang, "style": self.style, "fra": fra, **extra}
