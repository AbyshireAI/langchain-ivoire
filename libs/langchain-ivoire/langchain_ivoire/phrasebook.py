"""Phrasebook with sound-alike matching ("phrase snap").

A learner's accent or an ASR slip often turns a known phrase into something close but wrong
(``"bla de le kɛ"`` for ``"Bla di like."``). When an utterance is *clearly* one phrasebook entry, the voice
agent uses the entry's checked text and translations instead of machine translation.

Matching compares sound-alike skeletons (tones, spaces and punctuation dropped; ɛ/e, ɔ/o, b/p, d/t
folded; ɲ -> ny; doubled letters collapsed) and accepts only a whole-utterance match that is both close
(``min_score``) and well ahead of the next-best phrase (``margin``).
"""

from __future__ import annotations

import difflib
import json
import os
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union, cast

from langchain_ivoire._languages import resolve_language

__all__ = ["Phrase", "PhraseMatch", "Phrasebook", "PhrasebookLike", "sound_skeleton"]


def _skeleton_bci(s: str) -> str:
    s = unicodedata.normalize("NFD", (s or "").lower())
    s = unicodedata.normalize("NFC", "".join(c for c in s if not unicodedata.combining(c)))
    s = s.replace("ɲ", "ny").replace("ɛ", "e").replace("ɔ", "o").replace("p", "b").replace("t", "d")
    s = re.sub(r"[^a-z]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)


_SKELETONS: dict[str, Callable[[str], str]] = {"bci": _skeleton_bci}


def sound_skeleton(text: str, lang: str = "bci") -> str:
    """Sound-alike skeleton of ``text`` used for fuzzy phrase matching."""
    code = resolve_language(lang).code
    return _SKELETONS[code](text)


@dataclass(frozen=True)
class Phrase:
    """A phrasebook entry: the text in the Ivorian language plus checked translations."""

    id: str
    text: str
    translations: dict[str, str] = field(default_factory=dict, compare=False)
    """e.g. ``{"fra": "Venez manger.", "eng": "Come and eat."}``"""
    audio: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True)
class PhraseMatch:
    phrase: Phrase
    score: float
    heard: str
    runner_up: float = 0.0


PhrasebookLike = Union["Phrasebook", str, "os.PathLike[str]", Mapping[str, Any], Iterable[Any]]


class Phrasebook:
    """A list of phrases in one language with sound-alike lookup.

    Entries may be :class:`Phrase` objects or dicts in the demo format
    (``{"id", "bci", "fra", "eng", "audio"?}``; the text key is the language code, or ``"text"``).

    Args:
        min_score: minimum similarity (0..1) of the skeletons (demo: 0.72; 0 disables matching).
        margin: required lead over the second-best phrase (demo: 0.15).
    """

    def __init__(
        self,
        phrases: Iterable[Phrase | Mapping[str, Any]],
        lang: str = "bci",
        *,
        min_score: float = 0.72,
        margin: float = 0.15,
    ) -> None:
        self.lang = resolve_language(lang).code
        self.min_score = min_score
        self.margin = margin
        self.phrases: list[Phrase] = [self._coerce(p, i) for i, p in enumerate(phrases)]
        self._skel = [sound_skeleton(p.text, self.lang) for p in self.phrases]

    def _coerce(self, p: Phrase | Mapping[str, Any], i: int) -> Phrase:
        if isinstance(p, Phrase):
            return p
        text = p.get(self.lang) or p.get("text")
        if not text:
            raise ValueError(f"phrase #{i} has no {self.lang!r} (or 'text') field")
        tr = {k: str(p[k]) for k in ("fra", "eng") if p.get(k)}
        return Phrase(
            id=str(p.get("id", i)),
            text=str(text),
            translations=tr,
            audio=p.get("audio"),
            raw=dict(p),
        )

    @classmethod
    def from_json(cls, path: str | os.PathLike[str], lang: str = "bci", **kw: Any) -> Phrasebook:
        """Load ``{"phrases": [...]}`` or a bare list from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["phrases"] if isinstance(data, dict) else data, lang, **kw)

    @classmethod
    def coerce(cls, value: PhrasebookLike, lang: str = "bci") -> Phrasebook:
        if isinstance(value, Phrasebook):
            return value
        if isinstance(value, (str, os.PathLike)):
            return cls.from_json(value, lang)
        if isinstance(value, Mapping):
            return cls(value["phrases"], lang)
        return cls(list(cast(Iterable[Any], value)), lang)

    def __len__(self) -> int:
        return len(self.phrases)

    def __iter__(self) -> Any:
        return iter(self.phrases)

    def scores(self, text: str) -> list[tuple[float, Phrase]]:
        """Similarity to every length-compatible phrase, best first."""
        q = sound_skeleton(text, self.lang)
        if len(q) < 4:
            return []
        scored = []
        for ph, r in zip(self.phrases, self._skel, strict=True):
            if r and 0.6 <= len(q) / len(r) <= 1.6:
                scored.append((difflib.SequenceMatcher(None, q, r).ratio(), ph))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def match(self, text: str) -> PhraseMatch | None:
        """The phrase ``text`` clearly is, or ``None``."""
        if self.min_score <= 0:
            return None
        scored = self.scores(text)
        if not scored:
            return None
        s1, best = scored[0]
        s2 = scored[1][0] if len(scored) > 1 else 0.0
        if s1 >= self.min_score and s1 - s2 >= self.margin:
            return PhraseMatch(phrase=best, score=round(s1, 3), heard=text, runner_up=round(s2, 3))
        return None
