"""Language registry: which Ivorian languages exist in the API and which are implemented."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "LANGUAGES",
    "PIVOT_LANGUAGES",
    "SUPPORTED_LANGUAGES",
    "Language",
    "normalize_code",
    "resolve_language",
    "resolve_text_language",
]


@dataclass(frozen=True)
class Language:
    """An Ivorian language known to the package."""

    code: str
    """ISO 639-3 code, e.g. ``"bci"``."""
    name: str
    french_name: str
    status: Literal["available", "planned"]


LANGUAGES: dict[str, Language] = {
    "bci": Language("bci", "Baoulé", "baoulé", "available"),
    "any": Language("any", "Anyin (Agni)", "agni", "planned"),
    "dyu": Language("dyu", "Dioula (Jula)", "dioula", "planned"),
    "bev": Language("bev", "Bété (Daloa)", "bété de Daloa", "planned"),
    "btg": Language("btg", "Bété (Gagnoa)", "bété de Gagnoa", "planned"),
}
"""Ivorian languages, implemented (``available``) or on the roadmap (``planned``)."""

PIVOT_LANGUAGES: dict[str, str] = {"fra": "French", "eng": "English"}
"""Languages every Ivorian language is translated to/from (English always pivots through French)."""

SUPPORTED_LANGUAGES: tuple[str, ...] = tuple(
    c for c, lg in LANGUAGES.items() if lg.status == "available"
)

_ALIASES = {
    "baoule": "bci",
    "baoulé": "bci",
    "agni": "any",
    "anyin": "any",
    "dioula": "dyu",
    "jula": "dyu",
    "fr": "fra",
    "fre": "fra",
    "french": "fra",
    "français": "fra",
    "en": "eng",
    "english": "eng",
}


def normalize_code(code: str) -> str:
    """Lower-case a language code and map common aliases (``"fr"`` -> ``"fra"``, ``"baoule"`` -> ``"bci"``)."""
    c = (code or "").strip().lower()
    return _ALIASES.get(c, c)


def resolve_language(lang: str) -> Language:
    """Return the implemented Ivorian language for ``lang``.

    Raises:
        NotImplementedError: the language is planned but not implemented yet.
        ValueError: the code is unknown (or is French/English, which are not Ivorian languages here).
    """
    code = normalize_code(lang)
    lg = LANGUAGES.get(code)
    if lg is None:
        known = ", ".join(f"{c!r}" for c in LANGUAGES)
        raise ValueError(f"Unknown Ivorian language {lang!r}. Known codes: {known}.")
    if lg.status != "available":
        raise NotImplementedError(
            f"{lg.name} ({lg.code!r}) is planned but not implemented yet. "
            f"Available now: {', '.join(SUPPORTED_LANGUAGES)} (Baoulé)."
        )
    return lg


def resolve_text_language(code: str) -> str:
    """Resolve a translation endpoint: an implemented Ivorian language, ``"fra"`` or ``"eng"``."""
    c = normalize_code(code)
    if c in PIVOT_LANGUAGES:
        return c
    return resolve_language(c).code
