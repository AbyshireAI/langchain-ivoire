import pytest

from langchain_ivoire import LANGUAGES, SUPPORTED_LANGUAGES, resolve_language
from langchain_ivoire._languages import normalize_code, resolve_text_language


def test_baoule_is_the_only_available_language() -> None:
    assert SUPPORTED_LANGUAGES == ("bci",)
    assert resolve_language("bci").name == "Baoulé"
    assert resolve_language("Baoulé").code == "bci"


@pytest.mark.parametrize("code", ["any", "dyu", "bev", "btg", "agni", "dioula"])
def test_planned_languages_raise_not_implemented(code: str) -> None:
    with pytest.raises(NotImplementedError, match="planned"):
        resolve_language(code)


def test_unknown_language() -> None:
    with pytest.raises(ValueError, match="Unknown Ivorian language"):
        resolve_language("xyz")
    with pytest.raises(ValueError):
        resolve_language("fra")  # French is a pivot, not an Ivorian language


def test_aliases_and_text_languages() -> None:
    assert normalize_code(" FR ") == "fra"
    assert resolve_text_language("en") == "eng"
    assert resolve_text_language("bci") == "bci"
    assert all(lg.status in ("available", "planned") for lg in LANGUAGES.values())
