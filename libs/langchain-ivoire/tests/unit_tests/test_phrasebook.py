import json
from pathlib import Path

import pytest

from langchain_ivoire import Phrasebook, sound_skeleton


@pytest.fixture
def pb(phrasebook_path: Path) -> Phrasebook:
    return Phrasebook.from_json(phrasebook_path)


@pytest.mark.parametrize("heard", ["bla de le kɛ", "plade le kɛ", "Bla di like.", "bla di likɛ"])
def test_sound_alike_snaps_to_come_and_eat(pb: Phrasebook, heard: str) -> None:
    m = pb.match(heard)
    assert m is not None
    assert m.phrase.text == "Bla di like."
    assert m.phrase.translations["eng"] == "Come and eat."
    assert m.phrase.translations["fra"] == "Venez manger."
    assert m.score >= 0.72 and m.score - m.runner_up >= 0.15
    assert m.heard == heard


@pytest.mark.parametrize("heard", ["ɔ ti kekle sikasua", "", "bla", "n ti"])
def test_unrelated_or_short_utterances_do_not_match(pb: Phrasebook, heard: str) -> None:
    assert pb.match(heard) is None


def test_skeleton_folds_sound_alikes() -> None:
    assert sound_skeleton("Bla di like.") == "bladilike"
    assert sound_skeleton("plade le kɛ") == "bladeleke"
    assert sound_skeleton("Nnɔsua kpalɛ") == "nosuakbale"
    assert sound_skeleton("ɲɛ") == "nye"


def test_disable_and_formats(phrasebook_path: Path) -> None:
    data = json.loads(phrasebook_path.read_text(encoding="utf-8"))
    assert Phrasebook(data["phrases"], min_score=0).match("bla di like") is None
    assert len(Phrasebook.coerce(data)) == len(data["phrases"])
    assert len(Phrasebook.coerce(str(phrasebook_path))) == len(data["phrases"])
    with pytest.raises(ValueError, match="no 'bci'"):
        Phrasebook([{"id": "x", "fra": "Bonjour"}])
    with pytest.raises(NotImplementedError):
        Phrasebook([], lang="dyu")
