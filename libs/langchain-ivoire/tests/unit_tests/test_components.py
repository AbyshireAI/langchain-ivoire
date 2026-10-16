from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda

from langchain_ivoire import (
    FakeBackend,
    IvoireSpeaker,
    IvoireTranscriber,
    IvoireTranslator,
    RemoteBackend,
    Speech,
    Transcription,
    Translation,
)
from langchain_ivoire._audio import wav_info


# ---- transcriber ---------------------------------------------------------------------------
def test_transcriber_bytes(fake: FakeBackend, wav: bytes) -> None:
    t = IvoireTranscriber(lang="bci", backend=fake).invoke(wav)
    assert isinstance(t, Transcription)
    assert t.text == "bla di like" and t.lang == "bci"
    assert t.confidence == pytest.approx(0.95)
    assert t.audio_s == pytest.approx(1.5)
    assert not t.low_confidence
    assert fake.calls[0] == ("transcribe", {"bytes": len(wav), "lang": "bci"})


def test_transcriber_path_and_dict(fake: FakeBackend, wav: bytes, tmp_path: Path) -> None:
    p = tmp_path / "a.wav"
    p.write_bytes(wav)
    asr = IvoireTranscriber(backend=fake)
    assert asr.invoke(str(p)).metadata["source"] == str(p)
    assert asr.invoke({"audio": p}).text == "bla di like"
    with pytest.raises(FileNotFoundError):
        asr.invoke(str(tmp_path / "missing.wav"))
    with pytest.raises(ValueError, match="empty"):
        asr.invoke(b"")


def test_transcriber_low_confidence(wav: bytes) -> None:
    be = FakeBackend(transcripts=["n ti"], confidence=0.42)
    assert IvoireTranscriber(backend=be, low_confidence_threshold=0.8).invoke(wav).low_confidence
    t = IvoireTranscriber(backend=be, low_confidence_threshold=0.3).invoke(wav)
    assert not t.low_confidence


async def test_transcriber_async(fake: FakeBackend, wav: bytes) -> None:
    t = await IvoireTranscriber(backend=fake).ainvoke(wav)
    assert t.text == "bla di like"


def test_transcriber_batch(fake: FakeBackend, wav: bytes) -> None:
    out = IvoireTranscriber(backend=fake).batch([wav, wav])
    assert [t.text for t in out] == ["bla di like", "bla di like"]


def test_planned_language_component() -> None:
    with pytest.raises(NotImplementedError, match="planned"):
        IvoireTranscriber(lang="dyu", backend="fake")


def test_default_backend_is_remote() -> None:
    asr = IvoireTranscriber()
    assert isinstance(asr.backend, RemoteBackend)
    assert asr.backend.base_url == "https://api.abyshire.ci"


def test_backend_options_conflict(fake: FakeBackend) -> None:
    with pytest.raises(ValueError):
        IvoireTranscriber(backend=fake, api_key="k")


# ---- translator ----------------------------------------------------------------------------
def test_translator_default_direction(fake: FakeBackend) -> None:
    r = IvoireTranslator(lang="bci", backend=fake).invoke("Bla di like.")
    assert isinstance(r, Translation)
    assert (r.text, r.source, r.target, r.style) == ("Venez manger.", "bci", "fra", "everyday")


def test_translator_pivots_english_through_french(fake: FakeBackend) -> None:
    tr = IvoireTranslator(lang="bci", source="bci", target="eng", backend=fake)
    r = tr.invoke("Bla di like.")
    assert r.text == "Come and eat."
    assert r.via == (("fra", "Venez manger."), ("eng", "Come and eat."))
    assert set(r.timings_ms) == {"mt_bci_fra", "mt_fra_eng"}


def test_translator_style_and_dict_input(fake: FakeBackend) -> None:
    tr = IvoireTranslator(lang="bci", source="fra", target="bci", style="tale", backend=fake)
    r = tr.invoke({"text": "Venez manger.", "style": "formal"})
    assert r.style == "formal"
    assert fake.calls[-1][1]["style"] == "formal"
    with pytest.raises(ValueError, match="unknown style"):
        IvoireTranslator(backend=fake, style="poetry")
    with pytest.raises(ValueError):
        IvoireTranslator(backend=fake, source="bci", target="bci")


def test_translator_from_french_defaults_to_lang(fake: FakeBackend) -> None:
    assert IvoireTranslator(lang="bci", source="fra", backend=fake).target == "bci"


def test_translator_empty_text(fake: FakeBackend) -> None:
    assert IvoireTranslator(backend=fake).invoke("  ").text == ""
    assert fake.calls == []


def test_lcel_pipeline(fake: FakeBackend, wav: bytes) -> None:
    chain = (
        IvoireTranscriber(backend=fake)
        | IvoireTranslator(backend=fake, target="eng")
        | RunnableLambda(lambda t: t.text)
    )
    assert chain.invoke(wav) == "Come and eat."


async def test_translator_async(fake: FakeBackend) -> None:
    r = await IvoireTranslator(backend=fake, source="fra", target="bci").ainvoke("Venez manger.")
    assert r.text == "Bla di like."


# ---- speaker -------------------------------------------------------------------------------
def test_speaker(fake: FakeBackend, tmp_path: Path) -> None:
    s = IvoireSpeaker(lang="bci", backend=fake).invoke("Bla di like.")
    assert isinstance(s, Speech)
    assert s.audio[:4] == b"RIFF" and s.media_type == "audio/wav"
    info = wav_info(s.audio)
    assert info is not None and info[0] == 16000
    assert s.save(tmp_path / "x.wav").read_bytes() == s.audio
    assert "bytes" in s.to_dict()["audio"]


def test_speaker_accepts_translation(fake: FakeBackend) -> None:
    chain = IvoireTranslator(backend=fake, source="fra", target="bci") | IvoireSpeaker(backend=fake)
    assert chain.invoke("Venez manger.").text == "Bla di like."
    with pytest.raises(ValueError, match="translation is into"):
        IvoireSpeaker(backend=fake).invoke(Translation(text="x", source="bci", target="fra"))
    with pytest.raises(ValueError, match="nothing to say"):
        IvoireSpeaker(backend=fake).invoke(" ")


async def test_speaker_async(fake: FakeBackend) -> None:
    s = await IvoireSpeaker(backend=fake).ainvoke({"text": "Ee!"})
    assert s.text == "Ee!"
