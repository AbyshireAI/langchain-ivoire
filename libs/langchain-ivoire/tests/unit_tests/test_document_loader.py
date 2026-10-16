from pathlib import Path

import pytest
from langchain_core.documents import Document

from langchain_ivoire import FakeBackend, IvoireAudioLoader
from langchain_ivoire._audio import silent_wav


@pytest.fixture
def audio_dir(tmp_path: Path) -> Path:
    (tmp_path / "a.wav").write_bytes(silent_wav(1.0))
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.wav").write_bytes(silent_wav(2.0))
    (tmp_path / "notes.txt").write_text("not audio")
    return tmp_path


def test_loads_directory(audio_dir: Path) -> None:
    be = FakeBackend(transcripts=["bla di like", "nnɔsua kpalɛ"], confidence=0.9)
    docs = IvoireAudioLoader(audio_dir, lang="bci", backend=be).load()
    assert [Path(d.metadata["source"]).name for d in docs] == ["a.wav", "b.wav"]
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].page_content == "bla di like"
    assert docs[1].metadata["duration"] == pytest.approx(2.0)
    assert docs[0].metadata["confidence"] == pytest.approx(0.9)
    assert docs[0].metadata["lang"] == "bci"


def test_file_list_glob_and_skip_empty(audio_dir: Path) -> None:
    be = FakeBackend(transcripts=["", "ee"])
    docs = IvoireAudioLoader(
        [audio_dir / "a.wav", audio_dir / "sub" / "b.wav"], backend=be, skip_empty=True
    ).load()
    assert [d.page_content for d in docs] == ["ee"]
    be2 = FakeBackend()
    assert len(IvoireAudioLoader(audio_dir, glob="*.wav", backend=be2).load()) == 1
    with pytest.raises(FileNotFoundError):
        IvoireAudioLoader(audio_dir / "nope.wav", backend=be2).load()


async def test_async_load(audio_dir: Path) -> None:
    docs = [d async for d in IvoireAudioLoader(audio_dir, backend=FakeBackend()).alazy_load()]
    assert len(docs) == 2
