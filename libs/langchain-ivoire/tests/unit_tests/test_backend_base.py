"""The async-first backend contract: implement either side, get the other for free."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from langchain_ivoire import IvoireBackend, IvoireTranslator, IvoireUnsupportedError
from langchain_ivoire._audio import silent_wav
from langchain_ivoire.backends.base import run_sync
from langchain_ivoire.types import Speech, Transcription


class AsyncOnly(IvoireBackend):
    name = "async-only"

    async def atranscribe(self, audio: bytes, *, lang: str) -> Transcription:
        return Transcription(text=f"{len(audio)} bytes", lang=lang)

    async def asynthesize(self, text: str, *, lang: str) -> Speech:
        return Speech(audio=silent_wav(0.1), text=text, lang=lang)

    async def atranslate_hop(
        self, texts: Sequence[str], *, source: str, target: str, style: str
    ) -> list[str]:
        return [f"{source}>{target}:{t}" for t in texts]


class Nothing(IvoireBackend):
    name = "nothing"


def test_sync_wrappers_over_async() -> None:
    be = AsyncOnly()
    assert be.transcribe(b"abc", lang="bci").text == "3 bytes"
    assert be.synthesize("Ee!", lang="bci").text == "Ee!"
    r = IvoireTranslator(backend=be, target="eng").invoke("Bla di like.")
    assert r.text == "fra>eng:bci>fra:Bla di like."
    assert [h[0] for h in r.via] == ["fra", "eng"]


async def test_sync_wrappers_inside_a_running_loop() -> None:
    be = AsyncOnly()
    assert be.transcribe(b"abcd", lang="bci").text == "4 bytes"  # runs in a helper thread
    assert (await be.atranslate(["x"], source="fra", target="bci"))[0].text == "fra>bci:x"
    assert be.translate([], source="bci", target="fra") == []
    assert await be.atranslate([], source="bci", target="fra") == []


async def test_missing_capabilities_are_explicit() -> None:
    be = Nothing()
    with pytest.raises(IvoireUnsupportedError, match="transcribe"):
        be.transcribe(b"x", lang="bci")
    with pytest.raises(IvoireUnsupportedError, match="atranscribe"):
        await be.atranscribe(b"x", lang="bci")
    with pytest.raises(IvoireUnsupportedError, match="synthesize"):
        be.synthesize("x", lang="bci")
    with pytest.raises(IvoireUnsupportedError, match="asynthesize"):
        await be.asynthesize("x", lang="bci")
    with pytest.raises(IvoireUnsupportedError, match="translate_hop"):
        be.translate(["x"], source="bci", target="fra")
    with pytest.raises(IvoireUnsupportedError, match="atranslate_hop"):
        await be.atranslate(["x"], source="bci", target="fra")
    assert (await be.ahealth())["backend"] == "nothing"
    assert (await be.alanguages())[0]["code"] == "bci"
    assert await be.amodels() == {"backend": "nothing"}
    await be.aclose()


def test_run_sync() -> None:
    async def two() -> int:
        return 2

    assert run_sync(two()) == 2
