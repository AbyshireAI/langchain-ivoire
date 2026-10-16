from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pytest_socket import disable_socket, socket_allow_hosts

from langchain_ivoire import FakeBackend
from langchain_ivoire._audio import silent_wav

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Unit tests never touch the network (pytest-socket).

    On Windows asyncio's self-pipe is a loopback TCP pair, so loopback stays open there;
    everywhere else every non-Unix socket is refused.
    """
    if sys.platform == "win32":
        socket_allow_hosts(["127.0.0.1", "::1"], allow_unix_socket=True)
    else:
        disable_socket(allow_unix_socket=True)


@pytest.fixture(autouse=True)
def _no_env_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IVOIRE_API_KEY", raising=False)
    monkeypatch.delenv("IVOIRE_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # never read a settings file from this machine (tests pass settings explicitly)
    monkeypatch.setenv("IVOIRE_SETTINGS", "off")


@pytest.fixture
def fake() -> FakeBackend:
    return FakeBackend(
        transcripts=["bla di like"],
        translations={
            ("bci", "fra", "bla di like"): "Venez manger.",
            ("bci", "fra", "Bla di like."): "Venez manger.",
            ("fra", "eng", "Venez manger."): "Come and eat.",
            ("fra", "bci", "Venez manger."): "Bla di like.",
            ("eng", "fra", "Come and eat."): "Venez manger.",
            ("bci", "fra", "N tɛ, an ti sɛn?"): "Bonjour, comment allez-vous ?",
            ("fra", "bci", "Je vais bien, merci."): "N liɛ yo'a kpa, n yo.",
        },
    )


@pytest.fixture
def wav() -> bytes:
    return silent_wav(1.5)


@pytest.fixture
def phrasebook_path() -> Path:
    return FIXTURES / "phrasebook_bci.json"
