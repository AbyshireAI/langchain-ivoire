"""The MCP server, exercised through the SDK's in-memory client (no subprocess, no sockets)."""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest
import respx

pytest.importorskip("mcp")

from mcp import Client
from mcp.types import AudioContent

from langchain_ivoire import FakeBackend, RemoteBackend
from langchain_ivoire._audio import silent_wav
from langchain_ivoire.mcp_server import create_mcp_server, main

WAV_B64 = base64.b64encode(silent_wav(0.5)).decode()


@pytest.fixture
def fake() -> FakeBackend:
    return FakeBackend(
        transcripts=["bla di like"],
        translations={
            ("bci", "fra", "Bla di like."): "Venez manger.",
            ("fra", "eng", "Venez manger."): "Come and eat.",
        },
    )


def _text(res: Any) -> str:
    return "".join(c.text for c in res.content if c.type == "text")


async def test_lists_the_four_tools(fake: FakeBackend) -> None:
    async with Client(create_mcp_server(fake)) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {
        "ivoire_transcribe",
        "ivoire_translate",
        "ivoire_speak",
        "ivoire_languages",
    }
    props = tools["ivoire_transcribe"].input_schema["properties"]
    assert set(props) == {"audio_base64", "audio_url"}  # no file paths


async def test_translate(fake: FakeBackend) -> None:
    async with Client(create_mcp_server(fake)) as client:
        res = await client.call_tool("ivoire_translate", {"text": "Bla di like."})
        assert not res.is_error and _text(res) == "Venez manger."
        res = await client.call_tool(
            "ivoire_translate", {"text": "Bla di like.", "source": "bci", "target": "eng"}
        )
        assert _text(res) == "Come and eat."
        res = await client.call_tool("ivoire_translate", {"text": "  "})
        assert not res.is_error
        res = await client.call_tool("ivoire_translate", {"text": "x", "style": "poetry"})
        assert res.is_error and "unknown style" in _text(res)
        res = await client.call_tool("ivoire_translate", {"text": "x", "target": "dyu"})
        assert res.is_error and "planned" in _text(res)


async def test_speak_returns_mcp_audio(fake: FakeBackend) -> None:
    async with Client(create_mcp_server(fake)) as client:
        res = await client.call_tool("ivoire_speak", {"text": "Bla di like."})
        assert not res.is_error
        kinds = [c.type for c in res.content]
        assert kinds == ["text", "audio"]
        audio = res.content[1]
        assert isinstance(audio, AudioContent)
        assert audio.mime_type == "audio/wav"
        assert base64.b64decode(audio.data)[:4] == b"RIFF"
        res = await client.call_tool("ivoire_speak", {"text": " "})
        assert res.is_error and "empty" in _text(res)


async def test_transcribe(fake: FakeBackend) -> None:
    async with Client(create_mcp_server(fake)) as client:
        res = await client.call_tool("ivoire_transcribe", {"audio_base64": WAV_B64})
        assert not res.is_error
        assert res.structured_content["text"] == "bla di like"
        assert res.structured_content["confidence"] == pytest.approx(0.95)
        assert json.loads(_text(res))["lang"] == "bci"
        for args, msg in (
            ({}, "pass audio_base64 or audio_url"),
            ({"audio_base64": "!!"}, "base64"),
            ({"audio_url": "file:///etc/passwd"}, "http or https"),
            ({"audio_url": "http://127.0.0.1/x.wav"}, "private or local"),
        ):
            res = await client.call_tool("ivoire_transcribe", args)
            assert res.is_error and msg in _text(res), args


async def test_languages_and_backend_errors() -> None:
    async with Client(create_mcp_server(FakeBackend())) as client:
        res = await client.call_tool("ivoire_languages", {})
        assert not res.is_error and "bci" in _text(res)
    # an unreachable remote API becomes a readable tool error, without secrets
    be = RemoteBackend(base_url="https://api.example.test", api_key="sk-secret", max_retries=0)
    with respx.mock:
        respx.post("https://api.example.test/v1/translate").mock(
            side_effect=httpx.ConnectError("refused")
        )
        respx.get("https://api.example.test/v1/languages").mock(
            return_value=httpx.Response(401, json={"detail": "bad key"})
        )
        async with Client(create_mcp_server(be)) as client:
            res = await client.call_tool("ivoire_translate", {"text": "Bla di like."})
            assert res.is_error and "IvoireConnectionError" in _text(res)
            assert "sk-secret" not in _text(res)
            res = await client.call_tool("ivoire_languages", {})
            assert res.is_error and "IvoireAuthenticationError" in _text(res)
            assert "sk-secret" not in _text(res)


def test_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    runs: list[tuple[str, dict[str, Any]]] = []
    from mcp.server.mcpserver import MCPServer

    monkeypatch.setattr(MCPServer, "run", lambda self, t="stdio", **kw: runs.append((t, kw)))
    main(["--backend", "fake"])
    main(["--backend", "fake", "--transport", "streamable-http", "--port", "8123"])
    assert runs == [("stdio", {}), ("streamable-http", {"host": "127.0.0.1", "port": 8123})]
    with pytest.raises(SystemExit):
        main(["--version"])
