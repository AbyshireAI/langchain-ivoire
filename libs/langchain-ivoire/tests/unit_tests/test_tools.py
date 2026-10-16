from __future__ import annotations

import base64
import json
import socket
from typing import Any

import httpx
import pytest
import respx
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_tests.unit_tests import ToolsUnitTests
from pydantic import BaseModel

from langchain_ivoire import (
    FakeBackend,
    IvoireSpeakTool,
    IvoireTranscribeTool,
    IvoireTranslateTool,
    get_tools,
)
from langchain_ivoire._audio import silent_wav
from langchain_ivoire.audio import from_audio_block, to_audio_block
from tests.unit_tests._fakes import ToolCallingFake

FAKE = FakeBackend(translations={("bci", "fra", "Bla di like."): "Venez manger."})
WAV_B64 = base64.b64encode(silent_wav(0.5)).decode()


class TestTranslateToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[IvoireTranslateTool]:
        return IvoireTranslateTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"lang": "bci", "backend": FAKE}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "Bla di like.", "source_lang": "bci", "target_lang": "fra"}


class TestTranscribeToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[IvoireTranscribeTool]:
        return IvoireTranscribeTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"lang": "bci", "backend": FAKE}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"audio_base64": WAV_B64}


class TestSpeakToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[IvoireSpeakTool]:
        return IvoireSpeakTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"lang": "bci", "backend": FAKE}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "Bla di like."}


def test_translate_tool_runs() -> None:
    tool = IvoireTranslateTool(backend=FAKE)
    assert (
        tool.invoke({"text": "Bla di like.", "source_lang": "bci", "target_lang": "fra"})
        == "Venez manger."
    )
    assert tool.invoke({"text": "Bla di like."}) == "Venez manger."  # defaults bci -> fra


def test_transcribe_tool_base64_and_block() -> None:
    tool = IvoireTranscribeTool(backend=FakeBackend())
    assert tool.invoke({"audio_base64": WAV_B64}) == "bla di like"
    assert tool.invoke({"audio": to_audio_block(silent_wav(0.5))}) == "bla di like"
    assert tool.invoke({"audio_base64": "data:audio/wav;base64," + WAV_B64}) == "bla di like"


def test_transcribe_tool_hides_runtime_and_file_paths() -> None:
    tool = IvoireTranscribeTool(backend=FakeBackend())
    schema = tool.tool_call_schema
    assert isinstance(schema, type) and issubclass(schema, BaseModel)
    props = schema.model_json_schema()["properties"]
    assert set(props) == {"audio_url", "audio_base64", "audio"}
    assert "file_path" not in props


@pytest.mark.parametrize(
    "url",
    [
        "/etc/passwd",
        "file:///etc/passwd",
        "C:/Windows/win.ini",
        "ftp://example.com/a.wav",
        "http://127.0.0.1:8090/audio/x.wav",
        "http://[::1]/x.wav",
        "http://169.254.169.254/latest/meta-data",
        "http://10.0.0.5/x.wav",
        "http://localhost/x.wav",
        "http://user:pw@example.com/x.wav",
    ],
)
def test_transcribe_tool_refuses_paths_and_local_urls(url: str) -> None:
    tool = IvoireTranscribeTool(backend=FakeBackend())
    with pytest.raises(ValueError):
        tool.invoke({"audio_url": url})


def _resolve_to(monkeypatch: pytest.MonkeyPatch, ip: str) -> None:
    def fake_getaddrinfo(host: str, port: Any, *a: Any, **k: Any) -> list[Any]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


@respx.mock
def test_transcribe_tool_fetches_public_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _resolve_to(monkeypatch, "93.184.216.34")
    route = respx.get("https://audio.example.com/q.wav").mock(
        return_value=httpx.Response(200, content=silent_wav(0.5))
    )
    be = FakeBackend()
    out = IvoireTranscribeTool(backend=be).invoke({"audio_url": "https://audio.example.com/q.wav"})
    assert out == "bla di like" and route.called
    assert "authorization" not in route.calls[0].request.headers  # never sends the API key


@respx.mock
def test_transcribe_tool_rechecks_redirects_and_caps_size(monkeypatch: pytest.MonkeyPatch) -> None:
    _resolve_to(monkeypatch, "93.184.216.34")
    respx.get("https://audio.example.com/r").mock(
        return_value=httpx.Response(302, headers={"location": "http://127.0.0.1/secret"})
    )
    with pytest.raises(ValueError, match="private or local"):
        IvoireTranscribeTool(backend=FakeBackend()).invoke(
            {"audio_url": "https://audio.example.com/r"}
        )
    respx.get("https://audio.example.com/big.wav").mock(
        return_value=httpx.Response(200, content=b"x" * 2048)
    )
    with pytest.raises(ValueError, match="larger than"):
        IvoireTranscribeTool(backend=FakeBackend(), max_audio_bytes=1024).invoke(
            {"audio_url": "https://audio.example.com/big.wav"}
        )


@respx.mock
async def test_transcribe_tool_async_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _resolve_to(monkeypatch, "93.184.216.34")
    respx.get("https://audio.example.com/q.wav").mock(
        return_value=httpx.Response(200, content=silent_wav(0.5))
    )
    tool = IvoireTranscribeTool(backend=FakeBackend())
    assert await tool.ainvoke({"audio_url": "https://audio.example.com/q.wav"}) == "bla di like"
    _resolve_to(monkeypatch, "10.9.8.7")
    with pytest.raises(ValueError, match="private or local"):
        await tool.ainvoke({"audio_url": "https://lan.example.com/q.wav"})


def test_transcribe_tool_private_urls_opt_in() -> None:
    with respx.mock:
        respx.get("http://127.0.0.1:9/x.wav").mock(
            return_value=httpx.Response(200, content=silent_wav(0.3))
        )
        tool = IvoireTranscribeTool(backend=FakeBackend(), allow_private_urls=True)
        assert tool.invoke({"audio_url": "http://127.0.0.1:9/x.wav"}) == "bla di like"


def test_transcribe_tool_without_audio_fails_clearly() -> None:
    with pytest.raises(ValueError, match="no audio"):
        IvoireTranscribeTool(backend=FakeBackend()).invoke({})


@pytest.mark.parametrize("use_async", [False, True])
async def test_transcribe_tool_reads_latest_audio_from_agent_state(use_async: bool) -> None:
    """Inside create_agent the tool gets the user's voice message through ToolRuntime."""
    model = ToolCallingFake(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "ivoire_transcribe", "args": {}, "id": "t1"}],
                ),
                AIMessage(content="Il a dit : venez manger."),
            ]
        )
    )
    agent = create_agent(model, tools=[IvoireTranscribeTool(backend=FakeBackend())])
    voice = HumanMessage(
        content=[{"type": "text", "text": "Écoute :"}, dict(to_audio_block(silent_wav(0.4)))]
    )
    inputs: Any = {"messages": [voice]}
    out = await agent.ainvoke(inputs) if use_async else agent.invoke(inputs)
    tool_msg = next(m for m in out["messages"] if isinstance(m, ToolMessage))
    assert tool_msg.content == "bla di like" and tool_msg.status == "success"


def test_speak_tool_returns_audio_artifact() -> None:
    call = {
        "name": "ivoire_speak",
        "args": {"text": "Bla di like."},
        "id": "c1",
        "type": "tool_call",
    }
    msg = IvoireSpeakTool(backend=FakeBackend()).invoke(call)
    assert isinstance(msg, ToolMessage)
    assert str(msg.content).startswith("Spoke ")
    block = msg.artifact
    assert block["type"] == "audio" and block["mime_type"] == "audio/wav"
    assert block["extras"]["text"] == "Bla di like." and block["extras"]["lang"] == "bci"
    assert from_audio_block(block)[:4] == b"RIFF"
    json.dumps(msg.model_dump())  # the whole ToolMessage is JSON


async def test_speak_tool_artifact_survives_a_checkpointer() -> None:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    call = {"name": "ivoire_speak", "args": {"text": "Ee!"}, "id": "c2", "type": "tool_call"}
    msg = await IvoireSpeakTool(backend=FakeBackend()).ainvoke(call)
    serde = JsonPlusSerializer()
    back = serde.loads_typed(serde.dumps_typed(msg))
    assert back.artifact == msg.artifact


async def test_tools_async() -> None:
    assert (
        await IvoireTranslateTool(backend=FAKE).ainvoke({"text": "Bla di like."}) == "Venez manger."
    )


def test_get_tools_share_backend() -> None:
    be = FakeBackend()
    tools = get_tools("bci", be)
    assert [t.name for t in tools] == ["ivoire_translate", "ivoire_transcribe", "ivoire_speak"]
    assert all(t._get_backend() is be for t in tools)  # type: ignore[attr-defined]
    assert len(get_tools("bci", be, include_speech=False)) == 1
