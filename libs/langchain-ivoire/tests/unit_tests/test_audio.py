from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest
import respx
from langchain_core.messages import AIMessage, HumanMessage

from langchain_ivoire import FakeBackend, IvoireSpeaker
from langchain_ivoire._audio import silent_wav
from langchain_ivoire.audio import (
    afrom_audio_block,
    check_audio_url,
    decode_base64_audio,
    fetch_audio_url,
    from_audio_block,
    is_audio_block,
    latest_audio_block,
    to_audio_block,
)

WAV = silent_wav(0.25)
B64 = base64.b64encode(WAV).decode()


def test_round_trip_bytes_and_speech() -> None:
    block = to_audio_block(WAV)
    assert block["type"] == "audio" and block["mime_type"] == "audio/wav"
    assert block["id"].startswith("lc_")
    assert block["extras"] == {"sample_rate": 16000, "audio_s": 0.25}
    assert from_audio_block(block) == WAV
    json.dumps(block)

    speech = IvoireSpeaker(backend=FakeBackend()).invoke("Ee!")
    sb = to_audio_block(speech, source="test")
    assert sb["extras"]["text"] == "Ee!" and sb["extras"]["source"] == "test"
    assert from_audio_block(sb) == speech.audio
    assert to_audio_block(sb) == sb  # blocks pass through
    assert to_audio_block(b"OggS....", mime_type="audio/ogg")["mime_type"] == "audio/ogg"
    assert to_audio_block(b"\x00\x01")["mime_type"] == "application/octet-stream"


@pytest.mark.parametrize(
    "block",
    [
        {"type": "audio", "base64": B64, "mime_type": "audio/wav"},
        {"type": "audio", "source_type": "base64", "data": B64, "mime_type": "audio/wav"},
        {"type": "input_audio", "input_audio": {"data": B64, "format": "wav"}},
        {"type": "audio", "url": f"data:audio/wav;base64,{B64}"},
    ],
)
def test_block_shapes(block: dict[str, Any]) -> None:
    assert is_audio_block(block)
    assert from_audio_block(block) == WAV


def test_block_errors() -> None:
    with pytest.raises(ValueError, match="not an audio"):
        from_audio_block({"type": "image", "base64": B64})
    with pytest.raises(ValueError, match="fetch_url"):
        from_audio_block({"type": "audio", "url": "https://example.com/a.wav"})
    with pytest.raises(ValueError, match="file_id"):
        from_audio_block({"type": "audio", "file_id": "f1"})
    with pytest.raises(ValueError, match="not an audio"):
        to_audio_block({"type": "text", "text": "x"})
    with pytest.raises(ValueError, match="empty"):
        to_audio_block(b"")


def test_decode_base64_limits() -> None:
    with pytest.raises(ValueError, match="valid base64"):
        decode_base64_audio("!!!")
    with pytest.raises(ValueError, match="larger than"):
        decode_base64_audio(B64, max_bytes=10)
    with pytest.raises(ValueError, match="only base64"):
        decode_base64_audio("data:audio/wav,abc")
    with pytest.raises(ValueError, match="empty"):
        decode_base64_audio("")


def test_latest_audio_block() -> None:
    a1, a2 = to_audio_block(WAV), to_audio_block(silent_wav(0.5))
    msgs = [
        HumanMessage(content=[dict(a1)]),
        AIMessage(content="ok"),
        {"role": "user", "content": [{"type": "text", "text": "x"}, dict(a2)]},
        HumanMessage("plain text"),
    ]
    assert latest_audio_block(msgs) == a2
    assert latest_audio_block([HumanMessage("x")]) is None


def test_check_audio_url() -> None:
    assert check_audio_url("http://10.1.2.3/a.wav", allow_private=True)
    assert check_audio_url("https://93.184.216.34/a.wav")
    for bad in ("https:///a.wav", "http://box.local/a.wav", "http://[::ffff:127.0.0.1]/a"):
        with pytest.raises(ValueError):
            check_audio_url(bad)


@respx.mock
def test_fetch_follows_safe_redirects_and_limits() -> None:
    respx.get("https://93.184.216.34/a").mock(
        return_value=httpx.Response(301, headers={"location": "/b.wav"})
    )
    respx.get("https://93.184.216.34/b.wav").mock(return_value=httpx.Response(200, content=WAV))
    assert fetch_audio_url("https://93.184.216.34/a") == WAV
    respx.get("https://93.184.216.34/loop").mock(
        return_value=httpx.Response(302, headers={"location": "/loop"})
    )
    with pytest.raises(ValueError, match="too many redirects"):
        fetch_audio_url("https://93.184.216.34/loop")
    respx.get("https://93.184.216.34/404").mock(return_value=httpx.Response(404))
    with pytest.raises(ValueError, match="404"):
        fetch_audio_url("https://93.184.216.34/404")
    respx.get("https://93.184.216.34/empty").mock(return_value=httpx.Response(200, content=b""))
    with pytest.raises(ValueError, match="empty"):
        fetch_audio_url("https://93.184.216.34/empty")
    block = {"type": "audio", "url": "https://93.184.216.34/b.wav"}
    assert from_audio_block(block, fetch_url=True) == WAV


async def test_async_fetch() -> None:
    with respx.mock:
        respx.get("https://93.184.216.34/b.wav").mock(return_value=httpx.Response(200, content=WAV))
        respx.get("https://93.184.216.34/r").mock(
            return_value=httpx.Response(302, headers={"location": "/b.wav"})
        )
        respx.get("https://93.184.216.34/big").mock(
            return_value=httpx.Response(200, content=b"x" * 100)
        )
        respx.get("https://93.184.216.34/bad").mock(return_value=httpx.Response(500))
        respx.get("https://93.184.216.34/empty").mock(return_value=httpx.Response(200))
        block = {"type": "audio", "url": "https://93.184.216.34/r"}
        assert await afrom_audio_block(block, fetch_url=True) == WAV
        assert await afrom_audio_block(to_audio_block(WAV)) == WAV
        with pytest.raises(ValueError, match="fetch_url"):
            await afrom_audio_block(block)
        with pytest.raises(ValueError, match="larger than"):
            await afrom_audio_block(
                {"type": "audio", "url": "https://93.184.216.34/big"}, fetch_url=True, max_bytes=10
            )
        for path, msg in (("bad", "500"), ("empty", "empty")):
            with pytest.raises(ValueError, match=msg):
                await afrom_audio_block(
                    {"type": "audio", "url": f"https://93.184.216.34/{path}"}, fetch_url=True
                )
        with pytest.raises(ValueError, match="not an audio"):
            await afrom_audio_block({"type": "text"})
