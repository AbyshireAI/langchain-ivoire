from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from langchain_ivoire import (
    IvoireAuthenticationError,
    IvoireConfigError,
    IvoireConnectionError,
    IvoirePermissionError,
    IvoireRateLimitError,
    IvoireServerError,
    IvoireSpeaker,
    IvoireTranscriber,
    IvoireTranslator,
    IvoireUnsupportedError,
    RemoteBackend,
)
from langchain_ivoire._audio import silent_wav
from tests.unit_tests._api_fixtures import LISTEN, SPEAK, tts

SECRET = "sk-test-NEVER-PRINT-ME"
Handler = Callable[[httpx.Request], httpx.Response]


def make(handler: Handler, **kw: object) -> tuple[RemoteBackend, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def wrapped(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return handler(req)

    transport = httpx.MockTransport(wrapped)
    opts: dict[str, object] = {"backoff_factor": 0.0, "max_retries": 2, **kw}
    be = RemoteBackend(
        client=httpx.Client(transport=transport),
        async_client=httpx.AsyncClient(transport=transport),
        **opts,  # type: ignore[arg-type]
    )
    return be, seen


def server(req: httpx.Request) -> httpx.Response:
    path = req.url.path
    if path.endswith("/health"):
        return httpx.Response(200, json={"ok": True})
    if path.endswith("/listen_bci"):
        return httpx.Response(200, json=LISTEN)
    if path.endswith("/tts_bci"):
        return httpx.Response(200, json=tts(json.loads(req.content)["bci"]))
    if path.endswith("/audio/abc.wav"):
        return httpx.Response(
            200, content=silent_wav(0.5, 16000), headers={"content-type": "audio/wav"}
        )
    if path.endswith("/speak_bci"):
        return httpx.Response(200, json=SPEAK)
    if path.endswith("/translate"):  # an older server, without /v1/translate
        return httpx.Response(405, json={"detail": "Method Not Allowed"})
    return httpx.Response(404, json={"detail": "Not Found"})


def test_defaults_env_and_auth_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IVOIRE_API_KEY", SECRET)
    monkeypatch.setenv("IVOIRE_BASE_URL", "https://example.test/v1/")
    be, seen = make(server)
    assert be.base_url == "https://example.test/v1"
    assert be.health() == {"ok": True}
    assert seen[0].url == "https://example.test/v1/health"
    assert seen[0].headers["authorization"] == f"Bearer {SECRET}"
    assert seen[0].headers["user-agent"].startswith("langchain-ivoire/")
    assert SECRET not in repr(be) and "***" in repr(be)


def test_lan_without_key_sends_no_auth_header() -> None:
    be, seen = make(server, base_url="http://ivoire.example.test:8090")
    be.health()
    assert "authorization" not in seen[0].headers
    assert not be.has_api_key
    with pytest.raises(ValueError):
        RemoteBackend(base_url="ftp://x")


def test_transcribe_uploads_multipart_and_maps_fields() -> None:
    be, seen = make(server, api_key=SECRET)
    t = IvoireTranscriber(backend=be).invoke(silent_wav(1.0))
    req = seen[0]
    assert req.method == "POST" and req.url.path == "/listen_bci"
    assert b'name="audio"; filename="audio.wav"' in req.content
    assert t.text == "bla de le kɛ"  # raw ASR; the server's phrase snap is in metadata
    assert t.metadata["server_text"] == "Bla di like."
    assert t.metadata["translations"] == {"fra": "Venez manger.", "eng": "Come and eat."}
    assert t.confidence == pytest.approx(0.91) and t.audio_s == pytest.approx(1.2)


def test_synthesize_fetches_relative_audio_url() -> None:
    be, seen = make(server, api_key=SECRET)
    s = IvoireSpeaker(backend=be).invoke("Bla di like.")
    assert [r.url.path for r in seen] == ["/tts_bci", "/audio/abc.wav"]
    assert json.loads(seen[0].content) == {"bci": "Bla di like."}
    assert s.audio[:4] == b"RIFF" and s.sample_rate == 16000 and s.audio_s == pytest.approx(0.5)
    assert all(r.headers["authorization"] == f"Bearer {SECRET}" for r in seen)


def test_translate_uses_translate_endpoint_when_available() -> None:
    def h(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body == {
            "texts": ["Bla di like."],
            "source": "bci",
            "target": "fra",
            "style": "everyday",
        }
        return httpx.Response(200, json={"texts": ["Venez manger."]})

    be, _ = make(h)
    assert IvoireTranslator(backend=be).invoke("Bla di like.").text == "Venez manger."


def test_style_aliases_rename_the_style_on_the_wire(tmp_path: Path) -> None:
    sent: list[str] = []

    def h(req: httpx.Request) -> httpx.Response:
        sent.append(json.loads(req.content)["style"])
        return httpx.Response(200, json={"texts": ["Venez manger."]})

    be, _ = make(h, style_aliases={"formal": "register-2"})
    r = IvoireTranslator(backend=be, style="formal").invoke("Bla di like.")
    assert r.style == "formal" and sent == ["register-2"]  # the caller still sees "formal"
    IvoireTranslator(backend=be).invoke("Bla di like.")
    assert sent[-1] == "everyday"  # styles without an alias are sent as they are

    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"remote": {"style_aliases": {"tale": "story"}}}), encoding="utf-8")
    be, _ = make(h, settings=str(f))
    IvoireTranslator(backend=be, style="tale").invoke("Bla di like.")
    assert sent[-1] == "story"
    with pytest.raises(IvoireConfigError, match="style_aliases"):
        make(h, style_aliases={"formal": 2})


def test_translate_fallback_and_unsupported() -> None:
    be, seen = make(server)
    r = IvoireTranslator(backend=be, source="eng", target="bci").invoke("Come and eat.")
    assert r.text == "Bla di like." and r.via == (("fra", "Venez manger."), ("bci", "Bla di like."))
    assert json.loads(seen[-1].content) == {"text": "Come and eat.", "use_phrasebook": False}
    n = len(seen)
    with pytest.raises(IvoireUnsupportedError, match="no POST /v1/translate"):
        IvoireTranslator(backend=be).invoke("Bla di like.")
    assert len(seen) == n  # the missing endpoint is remembered, not probed again
    with pytest.raises(NotImplementedError):  # IvoireUnsupportedError is a NotImplementedError
        IvoireTranslator(backend=be, source="eng", target="bci", style="formal").invoke("x")


@pytest.mark.parametrize(
    ("status", "exc"),
    [(401, IvoireAuthenticationError), (403, IvoirePermissionError), (500, IvoireServerError)],
)
def test_error_mapping_without_leaking_key(status: int, exc: type[Exception]) -> None:
    be, seen = make(lambda r: httpx.Response(status, json={"error": "nope"}), api_key=SECRET)
    with pytest.raises(exc) as ei:
        be.health()
    assert SECRET not in str(ei.value) and "nope" in str(ei.value)
    assert len(seen) == 1  # not retried
    assert getattr(ei.value, "status_code", None) == status


def test_retries_429_and_503_then_succeeds() -> None:
    answers = iter(
        [
            httpx.Response(429, headers={"retry-after": "0"}, json={"error": "rate limit"}),
            httpx.Response(503, json={"error": "loading"}),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    be, seen = make(lambda r: next(answers))
    assert be.health() == {"ok": True}
    assert len(seen) == 3


def test_rate_limit_gives_up_after_retries() -> None:
    be, seen = make(
        lambda r: httpx.Response(429, headers={"retry-after": "0"}, json={"error": "slow down"}),
        max_retries=1,
    )
    with pytest.raises(IvoireRateLimitError) as ei:
        be.health()
    assert ei.value.retry_after == 0.0
    assert len(seen) == 2


def test_connection_error() -> None:
    def h(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    be, seen = make(h, max_retries=1)
    with pytest.raises(IvoireConnectionError):
        be.health()
    assert len(seen) == 2


def test_only_baoule() -> None:
    be, _ = make(server)
    with pytest.raises(IvoireUnsupportedError):
        be.synthesize("x", lang="dyu")


async def test_async_paths() -> None:
    answers = iter([httpx.Response(503), httpx.Response(200, json=LISTEN)])
    be, seen = make(lambda r: next(answers))
    t = await IvoireTranscriber(backend=be).ainvoke(silent_wav(1.0))
    assert t.metadata["server_text"] == "Bla di like." and len(seen) == 2
    be2, _ = make(server)
    s = await IvoireSpeaker(backend=be2).ainvoke("Bla di like.")
    assert s.audio[:4] == b"RIFF"
    assert (await be2.ahealth())["ok"] is True
