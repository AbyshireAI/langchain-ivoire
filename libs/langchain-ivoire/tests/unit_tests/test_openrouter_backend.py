"""OpenRouterBackend against a mocked OpenAI-compatible API (no network, no real keys)."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from langchain_ivoire import (
    IvoireApprovalRequiredError,
    IvoireAuthenticationError,
    IvoireEmbeddings,
    IvoireServerError,
    IvoireSpeaker,
    IvoireTranscriber,
    IvoireTranslator,
    IvoireUnsupportedError,
    OpenRouterBackend,
    get_backend,
)
from langchain_ivoire._audio import silent_wav
from langchain_ivoire.backends.openrouter import (
    EMBED_BATCH,
    OPENROUTER_BASE_URL,
    translation_instruction,
)
from langchain_ivoire.backends.remote import DEFAULT_BASE_URL

OR_SECRET = "sk-or-test-NEVER-PRINT-ME"
GW_SECRET = "sk-gw-test-NEVER-PRINT-ME"
GATEWAY = "https://gateway.example.test/v1"
Handler = Callable[[httpx.Request], httpx.Response]


def make(handler: Handler, **kw: object) -> tuple[OpenRouterBackend, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def wrapped(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return handler(req)

    transport = httpx.MockTransport(wrapped)
    opts: dict[str, object] = {"backoff_factor": 0.0, "max_retries": 2, **kw}
    be = OpenRouterBackend(
        client=httpx.Client(transport=transport),
        async_client=httpx.AsyncClient(transport=transport),
        **opts,  # type: ignore[arg-type]
    )
    return be, seen


def _translate(text: str, target: str) -> str:
    return {"Venez manger.": "Bla di like.", "Bla di like.": "Venez manger."}.get(
        text, f"[{target}] {text}"
    )


def api(req: httpx.Request) -> httpx.Response:
    path = req.url.path
    if path.endswith("/chat/completions"):
        body = json.loads(req.content)
        text = body["messages"][-1]["content"]
        return httpx.Response(
            200,
            json={
                "id": "gen-1",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": _translate(text, body["translation"]["target"]),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12},
            },
        )
    if path.endswith("/audio/transcriptions"):
        return httpx.Response(
            200,
            json={"text": " bla di like ", "confidence": 0.91, "usage": {"seconds": 1.5}},
            headers={"x-request-id": "req-asr"},
        )
    if path.endswith("/audio/speech"):
        return httpx.Response(
            200, content=silent_wav(0.5, 16000), headers={"content-type": "audio/wav"}
        )
    if path.endswith("/embeddings"):
        body = json.loads(req.content)
        rows = [
            {"object": "embedding", "index": i, "embedding": [float(i), float(len(t))]}
            for i, t in enumerate(body["input"])
        ]
        return httpx.Response(200, json={"data": list(reversed(rows)), "model": body["model"]})
    return httpx.Response(404, json={"error": {"message": "no such route", "code": 404}})


# ---- configuration and keys -------------------------------------------------------------------
def test_defaults_to_openrouter_and_its_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", OR_SECRET)
    monkeypatch.setenv("IVOIRE_API_KEY", GW_SECRET)
    monkeypatch.setenv("IVOIRE_BASE_URL", GATEWAY)  # belongs to the native backend only
    be, seen = make(api)
    assert be.base_url == OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
    assert be.has_api_key
    assert OR_SECRET not in repr(be) and "***" in repr(be)
    be.translate(["Venez manger."], source="fra", target="bci")
    assert str(seen[0].url) == "https://openrouter.ai/api/v1/chat/completions"
    assert seen[0].headers["authorization"] == f"Bearer {OR_SECRET}"


def test_gateway_base_url_uses_the_ivoire_key_never_the_openrouter_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", OR_SECRET)
    monkeypatch.setenv("IVOIRE_API_KEY", GW_SECRET)
    be, seen = make(api, base_url=GATEWAY + "/")
    assert be.base_url == GATEWAY
    be.translate(["Venez manger."], source="fra", target="bci")
    assert str(seen[0].url) == GATEWAY + "/chat/completions"
    assert seen[0].headers["authorization"] == f"Bearer {GW_SECRET}"
    assert all(OR_SECRET not in str(v) for r in seen for v in r.headers.values())


def test_no_key_sends_no_authorization() -> None:
    be, seen = make(api)
    assert not be.has_api_key
    be.translate(["Venez manger."], source="fra", target="bci")
    assert "authorization" not in seen[0].headers


def test_explicit_key_and_app_headers() -> None:
    be, seen = make(api, api_key=OR_SECRET, app_url="https://app.example.test", app_title="Demo")
    be.translate(["Venez manger."], source="fra", target="bci")
    h = seen[0].headers
    assert h["authorization"] == f"Bearer {OR_SECRET}"
    assert h["http-referer"] == "https://app.example.test"
    assert h["x-title"] == "Demo"


def test_rejects_bad_options() -> None:
    with pytest.raises(ValueError, match="translate_model"):
        OpenRouterBackend(translate_model=" ")
    with pytest.raises(ValueError, match="http"):
        OpenRouterBackend(base_url="ftp://example.test")


def test_get_backend_and_components_accept_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", OR_SECRET)
    be = get_backend("openrouter")
    assert isinstance(be, OpenRouterBackend) and be.has_api_key
    tr = IvoireTranslator(source="fra", target="bci", backend="openrouter", base_url=GATEWAY)
    assert isinstance(tr.backend, OpenRouterBackend)
    assert tr.backend.base_url == GATEWAY
    assert not tr.backend.has_api_key  # OPENROUTER_API_KEY is not sent to another host
    emb = IvoireEmbeddings().backend
    assert isinstance(emb, OpenRouterBackend)
    assert emb.base_url == DEFAULT_BASE_URL  # embeddings: the Abyshire API, not OpenRouter
    assert not emb.has_api_key  # OPENROUTER_API_KEY is not sent there


# ---- translation: chat completions -------------------------------------------------------------
def test_translate_convention_and_result() -> None:
    be, seen = make(api, translate_model="acme/custom-translate")
    out = IvoireTranslator(source="fra", target="bci", style="formal", backend=be).invoke(
        "Venez manger."
    )
    body = json.loads(seen[0].content)
    assert body["model"] == "acme/custom-translate"
    assert body["messages"] == [
        {"role": "system", "content": "ivoire-translate source=fra target=bci style=formal"},
        {"role": "user", "content": "Venez manger."},
    ]
    assert body["translation"] == {"source": "fra", "target": "bci", "style": "formal"}
    assert body["stream"] is False
    assert out.text == "Bla di like."
    assert (out.source, out.target, out.style) == ("fra", "bci", "formal")
    assert out.via == (("bci", "Bla di like."),)
    assert out.metadata["usage"]["total_tokens"] == 12
    assert out.metadata["request_id"] == "gen-1"
    assert out.metadata["model"] == "acme/custom-translate"


def test_default_model_ids() -> None:
    be, _ = make(api)
    assert be.translate_model == "abyshire/translate-baoule"
    assert be.listen_model == "abyshire/listen-baoule"
    assert be.speak_model == "abyshire/speak-baoule"
    assert be.embed_model == "abyshire/embed-baoule"


def test_english_goes_in_one_request_and_long_texts_are_split() -> None:
    be, seen = make(api)
    long = " ".join(["Une phrase assez longue pour remplir le texte."] * 40)
    [a, b] = be.translate(["Come and eat.", long], source="eng", target="bci")
    assert a.text == "[bci] Come and eat."
    first = json.loads(seen[0].content)
    assert first["translation"]["source"] == "eng"
    assert len(seen) > 2  # the long text went in several pieces
    assert all(len(json.loads(r.content)["messages"][-1]["content"]) <= 1000 for r in seen)
    assert b.metadata["usage"]["total_tokens"] == 12 * (len(seen) - 1)


def test_translate_validates_direction_and_style() -> None:
    be, seen = make(api)
    with pytest.raises(ValueError):
        be.translate(["x"], source="fra", target="fra")
    with pytest.raises(ValueError, match="style"):
        be.translate(["x"], source="fra", target="bci", style="poetic")
    assert be.translate([], source="fra", target="bci") == []
    assert seen == []


def test_reply_without_text_is_a_server_error() -> None:
    be, _ = make(lambda r: httpx.Response(200, json={"choices": []}))
    with pytest.raises(IvoireServerError, match="text reply"):
        be.translate(["x"], source="fra", target="bci")


async def test_atranslate() -> None:
    be, seen = make(api)
    [out] = await be.atranslate(["Bla di like."], source="bci", target="fra")
    assert out.text == "Venez manger."
    assert json.loads(seen[0].content)["messages"][0]["content"] == translation_instruction(
        "bci", "fra", "everyday"
    )
    await be.aclose()


# ---- speech ------------------------------------------------------------------------------------
def test_transcribe_multipart(wav: bytes) -> None:
    be, seen = make(api, base_url=GATEWAY)
    t = IvoireTranscriber(backend=be).invoke(wav)
    req = seen[0]
    assert str(req.url) == GATEWAY + "/audio/transcriptions"
    body = req.content
    assert b'name="model"' in body and b"abyshire/listen-baoule" in body
    assert b'name="language"' in body and b"bci" in body
    assert b'name="file"; filename="audio.wav"' in body
    assert t.text == "bla di like"
    assert t.confidence == pytest.approx(0.91)
    assert t.audio_s == pytest.approx(1.5)
    assert t.metadata["request_id"] == "req-asr"
    assert t.metadata["usage"] == {"seconds": 1.5}


async def test_atranscribe_and_lang_check(wav: bytes) -> None:
    be, _ = make(api, base_url=GATEWAY)
    t = await be.atranscribe(wav, lang="bci")
    assert t.text == "bla di like"
    with pytest.raises(IvoireUnsupportedError):
        be.transcribe(wav, lang="dyu")


def test_synthesize_wav() -> None:
    be, seen = make(api, base_url=GATEWAY, voice="narrator")
    sp = IvoireSpeaker(backend=be).invoke("Bla di like.")
    body = json.loads(seen[0].content)
    assert body == {
        "model": "abyshire/speak-baoule",
        "input": "Bla di like.",
        "voice": "narrator",
        "response_format": "wav",
    }
    assert "audio/*" in seen[0].headers["accept"]
    assert sp.audio[:4] == b"RIFF" and sp.sample_rate == 16000
    assert sp.audio_s == pytest.approx(0.5)
    assert sp.media_type == "audio/wav"


async def test_asynthesize_rejects_json_answer() -> None:
    be, _ = make(lambda r: httpx.Response(200, json={"audio_url": "/x.wav"}), base_url=GATEWAY)
    with pytest.raises(IvoireServerError, match="JSON instead of audio"):
        await be.asynthesize("Bla di like.", lang="bci")


# ---- embeddings --------------------------------------------------------------------------------
def test_embeddings_batches_order_and_kind() -> None:
    be, seen = make(api, base_url=GATEWAY)
    texts = [f"t{i}" for i in range(EMBED_BATCH + 3)]
    vecs = IvoireEmbeddings(backend=be).embed_documents(texts)
    assert len(seen) == 2
    first = json.loads(seen[0].content)
    assert first["model"] == "abyshire/embed-baoule"
    assert first["input_type"] == "document" and first["encoding_format"] == "float"
    assert len(first["input"]) == EMBED_BATCH
    assert vecs[1] == [1.0, 2.0]  # sorted back by index
    assert len(vecs) == len(texts)
    q = IvoireEmbeddings(backend=be).embed_query("Awe kuun min.")
    assert q == [0.0, 13.0]
    assert json.loads(seen[-1].content)["input_type"] == "query"


async def test_aembeddings_and_bad_kind() -> None:
    be, _ = make(api, base_url=GATEWAY)
    emb = IvoireEmbeddings(backend=be)
    assert await emb.aembed_query("abc") == [0.0, 3.0]
    assert len(await emb.aembed_documents(["a", "b"])) == 2
    with pytest.raises(ValueError, match="kind"):
        be.embed(["a"], kind="passage")


def test_embedding_count_mismatch() -> None:
    be, _ = make(lambda r: httpx.Response(200, json={"data": []}), base_url=GATEWAY)
    with pytest.raises(IvoireServerError, match="0 vectors for 1"):
        be.embed(["a"])


def test_other_backends_do_not_embed(fake: object) -> None:
    emb = IvoireEmbeddings(backend=fake)  # type: ignore[arg-type]
    with pytest.raises(IvoireUnsupportedError):
        emb.embed_query("a")


# ---- errors, retries, unsupported endpoints ----------------------------------------------------
def test_auth_error_hint_and_no_key_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", OR_SECRET)
    be, _ = make(
        lambda r: httpx.Response(
            401, json={"error": {"message": "No auth credentials", "code": 401}}
        )
    )
    with pytest.raises(IvoireAuthenticationError) as ei:
        be.translate(["x"], source="fra", target="bci")
    msg = str(ei.value)
    assert "No auth credentials" in msg and "OPENROUTER_API_KEY" in msg
    assert OR_SECRET not in msg and OR_SECRET not in repr(ei.value)


def test_429_is_retried_with_the_same_idempotency_key() -> None:
    calls = {"n": 0}

    def flaky(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": {}})
        return api(req)

    be, seen = make(flaky)
    [out] = be.translate(["Venez manger."], source="fra", target="bci")
    assert out.text == "Bla di like."
    assert len(seen) == 2
    assert seen[0].headers["idempotency-key"] == seen[1].headers["idempotency-key"]


def test_native_only_endpoints_and_local_info(wav: bytes) -> None:
    be, seen = make(api)
    for call in (
        lambda: be.speak_english("hi"),
        lambda: be.listen_english(wav),
        lambda: be.agent_text(fra="Bonjour"),
        lambda: be.fetch_audio("/audio/x.wav"),
    ):
        with pytest.raises(IvoireUnsupportedError, match="RemoteBackend"):
            call()
    assert be.health()["backend"] == "openrouter"
    assert any(lg["code"] == "bci" for lg in be.languages())
    assert seen == []


# ---- access model: translation public on OpenRouter, the rest by approval ----------------------
def test_openrouter_route_is_translation_only_by_default(wav: bytes) -> None:
    be, seen = make(api, api_key=OR_SECRET)
    assert be.via_openrouter
    assert be.translate(["Venez manger."], source="fra", target="bci")[0].text == "Bla di like."
    n = len(seen)
    for call, option in (
        (lambda: be.transcribe(wav, lang="bci"), "listen_model"),
        (lambda: be.synthesize("Bla di like.", lang="bci"), "speak_model"),
        (lambda: be.embed(["a"]), "embed_model"),
        (lambda: IvoireEmbeddings(backend=be).embed_query("a"), "embed_model"),
        (lambda: IvoireSpeaker(backend=be).invoke("Bla di like."), "speak_model"),
    ):
        with pytest.raises(IvoireApprovalRequiredError) as ei:
            call()
        msg = str(ei.value)
        assert "by approval" in msg and "api@abyshire.co.uk" in msg and option in msg
        assert OR_SECRET not in msg
    assert len(seen) == n  # nothing was sent
    assert isinstance(ei.value, IvoireUnsupportedError)


async def test_openrouter_async_gating(wav: bytes) -> None:
    be, seen = make(api)
    with pytest.raises(IvoireApprovalRequiredError):
        await be.atranscribe(wav, lang="bci")
    with pytest.raises(IvoireApprovalRequiredError):
        await be.asynthesize("x", lang="bci")
    with pytest.raises(IvoireApprovalRequiredError):
        await be.aembed(["x"])
    assert seen == []


def test_openrouter_granted_model_ids_are_allowed(wav: bytes) -> None:
    be, seen = make(
        api,
        listen_model="acme-org/baoule-listen",
        speak_model="acme-org/baoule-speak",
        embed_model="acme-org/granted-embed",
    )
    assert be.transcribe(wav, lang="bci").text == "bla di like"
    assert be.synthesize("Bla di like.", lang="bci").audio[:4] == b"RIFF"
    assert be.embed(["ab"]) == [[0.0, 2.0]]
    assert str(seen[0].url) == "https://openrouter.ai/api/v1/audio/transcriptions"
    assert b"acme-org/baoule-listen" in seen[0].content
    assert json.loads(seen[1].content)["model"] == "acme-org/baoule-speak"
    assert json.loads(seen[2].content)["model"] == "acme-org/granted-embed"


def test_gateway_serves_every_service_with_default_ids(wav: bytes) -> None:
    be, seen = make(api, base_url=GATEWAY, api_key=GW_SECRET)
    assert not be.via_openrouter
    be.transcribe(wav, lang="bci")
    be.synthesize("Bla di like.", lang="bci")
    be.embed(["a"])
    assert [r.url.path for r in seen] == [
        "/v1/audio/transcriptions",
        "/v1/audio/speech",
        "/v1/embeddings",
    ]


def test_gateway_base_url_without_v1_adds_it(wav: bytes) -> None:
    """The default hosted URL has no ``/v1``; the OpenAI-compatible routes live under it."""
    be, seen = make(api, base_url=DEFAULT_BASE_URL, api_key=GW_SECRET)
    assert be.base_url == "https://api.abyshire.ci" and not be.via_openrouter
    be.translate(["Venez manger."], source="fra", target="bci")
    be.transcribe(wav, lang="bci")
    be.synthesize("Bla di like.", lang="bci")
    be.embed(["a"])
    assert [str(r.url) for r in seen] == [
        "https://api.abyshire.ci/v1/chat/completions",
        "https://api.abyshire.ci/v1/audio/transcriptions",
        "https://api.abyshire.ci/v1/audio/speech",
        "https://api.abyshire.ci/v1/embeddings",
    ]


def test_embeddings_default_to_the_hosted_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IVOIRE_BASE_URL", raising=False)
    monkeypatch.setenv("IVOIRE_API_KEY", GW_SECRET)
    emb = IvoireEmbeddings()
    assert isinstance(emb.backend, OpenRouterBackend)
    assert emb.backend.base_url == DEFAULT_BASE_URL
    assert emb.backend._v1("/embeddings") == "/v1/embeddings"
