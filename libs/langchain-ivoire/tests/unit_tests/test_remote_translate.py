"""``POST /v1/translate``, ``GET /v1/languages`` and ``GET /v1/models`` against respx mocks."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest
import respx
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langchain_ivoire import (
    IvoireAPIError,
    IvoireAuthenticationError,
    IvoireConnectionError,
    IvoireIdempotencyConflictError,
    IvoireInvalidRequestError,
    IvoireNotFoundError,
    IvoirePermissionError,
    IvoireRateLimitError,
    IvoireServerError,
    IvoireTooLargeError,
    IvoireTranslator,
    IvoireUnsupportedError,
    IvoireUnsupportedLanguageError,
    IvoireUpstreamUnavailableError,
    RemoteBackend,
    with_language,
)
from langchain_ivoire.backends import remote as remote_mod
from tests.unit_tests._api_fixtures import LANGUAGES, MODELS, ok, problem

BASE = "https://api.example.test"
SECRET = "sk-test-NEVER-PRINT-ME"


@pytest.fixture
def be() -> RemoteBackend:
    return RemoteBackend(base_url=BASE, api_key=SECRET, backoff_factor=0.0, max_retries=2)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    seen: list[float] = []

    async def asleep(d: float) -> None:
        seen.append(d)

    monkeypatch.setattr(time, "sleep", seen.append)
    monkeypatch.setattr(asyncio, "sleep", asleep)
    return seen


# ---- happy paths -------------------------------------------------------------------------------
@respx.mock
def test_translate_request_and_result(be: RemoteBackend) -> None:
    route = respx.post(f"{BASE}/v1/translate").mock(return_value=ok(["Venez manger."]))
    r = IvoireTranslator(backend=be, style="tale").invoke("Bla di like.")
    req = route.calls[0].request
    assert json.loads(req.content) == {
        "texts": ["Bla di like."],
        "source": "bci",
        "target": "fra",
        "style": "tale",
    }
    assert req.headers["authorization"] == f"Bearer {SECRET}"
    assert len(req.headers["idempotency-key"]) == 36
    assert r.text == "Venez manger." and r.via == (("fra", "Venez manger."),)
    assert r.metadata["request_id"] == "req-1"
    assert r.metadata["model"]["mt"] == "abyshire/translate-baoule"
    assert r.metadata["usage"] == {"characters": 13, "segments": 1}
    assert r.timings_ms == {"mt": 12}
    assert be.last_request_id == "req-1"


@respx.mock
def test_base_url_with_v1_and_english_pivot() -> None:
    be = RemoteBackend(base_url=f"{BASE}/v1")
    route = respx.post(f"{BASE}/v1/translate").mock(
        return_value=ok(["Come and eat."], pivot=["Venez manger."])
    )
    r = IvoireTranslator(backend=be, target="eng").invoke("Bla di like.")
    assert route.called and "authorization" not in route.calls[0].request.headers
    assert r.via == (("fra", "Venez manger."), ("eng", "Come and eat."))


@respx.mock
def test_batches_of_32_keep_order(be: RemoteBackend) -> None:
    def echo(req: httpx.Request) -> httpx.Response:
        texts = json.loads(req.content)["texts"]
        assert len(texts) <= 32
        return ok([t.upper() for t in texts], rid=f"r{len(texts)}")

    route = respx.post(f"{BASE}/v1/translate").mock(side_effect=echo)
    texts = [f"text {i}" for i in range(70)]
    out = be.translate(texts, source="bci", target="fra")
    assert [len(json.loads(c.request.content)["texts"]) for c in route.calls] == [32, 32, 6]
    assert [t.text for t in out] == [t.upper() for t in texts]
    assert out[0].metadata["request_ids"] == ["r32", "r32", "r6"]
    keys = {c.request.headers["idempotency-key"] for c in route.calls}
    assert len(keys) == 3  # one key per logical request


@respx.mock
def test_long_texts_are_split_at_sentences_and_joined(be: RemoteBackend) -> None:
    def echo(req: httpx.Request) -> httpx.Response:
        texts = json.loads(req.content)["texts"]
        assert all(len(t) <= 1000 for t in texts)
        return ok([f"<{len(t)}>" for t in texts])

    respx.post(f"{BASE}/v1/translate").mock(side_effect=echo)
    long = " ".join(["Ɔ ti kpa."] * 300)  # 2999 characters
    one_word = "a" * 2500
    out = be.translate([long, "court", one_word], source="bci", target="fra")
    assert out[0].text.count("<") == 3 and out[0].source_text == long
    assert out[1].text == "<5>"
    assert out[2].text == "<1000> <1000> <500>"


async def test_async_translate_and_info(be: RemoteBackend) -> None:
    with respx.mock:
        respx.post(f"{BASE}/v1/translate").mock(return_value=ok(["Venez manger."]))
        respx.get(f"{BASE}/v1/languages").mock(return_value=httpx.Response(200, json=LANGUAGES))
        respx.get(f"{BASE}/v1/models").mock(return_value=httpx.Response(200, json=MODELS))
        r = await IvoireTranslator(backend=be).ainvoke("Bla di like.")
        assert r.text == "Venez manger." and r.metadata["request_id"] == "req-1"
        assert (await be.alanguages())[0]["code"] == "bci"
        assert (await be.amodels())["mt"]["name"] == "abyshire/translate-baoule"
        assert be.languages()[0]["mt"] == ["fra", "eng"]
        assert be.models()["mt"]["version"] == "2026-10-15"


@respx.mock
def test_with_language_works_with_the_remote_backend(be: RemoteBackend) -> None:
    table = {
        ("bci", "fra", "N tɛ, an ti sɛn?"): "Bonjour, comment allez-vous ?",
        ("fra", "bci", "Je vais bien, merci."): "N liɛ yo'a kpa, n yo.",
    }

    def mt(req: httpx.Request) -> httpx.Response:
        b = json.loads(req.content)
        return ok([table[(b["source"], b["target"], t)] for t in b["texts"]])

    route = respx.post(f"{BASE}/v1/translate").mock(side_effect=mt)
    chat = with_language(FakeListChatModel(responses=["Je vais bien, merci."]), backend=be)
    out = chat.invoke("N tɛ, an ti sɛn?")
    assert out.content == "N liɛ yo'a kpa, n yo."
    assert out.response_metadata["ivoire"]["input_fra"] == "Bonjour, comment allez-vous ?"
    assert len(route.calls) == 2


# ---- problem+json errors -----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("status", "code", "exc", "parents"),
    [
        (400, "invalid_request", IvoireInvalidRequestError, (ValueError,)),
        (422, "invalid_request", IvoireInvalidRequestError, (ValueError,)),
        (413, "too_large", IvoireTooLargeError, (IvoireInvalidRequestError, ValueError)),
        (400, "unsupported_language", IvoireUnsupportedLanguageError, (IvoireUnsupportedError,)),
        (401, "unauthorized", IvoireAuthenticationError, ()),
        (403, "forbidden", IvoirePermissionError, ()),
        (409, "idempotency_conflict", IvoireIdempotencyConflictError, ()),
        (429, "rate_limited", IvoireRateLimitError, ()),
        (503, "upstream_unavailable", IvoireUpstreamUnavailableError, (IvoireServerError,)),
        (500, "internal", IvoireServerError, ()),
    ],
)
@respx.mock
def test_problem_codes_map_to_typed_errors(
    status: int,
    code: str,
    exc: type[Exception],
    parents: tuple[type[Exception], ...],
    sleeps: list[float],
) -> None:
    be = RemoteBackend(base_url=BASE, api_key=SECRET, max_retries=0)
    respx.post(f"{BASE}/v1/translate").mock(
        return_value=problem(status, code, "the detail", **{"retry-after": "3"})
    )
    with pytest.raises(exc) as ei:
        be.translate(["x"], source="bci", target="fra")
    e = ei.value
    assert isinstance(e, IvoireAPIError) and all(isinstance(e, p) for p in parents)
    assert e.status_code == status and e.code == code
    assert e.detail == "the detail" and e.title == code.replace("_", " ")
    assert e.type and e.type.endswith(code)
    assert e.request_id == f"req-{code}" and f"req-{code}" in str(e)
    assert SECRET not in str(e) and SECRET not in repr(e.problem)
    if isinstance(e, IvoireRateLimitError):
        assert e.retry_after == 3.0
    assert sleeps == []  # max_retries=0
    assert not be._has_translate  # an error answer is not "endpoint missing"


@respx.mock
def test_status_without_problem_body_still_maps(be: RemoteBackend) -> None:
    respx.post(f"{BASE}/v1/translate").mock(
        return_value=httpx.Response(413, json={"detail": "too many texts"})
    )
    with pytest.raises(IvoireTooLargeError, match="too many texts") as ei:
        be.translate(["x"], source="bci", target="fra")
    assert ei.value.code is None


@respx.mock
def test_missing_endpoint_falls_back_and_is_remembered(be: RemoteBackend) -> None:
    tr = respx.post(f"{BASE}/v1/translate").mock(
        return_value=httpx.Response(404, json={"detail": "Not Found"})
    )
    with pytest.raises(IvoireUnsupportedError, match="no POST /v1/translate"):
        be.translate(["Bla di like."], source="bci", target="fra")
    with pytest.raises(IvoireUnsupportedError):
        be.translate(["Bla di like."], source="bci", target="fra")
    assert tr.call_count == 1
    speak = respx.post(f"{BASE}/speak_bci").mock(
        return_value=httpx.Response(200, json={"fra": "Venez manger.", "bci": "Bla di like."})
    )
    out = be.translate(["Come and eat."], source="eng", target="bci")
    assert out[0].text == "Bla di like." and speak.called


@respx.mock
def test_problem_404_is_an_error_not_a_missing_endpoint(be: RemoteBackend) -> None:
    respx.post(f"{BASE}/v1/translate").mock(return_value=problem(404, "not_found"))
    with pytest.raises(IvoireNotFoundError):
        be.translate(["x"], source="bci", target="fra")
    assert be._has_translate is None


@respx.mock
def test_wrong_number_of_texts_is_a_server_error(be: RemoteBackend) -> None:
    respx.post(f"{BASE}/v1/translate").mock(return_value=ok(["a", "b"]))
    with pytest.raises(IvoireServerError, match="returned 2 texts for 1"):
        be.translate(["x"], source="bci", target="fra")


# ---- retries -----------------------------------------------------------------------------------
@respx.mock
def test_retry_after_seconds_is_honoured_with_the_same_idempotency_key(
    be: RemoteBackend, sleeps: list[float]
) -> None:
    route = respx.post(f"{BASE}/v1/translate").mock(
        side_effect=[problem(429, "rate_limited", **{"retry-after": "2"}), ok(["Venez manger."])]
    )
    assert be.translate(["x"], source="bci", target="fra")[0].text == "Venez manger."
    assert sleeps == [2.0]
    k1, k2 = (c.request.headers["idempotency-key"] for c in route.calls)
    assert k1 == k2


@respx.mock
def test_retry_after_http_date_and_cap(sleeps: list[float]) -> None:
    be = RemoteBackend(base_url=BASE, backoff_factor=0.0, max_retries=2, max_backoff=5.0)
    later = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=60), usegmt=True)
    respx.post(f"{BASE}/v1/translate").mock(
        side_effect=[
            problem(429, "rate_limited", **{"retry-after": later}),
            problem(503, "upstream_unavailable", **{"retry-after": "soon"}),
            ok(["y"]),
        ]
    )
    assert be.translate(["x"], source="bci", target="fra")[0].text == "y"
    assert sleeps[0] == 5.0  # 60 s capped at max_backoff
    assert sleeps[1] == 0.0  # unparseable Retry-After -> exponential backoff (factor 0)


@respx.mock
def test_rate_limit_gives_up_after_retries(be: RemoteBackend, sleeps: list[float]) -> None:
    route = respx.post(f"{BASE}/v1/translate").mock(
        return_value=problem(429, "rate_limited", **{"retry-after": "1"})
    )
    with pytest.raises(IvoireRateLimitError) as ei:
        be.translate(["x"], source="bci", target="fra")
    assert route.call_count == 3 and sleeps == [1.0, 1.0]
    assert ei.value.retry_after == 1.0


@respx.mock
def test_bad_gateway_html_is_retried_then_raised(be: RemoteBackend, sleeps: list[float]) -> None:
    respx.post(f"{BASE}/v1/translate").mock(
        return_value=httpx.Response(502, text="<html>Bad Gateway</html>")
    )
    with pytest.raises(IvoireServerError, match="Bad Gateway"):
        be.translate(["x"], source="bci", target="fra")
    assert len(sleeps) == 2


@pytest.mark.parametrize(
    "error",
    [httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), httpx.RemoteProtocolError("eof")],
)
@respx.mock
def test_transport_errors_become_connection_errors(
    be: RemoteBackend, sleeps: list[float], error: Exception
) -> None:
    route = respx.post(f"{BASE}/v1/translate").mock(side_effect=error)
    with pytest.raises(IvoireConnectionError, match=r"cannot reach https://api\.example\.test"):
        be.translate(["x"], source="bci", target="fra")
    assert route.call_count == 3
    assert len({c.request.headers["idempotency-key"] for c in route.calls}) == 1


async def test_async_errors_and_retries(be: RemoteBackend, sleeps: list[float]) -> None:
    with respx.mock:
        respx.post(f"{BASE}/v1/translate").mock(
            side_effect=[problem(503, "upstream_unavailable"), ok(["z"])]
        )
        assert (await be.atranslate(["x"], source="bci", target="fra"))[0].text == "z"
        assert len(sleeps) == 1
    with respx.mock:
        respx.post(f"{BASE}/v1/translate").mock(return_value=problem(413, "too_large"))
        with pytest.raises(IvoireTooLargeError):
            await be.atranslate(["x"], source="bci", target="fra")
    with respx.mock:
        respx.post(f"{BASE}/v1/translate").mock(side_effect=httpx.ConnectError("down"))
        with pytest.raises(IvoireConnectionError):
            await be.atranslate(["x"], source="bci", target="fra")
    with respx.mock:
        be2 = RemoteBackend(base_url=BASE)
        respx.post(f"{BASE}/v1/translate").mock(return_value=httpx.Response(405))
        respx.post(f"{BASE}/speak_bci").mock(
            return_value=httpx.Response(200, json={"fra": "Venez manger."})
        )
        out = await be2.atranslate(["Come and eat."], source="eng", target="fra")
        assert out[0].text == "Venez manger."


def test_empty_and_same_language(be: RemoteBackend) -> None:
    assert be.translate([], source="bci", target="fra") == []
    with pytest.raises(ValueError):
        be.translate(["x"], source="bci", target="bci")


def test_split_long_helper() -> None:
    parts = remote_mod._split_long("Un. " * 400, limit=50)
    assert all(len(p) <= 50 for p in parts) and " ".join(parts) == ("Un. " * 400).strip()
    assert remote_mod._split_long("court") == ["court"]


def test_async_client_survives_separate_event_loops() -> None:
    """asyncio.run twice with one backend (each run closes its loop)."""
    be = RemoteBackend(base_url=BASE)
    with respx.mock:
        respx.post(f"{BASE}/v1/translate").mock(return_value=ok(["a"]))
        first = asyncio.run(be.atranslate(["x"], source="bci", target="fra"))
        second = asyncio.run(be.atranslate(["y"], source="bci", target="fra"))
    assert first[0].text == second[0].text == "a"
    asyncio.run(be.aclose())


@respx.mock
def test_long_audio_is_too_large(be: RemoteBackend) -> None:
    """The server answers 400 ``too_large`` for audio that is too long: the code wins."""
    from langchain_ivoire import IvoireTranscriber
    from langchain_ivoire._audio import silent_wav

    respx.post(f"{BASE}/listen_bci").mock(return_value=problem(400, "too_large", "audio > 30 s"))
    with pytest.raises(IvoireTooLargeError, match="audio > 30 s") as ei:
        IvoireTranscriber(backend=be).invoke(silent_wav(0.2))
    assert ei.value.status_code == 400 and ei.value.code == "too_large"
