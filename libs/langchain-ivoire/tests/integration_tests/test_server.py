"""Integration tests against a running server.

Configuration comes only from the environment (nothing is hard-coded):

    IVOIRE_TEST_BASE_URL   base URL of the server to test (required; the tests are skipped without it)
    IVOIRE_TEST_API_KEY    API key, if the server needs one (a LAN server usually does not)
    IVOIRE_TEST_AUDIO      a recording of "Bla di like." ("Come and eat."), for the ASR test

Run with ``pytest tests/integration_tests -m integration``. Skipped when the server is unreachable.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langchain_ivoire import (
    IvoireInvalidRequestError,
    IvoireLanguageMiddleware,
    IvoireSpeaker,
    IvoireTranscriber,
    IvoireTranslator,
    IvoireUnsupportedError,
    RemoteBackend,
    with_language,
)
from langchain_ivoire.exceptions import IvoireAPIError
from langchain_ivoire.phrasebook import sound_skeleton

pytestmark = pytest.mark.integration

BASE_URL = os.environ.get("IVOIRE_TEST_BASE_URL", "")


@pytest.fixture(scope="module")
def backend() -> RemoteBackend:
    if not BASE_URL:
        pytest.skip("set IVOIRE_TEST_BASE_URL to run the integration tests")
    try:
        r = httpx.get(BASE_URL.rstrip("/") + "/health", timeout=5)
    except httpx.HTTPError as e:
        pytest.skip(f"server not reachable at {BASE_URL}: {type(e).__name__}")
    if r.status_code != 200:
        pytest.skip(f"server at {BASE_URL} not ready ({r.status_code})")
    return RemoteBackend(
        base_url=BASE_URL, api_key=os.environ.get("IVOIRE_TEST_API_KEY", ""), timeout=120
    )


@pytest.fixture(scope="module")
def come_and_eat() -> Path:
    p = os.environ.get("IVOIRE_TEST_AUDIO", "")
    if not p or not Path(p).is_file():
        pytest.skip("set IVOIRE_TEST_AUDIO to a recording of 'Bla di like.'")
    return Path(p)


def test_health(backend: RemoteBackend) -> None:
    assert isinstance(backend.health().get("ok"), bool)


def test_listen_bci(backend: RemoteBackend, come_and_eat: Path) -> None:
    t = IvoireTranscriber(backend=backend).invoke(come_and_eat)
    server_text = t.metadata.get("server_text") or t.text
    assert sound_skeleton(server_text) == sound_skeleton("Bla di like.")
    assert t.metadata["translations"]["eng"].strip().rstrip(".!").lower() == "come and eat"
    assert t.confidence is not None and 0 < t.confidence <= 1
    assert t.audio_s and t.audio_s > 0.3


def test_speak_bci(backend: RemoteBackend) -> None:
    s = IvoireSpeaker(backend=backend).invoke("Bla di like.")
    assert s.audio[:4] == b"RIFF"
    assert s.sample_rate and s.audio_s and 0.3 < s.audio_s < 10


def test_english_to_baoule_via_speak_bci(backend: RemoteBackend) -> None:
    r = IvoireTranslator(backend=backend, source="eng", target="bci").invoke("Come and eat.")
    assert r.text and r.via[0][0] == "fra"


def test_round_trip_tts_then_asr(backend: RemoteBackend) -> None:
    s = IvoireSpeaker(backend=backend).invoke("Bla di like.")
    t = IvoireTranscriber(backend=backend).invoke(s.audio)
    assert t.text


def test_text_translation_gap_is_explicit(backend: RemoteBackend) -> None:
    try:
        r = IvoireTranslator(backend=backend).invoke("Bla di like.")
    except IvoireUnsupportedError as e:
        assert "/translate" in str(e)
    else:  # a newer server with POST /translate
        assert r.text


# ---- /v1 endpoints (servers with API version >= 1.1) ------------------------------------------
@pytest.fixture(scope="module")
def v1(backend: RemoteBackend) -> RemoteBackend:
    try:
        backend.languages()
    except IvoireAPIError as e:
        if e.status_code == 404:
            pytest.skip("server has no /v1 endpoints yet")
        raise
    return backend


def test_v1_languages_and_models(v1: RemoteBackend) -> None:
    (bci,) = [lg for lg in v1.languages() if lg["code"] == "bci"]
    assert set(bci["mt"]) >= {"fra", "eng"} and "everyday" in bci["styles"]
    assert "mt" in v1.models()


def test_v1_translate_round_trip(v1: RemoteBackend) -> None:
    out = v1.translate(["Bla di like.", "N tɛ, an ti sɛn?"], source="bci", target="eng")
    assert len(out) == 2 and all(t.text for t in out)
    assert out[0].via[0][0] == "fra" and out[0].via[-1][0] == "eng"
    assert out[0].metadata["request_id"] and out[0].metadata["usage"]["segments"] == 2
    back = IvoireTranslator(backend=v1, source="fra", target="bci", style="tale").invoke(
        "Venez manger."
    )
    assert back.text


async def test_v1_translate_async(v1: RemoteBackend) -> None:
    out = await v1.atranslate(["Venez manger."], source="fra", target="bci")
    assert out[0].text


def test_v1_problem_json_errors(v1: RemoteBackend) -> None:
    with pytest.raises(IvoireInvalidRequestError) as ei:
        v1.translate(["x" * 5000], source="bci", target="fra", style="poetry")
    assert ei.value.code in ("invalid_request", "too_large") and ei.value.request_id


def test_with_language_and_middleware_on_the_server(v1: RemoteBackend) -> None:
    chat = with_language(FakeListChatModel(responses=["Je vais bien, merci."]), backend=v1)
    reply = chat.invoke("N tɛ, an ti sɛn?")
    assert reply.content and reply.response_metadata["ivoire"]["input_fra"]
    mw = IvoireLanguageMiddleware(backend=v1)
    assert mw.pivot.to_french(["Bla di like."])[0].text


def test_v1_live_answers_match_the_vendored_spec(v1: RemoteBackend) -> None:
    """Live /v1 answers validate against ``tests/contract/openapi_v1.json``."""
    from tests.unit_tests.test_api_contract import validate

    ok_200 = ("responses", "200", "content", "application/json", "schema")
    with httpx.Client(base_url=v1.base_url, headers=v1._headers(), timeout=120) as c:
        r = c.get(v1._v1("/languages"))
        validate(r.json(), "paths", "/v1/languages", "get", *ok_200)
        body = {"texts": ["Bla di like."], "source": "bci", "target": "eng", "style": "everyday"}
        validate(body, "paths", "/v1/translate", "post", "requestBody", "content", "application/json",
                 "schema")  # fmt: skip
        r = c.post(v1._v1("/translate"), json=body)
        validate(r.json(), "paths", "/v1/translate", "post", *ok_200)
        assert r.headers.get("x-request-id")
        r = c.post(v1._v1("/translate"), json={**body, "style": "poetry"})
        assert r.status_code == 400
        assert r.headers["content-type"].startswith("application/problem+json")
        validate(r.json(), "components", "schemas", "Problem")
        assert r.json()["code"] == "invalid_request"
