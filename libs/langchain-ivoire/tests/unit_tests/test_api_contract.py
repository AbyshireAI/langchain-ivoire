"""Contract tests: the remote client and its test fixtures against the server's OpenAPI spec.

The spec is a vendored copy, ``tests/contract/openapi_v1.json`` (see ``tests/contract/README.md``
for how to refresh it). These tests check that

* every endpoint and method :class:`RemoteBackend` calls exists in the spec, with only declared
  query parameters and headers within their limits;
* every request body the client builds validates against the spec's request schema (JSON bodies
  with jsonschema, multipart bodies by their declared form fields);
* every response fixture the unit tests use (``_api_fixtures``) validates against the spec's
  response schema, including the ``Problem`` (RFC 9457) error schema, and every documented
  problem ``code`` maps to a typed exception;
* the response fields the client reads are declared in the spec.

Offline: requests go to an ``httpx.MockTransport``.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

import httpx
import pytest
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from langchain_ivoire import IvoireAPIError, RemoteBackend
from langchain_ivoire._audio import silent_wav
from langchain_ivoire.backends import remote as remote_mod
from tests.unit_tests import _api_fixtures as fx

SPEC_PATH = Path(__file__).resolve().parents[1] / "contract" / "openapi_v1.json"
SPEC: dict[str, Any] = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
SPEC_URI = "urn:ivoire:openapi-v1"
REGISTRY: Registry[Any] = Registry().with_resource(
    SPEC_URI, Resource.from_contents(SPEC, default_specification=DRAFT202012)
)
BASE = "https://api.example.test"  # like the hosted default (no /v1 suffix)
# Endpoints the client still offers for self-hosted servers; the hosted API does not serve them.
SELF_HOSTED_ONLY = {"/listen_eng", "/agent_text"}
METHODS = ("get", "post", "put", "patch", "delete")


# ---- spec helpers ------------------------------------------------------------------------------
def _pointer(*parts: str) -> str:
    return "#/" + "/".join(p.replace("~", "~0").replace("/", "~1") for p in parts)


def validate(instance: Any, *pointer: str) -> None:
    """Validate ``instance`` against the schema at ``pointer`` inside the spec (refs resolved)."""
    schema = {"$ref": SPEC_URI + _pointer(*pointer)}
    Draft202012Validator(schema, registry=REGISTRY).validate(instance)


def _template_regex(template: str) -> re.Pattern[str]:
    return re.compile("^" + re.sub(r"\\\{[^}]+\\\}", "[^/]+", re.escape(template)) + "$")


SPEC_PATHS = [(t, _template_regex(t)) for t in SPEC["paths"]]


def _rel(url_path: str) -> str:
    """A request path without the optional ``/v1`` prefix (native routes are served under both)."""
    return url_path[len("/v1") :] if url_path.startswith("/v1/") else url_path


def spec_path(url_path: str) -> str:
    """The spec path template for a request path. The spec lists each native route once, either at
    the root (``/listen_bci``) or under ``/v1`` (``/v1/translate``); the API serves both forms."""
    rel = _rel(url_path)
    for candidate in (url_path, rel, "/v1" + rel):
        matches = [t for t, rx in SPEC_PATHS if rx.match(candidate)]
        if matches:
            return str(matches[0])
    raise AssertionError(f"{url_path} is not in the spec (paths: {sorted(SPEC['paths'])})")


def problem_codes() -> set[str]:
    """The stable problem codes the spec lists in ``info.description``."""
    m = re.search(r"stable codes:\s*([a-z_,\s]+)\.", SPEC["info"]["description"], re.IGNORECASE)
    assert m, "the spec no longer lists its problem codes in info.description"
    return {c.strip() for c in m.group(1).split(",") if c.strip()}


def deref(node: Any) -> Any:
    """Follow a local ``$ref`` (``#/components/...``) inside the spec."""
    while isinstance(node, dict) and "$ref" in node:
        target: Any = SPEC
        for part in node["$ref"].removeprefix("#/").split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
        node = target
    return node


# The problem codes each status stands for when a response does not name its codes.
CODES_BY_STATUS = {
    400: ("invalid_request", "unsupported_language"),
    401: ("unauthorized",),
    403: ("forbidden",),
    404: ("not_found", "model_not_found"),
    405: ("method_not_allowed",),
    409: ("idempotency_conflict",),
    413: ("too_large",),
    429: ("rate_limited",),
    500: ("internal",),
    503: ("upstream_unavailable",),
}


def documented_errors() -> Iterator[tuple[str, str, int, str]]:
    """(path, method, status, code) for every problem+json answer of the spec: the codes its
    description names, else the codes of its status."""
    codes = problem_codes()
    for path, item in SPEC["paths"].items():
        for method, op in item.items():
            if method not in METHODS:
                continue
            for status, ref in op.get("responses", {}).items():
                resp = deref(ref)
                if "application/problem+json" not in resp.get("content", {}):
                    continue
                named = [
                    c for c in re.findall(r"[a-z_]+", resp.get("description", "")) if c in codes
                ]
                for code in named or CODES_BY_STATUS.get(int(status), ()):
                    if code in codes:
                        yield path, method, int(status), code


# ---- a spec-conforming mock server that records what the client sends --------------------------
def spec_server(seen: list[httpx.Request]) -> httpx.MockTransport:
    def handle(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        rel = _rel(req.url.path)
        if rel == "/health":
            return httpx.Response(200, json={"ok": True})
        if rel == "/languages":
            return httpx.Response(200, json=fx.LANGUAGES)
        if rel == "/models":
            return httpx.Response(200, json=fx.MODELS)
        if rel in ("/translate", "/translations"):
            b = json.loads(req.content)
            pivot = list(b["texts"]) if "fra" not in (b["source"], b["target"]) else None
            body = fx.translate_body(
                list(b["texts"]),
                source=b["source"],
                target=b["target"],
                style=b.get("style", "everyday"),
                pivot=pivot,
            )
            return httpx.Response(200, headers={"x-request-id": "req-1"}, json=body)
        if rel == "/listen_bci":
            return httpx.Response(200, json=fx.LISTEN)
        if rel == "/listen_eng":
            return httpx.Response(200, json={"eng": "Come and eat."})
        if rel == "/tts_bci":
            return httpx.Response(200, json=fx.tts(json.loads(req.content)["bci"]))
        if rel == "/speak_bci":
            return httpx.Response(200, json=fx.SPEAK)
        if rel == "/agent_text":
            return httpx.Response(200, json={"fra": "Bonjour.", "bci": "N tɛ."})
        if rel.startswith("/audio/"):
            return httpx.Response(
                200, content=silent_wav(0.3), headers={"content-type": "audio/wav"}
            )
        return fx.problem(404, "not_found")

    return httpx.MockTransport(handle)


def _exercise(be: RemoteBackend) -> None:
    wav = silent_wav(0.5)
    be.health()
    be.languages()
    be.models()
    be.translate(["Bla di like."], source="bci", target="fra", style="tale")
    be.translate(["Bla di like."], source="bci", target="eng")
    be.translate(["Venez manger."], source="fra", target="bci", style="formal")
    be.translate(
        [f"t {i}" for i in range(40)] + [" ".join(["Ɔ ti kpa."] * 150)], source="bci", target="fra"
    )
    be.transcribe(wav, lang="bci")
    be.synthesize("Bla di like.", lang="bci")
    be.speak_english("Come and eat.")
    be.listen_english(wav)
    be.agent_text(fra="Bonjour.", conv="c1")
    be.agent_text(eng="Hello.")
    be.fetch_audio("/audio/abc.wav")


async def _aexercise(be: RemoteBackend) -> None:
    await be.ahealth()
    await be.alanguages()
    await be.amodels()
    await be.atranslate(["Bla di like."], source="bci", target="fra")
    await be.atranscribe(silent_wav(0.5), lang="bci")
    await be.asynthesize("Bla di like.", lang="bci")


@pytest.fixture(scope="module")
def recorded() -> list[httpx.Request]:
    seen: list[httpx.Request] = []
    transport = spec_server(seen)
    be = RemoteBackend(
        base_url=BASE,
        api_key="sk-test-contract",
        client=httpx.Client(transport=transport),
        async_client=httpx.AsyncClient(transport=transport),
    )
    _exercise(be)
    asyncio.run(_aexercise(be))
    assert seen
    return seen


def _ids(reqs: list[httpx.Request]) -> list[str]:
    return [f"{r.method} {r.url.path}" for r in reqs]


# ---- requests ----------------------------------------------------------------------------------
def test_spec_is_openapi_3_1_with_the_default_server() -> None:
    assert SPEC["openapi"].startswith("3.1")
    assert remote_mod.DEFAULT_BASE_URL in [s["url"] for s in SPEC["servers"]]


def test_every_call_is_a_spec_endpoint_and_method(recorded: list[httpx.Request]) -> None:
    called = set()
    for req in recorded:
        if _rel(req.url.path) in SELF_HOSTED_ONLY:
            continue
        path = spec_path(req.url.path)
        assert req.method.lower() in SPEC["paths"][path], f"{req.method} {path} not in the spec"
        called.add((req.method, path))
    # the client covers every endpoint it documents
    assert {p for _, p in called} >= {
        "/v1/translate", "/v1/languages", "/v1/models", "/health", "/listen_bci",
        "/tts_bci", "/speak_bci", "/audio/{aid}.wav",
    }  # fmt: skip


def test_self_hosted_only_endpoints_are_not_in_the_hosted_spec(
    recorded: list[httpx.Request],
) -> None:
    assert {_rel(r.url.path) for r in recorded} >= SELF_HOSTED_ONLY
    for path in SELF_HOSTED_ONLY:
        assert path not in SPEC["paths"] and "/v1" + path not in SPEC["paths"]


def test_query_parameters_and_headers_are_declared(recorded: list[httpx.Request]) -> None:
    for req in recorded:
        if _rel(req.url.path) in SELF_HOSTED_ONLY:
            continue
        path = spec_path(req.url.path)
        op = SPEC["paths"][path][req.method.lower()]
        params = {(p["in"], p["name"].lower()): p for p in map(deref, op.get("parameters", []))}
        for name, _ in parse_qsl(req.url.query.decode()):
            assert ("query", name.lower()) in params, f"undeclared query {name!r} on {path}"
        for (where, name), p in params.items():
            if where == "path":
                continue
            if where == "header" and name in req.headers:
                Draft202012Validator(p["schema"]).validate(req.headers[name])
        if req.method == "POST":
            assert ("header", "idempotency-key") in params, f"{path} takes no Idempotency-Key"
            assert "idempotency-key" in req.headers
        assert req.headers["authorization"] == "Bearer sk-test-contract"  # the bearer scheme


def _multipart_fields(req: httpx.Request) -> dict[str, bytes]:
    ctype = req.headers["content-type"]
    assert ctype.startswith("multipart/form-data"), ctype
    boundary = ctype.split("boundary=", 1)[1].encode()
    fields: dict[str, bytes] = {}
    for part in req.content.split(b"--" + boundary):
        head, _, body = part.partition(b"\r\n\r\n")
        m = re.search(rb'name="([^"]+)"', head)
        if m:
            fields[m.group(1).decode()] = body.rstrip(b"\r\n")
    return fields


def test_request_bodies_validate(recorded: list[httpx.Request]) -> None:
    checked = 0
    for req in recorded:
        if _rel(req.url.path) in SELF_HOSTED_ONLY:
            continue
        path = spec_path(req.url.path)
        op = SPEC["paths"][path][req.method.lower()]
        body = op.get("requestBody")
        if body is None:
            assert not req.content, f"{req.method} {path} sends a body the spec does not declare"
            continue
        content = body["content"]
        if "application/json" in content:
            assert req.headers["content-type"].startswith("application/json")
            validate(json.loads(req.content), "paths", path, req.method.lower(), "requestBody",
                     "content", "application/json", "schema")  # fmt: skip
        else:
            schema = deref(content["multipart/form-data"]["schema"])
            fields = _multipart_fields(req)
            assert set(schema.get("required", [])) <= set(fields), (path, sorted(fields))
            assert set(fields) <= set(schema["properties"]), (path, sorted(fields))
            for name, value in fields.items():
                prop = schema["properties"][name]
                if prop.get("type") == "integer":
                    assert value.decode().lstrip("-").isdigit(), (name, value)
        checked += 1
    assert checked >= 10


def test_translate_batches_respect_the_spec_limits(recorded: list[httpx.Request]) -> None:
    item = SPEC["components"]["schemas"]["TranslateRequest"]["properties"]["texts"]
    assert item["maxItems"] == remote_mod.MAX_BATCH
    assert item["items"]["maxLength"] == remote_mod.MAX_TEXT_CHARS
    sizes = [
        len(json.loads(r.content)["texts"]) for r in recorded if r.url.path.endswith("/translate")
    ]
    assert max(sizes) == item["maxItems"]  # the 41-text call was split into 32 + 9


# ---- responses ---------------------------------------------------------------------------------
RESPONSE_FIXTURES: list[tuple[str, str, Any]] = [
    ("/listen_bci", "post", fx.LISTEN),
    ("/speak_bci", "post", fx.SPEAK),
    ("/tts_bci", "post", fx.tts("Bla di like.")),
    ("/v1/languages", "get", fx.LANGUAGES),
    ("/v1/models", "get", fx.MODELS),
    ("/v1/translate", "post", fx.translate_body(["Venez manger."])),
    (
        "/v1/translate",
        "post",
        fx.translate_body(["Come and eat."], target="eng", pivot=["Venez manger."]),
    ),
    ("/v1/translate", "post", json.loads(fx.ok(["a", "b"]).content)),
]


@pytest.mark.parametrize(("path", "method", "body"), RESPONSE_FIXTURES)
def test_fixture_responses_validate(path: str, method: str, body: Any) -> None:
    validate(
        body, "paths", path, method, "responses", "200", "content", "application/json", "schema"
    )


@pytest.mark.parametrize(("path", "method", "status", "code"), sorted(set(documented_errors())))
def test_documented_problems_validate_and_map_to_typed_errors(
    path: str, method: str, status: int, code: str
) -> None:
    resp = fx.problem(status, code, "the detail")
    assert resp.headers["content-type"] == "application/problem+json"
    schema = deref(SPEC["paths"][path][method]["responses"][str(status)])["content"][
        "application/problem+json"
    ]["schema"]
    if "$ref" in schema:
        schema = {"$ref": SPEC_URI + schema["$ref"]}
    Draft202012Validator(schema, registry=REGISTRY).validate(resp.json())
    req = httpx.Request(method.upper(), f"{BASE}{path}")
    resp.request = req
    with pytest.raises(IvoireAPIError) as ei:
        remote_mod._raise_for_status(resp)
    assert type(ei.value) is remote_mod._BY_CODE[code]
    assert ei.value.code == code and ei.value.status_code == status
    assert ei.value.request_id == f"req-{code}"


def test_documented_errors_cover_the_translate_endpoint() -> None:
    got = {(s, c) for p, _, s, c in documented_errors() if p == "/v1/translate"}
    assert {(400, "invalid_request"), (400, "unsupported_language"), (401, "unauthorized"),
            (403, "forbidden"), (409, "idempotency_conflict"), (413, "too_large"),
            (429, "rate_limited"), (503, "upstream_unavailable")} <= got  # fmt: skip


@pytest.mark.parametrize("code", sorted(problem_codes()))
def test_every_spec_problem_code_has_a_typed_error(code: str) -> None:
    fx_body = fx.problem_body(400, code)
    validate(fx_body, "components", "schemas", "Problem")
    assert code in remote_mod._BY_CODE, f"the client does not map problem code {code!r}"
    assert remote_mod._BY_CODE[code] is not IvoireAPIError


def test_fields_the_client_reads_are_declared() -> None:
    schemas = SPEC["components"]["schemas"]
    # ``heard_bci`` and ``phrase`` are read when present (self-hosted servers); the hosted API
    # does not send them.
    listen = set(schemas["ListenResponse"]["properties"])
    assert {"bci", "fra", "eng", "asr_confidence", "low_confidence", "audio_s",
            "route", "timings_ms", "usage"} <= listen  # fmt: skip
    translate = set(schemas["TranslateResponse"]["properties"])
    assert {"texts", "pivot", "model", "timings_ms", "usage", "request_id"} <= translate
    speak = set(schemas["SpeakResponse"]["properties"])
    assert {"fra", "bci", "audio_url", "timings_ms"} <= speak
    assert "languages" in schemas["LanguagesResponse"]["properties"]
    assert set(schemas["Problem"]["required"]) == {"type", "title", "status", "detail", "code"}
