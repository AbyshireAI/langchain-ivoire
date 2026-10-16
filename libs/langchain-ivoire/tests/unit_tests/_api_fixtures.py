"""Server answers used by the remote-backend unit tests.

Everything here follows the API contract in ``tests/contract/openapi_v1.json``;
``test_api_contract.py`` validates these bodies against the spec's response schemas, so a fixture
cannot drift from what the real server sends. (Tests that simulate *older* servers, without
``/v1/translate`` or problem+json, build their answers inline on purpose.)
"""

from __future__ import annotations

from typing import Any

import httpx

PROBLEM_HEADERS = {"content-type": "application/problem+json"}

LISTEN: dict[str, Any] = {  # POST /listen_bci -> ListenResponse
    "bci": "Bla di like.",
    "heard_bci": "bla de le kɛ",
    "fra": "Venez manger.",
    "eng": "Come and eat.",
    "asr_confidence": 0.91,
    "low_confidence": False,
    "audio_s": 1.2,
    "route": ["audio_in", "asr_bci", "phrase_bci", "out"],
    "phrase": {"id": "32_come_and_eat", "match": 0.84},
    "timings_ms": {"asr": 900},
    "usage": {"audio_seconds": 1.2, "characters": 12},
}

SPEAK: dict[str, Any] = {  # POST /speak_bci -> SpeakResponse
    "eng": "Come and eat.",
    "fra": "Venez manger.",
    "bci": "Bla di like.",
    "audio_url": "/audio/abc.wav",
    "audio_s": 0.5,
    "timings_ms": {"mt_eng_fra": 5},
    "route": ["text_in", "mt_eng_fra", "mt_fra_bci", "tts_bci", "out"],
    "usage": {"characters": 13},
}

LANGUAGES: dict[str, Any] = {  # GET /v1/languages -> LanguagesResponse
    "languages": [
        {
            "code": "bci",
            "name": "Baoulé",
            "asr": True,
            "tts": True,
            "mt": ["fra", "eng"],
            "styles": ["everyday", "tale", "formal"],
        }
    ]
}

MODELS: dict[str, Any] = {  # GET /v1/models -> ModelsResponse (OpenAI list + the native fields)
    "object": "list",
    "data": [
        {
            "id": "abyshire/translate-baoule",
            "object": "model",
            "created": 1760486400,
            "owned_by": "abyshire",
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "pricing": {"prompt": "0.000003", "completion": "0.000003"},
            "access": "public",
            "available": True,
        }
    ],
    "api_version": "2.0.0",
    "mt": {"name": "abyshire/translate-baoule", "version": "2026-10-15"},
}


def tts(text: str) -> dict[str, Any]:
    """POST /tts_bci -> TtsResponse (the client reads ``audio_url``)."""
    return {
        "bci": text,
        "audio_url": "/audio/abc.wav",
        "audio_s": 0.5,
        "timings_ms": {"tts": 30},
        "usage": {"characters": len(text)},
    }


def translate_body(
    texts: list[str],
    *,
    source: str = "bci",
    target: str = "fra",
    style: str = "everyday",
    pivot: list[str] | None = None,
    rid: str = "req-1",
) -> dict[str, Any]:
    """POST /v1/translate -> TranslateResponse."""
    return {
        "texts": texts,
        "source": source,
        "target": target,
        "style": style,
        "pivot": {"fra": pivot} if pivot is not None else None,
        "model": {
            "mt": "abyshire/translate-baoule",
            "pivot": "abyshire/translate-baoule",
            "tier": "commercial",
        },
        "timings_ms": {"mt": 12},
        "usage": {"characters": sum(len(t) for t in texts), "segments": len(texts)},
        "request_id": rid,
    }


def ok(texts: list[str], *, pivot: list[str] | None = None, rid: str = "req-1") -> httpx.Response:
    """A 200 answer of ``POST /v1/translate``."""
    return httpx.Response(
        200, headers={"x-request-id": rid}, json=translate_body(texts, pivot=pivot, rid=rid)
    )


def problem_body(status: int, code: str, detail: str = "nope") -> dict[str, Any]:
    """An RFC 9457 problem (the spec's ``Problem`` schema)."""
    return {
        "type": f"https://api.abyshire.ci/problems/{code}",
        "title": code.replace("_", " "),
        "status": status,
        "detail": detail,
        "code": code,
    }


def problem(status: int, code: str, detail: str = "nope", **headers: str) -> httpx.Response:
    """An ``application/problem+json`` error answer with an ``X-Request-Id``."""
    return httpx.Response(
        status,
        headers={**PROBLEM_HEADERS, "x-request-id": f"req-{code}", **headers},
        json=problem_body(status, code, detail),
    )
