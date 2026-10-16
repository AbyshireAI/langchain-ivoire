# API contract

`openapi_v1.json` is a vendored copy of the hosted API's OpenAPI 3.1 spec (API version 2.0.0,
refreshed 16 Oct 2026; the file name is kept so the tests and tools that load it need no change).
`tests/unit_tests/test_api_contract.py` checks the remote client and the unit-test fixtures
(`tests/unit_tests/_api_fixtures.py`) against it, offline; the integration tests also validate live
answers against it.

API 2.0.0 serves the native routes (`/v1/translate`, `/listen_bci`, `/tts_bci`, `/speak_bci`,
`/audio/{aid}.wav`, `/v1/languages`, `/v1/models`, `/health`; every native route also under `/v1`)
and the OpenAI-compatible routes (`/v1/chat/completions`, `/v1/audio/transcriptions`,
`/v1/audio/speech`, `/v1/embeddings`, `/v1/models`) at `https://api.abyshire.ci`. The translation
styles are `everyday`, `tale` and `formal`. `/listen_eng` and `/agent_text` are not part of the
hosted API; the client keeps them for self-hosted servers, and the contract tests check that they
stay out of this spec.

To refresh it after the API changes, take the spec the API publishes, convert it to JSON, keep only
the public server URL (`https://api.abyshire.ci`) and drop the operator-only readiness route
(`/ready`, schema `Ready`). Then run `pytest tests/unit_tests/test_api_contract.py`, fix the client
or the fixtures where they no longer match, and note the new API version here and in the CHANGELOG.
