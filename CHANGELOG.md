# Changelog

All notable changes to `langchain-ivoire` are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0,
minor versions may contain breaking changes; deprecated APIs keep working, with a warning, for at
least one minor release.

## [Unreleased]

## [0.1.0a3] - 2026-10-16

Third preview. The package is an API client: the models are served through the
Abyshire API only.

### Added

- `OpenRouterBackend` (`backend="openrouter"`): the components over the OpenAI-compatible API.
  Translation (`/chat/completions`, direction and style in a `system` message and a `translation`
  field) works through OpenRouter with `OPENROUTER_API_KEY` (coming soon). Speech recognition
  (`/audio/transcriptions`), the voice (`/audio/speech`) and embeddings (`/embeddings`) are
  available by approval through the Abyshire API (`base_url=`, `IVOIRE_API_KEY`); through
  OpenRouter they raise `IvoireApprovalRequiredError` unless a granted model id is configured.
  Model ids default to `abyshire/translate-baoule`, `abyshire/listen-baoule`,
  `abyshire/speak-baoule` and `abyshire/embed-baoule`.
- `IvoireEmbeddings`: Baoulé and French search embeddings as a LangChain `Embeddings` (by approval).
- `IvoireApprovalRequiredError`, and `IvoireBackend.embed()` / `aembed()` (optional capability).
- `ivoire-mcp --backend openrouter`.
- The problem code `model_not_found` maps to `IvoireNotFoundError`.
- Benchmarks record the latency of every request (p50, p90, mean and max).
- `ATTRIBUTION.md`: credits for third-party material under Creative Commons Attribution licences.

### Changed

- **The default base URL is the hosted API, `https://api.abyshire.ci`** (was a preview address).
  `RemoteBackend` and `IvoireEmbeddings` use it unless `base_url=` or `IVOIRE_BASE_URL` says
  otherwise. The OpenAI-compatible client adds `/v1` when the base URL has none.
- The vendored API contract (`tests/contract/openapi_v1.json`) is refreshed to API 2.0.0 (native and
  OpenAI-compatible routes); `/listen_eng` and `/agent_text` are not part of the hosted API.
- Error messages read OpenAI-style `{"error": {"message": ...}}` bodies.
- Model ids name the service first and the language last, so other languages fit later:
  `abyshire/translate-baoule`, `abyshire/listen-baoule`, `abyshire/speak-baoule` and
  `abyshire/embed-baoule`.
- Benchmarks re-measured through the hosted API (16 Oct 2026): test-set scores and request latency;
  see `docs/benchmarks.md`.

### Removed

- The offline option is not part of the public package (no `local` extra); use the hosted API.

## [0.1.0a2] - 2026-10-15

Second preview.

### Changed

- **Breaking:** the third translation style is now called `formal` (`STYLES` is
  `("everyday", "tale", "formal")`), in the components, the MCP server, the benchmark script and the
  vendored API contract.

### Added

- `RemoteBackend(style_aliases=...)`: send a style under another name to a server that names it
  differently; `Translation.style` keeps the caller's name. An optional JSON settings file
  (`IVOIRE_SETTINGS`, else `%APPDATA%\langchain-ivoire\settings.json` on Windows or
  `~/.config/langchain-ivoire/settings.json`) can hold them in its `"remote"` section.

## [0.1.0a1] - unreleased (set the date when tagging `v0.1.0a1`)

First public release, as a **preview** (alpha). API access is by request, and interfaces may change
before 0.1.0.

### Added

- `IvoireLanguageMiddleware` for `langchain.agents.create_agent`: translates the user's Baoulé to
  French before the agent (`before_agent`), gives the model a French view with a short French
  system hint (`wrap_model_call`), and translates the final answer back (`after_agent`), keeping
  the French in `additional_kwargs["ivoire"]`. Sync and async. Tool calls stay in French.
- Remote translation through `POST /v1/translate`: batches of at most 32 texts, long texts split at
  sentence ends, the `style` parameter, `Translation.metadata` with `request_id`, `model` and
  `usage`. `RemoteBackend.languages()` and `.models()` (`GET /v1/languages`, `GET /v1/models`).
- Typed RFC 9457 problem+json errors: `IvoireInvalidRequestError`, `IvoireTooLargeError`,
  `IvoireUnsupportedLanguageError`, `IvoireUpstreamUnavailableError`, `IvoireIdempotencyConflictError`
  (409); every API error carries the
  problem `code`, `title`, `detail` and the `X-Request-Id`.
- `Idempotency-Key` on every POST (stable across retries); `Retry-After` as seconds or an HTTP date.
- `langchain_ivoire.audio`: `to_audio_block()` / `from_audio_block()` for LangChain audio content
  blocks, and a guarded URL downloader.
- `ivoire-mcp`, an MCP server (`[mcp]` extra, official MCP Python SDK 2.x) with
  `ivoire_transcribe`, `ivoire_translate`, `ivoire_speak` (MCP audio content) and
  `ivoire_languages`, over stdio or streamable HTTP.
- API contract tests against a vendored copy of the server's OpenAPI 3.1 spec
  (`tests/contract/openapi_v1.json`, API 1.1.0): endpoints, methods, query parameters, headers and
  request bodies the client sends, and every response fixture (including problem+json errors),
  validated with jsonschema (test-only dependency).
- The problem codes `method_not_allowed` (405) and `length_required` (411) from the spec map to
  `IvoireNotFoundError` and `IvoireInvalidRequestError`.
- `benchmarks/`: chrF++ and CER/WER scoring with bootstrap 95% CIs and a public sample manifest.
- CI (ruff, mypy --strict, tests on Python 3.10–3.14 × Linux/Windows/macOS, lowest-direct versions,
  `[mcp]` extra, 85% coverage gate), a trusted-publishing release workflow with attestations and a
  CycloneDX SBOM, Dependabot, pre-commit, `SECURITY.md`, `CONTRIBUTING.md`, and a docs skeleton
  (mkdocs-material + mkdocstrings).

### Changed

- **Breaking:** `IvoireTranscribeTool` no longer accepts `file_path`. It takes `audio_base64`, an
  `audio` content block or a public `audio_url`, or reads the latest audio block from the agent
  state through `ToolRuntime`.
- **Breaking:** `IvoireSpeakTool`'s artifact is now a standard audio content block (JSON) instead of
  a `Speech` object.
- **Breaking:** the voice graph's `reply_audio` is an audio content block; the `audio` input also
  takes a block or base64 text.
- Backends are async-first: implement either the sync or the async method of each capability.
  `RemoteBackend` rebuilds its async HTTP client when used from a new event loop.
- `with_language` / `IvoireLanguageChat` share the pivot code with the middleware; the default
  hint mentions the `max_sentences` cap.
- Requires `langchain>=1.4,<2`, `langchain-core>=1.6,<2`, `langgraph>=1.2,<2`.

### Removed

- LAN addresses and local paths from defaults, docstrings, tests and the sdist. Integration tests
  read the server only from `IVOIRE_TEST_BASE_URL`.

### Security

- Unit tests block the network (pytest-socket). The transcribe tool and MCP server refuse local file
  paths and private-network URLs.

[Unreleased]: https://github.com/AbyshireAI/langchain-ivoire/compare/v0.1.0a3...HEAD
[0.1.0a3]: https://github.com/AbyshireAI/langchain-ivoire/compare/v0.1.0a2...v0.1.0a3
[0.1.0a2]: https://github.com/AbyshireAI/langchain-ivoire/compare/v0.1.0a1...v0.1.0a2
[0.1.0a1]: https://github.com/AbyshireAI/langchain-ivoire/releases/tag/v0.1.0a1
