# langchain-ivoire

> **Status: preview (`0.1.0a3`, alpha).** Access to the hosted API is by request (translation is the
> public service; speech recognition, the voice and embeddings are by approval). Interfaces may change
> before 0.1.0.

LangChain and LangGraph components for **Ivorian languages**. **Baoulé** (ISO 639-3 `bci`) works now.
Agni (`any`), Dioula (`dyu`) and Bété (`bev`, `btg`) are planned.

- `IvoireTranscriber` turns speech into Baoulé text and gives a confidence score.
- `IvoireTranslator` translates Baoulé to and from French, and to and from English through French. It has an `everyday`, `tale` or `formal` style.
- `IvoireSpeaker` turns Baoulé text into speech (WAV).
- `with_language(llm, lang="bci")` lets **any chat model** take Baoulé in and give Baoulé back.
- `create_voice_agent(llm, lang="bci")` is a prebuilt LangGraph voice agent. It runs listen → phrasebook snap → translate → LLM → translate back → speak.
- `IvoireAudioLoader` turns audio files into `Document`s that hold the transcripts.
- `IvoireLanguageMiddleware` makes any `create_agent` agent talk in Baoulé (the agent and its tools work in French).
- `IvoireTranslateTool`, `IvoireTranscribeTool` and `IvoireSpeakTool` are tools for agents. The transcribe tool takes base64, an audio content block or a public http(s) URL (never a local file path), or the latest voice message in the agent's state. The speak tool returns its audio as a standard audio content block artifact.
- `langchain_ivoire.audio` converts between bytes and LangChain audio content blocks (`to_audio_block`, `from_audio_block`).

All components are LangChain `Runnable`s, so they compose with `|` and work with `.batch()`, `.with_retry()`, tracing and LangGraph nodes.

## Install

```bash
pip install langchain-ivoire            # API client (light: langchain-core, langgraph, httpx)
pip install "langchain-ivoire[mcp]"     # + the ivoire-mcp server
```

While only the preview is on PyPI, pip installs it without `--pre`; pin it with `pip install "langchain-ivoire==0.1.0a3"`.

## Request an API key

The hosted API at `https://api.abyshire.ci` needs an API key issued by Abyshire.

- **Get a key / request approval: api@abyshire.co.uk** (your name, your organisation, the intended
  use, and whether it is research or commercial).
- **Translation** (`abyshire/translate-baoule`) is public. **Speech recognition**
  (`abyshire/listen-baoule`), **the voice** (`abyshire/speak-baoule`) and **embeddings**
  (`abyshire/embed-baoule`) are **by approval**.
- **Free trial quota during the alpha** (rate-limited per key). Commercial use is subject to an
  agreement with Abyshire.
- Translation via OpenRouter (coming soon).
- The models are served through the API only.
- No key yet? `backend="fake"` runs every component offline, so you can try the API and write tests
  first.

## Quick start

All examples use the remote backend by default. It reads `IVOIRE_API_KEY` (see [Request an API key](#request-an-api-key)) (and optionally `IVOIRE_BASE_URL`) from the environment. To try everything offline, pass `backend="fake"`.

### Transcribe

```python
from langchain_ivoire import IvoireTranscriber

asr = IvoireTranscriber(lang="bci")
t = asr.invoke("come_and_eat.wav")  # bytes, a path, or {"audio": ...}
print(t.text, t.confidence, t.low_confidence)  # e.g. 'bla di like' 0.962 False
```

### Translate

```python
from langchain_ivoire import IvoireTranslator

to_fr = IvoireTranslator(lang="bci")  # bci -> fra
to_bci = IvoireTranslator(lang="bci", source="fra", target="bci", style="everyday")
to_en = IvoireTranslator(lang="bci", source="bci", target="eng")  # pivots via French

r = to_en.invoke("Bla di like.")
r.text, r.via  # 'Come and eat.', (('fra', 'Venez manger.'), ('eng', 'Come and eat.'))
```

### Speak

```python
from langchain_ivoire import IvoireSpeaker, IvoireTranslator

say_in_baoule = IvoireTranslator(lang="bci", source="fra", target="bci") | IvoireSpeaker(lang="bci")
say_in_baoule.invoke("Venez manger.").save("venez_manger.wav")
```

### `IvoireLanguageMiddleware`: a Baoulé agent with `create_agent`

```python
from langchain.agents import create_agent
from langchain_ivoire import IvoireLanguageMiddleware

MODEL = "provider:model"  # any model id that init_chat_model accepts
agent = create_agent(
    MODEL,
    tools=[get_weather],
    middleware=[IvoireLanguageMiddleware("bci", max_sentences=2)],
)
out = agent.invoke({"messages": [{"role": "user", "content": "N tɛ, an ti sɛn?"}]})
out["messages"][-1].content  # the reply in Baoulé
out["messages"][-1].additional_kwargs["ivoire"]["fra"]  # what the model said in French
```

- `before_agent` translates the user's Baoulé messages to French. The message keeps its Baoulé `content`; the French is stored in `additional_kwargs["ivoire"]["fra"]`, so a turn is never translated twice.
- `wrap_model_call` gives the model a French view of the conversation, with a short French system hint appended to your system prompt (short, simple French, at most `max_sentences` sentences, no markdown). Tool calls, tool arguments and tool results stay in French.
- `after_agent` translates the final answer back to Baoulé and keeps the French in `additional_kwargs["ivoire"]`.
- Sync and async (`invoke`/`ainvoke`) both work. Put the middleware first in `middleware=[...]`.

Any chat model works: use the `provider:model` id of the model you already use (see `init_chat_model` in the LangChain docs), or pass a chat model instance.

Streaming tokens from the model node are French; the Baoulé reply appears when the agent finishes. See `examples/middleware_agent.py` (runs offline).

### `with_language`: any chat model, in Baoulé

```python
from langchain.chat_models import init_chat_model
from langchain_ivoire import with_language

model = init_chat_model("provider:model")  # any chat model
chat = with_language(model, lang="bci")
reply = chat.invoke("N tɛ, an ti sɛn?")  # "Hello, how are you?"
reply.content  # Baoulé reply
reply.response_metadata["ivoire"]  # {'input_fra': ..., 'output_fra': ..., 'timings_ms': {...}}
```

The wrapper is itself a `BaseChatModel`, so it works as a drop-in replacement. It accepts message lists, keeps earlier turns (it reuses their French from the metadata), and supports `bind_tools` (passed through to the inner model). It adds a short French system hint that asks the model to reply in short, simple French, because short sentences translate much better. Change the hint with `system_hint="..."` or turn it off with `system_hint=False`.

### Voice agent (LangGraph)

```python
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ivoire import create_voice_agent
from langchain_ivoire.audio import from_audio_block, to_audio_block

agent = create_voice_agent(
    llm, lang="bci", phrasebook="phrasebook.json", checkpointer=InMemorySaver()
)
question = to_audio_block(open("question.wav", "rb").read())
out = agent.invoke({"audio": question}, {"configurable": {"thread_id": "family-call"}})
out["transcript"], out["user_fra"], out["reply_fra"], out["reply"]
open("reply.wav", "wb").write(from_audio_block(out["reply_audio"]))
```

Audio in the state is a standard LangChain audio content block (`{"type": "audio", "base64": ..., "mime_type": "audio/wav"}`), so the whole state is JSON and checkpoints cleanly. The `audio` input also takes base64 text or raw bytes.

The graph does the following:

- `listen` (ASR) routes the turn to one of these nodes:
  - `phrase_snap`, when the utterance is clearly a phrasebook entry. The entry's checked text and translation are used, and MT is skipped.
  - `low_confidence`, which marks the turn and then continues.
  - `translate_in`.
- The turn then goes through `agent` (the LLM, French in and French out), `translate_out` and `speak`, and ends at `out`.
- If the LLM fails, the turn still returns the transcript and its French.

The phrase snap compares sound-alike skeletons. For example, it maps a learner's "bla de le kɛ" to the entry "Bla di like." ("Come and eat.").

### MCP server (any MCP client)

```bash
pip install "langchain-ivoire[mcp]"
ivoire-mcp                                            # stdio
ivoire-mcp --transport streamable-http --port 8000    # http://127.0.0.1:8000/mcp
ivoire-mcp --backend fake                             # offline, to try it out
```

It exposes four tools, built on the official MCP Python SDK:

| tool | does |
|---|---|
| `ivoire_transcribe` | Baoulé speech → text (`audio_base64`, or a public http(s) `audio_url`); returns text, confidence and `low_confidence` |
| `ivoire_translate` | text between `bci`, `fra` and `eng`, with a `style` |
| `ivoire_speak` | Baoulé text → speech, returned as MCP `audio` content (WAV) |
| `ivoire_languages` | what the backend supports |

Configuration comes from `IVOIRE_API_KEY` and `IVOIRE_BASE_URL`. Most MCP clients (desktop assistants, IDEs) take a JSON entry like this one in their config file (replace the placeholder; never commit a real key):

```json
{
  "mcpServers": {
    "baoule": {
      "command": "uvx",
      "args": ["--from", "langchain-ivoire[mcp]", "ivoire-mcp"],
      "env": {
        "IVOIRE_API_KEY": "<your-ivoire-api-key>",
        "IVOIRE_BASE_URL": "https://api.abyshire.ci"
      }
    }
  }
}
```

The transcribe tool refuses loopback and private-network URLs unless you start the server with `--allow-private-urls`.

### Audio documents

```python
from langchain_ivoire import IvoireAudioLoader

docs = IvoireAudioLoader("recordings/", lang="bci").load()
docs[0].metadata  # {'source': ..., 'duration': ..., 'confidence': ..., 'lang': 'bci', ...}
```

## Backends

| backend | what | needs |
|---|---|---|
| `remote` (default) | Abyshire API at `https://api.abyshire.ci` | an API key from Abyshire for the hosted API |
| `openrouter` | the OpenAI-compatible API: translation through OpenRouter, or every service directly at the Abyshire API with an approved key ([coming soon](#hosted-api-and-openrouter)) | `OPENROUTER_API_KEY` (translation via OpenRouter) or an Abyshire API key |
| `fake` | deterministic stand-in for tests and examples | nothing |

```python
IvoireTranscriber(lang="bci", api_key="...", base_url="https://api.abyshire.ci")
IvoireTranscriber(lang="bci", backend="fake")  # tests and examples, no network
```

The remote client translates with `POST /v1/translate`: at most 32 texts per request, texts longer than 1000 characters are split at sentence ends and joined again, and the `style` is sent along. Each `Translation` keeps the server's `request_id`, `model` and `usage` in `.metadata`. `RemoteBackend.languages()` and `.models()` read `GET /v1/languages` and `GET /v1/models`.

Errors are typed. The server sends RFC 9457 `application/problem+json`; every error keeps the problem `code`, `title`, `detail` and the `X-Request-Id` (`err.request_id`):

| error | problem `code` | status |
|---|---|---|
| `IvoireInvalidRequestError` (a `ValueError`) | `invalid_request`, `length_required` | 400, 411, 422 |
| `IvoireTooLargeError` | `too_large` | 413, or 400 for audio that is too long |
| `IvoireUnsupportedLanguageError` (a `NotImplementedError`) | `unsupported_language` | 400 |
| `IvoireAuthenticationError` | `unauthorized` | 401 |
| `IvoirePermissionError` | `forbidden` | 403 |
| `IvoireNotFoundError` | `not_found`, `model_not_found`, `method_not_allowed` | 404, 405 |
| `IvoireIdempotencyConflictError` | `idempotency_conflict` | 409 (not retried) |
| `IvoireRateLimitError` (`.retry_after`) | `rate_limited` | 429 |
| `IvoireUpstreamUnavailableError` | `upstream_unavailable` | 503 |
| `IvoireServerError` | `internal` | 5xx |
| `IvoireConnectionError` | | the server could not be reached |

429, 502, 503 and 504 answers and connection failures are retried with exponential backoff; `Retry-After` (seconds or a date) is honoured. Every POST carries an `Idempotency-Key` that stays the same across the retries of one call.

A server without `/v1/translate` (older versions) only offers English → French/Baoulé (through `/speak_bci`); other directions then raise `IvoireUnsupportedError`.

## Hosted API and OpenRouter

**Translation via OpenRouter (coming soon); speech, voice and embeddings by approval through the
Abyshire API.**

> **Coming soon.** The OpenAI-compatible endpoints and the OpenRouter listing below are not live
> yet. The package side is ready, so code written against it works unchanged once they open. Until
> then, use the default `remote` backend with an Abyshire API key.

The hosted API will also speak the OpenAI wire format, with one product id per service
(`GET /v1/models` lists them):

| product id | service | endpoint | component | access |
|---|---|---|---|---|
| `abyshire/translate-baoule` | French ↔ Baoulé translation (English through French), with styles | `POST /v1/chat/completions` | `IvoireTranslator` | OpenRouter (coming soon), or an Abyshire API key |
| `abyshire/listen-baoule` | Baoulé speech to text | `POST /v1/audio/transcriptions` | `IvoireTranscriber` | by approval, Abyshire API |
| `abyshire/speak-baoule` | Baoulé text to speech | `POST /v1/audio/speech` | `IvoireSpeaker` | by approval, Abyshire API |
| `abyshire/embed-baoule` | Baoulé and French search embeddings | `POST /v1/embeddings` | `IvoireEmbeddings` | by approval, Abyshire API |

**Translation through OpenRouter** (coming soon). Developers who already use OpenRouter need no
new account:

```python
# export OPENROUTER_API_KEY=...
from langchain_ivoire import IvoireTranslator

to_bci = IvoireTranslator(source="fra", target="bci", backend="openrouter")
to_bci.invoke("Venez manger.").text
```

**Speech, voice and embeddings by approval.** Request access from Abyshire (see
[Request an API key](#request-an-api-key)). With an approved key, call the Abyshire
API, either with the default `remote` backend or with the same OpenAI-compatible backend pointed at
it:

```python
from langchain_ivoire import IvoireEmbeddings, IvoireSpeaker, OpenRouterBackend

api = OpenRouterBackend(base_url="https://api.abyshire.ci")  # key from IVOIRE_API_KEY
IvoireSpeaker(backend=api).invoke("Bla di like.")
emb = IvoireEmbeddings()  # the Abyshire API by default, IVOIRE_API_KEY
```

With `backend="openrouter"`, the speech, voice and embedding components raise
`IvoireApprovalRequiredError` before anything is sent, unless you pass a model id your organisation
was granted (`OpenRouterBackend(listen_model=..., speak_model=..., embed_model=...)`).

- `OPENROUTER_API_KEY` is only ever sent to OpenRouter. With any other `base_url` the key comes from
  `api_key=` or `IVOIRE_API_KEY`. Keys never appear in logs, reprs or error messages.
- Model ids can be changed: `OpenRouterBackend(translate_model=..., voice=..., ...)`.
- The default `remote` backend's `base_url` can point at any Abyshire API URL you were given; it uses
  the native endpoints (`/v1/translate`, ...) rather than the OpenAI-compatible ones.

**Calling the translator without this package.** Any OpenAI-compatible client works. A chat
completion translates the text of its single `user` message; the direction and style go in a
`system` message of the form `ivoire-translate source=<code> target=<code> style=<style>` and in a
top-level `translation` field (the server reads the field first). Codes are `bci`, `fra` and `eng`;
styles are `everyday`, `tale` and `formal`. The reply's message content is the translation only.

```python
from openai import OpenAI

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key="...")
reply = client.chat.completions.create(
    model="abyshire/translate-baoule",
    messages=[
        {"role": "system", "content": "ivoire-translate source=fra target=bci style=everyday"},
        {"role": "user", "content": "Venez manger."},
    ],
    extra_body={"translation": {"source": "fra", "target": "bci", "style": "everyday"}},
)
print(reply.choices[0].message.content)
```

For approved keys: transcription requests send `language="bci"` (ISO 639-3), speech comes back as
WAV (`response_format="wav"`), and embedding requests send `input_type` (`"query"` for questions,
`"document"` for texts you index).

## Supported languages

| code | language | status |
|---|---|---|
| `bci` | Baoulé | available |
| `any` | Anyin (Agni) | planned |
| `dyu` | Dioula (Jula) | planned |
| `bev` / `btg` | Bété (Daloa / Gagnoa) | planned |

Planned languages raise `NotImplementedError` with a clear message.

## Benchmarks

Measured **through the hosted API** on 16 Oct 2026, one item per request (details, confidence
intervals and method: [`docs/benchmarks.md`](https://github.com/AbyshireAI/langchain-ivoire/blob/main/docs/benchmarks.md)):

| model | test set | result | latency p50 / p90 |
|---|---|---|---|
| `abyshire/translate-baoule` | held-out everyday sentences, n = 716 | chrF++ **36.5** bci → fra, **37.1** fra → bci | 237 / 442 ms |
| | held-out sentences from Baoulé tales, n = 218 | chrF++ 16.1 bci → fra, 19.4 fra → bci | 373 / 608 ms |
| `abyshire/listen-baoule` | **phone recordings of speakers not seen before** | **WER 19.3 %**, CER 8.3 % | 578 / 998 ms |
| | read speech, held-out sentences | WER 16.4 %, CER 7.7 % | 645 / 825 ms |

There is no standard public test set for Baoulé yet; the sets are small, so read the confidence
intervals. The voice and the embeddings have not been benchmarked through the API yet. The voice
has not been rated by native speakers yet, and the phrasebook entries have not been reviewed by a
native speaker.

Machine translation of a low-resource language makes mistakes. Do not use it where an error can
cause harm (medical, legal, safety) without a human interpreter.

## Model access

- The models are served through the API only.
- The hosted API needs an **API key from Abyshire**; see [Request an API key](#request-an-api-key).

## Licences

- **Code:** MIT (© Abyshire Technology Ltd). See `LICENSE`.
- **Models:** served through the Abyshire API under its own terms; nothing model-related ships in
  this package. Credits for third-party material: `ATTRIBUTION.md` in the repository.

## Privacy

- **No telemetry:** this package sends nothing anywhere except the requests you make to the backend you choose.
- API keys come only from the constructor or from environment variables, and they are never logged or included in error messages.
- The hosted API may keep its own service logs. See the API terms.

## Development

```bash
uv venv && uv pip install -e . pytest pytest-asyncio pytest-socket langchain-tests ruff mypy
pytest tests/unit_tests --disable-socket --allow-unix-socket
pytest tests/integration_tests -m integration          # needs a running server (IVOIRE_TEST_BASE_URL)
ruff check . && ruff format --check . && mypy langchain_ivoire
```
