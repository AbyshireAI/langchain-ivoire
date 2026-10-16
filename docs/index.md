# langchain-ivoire

!!! note "Status: preview (`0.1.0a3`, alpha)"
    Access to the hosted API is by request (translation is the public service; speech recognition, the voice and embeddings are by approval). Interfaces may change before 0.1.0.

LangChain and LangGraph components for **Ivorian languages**. **Baoulé** (ISO 639-3 `bci`) works now;
Agni, Dioula and Bété are planned.

- **Middleware:** `IvoireLanguageMiddleware` lets any `create_agent` agent talk with the user in
  Baoulé. The agent and its tools work in French.
- **Tools:** translate, transcribe and speak, for any agent.
- **Voice agent:** a prebuilt LangGraph graph: listen → translate → LLM → translate back → speak.
- **MCP server:** the same tools for any MCP client (desktop assistants, IDEs, other agents).
- **Runnables:** `IvoireTranscriber`, `IvoireTranslator` and `IvoireSpeaker`, which compose with `|`.

## Install

```bash
pip install langchain-ivoire            # remote API client (light: no torch)
pip install "langchain-ivoire[mcp]"     # + the ivoire-mcp server
```

While only the preview is on PyPI, pip installs it without `--pre`; pin it with `pip install "langchain-ivoire==0.1.0a3"`.

The remote backend reads `IVOIRE_API_KEY` (an API key from Abyshire) and, optionally,
`IVOIRE_BASE_URL` (another server running the same API). Pass `backend="fake"` to try everything
offline.

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

## Quickstart

```python
from langchain_ivoire import IvoireSpeaker, IvoireTranscriber, IvoireTranslator

text = IvoireTranscriber(lang="bci").invoke(open("question.wav", "rb").read()).text
french = IvoireTranslator(lang="bci").invoke(text).text                  # bci -> fra
english = IvoireTranslator(lang="bci", target="eng").invoke(text).text   # pivots through French

say = IvoireTranslator(lang="bci", source="fra", target="bci") | IvoireSpeaker(lang="bci")
say.invoke("Venez manger.").save("venez_manger.wav")
```

Any chat model, in Baoulé:

```python
from langchain_ivoire import with_language

chat = with_language(my_chat_model, lang="bci")
chat.invoke("N tɛ, an ti sɛn?").content   # Baoulé reply
```

For agents, see [Middleware](middleware.md). Errors are typed (see
[API reference](api.md#langchain_ivoire.exceptions)) and carry the server's `request_id`.

Machine translation of a low-resource language makes mistakes. Read [Limitations](limitations.md)
before you build on it.
