# Middleware

`IvoireLanguageMiddleware` makes an agent built with `langchain.agents.create_agent` talk with the
user in Baoulé, while the model, the tools and the tool arguments stay in French.

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
out["messages"][-1].content                             # Baoulé
out["messages"][-1].additional_kwargs["ivoire"]["fra"]  # the model's French
```

Any chat model works: use the `provider:model` id of the model you already use (see `init_chat_model` in the LangChain docs), or pass a chat model instance.

## How it works

| hook | what it does |
|---|---|
| `before_agent` / `abefore_agent` | translates every user message without French yet; the message keeps its Baoulé `content`, and the French goes into `additional_kwargs["ivoire"]["fra"]` (same message ID, so it is updated in place) |
| `wrap_model_call` / `awrap_model_call` | hands the model a French view of the conversation through `ModelRequest.override(messages=..., system_message=...)`, with a short French hint appended to your system prompt |
| `after_agent` / `aafter_agent` | translates the final AI message to Baoulé and keeps the French (and the raw reply, if cleaning changed it) in `additional_kwargs["ivoire"]` |

Messages from earlier turns already hold their French, so with a checkpointer only the new turn is
translated. Tool calls, tool results and intermediate AI messages are never translated.

## Options

| option | default | meaning |
|---|---|---|
| `lang` | `"bci"` | the Ivorian language |
| `style` | `"everyday"` | translation register: `everyday`, `tale`, `formal` |
| `system_hint` | `True` | `True`: the default French hint; a string: your hint; `False`: none |
| `max_sentences` | `2` | keep at most this many sentences of the French answer (short sentences translate better) |
| `clean` | `True` | strip markdown, emoji and brackets before translating |
| `backend`, `api_key`, `base_url`, `translator` | remote API | where translation runs |

Put the middleware **first** in `middleware=[...]` so every other middleware sees the French view.
Streamed tokens from the model node are French; the Baoulé answer appears when the agent finishes.

## Without an agent

`with_language(llm, lang="bci")` wraps a chat model directly and uses the same pivot code. Its
result keeps the French in `response_metadata["ivoire"]`.

A runnable example, which also works offline, is `libs/langchain-ivoire/examples/middleware_agent.py`.
