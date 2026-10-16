"""Baoulé in, Baoulé out: a ``create_agent`` agent with ``IvoireLanguageMiddleware``.

The agent and its tools work in French; the middleware translates the user's Baoulé before the
agent runs and the final answer back to Baoulé afterwards.

Offline (scripted model and fake translator; no key, no network)::

    python examples/middleware_agent.py

The fake translator only knows a few fixed sentences; anything else comes back marked
``[fra>bci] ...`` so you can see where a real translation would go.

With a real model and the hosted API (needs your model provider's LangChain package)::

    export IVOIRE_API_KEY=...        # translation API key from Abyshire
    export <PROVIDER>_API_KEY=...    # your model provider's key
    python examples/middleware_agent.py --model provider:model

``--model`` takes any ``provider:model`` id that ``init_chat_model`` accepts.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from langchain_ivoire import FakeBackend, IvoireLanguageMiddleware

QUESTION = "N tɛ, an ti sɛn?"  # "Hello, how are you?"


@tool
def get_weather(ville: str) -> str:
    """Donne la météo du jour pour une ville de Côte d'Ivoire."""
    return f"À {ville} : ensoleillé, 31 °C."


class ScriptedModel(GenericFakeChatModel):
    """Stands in for an LLM offline: calls the weather tool, then answers in French."""

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        return self


def offline_setup() -> tuple[Any, IvoireLanguageMiddleware]:
    model = ScriptedModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "get_weather", "args": {"ville": "Bouaké"}, "id": "1"}],
                ),
                AIMessage(content="Je vais bien, merci. À Bouaké, il fait beau aujourd'hui."),
            ]
        )
    )
    backend = FakeBackend(translations={("bci", "fra", QUESTION): "Bonjour, comment allez-vous ?"})
    return model, IvoireLanguageMiddleware("bci", backend=backend)


def main() -> None:
    ap = argparse.ArgumentParser(description="Baoulé agent with IvoireLanguageMiddleware")
    ap.add_argument("--model", help="provider:model for init_chat_model (default: offline)")
    ap.add_argument("--question", default=QUESTION)
    args = ap.parse_args()

    if args.model:
        from langchain.chat_models import init_chat_model

        model: Any = init_chat_model(args.model)
        middleware = IvoireLanguageMiddleware("bci")  # remote API, key from IVOIRE_API_KEY
    else:
        model, middleware = offline_setup()

    agent = create_agent(model, tools=[get_weather], middleware=[middleware])
    out = agent.invoke({"messages": [HumanMessage(args.question)]})

    for m in out["messages"]:
        print(f"{m.type:>5}: {m.content or getattr(m, 'tool_calls', '')}")
        fra = m.additional_kwargs.get("ivoire", {}).get("fra")
        if fra:
            print(f"       (fr) {fra}")


if __name__ == "__main__":
    main()
