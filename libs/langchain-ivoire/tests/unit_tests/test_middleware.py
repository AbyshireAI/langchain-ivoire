"""Contract tests for IvoireLanguageMiddleware with create_agent, a fake chat model and FakeBackend.

There is no standard langchain-tests suite for middleware, so these pin down the behaviour:
what the model sees, what ends up in the state, and that tool calls stay in French.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain_ivoire import FakeBackend, IvoireLanguageMiddleware, IvoireTranslator
from tests.unit_tests._fakes import ToolCallingFake

USER_BCI = "N tɛ, an ti sɛn?"
USER_FRA = "Bonjour, comment allez-vous ?"
REPLY_FRA = "Je vais bien, merci."
REPLY_BCI = "N liɛ yo'a kpa, n yo."


def _mt_calls(be: FakeBackend) -> list[tuple[str, str, list[str]]]:
    return [
        (c[1]["source"], c[1]["target"], c[1]["texts"]) for c in be.calls if c[0] == "translate"
    ]


async def _run(agent: Any, inputs: Any, use_async: bool, config: Any = None) -> Any:
    return await agent.ainvoke(inputs, config) if use_async else agent.invoke(inputs, config)


@pytest.mark.parametrize("use_async", [False, True])
async def test_round_trip(fake: FakeBackend, use_async: bool) -> None:
    model = ToolCallingFake(messages=iter([AIMessage(content=REPLY_FRA)]))
    agent = create_agent(model, middleware=[IvoireLanguageMiddleware("bci", backend=fake)])
    out = await _run(agent, {"messages": [HumanMessage(USER_BCI)]}, use_async)

    # the model saw French only, with the hint as system message
    (seen,) = model.seen
    assert isinstance(seen[0], SystemMessage)
    assert "français simple" in str(seen[0].content) and "baoulé" in str(seen[0].content)
    assert "deux phrases courtes au maximum" in str(seen[0].content)
    assert seen[1].content == USER_FRA and "ivoire" not in seen[1].additional_kwargs

    # the state: user message updated in place (Baoulé + French), reply translated in place
    msgs = out["messages"]
    assert len(msgs) == 2
    user, reply = msgs
    assert user.content == USER_BCI and user.additional_kwargs["ivoire"]["fra"] == USER_FRA
    assert isinstance(reply, AIMessage) and reply.content == REPLY_BCI
    meta = reply.additional_kwargs["ivoire"]
    assert meta["fra"] == REPLY_FRA and meta["lang"] == "bci" and meta["style"] == "everyday"
    assert "raw_fra" not in meta and "translate_out" in meta["timings_ms"]
    assert _mt_calls(fake) == [("bci", "fra", [USER_BCI]), ("fra", "bci", [REPLY_FRA])]


@tool
def get_weather(ville: str, jour: str) -> str:
    """Météo d'une ville pour un jour donné."""
    return f"À {ville}, {jour} : soleil, 31 °C."


@pytest.mark.parametrize("use_async", [False, True])
async def test_tool_calls_stay_in_french(fake: FakeBackend, use_async: bool) -> None:
    model = ToolCallingFake(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_weather",
                            "args": {"ville": "Bouaké", "jour": "demain"},
                            "id": "call-1",
                        }
                    ],
                ),
                AIMessage(content="Demain à Bouaké, il fera beau et chaud."),
            ]
        )
    )
    be = FakeBackend(
        translations={
            (
                "bci",
                "fra",
                "<bci: weather question about Bouaké>",
            ): "Quel temps fera-t-il à Bouaké ?",
        }
    )
    agent = create_agent(
        model, tools=[get_weather], middleware=[IvoireLanguageMiddleware(backend=be)]
    )
    out = await _run(
        agent, {"messages": [HumanMessage("<bci: weather question about Bouaké>")]}, use_async
    )
    msgs = out["messages"]
    assert [type(m).__name__ for m in msgs] == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "AIMessage",
    ]
    call, result = msgs[1], msgs[2]
    assert call.tool_calls[0]["args"] == {"ville": "Bouaké", "jour": "demain"}
    assert "ivoire" not in call.additional_kwargs  # intermediate messages are untouched
    assert isinstance(result, ToolMessage) and result.content == "À Bouaké, demain : soleil, 31 °C."
    # second model call: French question, French tool call and French tool result
    second = model.seen[1]
    assert [m.type for m in second] == ["system", "human", "ai", "tool"]
    assert second[1].content == "Quel temps fera-t-il à Bouaké ?"
    assert msgs[-1].content == "[fra>bci] Demain à Bouaké, il fera beau et chaud."
    assert msgs[-1].additional_kwargs["ivoire"]["fra"] == "Demain à Bouaké, il fera beau et chaud."
    # only the user's question and the final answer were translated, never the tool traffic
    assert _mt_calls(be) == [
        ("bci", "fra", ["<bci: weather question about Bouaké>"]),
        ("fra", "bci", ["Demain à Bouaké, il fera beau et chaud."]),
    ]


def test_multi_turn_reuses_stored_french(fake: FakeBackend) -> None:
    model = ToolCallingFake(
        messages=iter([AIMessage(content=REPLY_FRA), AIMessage("Venez manger.")])
    )
    agent = create_agent(
        model,
        middleware=[IvoireLanguageMiddleware(backend=fake)],
        checkpointer=InMemorySaver(),
    )
    cfg: Any = {"configurable": {"thread_id": "t1"}}
    agent.invoke({"messages": [HumanMessage(USER_BCI)]}, cfg)
    n = len(fake.calls)
    out = agent.invoke({"messages": [HumanMessage("Bla di like.")]}, cfg)
    second = model.seen[1]
    assert [m.content for m in second[1:]] == [USER_FRA, REPLY_FRA, "Venez manger."]
    assert _mt_calls(fake)[n:] == [
        ("bci", "fra", ["Bla di like."]),
        ("fra", "bci", ["Venez manger."]),
    ]
    assert [m.content for m in out["messages"]] == [
        USER_BCI,
        REPLY_BCI,
        "Bla di like.",
        "Bla di like.",
    ]


def test_hint_is_appended_to_the_agent_system_prompt(fake: FakeBackend) -> None:
    model = ToolCallingFake(messages=iter([AIMessage(content="Oui.")]))
    agent = create_agent(
        model,
        system_prompt="Tu es un médecin.",
        middleware=[IvoireLanguageMiddleware(backend=fake, max_sentences=1)],
    )
    agent.invoke({"messages": [HumanMessage("Bla di like.")]})
    system = model.seen[0][0]
    assert str(system.content).startswith("Tu es un médecin.\n\n")
    assert "une seule phrase courte" in str(system.content)


@pytest.mark.parametrize(("hint", "expected"), [(False, None), ("Sois bref.", "Sois bref.")])
def test_custom_or_no_hint(fake: FakeBackend, hint: Any, expected: str | None) -> None:
    model = ToolCallingFake(messages=iter([AIMessage(content="Oui.")]))
    agent = create_agent(
        model, middleware=[IvoireLanguageMiddleware(backend=fake, system_hint=hint)]
    )
    agent.invoke({"messages": [HumanMessage("Bla di like.")]})
    first = model.seen[0][0]
    if expected is None:
        assert not isinstance(first, SystemMessage)
    else:
        assert isinstance(first, SystemMessage) and first.content == expected


def test_reply_cleaning_and_sentence_cap(fake: FakeBackend) -> None:
    model = ToolCallingFake(
        messages=iter([AIMessage(content="**Bonjour !** 😀 Je vais bien. Et vous ? Merci.")])
    )
    agent = create_agent(model, middleware=[IvoireLanguageMiddleware(backend=fake)])
    out = agent.invoke({"messages": [HumanMessage("Bla di like.")]})
    meta = out["messages"][-1].additional_kwargs["ivoire"]
    assert meta["fra"] == "Bonjour ! Je vais bien."
    assert meta["raw_fra"].startswith("**Bonjour")
    assert out["messages"][-1].content == "[fra>bci] Bonjour ! Je vais bien."


def test_no_cap_and_no_cleaning(fake: FakeBackend) -> None:
    model = ToolCallingFake(messages=iter([AIMessage(content="**Un.** Deux. Trois.")]))
    mw = IvoireLanguageMiddleware(backend=fake, max_sentences=None, clean=False)
    out = create_agent(model, middleware=[mw]).invoke({"messages": [HumanMessage("Bla di like.")]})
    assert out["messages"][-1].additional_kwargs["ivoire"]["fra"] == "**Un.** Deux. Trois."


def test_list_content_keeps_other_blocks(fake: FakeBackend) -> None:
    model = ToolCallingFake(messages=iter([AIMessage(content=REPLY_FRA)]))
    agent = create_agent(model, middleware=[IvoireLanguageMiddleware(backend=fake)])
    audio = {"type": "audio", "base64": "UklGRg==", "mime_type": "audio/wav"}
    msg = HumanMessage(content=[{"type": "text", "text": USER_BCI}, audio])
    out = agent.invoke({"messages": [msg]})
    seen_user = model.seen[0][1]
    assert seen_user.content == [{"type": "text", "text": USER_FRA}, audio]
    assert out["messages"][0].content == msg.content


def test_empty_final_reply_is_left_alone(fake: FakeBackend) -> None:
    model = ToolCallingFake(messages=iter([AIMessage(content="")]))
    agent = create_agent(model, middleware=[IvoireLanguageMiddleware(backend=fake)])
    out = agent.invoke({"messages": [HumanMessage("Bla di like.")]})
    assert (
        out["messages"][-1].content == "" and "ivoire" not in out["messages"][-1].additional_kwargs
    )
    assert all(c[1]["source"] == "bci" for c in fake.calls)


def test_graph_has_both_hooks(fake: FakeBackend) -> None:
    agent = create_agent(
        ToolCallingFake(messages=iter([])), middleware=[IvoireLanguageMiddleware(backend=fake)]
    )
    nodes = set(agent.get_graph().nodes)
    assert {
        "IvoireLanguageMiddleware.before_agent",
        "IvoireLanguageMiddleware.after_agent",
    } <= nodes


def test_options(fake: FakeBackend) -> None:
    with pytest.raises(NotImplementedError, match="planned"):
        IvoireLanguageMiddleware("dyu", backend=fake)
    tr = IvoireTranslator(backend=fake, style="tale")
    mw = IvoireLanguageMiddleware(translator=tr, style="tale")
    assert mw.translator is tr and mw.pivot.style == "tale"
    with pytest.raises(ValueError):
        IvoireLanguageMiddleware(backend=fake, style="poetry").pivot.to_french(["x"])
