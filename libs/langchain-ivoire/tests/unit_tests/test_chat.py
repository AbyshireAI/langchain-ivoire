from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import Field

from langchain_ivoire import FakeBackend, IvoireLanguageChat, IvoireTranslator, with_language
from langchain_ivoire.chat import clean_reply


class Recorder(BaseCallbackHandler):
    """Captures the messages every chat model run starts with (wrapper first, then inner model)."""

    def __init__(self) -> None:
        self.runs: list[tuple[list[BaseMessage], UUID | None]] = []

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.runs.append((messages[0], parent_run_id))


def test_round_trip_through_french(fake: FakeBackend) -> None:
    llm = FakeListChatModel(responses=["Je vais bien, merci."])
    chat = with_language(llm, lang="bci", backend=fake)
    rec = Recorder()
    out = chat.invoke("N tɛ, an ti sɛn?", config={"callbacks": [rec]})

    assert isinstance(out, AIMessage)
    assert out.content == "N liɛ yo'a kpa, n yo."
    meta = out.response_metadata["ivoire"]
    assert meta["lang"] == "bci"
    assert meta["input_fra"] == "Bonjour, comment allez-vous ?"
    assert meta["output_fra"] == "Je vais bien, merci."
    assert set(meta["timings_ms"]) == {"translate_in", "llm", "translate_out", "total"}

    # the wrapper run sees Baoulé; the inner model run (a child of it) sees French + system hint
    (outer, outer_parent), (inner, inner_parent) = rec.runs
    assert outer[-1].content == "N tɛ, an ti sɛn?"
    assert isinstance(inner[0], SystemMessage) and "français simple" in str(inner[0].content)
    assert "baoulé" in str(inner[0].content)
    assert inner[-1].content == "Bonjour, comment allez-vous ?"
    assert inner_parent is not None and outer_parent is None


def test_history_reuses_french_from_metadata(fake: FakeBackend) -> None:
    llm = GenericFakeChatModel(messages=iter(["Je vais bien, merci.", "Venez manger."]))
    chat = with_language(llm, backend=fake)
    first = chat.invoke([HumanMessage("N tɛ, an ti sɛn?")])
    n_calls = len(fake.calls)
    second = chat.invoke([HumanMessage("N tɛ, an ti sɛn?"), first, HumanMessage("Bla di like.")])
    assert second.content == "Bla di like."
    new = [c for c in fake.calls[n_calls:] if c[1]["source"] == "bci"]
    # only the new user turn is translated: the first turn is cached, the AI turn has its French
    assert [c[1]["texts"] for c in new] == [["Bla di like."]]


def test_existing_system_message_gets_the_hint(fake: FakeBackend) -> None:
    rec = Recorder()
    chat = with_language(FakeListChatModel(responses=["Oui."]), backend=fake)
    chat.invoke(
        [SystemMessage("Tu es un médecin."), HumanMessage("Bla di like.")], {"callbacks": [rec]}
    )
    inner = rec.runs[1][0]
    assert sum(isinstance(m, SystemMessage) for m in inner) == 1
    assert str(inner[0].content).startswith("Tu es un médecin.")
    assert "français simple" in str(inner[0].content)


def test_custom_or_no_hint(fake: FakeBackend) -> None:
    rec = Recorder()
    chat = with_language(FakeListChatModel(responses=["Oui."]), backend=fake, system_hint=False)
    chat.invoke("Bla di like.", {"callbacks": [rec]})
    assert not any(isinstance(m, SystemMessage) for m in rec.runs[1][0])
    chat2 = with_language(
        FakeListChatModel(responses=["Oui."]), backend=fake, system_hint="Sois bref."
    )
    assert chat2.system_hint == "Sois bref."


def test_reply_cleaning_and_sentence_cap(fake: FakeBackend) -> None:
    llm = FakeListChatModel(responses=["**Bonjour !** 😀 Je vais bien. Et vous ? Merci."])
    chat = with_language(llm, backend=fake, max_sentences=2)
    meta = chat.invoke("Bla di like.").response_metadata["ivoire"]
    assert meta["output_fra"] == "Bonjour ! Je vais bien."
    assert meta["raw_output_fra"].startswith("**Bonjour")
    assert clean_reply("- un\n- deux") == "un deux"


async def test_async(fake: FakeBackend) -> None:
    chat = with_language(FakeListChatModel(responses=["Je vais bien, merci."]), backend=fake)
    out = await chat.ainvoke("N tɛ, an ti sɛn?")
    assert out.content == "N liɛ yo'a kpa, n yo."


def test_stream_falls_back_to_one_chunk(fake: FakeBackend) -> None:
    chat = with_language(FakeListChatModel(responses=["Je vais bien, merci."]), backend=fake)
    chunks = list(chat.stream("N tɛ, an ti sɛn?"))
    assert "".join(str(c.content) for c in chunks) == "N liɛ yo'a kpa, n yo."


def test_is_a_chat_model_and_planned_language(fake: FakeBackend) -> None:
    chat = with_language(FakeListChatModel(responses=["x"]), backend=fake)
    assert isinstance(chat, BaseChatModel) and isinstance(chat, IvoireLanguageChat)
    with pytest.raises(NotImplementedError, match="planned"):
        with_language(FakeListChatModel(responses=["x"]), lang="any", backend=fake)
    chat2 = with_language(
        FakeListChatModel(responses=["x"]), translator=IvoireTranslator(backend=fake)
    )
    assert chat2.translator.backend is fake


class ToolFake(GenericFakeChatModel):
    bound: list[Any] = Field(default_factory=list)

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self.model_copy(update={"bound": list(tools)})


def test_bind_tools_delegates(fake: FakeBackend) -> None:
    def get_weather(city: str) -> str:
        """Weather for a city."""
        return "sunny"

    tool_call = AIMessage(
        content="", tool_calls=[{"name": "get_weather", "args": {"city": "Bouaké"}, "id": "1"}]
    )
    chat = with_language(ToolFake(messages=iter([tool_call])), backend=fake).bind_tools(
        [get_weather]
    )
    assert isinstance(chat, IvoireLanguageChat)
    assert chat.llm.bound == [get_weather]  # type: ignore[attr-defined]
    out = chat.invoke("Bla di like.")
    assert out.content == "" and out.tool_calls[0]["args"] == {"city": "Bouaké"}

    with pytest.raises(NotImplementedError):
        with_language(FakeListChatModel(responses=["x"]), backend=fake).bind_tools([get_weather])
