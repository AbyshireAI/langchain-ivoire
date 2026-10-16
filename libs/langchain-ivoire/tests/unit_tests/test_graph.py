from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from langchain_ivoire import FakeBackend, create_voice_agent
from langchain_ivoire.audio import from_audio_block, to_audio_block


def _edges(graph: Any) -> set[tuple[str, str]]:
    return {(e.source, e.target) for e in graph.get_graph().edges}


def test_graph_structure_with_phrasebook(fake: FakeBackend, phrasebook_path: Path) -> None:
    g = create_voice_agent(
        FakeListChatModel(responses=["x"]), phrasebook=phrasebook_path, backend=fake
    )
    nodes = set(g.get_graph().nodes)
    assert nodes == {
        "__start__",
        "listen",
        "phrase_snap",
        "low_confidence",
        "translate_in",
        "agent",
        "translate_out",
        "speak",
        "out",
        "__end__",
    }
    assert _edges(g) == {
        ("__start__", "listen"),
        ("listen", "out"),
        ("listen", "phrase_snap"),
        ("listen", "low_confidence"),
        ("listen", "translate_in"),
        ("phrase_snap", "agent"),
        ("phrase_snap", "translate_in"),
        ("low_confidence", "translate_in"),
        ("translate_in", "agent"),
        ("agent", "translate_out"),
        ("agent", "out"),
        ("translate_out", "speak"),
        ("speak", "out"),
        ("out", "__end__"),
    }


def test_graph_structure_minimal(fake: FakeBackend) -> None:
    g = create_voice_agent(FakeListChatModel(responses=["x"]), backend=fake, speak=False)
    nodes = set(g.get_graph().nodes)
    assert "phrase_snap" not in nodes and "speak" not in nodes
    assert ("translate_out", "out") in _edges(g)
    assert "listen" in g.get_graph().draw_mermaid()


def test_phrase_snap_path(phrasebook_path: Path, wav: bytes) -> None:
    be = FakeBackend(
        transcripts=["bla de le kɛ"],
        translations={("fra", "bci", "Oui, merci."): "Cɛcɛ, n yo."},
        confidence=0.55,  # low, but a clear phrase match wins (as in the demo)
    )
    g = create_voice_agent(
        FakeListChatModel(responses=["Oui, merci."]), phrasebook=phrasebook_path, backend=be
    )
    out = g.invoke({"audio": wav})
    assert out["route"] == ["listen", "phrase_snap", "agent", "translate_out", "speak", "out"]
    assert out["heard"] == "bla de le kɛ"
    assert out["transcript"] == "Bla di like."
    assert out["user_fra"] == "Venez manger."  # from the phrasebook, no MT
    assert not any(c[0] == "translate" and c[1]["source"] == "bci" for c in be.calls)
    assert out["phrase"]["translations"]["eng"] == "Come and eat."
    assert out["reply"] == "Cɛcɛ, n yo."
    assert from_audio_block(out["reply_audio"])[:4] == b"RIFF"
    assert out["reply_audio"]["mime_type"] == "audio/wav"
    assert out["low_confidence"] is False
    assert {"listen", "phrase_snap", "agent", "translate_out", "speak", "total"} <= set(
        out["timings_ms"]
    )


def test_ok_path(fake: FakeBackend, wav: bytes) -> None:
    g = create_voice_agent(FakeListChatModel(responses=["Je vais bien, merci."]), backend=fake)
    out = g.invoke({"audio": wav})
    assert out["route"] == ["listen", "translate_in", "agent", "translate_out", "speak", "out"]
    assert out["user_fra"] == "Venez manger."
    assert out["reply_fra"] == "Je vais bien, merci."
    assert out["reply"] == "N liɛ yo'a kpa, n yo."


def test_low_confidence_path(wav: bytes) -> None:
    be = FakeBackend(transcripts=["ɔ ti kekle sikasua"], confidence=0.4)
    g = create_voice_agent(FakeListChatModel(responses=["Pouvez-vous répéter ?"]), backend=be)
    out = g.invoke({"audio": wav})
    assert out["route"][:3] == ["listen", "low_confidence", "translate_in"]
    assert out["low_confidence"] is True
    assert out["reply"] == "[fra>bci] Pouvez-vous répéter ?"


def test_nothing_recognised(wav: bytes) -> None:
    be = FakeBackend(transcripts=[""])
    g = create_voice_agent(FakeListChatModel(responses=["x"]), backend=be)
    out = g.invoke({"audio": wav})
    assert out["route"] == ["listen", "out"]
    assert out["reply"] == ""


def test_text_input_skips_asr(fake: FakeBackend) -> None:
    g = create_voice_agent(
        FakeListChatModel(responses=["Je vais bien, merci."]), backend=fake, speak=False
    )
    out = g.invoke({"text": "N tɛ, an ti sɛn?"})
    assert out["user_fra"] == "Bonjour, comment allez-vous ?"
    assert out["reply"] == "N liɛ yo'a kpa, n yo."
    assert not any(c[0] == "transcribe" for c in fake.calls)


def test_llm_failure_still_returns_translation(fake: FakeBackend, wav: bytes) -> None:
    def boom(_: Any) -> Any:
        raise TimeoutError("provider down")

    g = create_voice_agent(RunnableLambda(boom), backend=fake)
    out = g.invoke({"audio": wav})
    assert out["route"] == ["listen", "translate_in", "agent_failed", "out"]
    assert out["user_fra"] == "Venez manger."
    assert "TimeoutError" in out["error"]
    assert not out.get("reply_audio")


def test_memory_with_checkpointer(fake: FakeBackend, wav: bytes) -> None:
    seen: list[list[BaseMessage]] = []
    replies = iter(["Je vais bien, merci.", "Venez manger."])

    def llm(msgs: list[BaseMessage]) -> str:
        seen.append(msgs)
        return next(replies)

    g = create_voice_agent(RunnableLambda(llm), backend=fake, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    g.invoke({"audio": wav}, cfg)
    out = g.invoke({"text": "N tɛ, an ti sɛn?"}, cfg)
    # second turn: system + 2 earlier French messages + the new one
    assert [m.type for m in seen[1]] == ["system", "human", "ai", "human"]
    assert len(out["messages"]) == 4
    assert out["route"][0] == "listen" and out["route"].count("listen") == 1  # reset per turn
    assert out["phrase"] is None and out["audio"] is None


async def test_async_invoke(fake: FakeBackend, wav: bytes) -> None:
    g = create_voice_agent(FakeListChatModel(responses=["Je vais bien, merci."]), backend=fake)
    out = await g.ainvoke({"audio": wav})
    assert out["reply"] == "N liɛ yo'a kpa, n yo."


def test_planned_language() -> None:
    with pytest.raises(NotImplementedError):
        create_voice_agent(FakeListChatModel(responses=["x"]), lang="dyu", backend="fake")


def test_state_holds_audio_blocks_and_is_json(fake: FakeBackend, wav: bytes) -> None:
    import base64
    import json

    g = create_voice_agent(
        FakeListChatModel(responses=["Je vais bien, merci."] * 3),
        backend=fake,
        checkpointer=InMemorySaver(),
    )
    cfg: Any = {"configurable": {"thread_id": "json"}}
    for audio in (to_audio_block(wav), base64.b64encode(wav).decode(), wav):
        out = g.invoke({"audio": audio}, cfg)
        assert out["reply"] == "N liɛ yo'a kpa, n yo." and out["audio"] is None
    state = dict(g.get_state(cfg).values)
    state.pop("messages")
    json.dumps(state)  # no bytes, no dataclasses
    assert state["reply_audio"]["type"] == "audio"


def test_bad_audio_input(fake: FakeBackend) -> None:
    g = create_voice_agent(FakeListChatModel(responses=["x"]), backend=fake)
    with pytest.raises(TypeError):
        g.invoke({"audio": 3.5})
    with pytest.raises(ValueError, match="base64"):
        g.invoke({"audio": "not base64!"})
