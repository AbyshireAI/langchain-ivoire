r"""Prebuilt LangGraph voice agent: speech in an Ivorian language in, spoken reply out.

::

    START -> listen -(nothing)-------------------------------------------------> out -> END
                    |-(phrase)--> phrase_snap -(has French)--> agent
                    |                         \-(no French)--> translate_in
                    |-(low)-----> low_confidence -> translate_in -> agent
                    \-(ok)-----------------------> translate_in -> agent
             agent -(ok)--> translate_out -> speak -> out          (speak only if speak=True)
                   \-(LLM failed)--------------------> out

Mirrors the Abyshire demo graph: ASR -> (sound-alike phrasebook snap) -> lang->French -> LLM (French in,
short simple French out) -> French->lang -> TTS. Low-confidence turns are marked and still translated
(an app can ask the speaker to repeat). The LLM sees only French; its conversation (French) is kept in
``messages``, so pass a checkpointer and a ``thread_id`` for multi-turn memory.
"""

from __future__ import annotations

import time
from collections.abc import Hashable
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.content import AudioContentBlock
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from langchain_ivoire._languages import resolve_language
from langchain_ivoire._pivot import build_system_hint, clean_reply, message_text
from langchain_ivoire.audio import decode_base64_audio, from_audio_block, to_audio_block
from langchain_ivoire.backends import BackendLike
from langchain_ivoire.phrasebook import Phrasebook, PhrasebookLike
from langchain_ivoire.speaker import IvoireSpeaker
from langchain_ivoire.transcriber import IvoireTranscriber
from langchain_ivoire.translator import IvoireTranslator

__all__ = ["VoiceAgentState", "create_voice_agent"]

_RESET = "__reset__"


def _route_reducer(a: list[str] | None, b: list[str] | None) -> list[str]:
    b = list(b or [])
    if b[:1] == [_RESET]:  # a new turn starts
        return b[1:]
    return list(a or []) + b


def _timings_reducer(a: dict[str, int] | None, b: dict[str, int] | None) -> dict[str, int]:
    b = dict(b or {})
    if b.pop(_RESET, None) is not None:
        return b
    return {**(a or {}), **b}


class VoiceAgentState(TypedDict, total=False):
    """State of the voice agent graph. Inputs: ``audio`` or ``text`` (already in ``lang``).

    Audio is kept as standard LangChain audio content blocks (base64 inside), so the whole state is
    JSON-serialisable and checkpoints cleanly.
    """

    audio: AudioContentBlock | str | bytes | None
    """Input speech: an audio content block (recommended, see
    :func:`langchain_ivoire.audio.to_audio_block`), base64 text, or raw bytes. Cleared by ``listen``."""
    text: str | None
    transcript: str
    """What was understood, in ``lang`` (the phrase's checked text after a phrase snap)."""
    heard: str
    """Raw ASR text when the turn was snapped to a phrase."""
    asr_confidence: float | None
    low_confidence: bool
    phrase: dict[str, Any] | None
    user_fra: str
    reply_fra: str
    reply: str
    """The agent's reply in ``lang``."""
    reply_audio: AudioContentBlock | None
    """The spoken reply as an audio content block (``from_audio_block`` gives the WAV bytes)."""
    error: str | None
    messages: Annotated[list[AnyMessage], add_messages]
    """The (French) conversation seen by the LLM."""
    route: Annotated[list[str], _route_reducer]
    timings_ms: Annotated[dict[str, int], _timings_reducer]


def _audio_bytes(audio: Any) -> bytes | None:
    """Input audio as bytes: a content block (URLs must be public http/https), base64 or bytes."""
    if not audio:
        return None
    if isinstance(audio, (bytes, bytearray, memoryview)):
        return bytes(audio)
    if isinstance(audio, str):
        return decode_base64_audio(audio)
    if isinstance(audio, dict):
        return from_audio_block(audio, fetch_url=True)
    raise TypeError(f"unsupported audio input: {type(audio).__name__}")


def _ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


def create_voice_agent(
    llm: Runnable[Any, Any],
    lang: str = "bci",
    phrasebook: PhrasebookLike | None = None,
    *,
    backend: BackendLike = None,
    api_key: str | None = None,
    base_url: str | None = None,
    transcriber: IvoireTranscriber | None = None,
    translator: IvoireTranslator | None = None,
    speaker: IvoireSpeaker | None = None,
    style: str = "everyday",
    low_confidence_threshold: float = 0.80,
    system_prompt: str | None = None,
    speak: bool = True,
    max_history_messages: int = 20,
    checkpointer: Any = None,
    name: str = "ivoire_voice_agent",
) -> Any:
    """Build and compile the voice agent graph.

    Example:
        .. code-block:: python

            from langchain_ivoire import create_voice_agent

            agent = create_voice_agent(llm, lang="bci", phrasebook="phrasebook.json")
            wav = open("question.wav", "rb").read()
            out = agent.invoke({"audio": to_audio_block(wav)})
            out["transcript"], out["user_fra"], out["reply_fra"], out["reply"]
            open("reply.wav", "wb").write(from_audio_block(out["reply_audio"]))

    Args:
        llm: any chat model; it receives French and should answer in French.
        lang: the Ivorian language (``"bci"``).
        phrasebook: a :class:`Phrasebook`, a JSON path or a list of entries; enables the
            ``phrase_snap`` node.
        backend: shared backend for the default components.
        api_key: See ``backend``.
        base_url: See ``backend``.
        transcriber: custom components (override the backend options).
        translator: See ``transcriber``.
        speaker: See ``transcriber``.
        style: translation register.
        low_confidence_threshold: ASR confidence below which the turn is marked ``low_confidence``.
        system_prompt: French system prompt for the LLM (default: a short-simple-French hint).
        speak: add the ``speak`` node (TTS of the reply).
        max_history_messages: French messages of earlier turns passed to the LLM.
        checkpointer: LangGraph checkpointer for multi-turn memory.

    Returns:
        A compiled LangGraph graph (``invoke``/``ainvoke``/``stream`` with :class:`VoiceAgentState`).
    """
    code = resolve_language(lang).code
    shared = None
    if transcriber is None or translator is None or (speak and speaker is None):
        from langchain_ivoire._base import make_backend

        shared = make_backend(backend, api_key, base_url)
    asr = transcriber or IvoireTranscriber(
        code, backend=shared, low_confidence_threshold=low_confidence_threshold
    )
    mt = translator or IvoireTranslator(code, style=style, backend=shared)
    tts = speaker or (IvoireSpeaker(code, backend=shared) if speak else None)
    pb = Phrasebook.coerce(phrasebook, code) if phrasebook is not None else None
    prompt = system_prompt or build_system_hint(code, True) or ""

    # ---- nodes -------------------------------------------------------------------------------
    def listen(state: VoiceAgentState, config: RunnableConfig) -> dict[str, Any]:
        t0 = time.perf_counter()
        fresh: dict[str, Any] = {
            "audio": None,
            "text": None,
            "heard": "",
            "phrase": None,
            "low_confidence": False,
            "user_fra": "",
            "reply_fra": "",
            "reply": "",
            "reply_audio": None,
            "error": None,
        }
        audio = _audio_bytes(state.get("audio"))
        if audio:
            t = asr.invoke(audio, config)
            fresh.update(transcript=t.text, asr_confidence=t.confidence)
            low = t.low_confidence
        else:
            fresh.update(transcript=(state.get("text") or "").strip(), asr_confidence=None)
            low = False
        fresh["low_confidence"] = low
        fresh["route"] = [_RESET, "listen"]
        fresh["timings_ms"] = {_RESET: 0, "listen": _ms(t0)}
        return fresh

    def route_listen(state: VoiceAgentState) -> str:
        if not state.get("transcript"):
            return "nothing"
        if pb is not None and pb.match(state["transcript"]):
            return "phrase"
        return "low" if state.get("low_confidence") else "ok"

    def phrase_snap(state: VoiceAgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        assert pb is not None
        m = pb.match(state["transcript"])
        assert m is not None
        ph = m.phrase
        return {
            "heard": state["transcript"],
            "transcript": ph.text,
            "user_fra": ph.translations.get("fra", ""),
            "low_confidence": False,
            "phrase": {
                "id": ph.id,
                "text": ph.text,
                "translations": dict(ph.translations),
                "score": m.score,
                "runner_up": m.runner_up,
                "heard": m.heard,
            },
            "route": ["phrase_snap"],
            "timings_ms": {"phrase_snap": _ms(t0)},
        }

    def route_phrase(state: VoiceAgentState) -> str:
        return "agent" if state.get("user_fra") else "translate_in"

    def low_confidence(state: VoiceAgentState) -> dict[str, Any]:
        return {"low_confidence": True, "route": ["low_confidence"]}

    def translate_in(state: VoiceAgentState, config: RunnableConfig) -> dict[str, Any]:
        t0 = time.perf_counter()
        tr = mt.invoke({"text": state["transcript"], "source": code, "target": "fra"}, config)
        return {
            "user_fra": tr.text,
            "route": ["translate_in"],
            "timings_ms": {"translate_in": _ms(t0)},
        }

    def agent(state: VoiceAgentState, config: RunnableConfig) -> dict[str, Any]:
        t0 = time.perf_counter()
        history = (
            list(state.get("messages") or [])[-max_history_messages:]
            if max_history_messages
            else []
        )
        user = HumanMessage(content=state["user_fra"])
        msgs: list[BaseMessage] = [SystemMessage(content=prompt), *history, user]
        try:
            reply = llm.invoke(msgs, config)
        except Exception as e:  # the turn still returns the transcript and its translation
            return {
                "error": f"LLM call failed: {type(e).__name__}: {str(e)[:200]}",
                "route": ["agent_failed"],
                "timings_ms": {"agent": _ms(t0)},
            }
        raw = message_text(reply) if isinstance(reply, BaseMessage) else str(reply)
        fra = clean_reply(raw)
        return {
            "reply_fra": fra,
            "messages": [user, AIMessage(content=fra)],
            "route": ["agent"],
            "timings_ms": {"agent": _ms(t0)},
        }

    def route_agent(state: VoiceAgentState) -> str:
        return "fail" if state.get("error") or not state.get("reply_fra") else "ok"

    def translate_out(state: VoiceAgentState, config: RunnableConfig) -> dict[str, Any]:
        t0 = time.perf_counter()
        tr = mt.invoke({"text": state["reply_fra"], "source": "fra", "target": code}, config)
        return {
            "reply": tr.text,
            "route": ["translate_out"],
            "timings_ms": {"translate_out": _ms(t0)},
        }

    def speak_node(state: VoiceAgentState, config: RunnableConfig) -> dict[str, Any]:
        t0 = time.perf_counter()
        assert tts is not None
        if not state.get("reply"):
            return {"route": ["speak_skipped"]}
        s = tts.invoke(state["reply"], config)
        return {
            "reply_audio": to_audio_block(s),
            "route": ["speak"],
            "timings_ms": {"speak": _ms(t0)},
        }

    def out(state: VoiceAgentState) -> dict[str, Any]:
        total = sum(v for k, v in (state.get("timings_ms") or {}).items() if k != "total")
        return {"route": ["out"], "timings_ms": {"total": total}}

    # ---- wiring ------------------------------------------------------------------------------
    g = StateGraph(VoiceAgentState)
    g.add_node("listen", listen)
    g.add_node("low_confidence", low_confidence)
    g.add_node("translate_in", translate_in)
    g.add_node("agent", agent)
    g.add_node("translate_out", translate_out)
    g.add_node("out", out)
    g.add_edge(START, "listen")
    branches: dict[Hashable, str] = {
        "nothing": "out",
        "low": "low_confidence",
        "ok": "translate_in",
    }
    if pb is not None:
        g.add_node("phrase_snap", phrase_snap)
        branches["phrase"] = "phrase_snap"
        g.add_conditional_edges(
            "phrase_snap", route_phrase, {"agent": "agent", "translate_in": "translate_in"}
        )
    g.add_conditional_edges("listen", route_listen, branches)
    g.add_edge("low_confidence", "translate_in")
    g.add_edge("translate_in", "agent")
    g.add_conditional_edges("agent", route_agent, {"ok": "translate_out", "fail": "out"})
    if tts is not None:
        g.add_node("speak", speak_node)
        g.add_edge("translate_out", "speak")
        g.add_edge("speak", "out")
    else:
        g.add_edge("translate_out", "out")
    g.add_edge("out", END)
    return g.compile(checkpointer=checkpointer, name=name)
