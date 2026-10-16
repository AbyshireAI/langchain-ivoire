"""``IvoireLanguageMiddleware``: an agent that talks with the user in an Ivorian language.

For agents built with :func:`langchain.agents.create_agent`::

    user (Baoulé) --before_agent: bci->fra--> agent loop in French (model, tools) --after_agent:
    fra->bci--> final reply (Baoulé)

* ``before_agent`` translates every user message that has no French yet. The message keeps its
  Baoulé ``content``; the French goes into ``additional_kwargs["ivoire"]["fra"]`` and the message
  is updated in place (same ID, so ``add_messages`` replaces it rather than appending).
* ``wrap_model_call`` hands the model a French view of the conversation
  (``ModelRequest.override(messages=..., system_message=...)``) with a short French system hint.
  Tool calls, tool arguments and tool results therefore stay in French.
* ``after_agent`` translates the final AI message to Baoulé and keeps the French (and the model's
  raw reply, if cleaning changed it) in ``additional_kwargs["ivoire"]``.

Example:
    .. code-block:: python

        from langchain.agents import create_agent
        from langchain_ivoire import IvoireLanguageMiddleware

        model = "provider:model"  # any model id that init_chat_model accepts
        agent = create_agent(
            model,
            tools=[get_weather],
            middleware=[IvoireLanguageMiddleware("bci")],
        )
        out = agent.invoke({"messages": [{"role": "user", "content": "N tɛ, an ti sɛn?"}]})
        out["messages"][-1].content                          # Baoulé reply
        out["messages"][-1].additional_kwargs["ivoire"]["fra"]  # what the model said in French

Put it first in ``middleware=[...]`` so that every other middleware sees the French view.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, cast

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage
from langgraph.runtime import Runtime

from langchain_ivoire._languages import resolve_language
from langchain_ivoire._pivot import (
    IVOIRE_KEY,
    LanguagePivot,
    build_system_hint,
    french_view,
    message_text,
    stored_french,
    with_text,
)
from langchain_ivoire.backends import BackendLike
from langchain_ivoire.translator import IvoireTranslator

__all__ = ["IvoireLanguageMiddleware"]


def _ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


class IvoireLanguageMiddleware(AgentMiddleware[AgentState[Any], Any, Any]):
    """Translate the user's messages to French before the agent, and the reply back after it.

    Args:
        lang: the Ivorian language (``"bci"``).
        style: translation register (``"everyday"``, ``"tale"`` or ``"formal"``).
        backend: translation backend (default: the remote API).
        api_key: See ``backend``.
        base_url: See ``backend``.
        translator: a ready :class:`~langchain_ivoire.IvoireTranslator` (overrides the backend
            options).
        system_hint: ``True`` adds a short French hint (reply in short simple French, at most
            ``max_sentences`` sentences, no markdown); a string is used as the hint; ``False`` or
            ``None`` adds none. The hint is appended to the agent's own system prompt.
        max_sentences: keep at most this many sentences of the final French reply (``None``: all).
        clean: strip markdown, emoji and brackets from the final French reply before translating.
    """

    def __init__(
        self,
        lang: str = "bci",
        *,
        style: str = "everyday",
        backend: BackendLike = None,
        api_key: str | None = None,
        base_url: str | None = None,
        translator: IvoireTranslator | None = None,
        system_hint: str | bool | None = True,
        max_sentences: int | None = 2,
        clean: bool = True,
    ) -> None:
        super().__init__()
        code = resolve_language(lang).code
        if translator is None:
            translator = IvoireTranslator(
                code, style=style, backend=backend, api_key=api_key, base_url=base_url
            )
        elif translator.lang != code:
            raise ValueError(f"translator is for {translator.lang!r}, not {code!r}")
        self.lang = code
        self.pivot = LanguagePivot(
            translator,
            style=style,
            system_hint=build_system_hint(code, system_hint, max_sentences),
            clean=clean,
            max_sentences=max_sentences,
        )

    @property
    def translator(self) -> IvoireTranslator:
        return self.pivot.translator

    # ---- before_agent: user messages -> French -------------------------------------------------
    @staticmethod
    def _pending(messages: Sequence[AnyMessage]) -> list[tuple[BaseMessage, str]]:
        out: list[tuple[BaseMessage, str]] = []
        for m in messages:
            if isinstance(m, HumanMessage) and m.id and stored_french(m) is None:
                txt = message_text(m)
                if txt.strip():
                    out.append((m, txt))
        return out

    def _tag_users(
        self, todo: list[tuple[BaseMessage, str]], french: dict[str, str], ms: int
    ) -> dict[str, Any] | None:
        if not todo:
            return None
        updated = []
        for m, txt in todo:
            meta = self.pivot.meta(french[txt], timings_ms={"translate_in": ms})
            updated.append(
                m.model_copy(
                    update={"additional_kwargs": {**m.additional_kwargs, IVOIRE_KEY: meta}}
                )
            )
        return {"messages": updated}

    def before_agent(self, state: AgentState[Any], runtime: Runtime[Any]) -> dict[str, Any] | None:
        todo = self._pending(state["messages"])
        if not todo:
            return None
        texts = list(dict.fromkeys(t for _, t in todo))
        t0 = time.perf_counter()
        french = {src: tr.text for src, tr in zip(texts, self.pivot.to_french(texts), strict=True)}
        return self._tag_users(todo, french, _ms(t0))

    async def abefore_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        todo = self._pending(state["messages"])
        if not todo:
            return None
        texts = list(dict.fromkeys(t for _, t in todo))
        t0 = time.perf_counter()
        trs = await self.pivot.ato_french(texts)
        french = {src: tr.text for src, tr in zip(texts, trs, strict=True)}
        return self._tag_users(todo, french, _ms(t0))

    # ---- wrap_model_call: the model sees French only -------------------------------------------
    def _french_request(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        return request.override(
            messages=cast("list[AnyMessage]", french_view(request.messages)),
            system_message=self.pivot.merge_hint(request.system_message),
        )

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelCallResult:
        return handler(self._french_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelCallResult:
        return await handler(self._french_request(request))

    # ---- after_agent: final reply -> the user's language ---------------------------------------
    @staticmethod
    def _final_reply(messages: Sequence[AnyMessage]) -> AIMessage | None:
        if not messages:
            return None
        last = messages[-1]
        if (
            isinstance(last, AIMessage)
            and last.id
            and not last.tool_calls
            and stored_french(last) is None
            and message_text(last).strip()
        ):
            return last
        return None

    def _tag_reply(self, msg: AIMessage, raw: str, fra: str, text: str, ms: int) -> dict[str, Any]:
        extra: dict[str, Any] = {"timings_ms": {"translate_out": ms}}
        if raw != fra:
            extra["raw_fra"] = raw
        meta = self.pivot.meta(fra, **extra)
        new = with_text(msg, text, additional_kwargs={**msg.additional_kwargs, IVOIRE_KEY: meta})
        return {"messages": [new]}

    def after_agent(self, state: AgentState[Any], runtime: Runtime[Any]) -> dict[str, Any] | None:
        msg = self._final_reply(state["messages"])
        if msg is None:
            return None
        raw = message_text(msg)
        fra = self.pivot.finish_french(raw)
        t0 = time.perf_counter()
        text = self.pivot.from_french(fra)
        return self._tag_reply(msg, raw, fra, text, _ms(t0))

    async def aafter_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        msg = self._final_reply(state["messages"])
        if msg is None:
            return None
        raw = message_text(msg)
        fra = self.pivot.finish_french(raw)
        t0 = time.perf_counter()
        text = await self.pivot.afrom_french(fra)
        return self._tag_reply(msg, raw, fra, text, _ms(t0))
