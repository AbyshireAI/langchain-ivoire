"""``with_language``: make any chat model talk in an Ivorian language by pivoting through French.

    user (Baoulé) --bci->fra--> chat model (French in, short simple French out) --fra->bci--> AIMessage (Baoulé)

The wrapped model never sees Baoulé, so any chat model works (hosted or local).
For agents built with ``langchain.agents.create_agent``, use
:class:`~langchain_ivoire.IvoireLanguageMiddleware` instead; both share the pivot code in
``langchain_ivoire._pivot``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManager,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import ConfigDict, PrivateAttr

from langchain_ivoire._languages import resolve_language
from langchain_ivoire._pivot import (
    DEFAULT_SYSTEM_HINT,
    IVOIRE_KEY,
    LanguagePivot,
    build_system_hint,
    clean_reply,
    message_text,
    stored_french,
    with_text,
)
from langchain_ivoire.backends import BackendLike
from langchain_ivoire.translator import IvoireTranslator

__all__ = [
    "DEFAULT_SYSTEM_HINT",
    "IvoireLanguageChat",
    "clean_reply",
    "message_text",
    "with_language",
]

_CACHE_MAX = 512


def _child_callbacks(
    run_manager: CallbackManagerForLLMRun | AsyncCallbackManagerForLLMRun | None,
) -> CallbackManager | None:
    """Callbacks for the inner model, nested under this run (LLM run managers have no get_child)."""
    if run_manager is None:
        return None
    manager = CallbackManager(handlers=[], parent_run_id=run_manager.run_id)
    manager.set_handlers(list(run_manager.inheritable_handlers))
    manager.add_tags(list(run_manager.inheritable_tags))
    manager.add_metadata(dict(run_manager.inheritable_metadata))
    return manager


def _ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


class IvoireLanguageChat(BaseChatModel):
    """A chat model that speaks ``lang`` by wrapping a French-speaking chat model.

    Human messages (and earlier assistant turns) are translated ``lang`` -> French, a French system
    hint asks the model for short simple French, and the reply is translated back. The returned
    :class:`~langchain_core.messages.AIMessage` holds the ``lang`` text; the French texts and
    per-stage timings are in ``response_metadata["ivoire"]``. Use :func:`with_language` to build one.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Runnable[Any, Any]
    """The wrapped chat model (or any Runnable that takes a list of messages)."""
    translator: IvoireTranslator
    lang: str = "bci"
    style: str = "everyday"
    system_hint: str | None = None
    """French system hint; ``None`` adds none."""
    clean: bool = True
    """Strip markdown/emoji from the French reply before translating it."""
    max_sentences: int | None = None
    """Keep at most this many sentences of the reply (the demo uses 2)."""

    _fra_cache: dict[str, str] = PrivateAttr(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "ivoire-language"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        inner = getattr(self.llm, "_llm_type", type(self.llm).__name__)
        return {"lang": self.lang, "style": self.style, "llm": inner}

    @property
    def pivot(self) -> LanguagePivot:
        """The shared pivot (translation, hint, reply cleaning) for this model's settings."""
        return LanguagePivot(
            self.translator,
            style=self.style,
            system_hint=self.system_hint,
            clean=self.clean,
            max_sentences=self.max_sentences,
        )

    # ---- input side --------------------------------------------------------------------------
    def _plan(self, messages: Sequence[BaseMessage]) -> tuple[list[str | None], list[str]]:
        """For each message: its text (None = keep as is), plus texts still to translate."""
        known: list[str | None] = []
        todo: list[str] = []
        for m in messages:
            if isinstance(m, (HumanMessage, AIMessage)):
                txt = message_text(m)
                fra = stored_french(m)
                if fra is not None:
                    self._remember(txt, fra)
                elif txt.strip() and txt not in self._fra_cache:
                    todo.append(txt)
                known.append(txt)
            else:
                known.append(None)
        return known, list(dict.fromkeys(todo))

    def _remember(self, text: str, fra: str) -> None:
        if len(self._fra_cache) >= _CACHE_MAX:
            self._fra_cache.pop(next(iter(self._fra_cache)))
        self._fra_cache[text] = fra

    def _assemble(
        self, messages: Sequence[BaseMessage], known: list[str | None], pivot: LanguagePivot
    ) -> tuple[list[BaseMessage], str]:
        out: list[BaseMessage] = []
        last_user_fra = ""
        for m, txt in zip(messages, known, strict=True):
            if txt is None or not txt.strip():
                out.append(m)
                continue
            fra = self._fra_cache.get(txt, txt)
            extra = {k: v for k, v in m.additional_kwargs.items() if k != IVOIRE_KEY}
            out.append(with_text(m, fra, additional_kwargs=extra))
            if isinstance(m, HumanMessage):
                last_user_fra = fra
        if self.system_hint:
            first = out[0] if out and isinstance(out[0], SystemMessage) else None
            hinted = pivot.merge_hint(first)
            assert hinted is not None
            if first is not None:
                out[0] = hinted
            else:
                out.insert(0, hinted)
        return out, last_user_fra

    # ---- output side -------------------------------------------------------------------------
    def _result(
        self,
        reply: Any,
        user_fra: str,
        reply_bci: str,
        reply_fra: str,
        raw_fra: str,
        timings: dict[str, int],
    ) -> ChatResult:
        base = reply if isinstance(reply, BaseMessage) else AIMessage(content=str(reply))
        meta = dict(base.response_metadata or {})
        meta[IVOIRE_KEY] = {
            "lang": self.lang,
            "style": self.style,
            "backend": self.translator.backend.name,
            "input_fra": user_fra,
            "output_fra": reply_fra,
            **({"raw_output_fra": raw_fra} if raw_fra != reply_fra else {}),
            "timings_ms": timings,
        }
        kw: dict[str, Any] = {"content": reply_bci, "response_metadata": meta, "id": base.id}
        if isinstance(base, AIMessage):
            kw["tool_calls"] = base.tool_calls
            kw["invalid_tool_calls"] = base.invalid_tool_calls
            kw["usage_metadata"] = base.usage_metadata
        msg = AIMessage(**kw)
        if reply_bci:
            self._remember(reply_bci, reply_fra)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _llm_kwargs(self, stop: list[str] | None, kwargs: dict[str, Any]) -> dict[str, Any]:
        return {**({"stop": stop} if stop is not None else {}), **kwargs}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        t_all = time.perf_counter()
        timings: dict[str, int] = {}
        pivot = self.pivot
        known, todo = self._plan(messages)
        t0 = time.perf_counter()
        if todo:
            for src, tr in zip(todo, pivot.to_french(todo), strict=True):
                self._remember(src, tr.text)
        timings["translate_in"] = _ms(t0)
        fr_messages, user_fra = self._assemble(messages, known, pivot)
        config: RunnableConfig = {"callbacks": _child_callbacks(run_manager)}
        t0 = time.perf_counter()
        reply = self.llm.invoke(fr_messages, config, **self._llm_kwargs(stop, kwargs))
        timings["llm"] = _ms(t0)
        raw = message_text(reply) if isinstance(reply, BaseMessage) else str(reply)
        reply_fra = pivot.finish_french(raw)
        t0 = time.perf_counter()
        bci = pivot.from_french(reply_fra)
        timings["translate_out"] = _ms(t0)
        timings["total"] = _ms(t_all)
        return self._result(reply, user_fra, bci, reply_fra, raw, timings)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        t_all = time.perf_counter()
        timings: dict[str, int] = {}
        pivot = self.pivot
        known, todo = self._plan(messages)
        t0 = time.perf_counter()
        if todo:
            for src, tr in zip(todo, await pivot.ato_french(todo), strict=True):
                self._remember(src, tr.text)
        timings["translate_in"] = _ms(t0)
        fr_messages, user_fra = self._assemble(messages, known, pivot)
        config: RunnableConfig = {"callbacks": _child_callbacks(run_manager)}
        t0 = time.perf_counter()
        reply = await self.llm.ainvoke(fr_messages, config, **self._llm_kwargs(stop, kwargs))
        timings["llm"] = _ms(t0)
        raw = message_text(reply) if isinstance(reply, BaseMessage) else str(reply)
        reply_fra = pivot.finish_french(raw)
        t0 = time.perf_counter()
        bci = await pivot.afrom_french(reply_fra)
        timings["translate_out"] = _ms(t0)
        timings["total"] = _ms(t_all)
        return self._result(reply, user_fra, bci, reply_fra, raw, timings)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> IvoireLanguageChat:
        """Bind tools on the wrapped model (tool arguments stay in French/the model's language)."""
        bind = getattr(self.llm, "bind_tools", None)
        if bind is None:
            raise NotImplementedError(f"{type(self.llm).__name__} does not support bind_tools")
        return self.model_copy(update={"llm": bind(tools, **kwargs)})


def with_language(
    llm: Runnable[Any, Any],
    lang: str = "bci",
    *,
    style: str = "everyday",
    backend: BackendLike = None,
    api_key: str | None = None,
    base_url: str | None = None,
    translator: IvoireTranslator | None = None,
    system_hint: str | bool | None = True,
    clean: bool = True,
    max_sentences: int | None = None,
) -> IvoireLanguageChat:
    """Wrap ``llm`` so it can be talked to in ``lang`` (Baoulé: ``"bci"``).

    Example:
        .. code-block:: python

            from langchain.chat_models import init_chat_model
            from langchain_ivoire import with_language

            model = init_chat_model("provider:model")  # any chat model
            chat = with_language(model, lang="bci")
            msg = chat.invoke("N tɛ, an ti sɛn?")
            msg.content                                # Baoulé reply
            msg.response_metadata["ivoire"]["output_fra"]  # what the model said in French

    Args:
        llm: any chat model (or Runnable taking a list of messages).
        lang: the Ivorian language.
        style: translation register (``"everyday"``, ``"tale"``, ``"formal"``).
        backend: translation backend (default: remote API).
        api_key: See ``backend``.
        base_url: See ``backend``.
        translator: a ready :class:`IvoireTranslator` (overrides backend options).
        system_hint: ``True`` = :data:`DEFAULT_SYSTEM_HINT` (mentions ``max_sentences``), a string
            = your own French hint, ``False``/``None`` = no hint.
        clean: strip markdown/emoji from the French reply before translating.
        max_sentences: keep only the first N sentences of the reply.
    """
    lg = resolve_language(lang)
    if translator is None:
        translator = IvoireTranslator(
            lang=lg.code, style=style, backend=backend, api_key=api_key, base_url=base_url
        )
    elif translator.lang != lg.code:
        raise ValueError(f"translator is for {translator.lang!r}, not {lg.code!r}")
    return IvoireLanguageChat(
        llm=llm,
        translator=translator,
        lang=lg.code,
        style=style,
        system_hint=build_system_hint(lg.code, system_hint, max_sentences),
        clean=clean,
        max_sentences=max_sentences,
    )
