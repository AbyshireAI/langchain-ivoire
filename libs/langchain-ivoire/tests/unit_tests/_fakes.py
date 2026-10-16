"""Test doubles shared by the unit tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import BaseMessage
from pydantic import Field


class ToolCallingFake(GenericFakeChatModel):
    """A scripted chat model that accepts ``bind_tools`` and records what it was sent."""

    seen: list[list[BaseMessage]] = Field(default_factory=list)
    bound: list[Any] = Field(default_factory=list)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        self.bound[:] = list(tools)
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)
