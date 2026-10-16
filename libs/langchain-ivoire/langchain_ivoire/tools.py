"""LangChain tools for agents: translate, transcribe a recording, speak a text.

Security: the transcribe tool never reads local files. It takes audio as base64, a standard audio
content block, or a public ``http``/``https`` URL, or (inside a LangGraph/``create_agent`` agent)
the most recent audio block the user sent in the conversation.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.messages.content import AudioContentBlock
from langchain_core.tools import BaseTool, InjectedToolArg
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from langchain_ivoire._base import make_backend
from langchain_ivoire._languages import resolve_language, resolve_text_language
from langchain_ivoire.audio import (
    MAX_AUDIO_BYTES,
    afetch_audio_url,
    decode_base64_audio,
    fetch_audio_url,
    from_audio_block,
    is_audio_block,
    latest_audio_block,
    to_audio_block,
)
from langchain_ivoire.backends import IvoireBackend
from langchain_ivoire.speaker import IvoireSpeaker
from langchain_ivoire.transcriber import IvoireTranscriber
from langchain_ivoire.translator import IvoireTranslator

__all__ = [
    "IvoireSpeakTool",
    "IvoireTranscribeTool",
    "IvoireTranslateTool",
    "SpeakInput",
    "TranscribeInput",
    "TranslateInput",
    "get_tools",
]


class TranslateInput(BaseModel):
    text: str = Field(description="The text to translate.")
    source_lang: str = Field(
        default="bci", description="Language of the text: 'bci' (Baoulé), 'fra' or 'eng'."
    )
    target_lang: str = Field(
        default="fra", description="Language to translate into: 'bci' (Baoulé), 'fra' or 'eng'."
    )


class TranscribeInput(BaseModel):
    audio_url: str | None = Field(
        default=None,
        description="Public http(s) URL of a recording (wav, mp3, ogg, webm, m4a, flac).",
    )
    audio_base64: str | None = Field(
        default=None, description="The recording itself, base64-encoded."
    )
    audio: dict[str, Any] | None = Field(
        default=None,
        description="A standard audio content block ({'type': 'audio', 'base64'|'url': ...}).",
    )
    runtime: Annotated[Any, InjectedToolArg()] = None
    """Injected by LangGraph's ToolNode (``ToolRuntime``); never shown to the model."""


class SpeakInput(BaseModel):
    text: str = Field(description="Text to speak aloud, written in the Ivorian language.")


class _IvoireTool(BaseTool):
    """Shared fields: language, backend (or remote API options)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    lang: str = "bci"
    backend: Any = Field(default=None, exclude=True)
    """An :class:`IvoireBackend`, ``"remote"``, ``"openrouter"``, ``"fake"`` or ``None`` (remote)."""
    base_url: str | None = None
    _backend: IvoireBackend | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        self.lang = resolve_language(self.lang).code

    def _get_backend(self) -> IvoireBackend:
        if self._backend is None:
            self._backend = make_backend(self.backend, None, self.base_url)
        return self._backend


class IvoireTranslateTool(_IvoireTool):
    """Translate between an Ivorian language (Baoulé) and French or English."""

    name: str = "ivoire_translate"
    description: str = (
        "Translate text between Baoulé ('bci'), French ('fra') and English ('eng'). "
        "Input: the text, its language and the target language. Returns the translation."
    )
    args_schema: ArgsSchema | None = TranslateInput
    style: str = "everyday"

    def _translator(self) -> IvoireTranslator:
        return IvoireTranslator(self.lang, style=self.style, backend=self._get_backend())

    def _run(
        self,
        text: str,
        source_lang: str = "bci",
        target_lang: str = "fra",
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        src, tgt = resolve_text_language(source_lang), resolve_text_language(target_lang)
        return self._translator().invoke({"text": text, "source": src, "target": tgt}).text

    async def _arun(
        self,
        text: str,
        source_lang: str = "bci",
        target_lang: str = "fra",
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        src, tgt = resolve_text_language(source_lang), resolve_text_language(target_lang)
        r = await self._translator().ainvoke({"text": text, "source": src, "target": tgt})
        return r.text


class IvoireTranscribeTool(_IvoireTool):
    """Transcribe a recording of Ivorian-language speech.

    The audio comes from, in order: ``audio_base64``, an ``audio`` content block, an ``audio_url``
    (http/https only; private and local addresses are refused unless ``allow_private_urls``), or,
    when the tool runs inside a LangGraph agent, the latest audio block in the conversation state.
    Local file paths are deliberately not accepted, so a model cannot make the tool read files.
    """

    name: str = "ivoire_transcribe"
    description: str = (
        "Transcribe a Baoulé voice recording into Baoulé text. Give audio_url (a public http/https "
        "link) or audio_base64. With no arguments, the latest recording the user sent in this "
        "conversation is used."
    )
    args_schema: ArgsSchema | None = TranscribeInput
    allow_private_urls: bool = False
    """Allow URLs on loopback/private networks (for trusted, local deployments only)."""
    max_audio_bytes: int = MAX_AUDIO_BYTES

    def _local_audio(
        self, audio_base64: str | None, audio: dict[str, Any] | None, runtime: Any
    ) -> tuple[bytes | None, str | None]:
        """Audio that needs no download, or a URL still to fetch."""
        if audio_base64:
            return decode_base64_audio(audio_base64, max_bytes=self.max_audio_bytes), None
        block = audio
        if block is None and runtime is not None:
            state = getattr(runtime, "state", None)
            messages = (
                state.get("messages")
                if isinstance(state, dict)
                else getattr(state, "messages", None)
            )
            block = latest_audio_block(messages or [])
        if block is None:
            raise ValueError(
                "no audio: pass audio_url, audio_base64 or an audio block "
                "(or send a voice message in the conversation)"
            )
        if not is_audio_block(block):
            raise ValueError("'audio' must be an audio content block")
        url = block.get("url")
        if url and not str(url).startswith("data:") and not block.get("base64"):
            return None, str(url)
        return from_audio_block(block, max_bytes=self.max_audio_bytes), None

    def _run(
        self,
        audio_url: str | None = None,
        audio_base64: str | None = None,
        audio: dict[str, Any] | None = None,
        runtime: Any = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        data, url = (
            (None, audio_url) if audio_url else self._local_audio(audio_base64, audio, runtime)
        )
        if data is None:
            assert url is not None
            data = fetch_audio_url(
                url, allow_private=self.allow_private_urls, max_bytes=self.max_audio_bytes
            )
        return IvoireTranscriber(self.lang, backend=self._get_backend()).invoke(data).text

    async def _arun(
        self,
        audio_url: str | None = None,
        audio_base64: str | None = None,
        audio: dict[str, Any] | None = None,
        runtime: Any = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        data, url = (
            (None, audio_url) if audio_url else self._local_audio(audio_base64, audio, runtime)
        )
        if data is None:
            assert url is not None
            data = await afetch_audio_url(
                url, allow_private=self.allow_private_urls, max_bytes=self.max_audio_bytes
            )
        t = await IvoireTranscriber(self.lang, backend=self._get_backend()).ainvoke(data)
        return t.text


class IvoireSpeakTool(_IvoireTool):
    """Speak a text aloud; the audio is the tool message's artifact.

    The artifact is a standard LangChain audio content block
    (``{"type": "audio", "base64": ..., "mime_type": "audio/wav", "extras": {...}}``), so it is
    JSON-serialisable and survives a checkpointer. Read the bytes with
    :func:`langchain_ivoire.audio.from_audio_block`.
    """

    name: str = "ivoire_speak"
    description: str = (
        "Speak a Baoulé text aloud (text-to-speech). Use it when the user should hear the answer. "
        "Input must already be in Baoulé."
    )
    args_schema: ArgsSchema | None = SpeakInput
    response_format: str = "content_and_artifact"  # type: ignore[assignment]

    def _run(
        self, text: str, run_manager: CallbackManagerForToolRun | None = None
    ) -> tuple[str, AudioContentBlock]:
        s = IvoireSpeaker(self.lang, backend=self._get_backend()).invoke(text)
        return f"Spoke {s.audio_s or 0:.1f} s of audio.", to_audio_block(s)

    async def _arun(
        self, text: str, run_manager: AsyncCallbackManagerForToolRun | None = None
    ) -> tuple[str, AudioContentBlock]:
        s = await IvoireSpeaker(self.lang, backend=self._get_backend()).ainvoke(text)
        return f"Spoke {s.audio_s or 0:.1f} s of audio.", to_audio_block(s)


def get_tools(
    lang: str = "bci", backend: Any = None, *, include_speech: bool = True
) -> list[BaseTool]:
    """All tools for ``lang`` sharing one backend."""
    be = make_backend(backend, None, None)
    tools: list[BaseTool] = [IvoireTranslateTool(lang=lang, backend=be)]
    if include_speech:
        tools += [
            IvoireTranscribeTool(lang=lang, backend=be),
            IvoireSpeakTool(lang=lang, backend=be),
        ]
    return tools
