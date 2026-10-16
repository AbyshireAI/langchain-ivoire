"""LangChain integration for Ivorian languages: speech recognition, translation, speech synthesis,
a language wrapper for any chat model and a prebuilt voice agent. Baoulé (``"bci"``) is implemented;
Agni, Dioula and Bété are planned.
"""

from langchain_ivoire._languages import (
    LANGUAGES,
    SUPPORTED_LANGUAGES,
    Language,
    resolve_language,
)
from langchain_ivoire._version import __version__
from langchain_ivoire.backends import (
    FakeBackend,
    IvoireBackend,
    OpenRouterBackend,
    RemoteBackend,
    get_backend,
)
from langchain_ivoire.chat import IvoireLanguageChat, with_language
from langchain_ivoire.document_loaders import IvoireAudioLoader
from langchain_ivoire.embeddings import IvoireEmbeddings
from langchain_ivoire.exceptions import (
    IvoireAPIError,
    IvoireApprovalRequiredError,
    IvoireAuthenticationError,
    IvoireConfigError,
    IvoireConnectionError,
    IvoireDependencyError,
    IvoireError,
    IvoireIdempotencyConflictError,
    IvoireInvalidRequestError,
    IvoireNotFoundError,
    IvoirePermissionError,
    IvoireRateLimitError,
    IvoireServerError,
    IvoireTooLargeError,
    IvoireUnsupportedError,
    IvoireUnsupportedLanguageError,
    IvoireUpstreamUnavailableError,
)
from langchain_ivoire.graph import VoiceAgentState, create_voice_agent
from langchain_ivoire.middleware import IvoireLanguageMiddleware
from langchain_ivoire.phrasebook import Phrase, Phrasebook, PhraseMatch, sound_skeleton
from langchain_ivoire.speaker import IvoireSpeaker
from langchain_ivoire.tools import (
    IvoireSpeakTool,
    IvoireTranscribeTool,
    IvoireTranslateTool,
    get_tools,
)
from langchain_ivoire.transcriber import IvoireTranscriber
from langchain_ivoire.translator import IvoireTranslator
from langchain_ivoire.types import Speech, Transcription, Translation

__all__ = [
    "LANGUAGES",
    "SUPPORTED_LANGUAGES",
    "FakeBackend",
    "IvoireAPIError",
    "IvoireApprovalRequiredError",
    "IvoireAudioLoader",
    "IvoireAuthenticationError",
    "IvoireBackend",
    "IvoireConfigError",
    "IvoireConnectionError",
    "IvoireDependencyError",
    "IvoireEmbeddings",
    "IvoireError",
    "IvoireIdempotencyConflictError",
    "IvoireInvalidRequestError",
    "IvoireLanguageChat",
    "IvoireLanguageMiddleware",
    "IvoireNotFoundError",
    "IvoirePermissionError",
    "IvoireRateLimitError",
    "IvoireServerError",
    "IvoireSpeakTool",
    "IvoireSpeaker",
    "IvoireTooLargeError",
    "IvoireTranscribeTool",
    "IvoireTranscriber",
    "IvoireTranslateTool",
    "IvoireTranslator",
    "IvoireUnsupportedError",
    "IvoireUnsupportedLanguageError",
    "IvoireUpstreamUnavailableError",
    "Language",
    "OpenRouterBackend",
    "Phrase",
    "PhraseMatch",
    "Phrasebook",
    "RemoteBackend",
    "Speech",
    "Transcription",
    "Translation",
    "VoiceAgentState",
    "__version__",
    "create_voice_agent",
    "get_backend",
    "get_tools",
    "resolve_language",
    "sound_skeleton",
    "with_language",
]
