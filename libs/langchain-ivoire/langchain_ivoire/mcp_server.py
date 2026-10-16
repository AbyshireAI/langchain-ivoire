"""MCP server exposing the Baoulé tools to any MCP client (desktop assistants, IDEs, other agents).

Install the extra and run it (stdio by default, or streamable HTTP)::

    pip install "langchain-ivoire[mcp]"
    ivoire-mcp                                            # stdio, for desktop MCP clients
    ivoire-mcp --transport streamable-http --port 8000    # http://127.0.0.1:8000/mcp

Tools:

* ``ivoire_transcribe``: Baoulé speech -> text. Audio as base64 or a public http(s) URL.
* ``ivoire_translate``: text between Baoulé (``bci``), French (``fra``) and English (``eng``).
* ``ivoire_speak``: Baoulé text -> speech, returned as MCP ``audio`` content (WAV).
* ``ivoire_languages``: what the backend supports.

Configuration comes from the environment: ``IVOIRE_API_KEY`` (hosted API key) and
``IVOIRE_BASE_URL`` (another server, e.g. a self-hosted one). Built on the official MCP Python SDK
(``mcp`` 2.x, ``mcp.server.mcpserver.MCPServer``).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

from langchain_ivoire._languages import resolve_language, resolve_text_language
from langchain_ivoire._version import __version__
from langchain_ivoire.audio import MAX_AUDIO_BYTES, afetch_audio_url, decode_base64_audio
from langchain_ivoire.backends import BackendLike, IvoireBackend, get_backend
from langchain_ivoire.backends.base import STYLES
from langchain_ivoire.exceptions import IvoireDependencyError, IvoireError

try:
    from mcp.server.mcpserver import Audio, MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise IvoireDependencyError(
        'The MCP server needs the "mcp" package: pip install "langchain-ivoire[mcp]"'
    ) from e

__all__ = ["create_mcp_server", "main"]

INSTRUCTIONS = (
    "Tools for Baoulé (bci), a language of Côte d'Ivoire. Translate between Baoulé, French and "
    "English, transcribe Baoulé speech and speak Baoulé text. Translation is machine translation "
    "of a low-resource language and makes mistakes; English always goes through French. Do not "
    "use it where an error could cause harm (medical, legal, safety) without a human interpreter."
)


def _fail(e: Exception) -> ToolError:
    """A tool error the client's model can read (messages never contain the API key)."""
    return ToolError(f"{type(e).__name__}: {e}")


def create_mcp_server(
    backend: BackendLike = None,
    *,
    lang: str = "bci",
    allow_private_urls: bool = False,
    max_audio_bytes: int = MAX_AUDIO_BYTES,
) -> MCPServer:
    """Build the MCP server.

    Args:
        backend: ``None``/``"remote"`` (``IVOIRE_API_KEY``/``IVOIRE_BASE_URL``), ``"openrouter"``
            (``OPENROUTER_API_KEY``), ``"fake"`` or an :class:`~langchain_ivoire.IvoireBackend`.
        lang: the Ivorian language served (``"bci"``).
        allow_private_urls: let ``ivoire_transcribe`` fetch audio from loopback/private addresses.
        max_audio_bytes: largest audio accepted.
    """
    code = resolve_language(lang).code
    be: IvoireBackend = get_backend(backend)
    server = MCPServer(
        name="ivoire",
        title="Baoulé (langchain-ivoire)",
        version=__version__,
        instructions=INSTRUCTIONS,
    )

    @server.tool(
        name="ivoire_transcribe",
        description=(
            "Transcribe a Baoulé voice recording into Baoulé text. Pass the audio as audio_base64 "
            "(wav, mp3, ogg, webm, m4a or flac) or as audio_url (a public http/https link). "
            "Returns the text, a confidence between 0 and 1, and low_confidence (ask the speaker "
            "to repeat when it is true)."
        ),
    )
    async def ivoire_transcribe(
        audio_base64: str | None = None, audio_url: str | None = None
    ) -> dict[str, Any]:
        try:
            if audio_base64:
                data = decode_base64_audio(audio_base64, max_bytes=max_audio_bytes)
            elif audio_url:
                data = await afetch_audio_url(
                    audio_url, allow_private=allow_private_urls, max_bytes=max_audio_bytes
                )
            else:
                raise ValueError("pass audio_base64 or audio_url")
            t = await be.atranscribe(data, lang=code)
        except (IvoireError, ValueError) as e:
            raise _fail(e) from None
        return {
            "text": t.text,
            "lang": t.lang,
            "confidence": t.confidence,
            "low_confidence": t.low_confidence,
            "audio_s": t.audio_s,
        }

    @server.tool(
        name="ivoire_translate",
        description=(
            "Translate text between Baoulé ('bci'), French ('fra') and English ('eng'). "
            "style: 'everyday' (default), 'tale' (story register) or 'formal'. Short, simple "
            "sentences translate best."
        ),
    )
    async def ivoire_translate(
        text: str, source: str = "bci", target: str = "fra", style: str = "everyday"
    ) -> str:
        try:
            src, tgt = resolve_text_language(source), resolve_text_language(target)
            if style not in STYLES:
                raise ValueError(f"unknown style {style!r}; use one of {', '.join(STYLES)}")
            if not text.strip():
                return ""
            out = await be.atranslate([text], source=src, target=tgt, style=style)
        except (IvoireError, ValueError, NotImplementedError) as e:
            raise _fail(e) from None
        return out[0].text

    @server.tool(
        name="ivoire_speak",
        description=(
            "Speak a Baoulé text aloud (text-to-speech). The text must already be in Baoulé; "
            "translate it first with ivoire_translate if needed. Returns WAV audio."
        ),
        structured_output=False,
    )
    async def ivoire_speak(text: str) -> list[Any]:
        if not text.strip():
            raise ToolError("nothing to say (empty text)")
        try:
            s = await be.asynthesize(text.strip(), lang=code)
        except IvoireError as e:
            raise _fail(e) from None
        return [f"Spoke {s.audio_s or 0:.1f} s of Baoulé audio.", Audio(data=s.audio, format="wav")]

    @server.tool(
        name="ivoire_languages",
        description="List the languages served, with their speech and translation support.",
    )
    async def ivoire_languages() -> list[dict[str, Any]]:
        try:
            return await be.alanguages()
        except IvoireError as e:
            raise _fail(e) from None

    return server


def main(argv: Sequence[str] | None = None) -> None:
    """Console entry point: ``ivoire-mcp``."""
    ap = argparse.ArgumentParser(
        prog="ivoire-mcp",
        description="MCP server for Baoulé speech and translation (langchain-ivoire). "
        "Configure with IVOIRE_API_KEY and IVOIRE_BASE_URL.",
    )
    ap.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    ap.add_argument("--host", default="127.0.0.1", help="HTTP bind address (streamable-http)")
    ap.add_argument("--port", type=int, default=8000, help="HTTP port (streamable-http)")
    ap.add_argument(
        "--backend",
        choices=["remote", "openrouter", "fake"],
        default="remote",
        help="remote API (default), the OpenAI-compatible API via OpenRouter "
        "(OPENROUTER_API_KEY), or an offline fake for trying it out",
    )
    ap.add_argument(
        "--allow-private-urls",
        action="store_true",
        help="let ivoire_transcribe fetch audio from loopback/LAN addresses",
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = ap.parse_args(argv)
    server = create_mcp_server(args.backend, allow_private_urls=args.allow_private_urls)
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
