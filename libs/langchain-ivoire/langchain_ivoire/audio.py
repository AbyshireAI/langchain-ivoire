"""Audio as standard LangChain content blocks.

LangChain represents audio in messages as an ``AudioContentBlock``
(``{"type": "audio", "base64": ..., "mime_type": ...}`` or ``{"type": "audio", "url": ...}``),
built with :func:`langchain_core.messages.content.create_audio_block`. Everything in this package
that stores or returns audio (the speak tool's artifact, the voice graph state, the MCP server) uses
these blocks, because they are plain JSON and a LangGraph checkpointer can serialise them.

Example:
    .. code-block:: python

        from langchain_ivoire.audio import from_audio_block, to_audio_block

        block = to_audio_block(open("question.wav", "rb").read())
        HumanMessage(content=[block])            # send it to an agent
        wav = from_audio_block(block)            # bytes again
"""

from __future__ import annotations

import base64
import binascii
import ipaddress
import socket
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
from langchain_core.messages import BaseMessage
from langchain_core.messages.content import AudioContentBlock, create_audio_block

from langchain_ivoire._audio import sniff_media_type, wav_info
from langchain_ivoire.types import Speech

__all__ = [
    "MAX_AUDIO_BYTES",
    "afetch_audio_url",
    "afrom_audio_block",
    "check_audio_url",
    "decode_base64_audio",
    "fetch_audio_url",
    "from_audio_block",
    "is_audio_block",
    "latest_audio_block",
    "to_audio_block",
]

MAX_AUDIO_BYTES = 25 * 1024 * 1024
"""Largest audio accepted from base64 or a URL (25 MiB)."""

_MAX_REDIRECTS = 3


def to_audio_block(
    audio: bytes | bytearray | memoryview | Speech | Mapping[str, Any],
    *,
    mime_type: str | None = None,
    **extras: Any,
) -> AudioContentBlock:
    """Wrap audio as a standard ``AudioContentBlock`` (base64 inline).

    Args:
        audio: encoded audio bytes, a :class:`~langchain_ivoire.Speech`, or an existing block
            (returned unchanged).
        mime_type: overrides the type sniffed from the bytes.
        **extras: stored under the block's ``extras`` (``None`` values are dropped). For a
            ``Speech`` the text, language, sample rate and duration are added automatically.
    """
    if isinstance(audio, Mapping):
        if not is_audio_block(audio):
            raise ValueError("not an audio content block")
        return dict(audio)  # type: ignore[return-value]
    if isinstance(audio, Speech):
        extras = {
            "text": audio.text,
            "lang": audio.lang,
            "sample_rate": audio.sample_rate,
            "audio_s": audio.audio_s,
            **extras,
        }
        mime_type = mime_type or audio.media_type
        data = audio.audio
    else:
        data = bytes(audio)
    if not data:
        raise ValueError("empty audio")
    if mime_type is None:
        mime_type = sniff_media_type(data)[1]
        if mime_type == "audio/wav" and "sample_rate" not in extras:
            info = wav_info(data)
            if info is not None:
                extras = {"sample_rate": info[0], "audio_s": info[1], **extras}
    return create_audio_block(
        base64=base64.b64encode(data).decode("ascii"), mime_type=mime_type, **extras
    )


def is_audio_block(obj: Any) -> bool:
    """``True`` for a standard audio block, a legacy (v0) one, or an OpenAI ``input_audio`` block."""
    return isinstance(obj, Mapping) and obj.get("type") in ("audio", "input_audio")


def decode_base64_audio(data: str, *, max_bytes: int = MAX_AUDIO_BYTES) -> bytes:
    """Decode base64 (or a ``data:audio/...;base64,`` URI) with a size check."""
    s = data.strip()
    if s.startswith("data:"):
        head, _, s = s.partition(",")
        if ";base64" not in head:
            raise ValueError("only base64 data: URIs are supported")
    if len(s) > (max_bytes * 4) // 3 + 8:
        raise ValueError(f"audio larger than {max_bytes} bytes")
    try:
        out = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("audio is not valid base64") from None
    if not out:
        raise ValueError("empty audio")
    return out


def _block_source(block: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """``(base64, url)`` of a block in any of the supported shapes."""
    if block.get("type") == "input_audio":  # OpenAI chat-completions shape
        inner = block.get("input_audio") or {}
        return inner.get("data"), None
    if block.get("base64"):
        return str(block["base64"]), None
    if block.get("url"):
        return None, str(block["url"])
    st = block.get("source_type")  # langchain-core v0 shape
    if st == "base64" and block.get("data"):
        return str(block["data"]), None
    if st == "url" and block.get("url"):
        return None, str(block["url"])
    return None, None


def from_audio_block(
    block: Mapping[str, Any],
    *,
    fetch_url: bool = False,
    allow_private_urls: bool = False,
    max_bytes: int = MAX_AUDIO_BYTES,
    timeout: float = 30.0,
) -> bytes:
    """Return the audio bytes of a content block.

    Base64 blocks are decoded. URL blocks are downloaded only when ``fetch_url=True`` (http/https,
    public addresses unless ``allow_private_urls``; see :func:`fetch_audio_url`). Blocks that only
    hold a provider ``file_id`` cannot be read.
    """
    if not is_audio_block(block):
        raise ValueError("not an audio content block")
    b64, url = _block_source(block)
    if b64:
        return decode_base64_audio(b64, max_bytes=max_bytes)
    if url:
        if url.startswith("data:"):
            return decode_base64_audio(url, max_bytes=max_bytes)
        if not fetch_url:
            raise ValueError("audio block holds a URL; pass fetch_url=True to download it")
        return fetch_audio_url(
            url, allow_private=allow_private_urls, max_bytes=max_bytes, timeout=timeout
        )
    raise ValueError("audio block has no base64 data or URL (file_id blocks are not supported)")


async def afrom_audio_block(
    block: Mapping[str, Any],
    *,
    fetch_url: bool = False,
    allow_private_urls: bool = False,
    max_bytes: int = MAX_AUDIO_BYTES,
    timeout: float = 30.0,
) -> bytes:
    """Async :func:`from_audio_block`."""
    if not is_audio_block(block):
        raise ValueError("not an audio content block")
    b64, url = _block_source(block)
    if url and not url.startswith("data:") and not b64:
        if not fetch_url:
            raise ValueError("audio block holds a URL; pass fetch_url=True to download it")
        return await afetch_audio_url(
            url, allow_private=allow_private_urls, max_bytes=max_bytes, timeout=timeout
        )
    return from_audio_block(block, max_bytes=max_bytes)


def latest_audio_block(messages: Iterable[Any]) -> dict[str, Any] | None:
    """The most recent audio block in a list of messages (newest message first, last block first)."""
    for m in reversed(list(messages)):
        content = m.content if isinstance(m, BaseMessage) else (m or {}).get("content")
        if isinstance(content, list):
            for b in reversed(content):
                if is_audio_block(b):
                    return dict(b)
    return None


# ---- URL download with SSRF guards -----------------------------------------------------------
def _blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def check_audio_url(url: str, *, allow_private: bool = False) -> str:
    """Validate a URL before downloading audio from it.

    Only ``http``/``https`` are allowed. Unless ``allow_private`` is set, hosts that are or resolve
    to loopback, private, link-local or reserved addresses are refused, so an LLM cannot use the
    transcribe tool to reach services on your machine or LAN. (DNS can still change between this
    check and the download; run untrusted agents behind an egress proxy if that matters to you.)

    Returns the URL unchanged; raises ``ValueError`` otherwise.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ValueError("audio URLs must use http or https")
    host = parts.hostname
    if not host:
        raise ValueError("audio URL has no host")
    if parts.username or parts.password:
        raise ValueError("audio URLs must not contain credentials")
    if allow_private:
        return url
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError(f"refusing to fetch audio from a local host ({host})")
    try:
        ips = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, parts.port or 443, proto=socket.IPPROTO_TCP)
        except OSError:
            raise ValueError(f"cannot resolve audio URL host {host!r}") from None
        ips = [ipaddress.ip_address(str(i[4][0]).split("%")[0]) for i in infos]
    if any(_blocked_ip(ip) for ip in ips):
        raise ValueError(f"refusing to fetch audio from a private or local address ({host})")
    return url


def _read_limited(chunks: Iterable[bytes], max_bytes: int) -> bytes:
    buf = bytearray()
    for c in chunks:
        buf += c
        if len(buf) > max_bytes:
            raise ValueError(f"audio larger than {max_bytes} bytes")
    if not buf:
        raise ValueError("empty audio")
    return bytes(buf)


def fetch_audio_url(
    url: str,
    *,
    allow_private: bool = False,
    max_bytes: int = MAX_AUDIO_BYTES,
    timeout: float = 30.0,
    client: httpx.Client | None = None,
) -> bytes:
    """Download audio from an http(s) URL with the checks of :func:`check_audio_url`.

    No credentials are sent (this never uses the Ivoire API client), redirects are re-checked
    (at most 3) and the body is capped at ``max_bytes``.
    """
    own = client is None
    c = client or httpx.Client(timeout=timeout, follow_redirects=False)
    try:
        for _ in range(_MAX_REDIRECTS + 1):
            check_audio_url(url, allow_private=allow_private)
            with c.stream("GET", url) as r:
                if r.is_redirect and "location" in r.headers:
                    url = urljoin(url, r.headers["location"])
                    continue
                if r.status_code >= 400:
                    raise ValueError(f"audio URL answered {r.status_code}")
                return _read_limited(r.iter_bytes(), max_bytes)
        raise ValueError("too many redirects")
    finally:
        if own:
            c.close()


async def afetch_audio_url(
    url: str,
    *,
    allow_private: bool = False,
    max_bytes: int = MAX_AUDIO_BYTES,
    timeout: float = 30.0,
    client: httpx.AsyncClient | None = None,
) -> bytes:
    """Async :func:`fetch_audio_url`."""
    own = client is None
    c = client or httpx.AsyncClient(timeout=timeout, follow_redirects=False)
    try:
        for _ in range(_MAX_REDIRECTS + 1):
            check_audio_url(url, allow_private=allow_private)
            async with c.stream("GET", url) as r:
                if r.is_redirect and "location" in r.headers:
                    url = urljoin(url, r.headers["location"])
                    continue
                if r.status_code >= 400:
                    raise ValueError(f"audio URL answered {r.status_code}")
                buf = bytearray()
                async for chunk in r.aiter_bytes():
                    buf += chunk
                    if len(buf) > max_bytes:
                        raise ValueError(f"audio larger than {max_bytes} bytes")
                if not buf:
                    raise ValueError("empty audio")
                return bytes(buf)
        raise ValueError("too many redirects")
    finally:
        if own:
            await c.aclose()
