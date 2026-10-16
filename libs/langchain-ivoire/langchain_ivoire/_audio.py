"""Small stdlib-only audio helpers (no numpy/torch needed)."""

from __future__ import annotations

import io
import os
import wave
from pathlib import Path
from typing import Any, Union

AudioInput = Union[bytes, bytearray, memoryview, str, "os.PathLike[str]", "dict[str, Any]"]
"""Raw audio bytes, a path to an audio file, or ``{"audio": <bytes|path>}``."""


def read_audio(audio: Any) -> tuple[bytes, str | None]:
    """Normalise an :data:`AudioInput` to ``(bytes, source)``; ``source`` is the path if one was given."""
    if isinstance(audio, dict):
        if "audio" not in audio:
            raise ValueError("audio dict input needs an 'audio' key (bytes or a file path)")
        return read_audio(audio["audio"])
    if isinstance(audio, (bytes, bytearray, memoryview)):
        data = bytes(audio)
        if not data:
            raise ValueError("empty audio")
        return data, None
    if isinstance(audio, (str, os.PathLike)):
        p = Path(audio)
        if not p.is_file():
            raise FileNotFoundError(f"audio file not found: {p}")
        return p.read_bytes(), str(p)
    raise TypeError(f"unsupported audio input type: {type(audio).__name__}")


def sniff_media_type(data: bytes) -> tuple[str, str]:
    """Guess ``(filename, media type)`` from magic bytes (used for multipart uploads)."""
    head = data[:12]
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "audio.wav", "audio/wav"
    if head[:4] == b"OggS":
        return "audio.ogg", "audio/ogg"
    if head[:4] == b"\x1aE\xdf\xa3":
        return "audio.webm", "audio/webm"
    if head[:4] == b"fLaC":
        return "audio.flac", "audio/flac"
    if head[4:8] == b"ftyp":
        return "audio.m4a", "audio/mp4"
    if head[:3] == b"ID3" or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio.mp3", "audio/mpeg"
    return "audio.bin", "application/octet-stream"


def wav_info(data: bytes) -> tuple[int, float] | None:
    """``(sample_rate, seconds)`` for WAV bytes, ``None`` if not a readable PCM WAV."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            sr = w.getframerate()
            return sr, round(w.getnframes() / float(sr), 2)
    except (wave.Error, EOFError, ZeroDivisionError):
        return None


def silent_wav(seconds: float, sample_rate: int = 16000) -> bytes:
    """A silent mono 16-bit WAV (used by the fake backend)."""
    n = max(1, int(seconds * sample_rate))
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * n)
    return b.getvalue()
