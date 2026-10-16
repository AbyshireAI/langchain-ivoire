"""Result types returned by the components and backends."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Speech", "Transcription", "Translation"]


@dataclass(frozen=True)
class Transcription:
    """Speech recognised as text in an Ivorian language."""

    text: str
    lang: str
    confidence: float | None = None
    """Mean per-token posterior of the recogniser (0..1); ``None`` if the backend has none."""
    segments: tuple[str, ...] = ()
    """Text per pause-delimited chunk (long audio is split at pauses)."""
    audio_s: float | None = None
    low_confidence: bool = False
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __str__(self) -> str:
        return self.text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Translation:
    """A translated text; ``via`` lists every hop (English always pivots through French)."""

    text: str
    source: str
    target: str
    source_text: str = ""
    style: str = "everyday"
    via: tuple[tuple[str, str], ...] = ()
    """(language, text) after each hop, e.g. ``(("fra", "Venez manger."), ("eng", "Come and eat."))``."""
    timings_ms: dict[str, int] = field(default_factory=dict, compare=False)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)
    """Backend details, e.g. the remote API's ``request_id``, ``model`` and ``usage``."""

    def __str__(self) -> str:
        return self.text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Speech:
    """Synthesised speech: WAV bytes plus what was said."""

    audio: bytes = field(repr=False)
    text: str
    lang: str
    sample_rate: int | None = None
    audio_s: float | None = None
    media_type: str = "audio/wav"
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __str__(self) -> str:
        return self.text

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write the audio to ``path`` and return it."""
        p = Path(path)
        p.write_bytes(self.audio)
        return p

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["audio"] = f"<{len(self.audio)} bytes>"
        return d
