"""Document loader: audio files -> transcripts as LangChain Documents."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from langchain_ivoire.backends import BackendLike
from langchain_ivoire.transcriber import IvoireTranscriber
from langchain_ivoire.types import Transcription

__all__ = ["AUDIO_EXTENSIONS", "IvoireAudioLoader"]

AUDIO_EXTENSIONS = (".wav", ".mp3", ".ogg", ".oga", ".opus", ".webm", ".m4a", ".flac")


class IvoireAudioLoader(BaseLoader):
    """Transcribe audio files into :class:`~langchain_core.documents.Document` objects.

    ``page_content`` is the transcript; metadata holds ``source``, ``duration`` (seconds),
    ``confidence``, ``low_confidence``, ``lang`` and ``segments``.

    Example:
        .. code-block:: python

            from langchain_ivoire import IvoireAudioLoader

            docs = IvoireAudioLoader("recordings/", lang="bci").load()

    Args:
        file_path: a file, a directory (searched with ``glob``), or a list of files.
        lang: the Ivorian language spoken.
        glob: pattern for directories (default: every file with a known audio extension, recursive).
        transcriber: a ready :class:`IvoireTranscriber` (overrides the backend options).
        skip_empty: skip files where no speech was recognised.
    """

    def __init__(
        self,
        file_path: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        lang: str = "bci",
        *,
        glob: str | None = None,
        backend: BackendLike = None,
        api_key: str | None = None,
        base_url: str | None = None,
        transcriber: IvoireTranscriber | None = None,
        skip_empty: bool = False,
    ) -> None:
        self.file_path = file_path
        self.glob = glob
        self.skip_empty = skip_empty
        self.transcriber = transcriber or IvoireTranscriber(
            lang, backend=backend, api_key=api_key, base_url=base_url
        )

    def _files(self) -> list[Path]:
        roots = (
            [self.file_path]
            if isinstance(self.file_path, (str, os.PathLike))
            else list(self.file_path)
        )
        files: list[Path] = []
        for r in roots:
            p = Path(r)
            if p.is_dir():
                cands = p.glob(self.glob) if self.glob else p.rglob("*")
                files += sorted(
                    f
                    for f in cands
                    if f.is_file() and (self.glob or f.suffix.lower() in AUDIO_EXTENSIONS)
                )
            elif p.is_file():
                files.append(p)
            else:
                raise FileNotFoundError(f"no such file or directory: {p}")
        return files

    def _doc(self, path: Path, t: Transcription) -> Document | None:
        if self.skip_empty and not t.text:
            return None
        return Document(
            page_content=t.text,
            metadata={
                "source": str(path),
                "duration": t.audio_s,
                "confidence": t.confidence,
                "low_confidence": t.low_confidence,
                "lang": t.lang,
                "segments": list(t.segments),
            },
        )

    def lazy_load(self) -> Iterator[Document]:
        for f in self._files():
            d = self._doc(f, self.transcriber.invoke(str(f)))
            if d is not None:
                yield d

    async def alazy_load(self) -> AsyncIterator[Document]:
        for f in self._files():
            d = self._doc(f, await self.transcriber.ainvoke(str(f)))
            if d is not None:
                yield d
