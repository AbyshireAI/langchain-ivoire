"""Shared plumbing for the components."""

from __future__ import annotations

from typing import Any

from langchain_ivoire._languages import resolve_language
from langchain_ivoire.backends import BackendLike, IvoireBackend, get_backend


def make_backend(
    backend: BackendLike, api_key: str | None, base_url: str | None, **extra: Any
) -> IvoireBackend:
    if isinstance(backend, IvoireBackend) or backend in ("local", "fake"):
        if api_key is not None or base_url is not None:
            raise ValueError("api_key/base_url only apply to the remote and openrouter backends")
        return get_backend(backend, **extra)
    return get_backend(backend, api_key=api_key, base_url=base_url, **extra)


class BackendComponent:
    """Holds the language and the backend; subclasses are LangChain Runnables."""

    lang: str
    backend: IvoireBackend

    def _setup(
        self, lang: str, backend: BackendLike, api_key: str | None, base_url: str | None
    ) -> None:
        self.lang = resolve_language(lang).code
        self.backend = make_backend(backend, api_key, base_url)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(lang={self.lang!r}, backend={self.backend!r})"
