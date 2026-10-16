"""Backends: ``"remote"`` (default, Abyshire API or a LAN server), ``"openrouter"`` (the same models
over the OpenAI-compatible API, through OpenRouter or the hosted gateway), ``"local"`` (not publicly
available) and ``"fake"`` (tests)."""

from __future__ import annotations

from typing import Any, Literal, Union

from langchain_ivoire.backends.base import STYLES, IvoireBackend, pivot_path
from langchain_ivoire.backends.fake import FakeBackend
from langchain_ivoire.backends.openrouter import OPENROUTER_BASE_URL, OpenRouterBackend
from langchain_ivoire.backends.remote import DEFAULT_BASE_URL, RemoteBackend
from langchain_ivoire.exceptions import IvoireUnsupportedError

__all__ = [
    "DEFAULT_BASE_URL",
    "OPENROUTER_BASE_URL",
    "STYLES",
    "BackendLike",
    "FakeBackend",
    "IvoireBackend",
    "OpenRouterBackend",
    "RemoteBackend",
    "get_backend",
    "pivot_path",
]

BackendLike = Union[IvoireBackend, Literal["remote", "openrouter", "local", "fake"], None]


def get_backend(backend: BackendLike = None, **kwargs: Any) -> IvoireBackend:
    """Resolve a backend instance.

    ``None`` or ``"remote"`` -> :class:`RemoteBackend` (``api_key``/``base_url`` kwargs or the
    ``IVOIRE_API_KEY``/``IVOIRE_BASE_URL`` env vars); ``"openrouter"`` -> :class:`OpenRouterBackend`
    (``OPENROUTER_API_KEY``); ``"local"`` -> not publicly available
    (raises :class:`~langchain_ivoire.IvoireUnsupportedError`);
    ``"fake"`` -> :class:`FakeBackend`. An instance is returned as is (kwargs must then be empty).
    """
    if isinstance(backend, IvoireBackend):
        if any(v is not None for v in kwargs.values()):
            raise ValueError("pass either a backend instance or backend options, not both")
        return backend
    opts = {k: v for k, v in kwargs.items() if v is not None}
    if backend in (None, "remote"):
        return RemoteBackend(**opts)
    if backend == "openrouter":
        return OpenRouterBackend(**opts)
    if backend == "local":
        from langchain_ivoire.backends.local import OFFLINE_UNAVAILABLE

        raise IvoireUnsupportedError(OFFLINE_UNAVAILABLE)
    if backend == "fake":
        return FakeBackend(**opts)
    raise ValueError(
        f"unknown backend {backend!r}; use 'remote', 'openrouter', 'local', 'fake' or an instance"
    )
