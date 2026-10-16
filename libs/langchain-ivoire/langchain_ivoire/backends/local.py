"""The offline option is not publicly available.

``langchain-ivoire`` is a client for the hosted Abyshire API: use ``backend="remote"`` (the
default) or ``backend="openrouter"``. ``backend="fake"`` runs every component without a network
for tests and examples.
"""

from __future__ import annotations

from typing import Any

from langchain_ivoire.exceptions import IvoireUnsupportedError

__all__ = ["OFFLINE_UNAVAILABLE", "LocalBackend"]

OFFLINE_UNAVAILABLE = (
    "The offline option is not publicly available. Use the hosted Abyshire API "
    "(backend='remote', the default) or backend='openrouter'."
)


class LocalBackend:
    """Placeholder: the offline option is not publicly available."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise IvoireUnsupportedError(OFFLINE_UNAVAILABLE)
