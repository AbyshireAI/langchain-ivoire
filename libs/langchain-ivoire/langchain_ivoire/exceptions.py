"""Typed exceptions raised by langchain-ivoire.

API errors carry the server's RFC 9457 problem details when it sends them (``code``, ``title``,
``detail``, ``type``) and the ``X-Request-Id`` of the failed request (``request_id``), so a failure
can be matched with the server logs. They never contain the API key.

=================================  ==========================  =========
exception                          problem ``code``            status
=================================  ==========================  =========
:class:`IvoireInvalidRequestError` ``invalid_request``,         400, 411, 422
                                   ``length_required``
:class:`IvoireTooLargeError`       ``too_large``               413
:class:`IvoireUnsupportedLanguageError` ``unsupported_language``  400
:class:`IvoireAuthenticationError` ``unauthorized``            401
:class:`IvoirePermissionError`     ``forbidden``               403
:class:`IvoireNotFoundError`       ``not_found``,              404, 405
                                   ``model_not_found``,
                                   ``method_not_allowed``
:class:`IvoireIdempotencyConflictError` ``idempotency_conflict``  409
:class:`IvoireRateLimitError`      ``rate_limited``            429
:class:`IvoireUpstreamUnavailableError` ``upstream_unavailable``  503
:class:`IvoireServerError`         ``internal``                5xx
=================================  ==========================  =========
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "IvoireAPIError",
    "IvoireApprovalRequiredError",
    "IvoireAuthenticationError",
    "IvoireConfigError",
    "IvoireConnectionError",
    "IvoireDependencyError",
    "IvoireError",
    "IvoireIdempotencyConflictError",
    "IvoireInvalidRequestError",
    "IvoireNotFoundError",
    "IvoirePermissionError",
    "IvoireRateLimitError",
    "IvoireServerError",
    "IvoireTooLargeError",
    "IvoireUnsupportedError",
    "IvoireUnsupportedLanguageError",
    "IvoireUpstreamUnavailableError",
]


class IvoireError(Exception):
    """Base class of every error raised by this package."""


class IvoireConfigError(IvoireError, ValueError):
    """Missing or invalid configuration (model path, base URL, ...)."""


class IvoireDependencyError(IvoireError, ImportError):
    """An optional dependency is missing (e.g. the ``[mcp]`` extra is not installed)."""


class IvoireUnsupportedError(IvoireError, NotImplementedError):
    """The selected backend cannot do this (e.g. a translation direction the server does not expose)."""


class IvoireApprovalRequiredError(IvoireUnsupportedError):
    """The service is available by approval only, and this route was not approved for you.

    Raised before any request is sent, e.g. for speech recognition, the voice or embeddings through
    OpenRouter without a model id your organisation was granted. Request access from Abyshire.
    """


class IvoireConnectionError(IvoireError):
    """The API could not be reached (DNS, refused connection, timeout)."""


class IvoireAPIError(IvoireError):
    """The API answered with an error status.

    Attributes:
        status_code: HTTP status.
        code: the problem ``code`` (``"invalid_request"``, ``"rate_limited"``, ...) or ``None``.
        title: the other RFC 9457 fields, when the server sent them.
        detail: See ``title``.
        type: See ``title``.
        request_id: the response's ``X-Request-Id``, to quote when reporting a problem.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        *,
        code: str | None = None,
        problem: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.problem: dict[str, Any] = dict(problem or {})
        self.title: str | None = self.problem.get("title")
        self.detail: str | None = self.problem.get("detail")
        self.type: str | None = self.problem.get("type")
        self.request_id = request_id


class IvoireInvalidRequestError(IvoireAPIError, ValueError):
    """400/422 ``invalid_request``: the server rejected the request as malformed."""


class IvoireTooLargeError(IvoireInvalidRequestError):
    """413 ``too_large``: too many texts, a text too long, or audio too long."""


class IvoireUnsupportedLanguageError(IvoireAPIError, IvoireUnsupportedError):
    """``unsupported_language``: the server does not serve this language or direction."""


class IvoireIdempotencyConflictError(IvoireAPIError):
    """409 ``idempotency_conflict``: an ``Idempotency-Key`` was reused with a different body.

    The client generates a fresh key per call (reused only by that call's retries), so this means a
    caller-supplied key or a server bug, not a transient error; it is not retried.
    """


class IvoireAuthenticationError(IvoireAPIError):
    """401 ``unauthorized``: missing or invalid API key."""


class IvoirePermissionError(IvoireAPIError):
    """403: the key is valid but not allowed to do this."""


class IvoireNotFoundError(IvoireAPIError):
    """404: unknown endpoint or expired resource."""


class IvoireRateLimitError(IvoireAPIError):
    """429 ``rate_limited``: too many requests (still failing after the configured retries).

    ``retry_after`` holds the server's ``Retry-After`` in seconds, when given.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = 429,
        retry_after: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, status_code, **kwargs)
        self.retry_after = retry_after


class IvoireServerError(IvoireAPIError):
    """5xx ``internal``: the server failed (still failing after retries for 503)."""


class IvoireUpstreamUnavailableError(IvoireServerError):
    """503 ``upstream_unavailable``: a model behind the API is loading or down; retry later."""
