"""Baoulé and French search embeddings as a LangChain :class:`~langchain_core.embeddings.Embeddings`."""

from __future__ import annotations

import os

from langchain_core.embeddings import Embeddings

from langchain_ivoire._base import make_backend
from langchain_ivoire.backends import BackendLike, IvoireBackend, OpenRouterBackend
from langchain_ivoire.backends.remote import DEFAULT_BASE_URL

__all__ = ["IvoireEmbeddings"]


class IvoireEmbeddings(Embeddings):
    """Baoulé and French texts in one vector space, so a Baoulé question finds French documents.

    Served by the hosted API's OpenAI-compatible ``/embeddings`` endpoint (``abyshire/embed-baoule``)
    and available **by approval**: request access from Abyshire (api@abyshire.co.uk). The default
    backend calls the Abyshire API with ``IVOIRE_API_KEY`` (and ``IVOIRE_BASE_URL``, if set).

    Example:
        .. code-block:: python

            from langchain_core.vectorstores import InMemoryVectorStore
            from langchain_ivoire import IvoireEmbeddings

            store = InMemoryVectorStore(IvoireEmbeddings())   # approved IVOIRE_API_KEY in the env
            store.add_texts(["Le repas du soir est servi à 19 heures."])
            store.similarity_search("Awe kuun min.")

    Args:
        backend: ``None`` (default: the Abyshire API) or any
            :class:`~langchain_ivoire.IvoireBackend` that implements ``embed``, e.g. an
            :class:`~langchain_ivoire.OpenRouterBackend` with an ``embed_model`` your organisation
            was granted on OpenRouter.
        api_key: See :class:`~langchain_ivoire.OpenRouterBackend`.
        base_url: See :class:`~langchain_ivoire.OpenRouterBackend`.
    """

    def __init__(
        self,
        *,
        backend: BackendLike = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if backend is None:
            url = base_url or os.environ.get("IVOIRE_BASE_URL") or DEFAULT_BASE_URL
            self.backend: IvoireBackend = OpenRouterBackend(api_key=api_key, base_url=url)
        else:
            self.backend = make_backend(backend, api_key, base_url)

    def __repr__(self) -> str:
        return f"IvoireEmbeddings(backend={self.backend!r})"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.backend.embed(texts, kind="document")

    def embed_query(self, text: str) -> list[float]:
        return self.backend.embed([text], kind="query")[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.backend.aembed(texts, kind="document")

    async def aembed_query(self, text: str) -> list[float]:
        return (await self.backend.aembed([text], kind="query"))[0]
