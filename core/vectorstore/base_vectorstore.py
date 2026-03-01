from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class VectorDocument(BaseModel):
    """A single embedded chunk ready for indexing."""

    id: str                         # Chunk.id  (deterministic md5)
    embedding: list[float]          # dense vector from embedder
    content: str                    # Chunk.content — stored in metadata so search returns it
    metadata: dict[str, Any]        # Chunk.metadata (includes source, chunk_index, strategy …)


class SearchResult(BaseModel):
    """A single result returned by a similarity search."""

    id: str
    content: str                    # always populated — sourced from stored metadata
    metadata: dict[str, Any]
    score: float                    # similarity score (higher = more similar)
    source: str                     # original file path


class BaseVectorStore(ABC):
    """Abstract interface for vector store backends.

    Both PineconeStore and QdrantStore must be drop-in replaceable.
    Errors are caught, logged, and re-raised as domain exceptions so
    callers always receive a clean, typed error rather than a raw SDK error.
    """

    @abstractmethod
    def upsert(self, documents: list[VectorDocument]) -> None:
        """Insert or update *documents* in the vector store.

        Must batch internally — never send all documents in a single request.
        """

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return the *top_k* most similar documents to *query_vector*.

        Args:
            query_vector: Dense embedding of the query text.
            top_k:        Number of results to return.
            filters:      Optional metadata filter (provider-specific semantics).
        """

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete documents by their IDs (no-op for unknown IDs)."""

    @abstractmethod
    def delete_by_source(self, source: str) -> None:
        """Delete every document whose ``source`` metadata field equals *source*."""
