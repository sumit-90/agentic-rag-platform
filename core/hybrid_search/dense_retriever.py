import time
from typing import Any

import structlog

from core.vectorstore.base_vectorstore import BaseVectorStore, SearchResult

logger = structlog.get_logger(__name__)


class DenseRetriever:
    """Thin wrapper over :class:`BaseVectorStore` that adds structured logging."""

    def __init__(self, vector_store: BaseVectorStore) -> None:
        self._store = vector_store

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        log = logger.bind(
            vector_dim=len(query_vector),
            top_k=top_k,
            has_filters=filters is not None,
        )
        t0 = time.perf_counter()
        results = self._store.search(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
        )
        log.info(
            "dense_retriever_search_complete",
            result_count=len(results),
            duration_s=round(time.perf_counter() - t0, 4),
        )
        return results
