import time
from typing import Any

import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
)

from config.settings import settings
from core.vectorstore.base_vectorstore import (
    BaseVectorStore,
    SearchResult,
    VectorDocument,
)

logger = structlog.get_logger(__name__)


class QdrantStore(BaseVectorStore):
    """Vector store backed by Qdrant (qdrant-client v1.16+)."""

    def __init__(self) -> None:
        self._client = QdrantClient(
            url=settings.qdrant.url,
            port=settings.qdrant.port,
        )
        self._collection = settings.qdrant.collection_name
        self._batch_size = settings.vectorstore.upsert_batch_size

    # ------------------------------------------------------------------
    # BaseVectorStore interface
    # ------------------------------------------------------------------

    def upsert(self, documents: list[VectorDocument]) -> None:
        if not documents:
            return

        log = logger.bind(
            store="qdrant",
            collection=self._collection,
            total_docs=len(documents),
        )
        log.info("qdrant_upsert_start")
        start = time.perf_counter()

        try:
            for batch_start in range(0, len(documents), self._batch_size):
                batch = documents[batch_start : batch_start + self._batch_size]
                points = [
                    PointStruct(
                        id=doc.id,
                        vector=doc.embedding,
                        payload={**doc.metadata, "content": doc.content},
                    )
                    for doc in batch
                ]
                self._client.upsert(
                    collection_name=self._collection,
                    points=points,
                )
        except Exception as exc:
            log.error("qdrant_upsert_failed", error=str(exc))
            raise RuntimeError(f"Qdrant upsert failed: {exc}") from exc

        duration = time.perf_counter() - start
        log.info("qdrant_upsert_complete", duration_s=round(duration, 4))

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        log = logger.bind(store="qdrant", collection=self._collection, top_k=top_k)
        start = time.perf_counter()

        query_filter = self._build_filter(filters) if filters else None

        try:
            hits = self._client.search(
                collection_name=self._collection,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
        except Exception as exc:
            log.error("qdrant_search_failed", error=str(exc))
            raise RuntimeError(f"Qdrant search failed: {exc}") from exc

        results: list[SearchResult] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            results.append(
                SearchResult(
                    id=str(hit.id),
                    content=payload.pop("content", ""),
                    metadata=payload,
                    score=hit.score,
                    source=payload.get("source", ""),
                )
            )

        duration = time.perf_counter() - start
        log.info(
            "qdrant_search_complete",
            result_count=len(results),
            duration_s=round(duration, 4),
        )
        return results

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        log = logger.bind(store="qdrant", count=len(ids))
        try:
            self._client.delete(
                collection_name=self._collection,
                points_selector=PointIdsList(points=ids),
            )
        except Exception as exc:
            log.error("qdrant_delete_failed", error=str(exc))
            raise RuntimeError(f"Qdrant delete failed: {exc}") from exc
        log.info("qdrant_delete_complete")

    def delete_by_source(self, source: str) -> None:
        log = logger.bind(store="qdrant", source=source)
        log.info("qdrant_delete_by_source_start")
        source_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source))]
        )
        try:
            self._client.delete(
                collection_name=self._collection,
                points_selector=source_filter,
            )
        except Exception as exc:
            log.error("qdrant_delete_by_source_failed", error=str(exc))
            raise RuntimeError(f"Qdrant delete_by_source failed: {exc}") from exc
        log.info("qdrant_delete_by_source_complete")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> Filter:
        """Convert a flat ``{field: value}`` dict into a Qdrant ``Filter``."""
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ]
        return Filter(must=conditions)
