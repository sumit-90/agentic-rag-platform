import time
from typing import Any

import structlog
from pinecone import Pinecone

from config.settings import settings
from core.vectorstore.base_vectorstore import (
    BaseVectorStore,
    SearchResult,
    VectorDocument,
)

logger = structlog.get_logger(__name__)


class PineconeStore(BaseVectorStore):
    """Vector store backed by Pinecone (client v3+)."""

    def __init__(self) -> None:
        pc = Pinecone(api_key=settings.pinecone.api_key.get_secret_value())
        self._index = pc.Index(settings.pinecone.index_name)
        self._namespace = settings.pinecone.namespace
        self._batch_size = settings.vectorstore.upsert_batch_size

    # ------------------------------------------------------------------
    # BaseVectorStore interface
    # ------------------------------------------------------------------

    def upsert(self, documents: list[VectorDocument]) -> None:
        if not documents:
            return

        log = logger.bind(
            store="pinecone",
            index=settings.pinecone.index_name,
            namespace=self._namespace,
            total_docs=len(documents),
        )
        log.info("pinecone_upsert_start")
        start = time.perf_counter()

        try:
            for batch_start in range(0, len(documents), self._batch_size):
                batch = documents[batch_start : batch_start + self._batch_size]
                vectors = [
                    {
                        "id": doc.id,
                        "values": doc.embedding,
                        "metadata": {**doc.metadata, "content": doc.content},
                    }
                    for doc in batch
                ]
                self._index.upsert(vectors=vectors, namespace=self._namespace)
        except Exception as exc:
            log.error("pinecone_upsert_failed", error=str(exc))
            raise RuntimeError(f"Pinecone upsert failed: {exc}") from exc

        duration = time.perf_counter() - start
        log.info("pinecone_upsert_complete", duration_s=round(duration, 4))

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        log = logger.bind(
            store="pinecone",
            index=settings.pinecone.index_name,
            top_k=top_k,
        )
        start = time.perf_counter()

        try:
            response = self._index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filters,
                include_metadata=True,
                namespace=self._namespace,
            )
        except Exception as exc:
            log.error("pinecone_search_failed", error=str(exc))
            raise RuntimeError(f"Pinecone search failed: {exc}") from exc

        results: list[SearchResult] = []
        for match in response.matches:
            meta = match.metadata or {}
            results.append(
                SearchResult(
                    id=match.id,
                    content=meta.pop("content", ""),
                    metadata=meta,
                    score=match.score,
                    source=meta.get("source", ""),
                )
            )

        duration = time.perf_counter() - start
        log.info(
            "pinecone_search_complete",
            result_count=len(results),
            duration_s=round(duration, 4),
        )
        return results

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        log = logger.bind(store="pinecone", count=len(ids))
        try:
            self._index.delete(ids=ids, namespace=self._namespace)
        except Exception as exc:
            log.error("pinecone_delete_failed", error=str(exc))
            raise RuntimeError(f"Pinecone delete failed: {exc}") from exc
        log.info("pinecone_delete_complete")

    def delete_by_source(self, source: str) -> None:
        log = logger.bind(store="pinecone", source=source)
        log.info("pinecone_delete_by_source_start")
        try:
            self._index.delete(
                filter={"source": {"$eq": source}},
                namespace=self._namespace,
            )
        except Exception as exc:
            log.error("pinecone_delete_by_source_failed", error=str(exc))
            raise RuntimeError(f"Pinecone delete_by_source failed: {exc}") from exc
        log.info("pinecone_delete_by_source_complete")
