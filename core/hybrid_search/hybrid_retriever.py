import time
from typing import Any

import structlog

from core.embeddings.base_embedder import BaseEmbedder
from core.hybrid_search.dense_retriever import DenseRetriever
from core.hybrid_search.sparse_retriever import SparseRetriever
from core.vectorstore.base_vectorstore import SearchResult

logger = structlog.get_logger(__name__)

RRF_K = 60

_VALID_MODES = {"dense", "sparse", "hybrid"}


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        embedder: BaseEmbedder,
    ) -> None:
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._embedder = embedder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        mode: str = "hybrid",
    ) -> list[SearchResult]:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Unknown retrieval mode '{mode}'. Valid modes: {', '.join(sorted(_VALID_MODES))}"
            )

        log = logger.bind(mode=mode, query=query, top_k=top_k)
        t0 = time.perf_counter()

        if mode == "dense":
            results = self._run_dense(query, top_k, filters)
            dense_count, sparse_count = len(results), 0

        elif mode == "sparse":
            results = self._sparse.search(query, top_k)
            dense_count, sparse_count = 0, len(results)

        else:  # hybrid
            fetch_k = top_k * 2
            dense_results = self._run_dense(query, fetch_k, filters)
            sparse_results = self._sparse.search(query, fetch_k)
            results = _reciprocal_rank_fusion(dense_results, sparse_results, top_k)
            dense_count, sparse_count = len(dense_results), len(sparse_results)

        log.info(
            "hybrid_retriever_search_complete",
            dense_count=dense_count,
            sparse_count=sparse_count,
            final_count=len(results),
            duration_s=round(time.perf_counter() - t0, 4),
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_dense(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[SearchResult]:
        query_vector = self._embedder.embed(query)
        return self._dense.search(query_vector, top_k, filters)


# ------------------------------------------------------------------
# Reciprocal Rank Fusion — module-level so it's independently testable
# ------------------------------------------------------------------

def _reciprocal_rank_fusion(
    dense_results: list[SearchResult],
    sparse_results: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    """Merge *dense_results* and *sparse_results* using RRF (k=60).

    Each document receives a score contribution of ``1 / (RRF_K + rank + 1)``
    from every list it appears in.  Documents are then sorted by total RRF
    score descending and the top *top_k* are returned.
    """
    scores: dict[str, float] = {}
    docs: dict[str, SearchResult] = {}

    for rank, result in enumerate(dense_results):
        scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (RRF_K + rank + 1)
        docs[result.id] = result

    for rank, result in enumerate(sparse_results):
        scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (RRF_K + rank + 1)
        docs[result.id] = result

    ranked_ids = sorted(scores, key=lambda id_: scores[id_], reverse=True)

    # Propagate the fused RRF score back onto the SearchResult so callers
    # can inspect it — score_threshold filtering in RetrievalService uses it.
    fused: list[SearchResult] = []
    for id_ in ranked_ids[:top_k]:
        result = docs[id_].model_copy(update={"score": scores[id_]})
        fused.append(result)

    return fused
