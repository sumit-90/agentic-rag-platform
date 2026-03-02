import time
from typing import Any
import json
import structlog

from config.settings import settings
from core.cache.base_cache import BaseCache
from core.hybrid_search.hybrid_retriever import HybridRetriever
from core.vectorstore.base_vectorstore import SearchResult

logger = structlog.get_logger(__name__)


class RetrievalService:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        cache: BaseCache | None = None,
    ) -> None:
        self._retriever = hybrid_retriever
        self._cache = cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        mode: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        resolved_top_k = top_k if top_k is not None else settings.retrieval.top_k
        resolved_mode = mode if mode is not None else settings.retrieval.mode

        log = logger.bind(
            query=query,
            mode=resolved_mode,
            top_k=resolved_top_k,
        )
        t0 = time.perf_counter()

        # Cache lookup — key encodes all parameters that affect the result
        filters_str = json.dumps(filters, sort_keys=True) if filters else "none"
        cache_key = BaseCache.make_key("retrieval", query, resolved_mode, str(resolved_top_k), filters_str)
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                log.info(
                    "retrieval_cache_hit",
                    result_count=len(cached),
                    duration_s=round(time.perf_counter() - t0, 4),
                )
                return cached

        log.debug("retrieval_cache_miss")

        # Run hybrid / dense / sparse retrieval
        results = self._retriever.search(
            query=query,
            top_k=resolved_top_k,
            filters=filters,
            mode=resolved_mode,
        )

        # Apply score threshold — discard low-confidence results
        threshold = settings.retrieval.score_threshold
        if resolved_mode == "dense":
            filtered = [r for r in results if r.score >= threshold]
        else:
            filtered = results

        if len(filtered) < len(results):
            log.debug(
                "retrieval_threshold_applied",
                before=len(results),
                after=len(filtered),
                threshold=threshold,
            )

        # Populate cache with the filtered results
        if self._cache is not None:
            self._cache.set(cache_key, filtered)

        log.info(
            "retrieval_complete",
            result_count=len(filtered),
            duration_s=round(time.perf_counter() - t0, 4),
        )
        return filtered
