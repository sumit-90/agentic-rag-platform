import time

import cohere
import structlog

from config.settings import settings
from core.vectorstore.base_vectorstore import SearchResult

logger = structlog.get_logger(__name__)


class RerankingService:
    def __init__(
        self,
        model: str | None = None,
        top_n: int | None = None,
    ) -> None:
        self._client = cohere.Client(
            api_key=settings.cohere.api_key.get_secret_value()
        )
        self._model = model or settings.cohere.rerank_model
        self._top_n = top_n if top_n is not None else settings.cohere.top_n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int | None = None,
    ) -> list[SearchResult]:
        if not results:
            return []

        resolved_top_n = top_n if top_n is not None else self._top_n

        log = logger.bind(
            query=query,
            input_count=len(results),
            top_n=resolved_top_n,
            model=self._model,
        )
        t0 = time.perf_counter()

        response = self._client.rerank(
            model=self._model,
            query=query,
            documents=[r.content for r in results],
            top_n=resolved_top_n,
        )

        reranked = [
            results[item.index].model_copy(update={"score": item.relevance_score})
            for item in response.results
        ]

        log.info(
            "reranking_complete",
            output_count=len(reranked),
            duration_s=round(time.perf_counter() - t0, 4),
        )
        return reranked
