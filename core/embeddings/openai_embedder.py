import time

import structlog
from langchain_openai import OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from core.embeddings.base_embedder import BaseEmbedder

logger = structlog.get_logger(__name__)

_MAX_RETRIES = 3
_WAIT_MIN_S = 1
_WAIT_MAX_S = 10


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, batch_size: int | None = None) -> None:
        self._model = settings.embedding.model_name
        self._batch_size = batch_size if batch_size is not None else settings.embedding.batch_size
        self._client = OpenAIEmbeddings(
            api_key=settings.openai.api_key.get_secret_value(),
            model=self._model,
        )

    # ------------------------------------------------------------------
    # BaseEmbedder interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model

    def embed(self, text: str) -> list[float]:
        self._validate_text(text)
        return self._embed_query_with_retry(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not self._validate_batch(texts):
            return []

        total = len(texts)
        # Send in a single shot when everything fits in one batch
        if total <= self._batch_size:
            t0 = time.perf_counter()
            result = self._embed_documents_with_retry(texts)
            logger.info(
                "openai_embedder_batch_complete",
                batch_num=1,
                num_batches=1,
                batch_size=total,
                model=self._model,
                duration_s=round(time.perf_counter() - t0, 4),
            )
            return result


        results: list[list[float]] = []
        num_batches = (total + self._batch_size - 1) // self._batch_size

        for batch_num in range(num_batches):
            start_idx = batch_num * self._batch_size
            batch = texts[start_idx : start_idx + self._batch_size]

            log = logger.bind(
                batch_num=batch_num + 1,
                num_batches=num_batches,
                batch_size=len(batch),
                model=self._model,
            )
            t0 = time.perf_counter()
            vectors = self._embed_documents_with_retry(batch)
            duration = time.perf_counter() - t0

            log.info("openai_embedder_batch_complete", duration_s=round(duration, 4))
            results.extend(vectors)

        return results

    # ------------------------------------------------------------------
    # Retry-wrapped private helpers
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=_WAIT_MIN_S, max=_WAIT_MAX_S),
        reraise=True,
    )
    def _embed_query_with_retry(self, text: str) -> list[float]:
        return self._client.embed_query(text)

    @retry(
        stop=stop_after_attempt(_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=_WAIT_MIN_S, max=_WAIT_MAX_S),
        reraise=True,
    )
    def _embed_documents_with_retry(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)
