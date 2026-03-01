import time

import structlog
from sentence_transformers import SentenceTransformer

from core.embeddings.base_embedder import BaseEmbedder

logger = structlog.get_logger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_BATCH_SIZE = 64


class HFEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        logger.info("hf_model_loading", model=model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("hf_model_ready", model=model_name)


    # ------------------------------------------------------------------
    # BaseEmbedder interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, text: str) -> list[float]:
        self._validate_text(text)
        vector = self._model.encode(text, show_progress_bar=False)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not self._validate_batch(texts):
            return []

        log = logger.bind(
            model=self._model_name,
            batch_size=self._batch_size,
            total_texts=len(texts),
        )
        t0 = time.perf_counter()
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        duration = time.perf_counter() - t0

        log.info("hf_embedder_batch_complete", duration_s=round(duration, 4))
        return [v.tolist() for v in vectors]
