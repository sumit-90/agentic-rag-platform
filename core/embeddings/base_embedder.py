from abc import ABC, abstractmethod

import structlog

logger = structlog.get_logger(__name__)


class BaseEmbedder(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the active embedding model identifier."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text string and return its vector."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in one optimised call and return their vectors."""

    # ------------------------------------------------------------------
    # Shared guard — called by subclasses before hitting any API/model
    # ------------------------------------------------------------------

    def _validate_text(self, text: str) -> None:
        """Raise ValueError if *text* is an empty or whitespace-only string."""
        if not text.strip():
            logger.warning("embedder_empty_text_input", embedder=type(self).__name__)
            raise ValueError("Cannot embed an empty or whitespace-only string.")

    def _validate_batch(self, texts: list[str]) -> bool:
        """
        Return False (caller should return []) for an empty list.
        Raises ValueError for any blank item inside the list.
        """
        if not texts:
            logger.warning("embedder_empty_batch", embedder=type(self).__name__)
            return False
        for i, t in enumerate(texts):
            if not t.strip():
                logger.warning(
                    "embedder_blank_item_in_batch",
                    embedder=type(self).__name__,
                    index=i,
                )
                raise ValueError(
                    f"Empty string at index {i} — cannot embed blank text."
                )
        return True
