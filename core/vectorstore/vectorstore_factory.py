import structlog

from config.settings import settings
from core.vectorstore.base_vectorstore import BaseVectorStore
from core.vectorstore.pinecone_store import PineconeStore
from core.vectorstore.qdrant_store import QdrantStore

logger = structlog.get_logger(__name__)

_REGISTRY: dict[str, type[BaseVectorStore]] = {
    "pinecone": PineconeStore,
    "qdrant": QdrantStore,
}


class VectorStoreFactory:
    @staticmethod
    def get_store(provider: str | None = None) -> BaseVectorStore:
        """Return a vector store instance for *provider*.

        If *provider* is omitted, ``settings.vectorstore.provider`` is used.
        Raises ``ValueError`` for unknown provider names.
        """
        resolved = provider or settings.vectorstore.provider
        store_cls = _REGISTRY.get(resolved)
        if store_cls is None:
            supported = ", ".join(sorted(_REGISTRY))
            raise ValueError(
                f"Unknown vector store provider '{resolved}'. "
                f"Supported providers: {supported}"
            )
        logger.debug("vectorstore_resolved", provider=resolved, store=store_cls.__name__)
        return store_cls()
