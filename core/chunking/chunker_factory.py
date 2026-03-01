import structlog

from core.chunking.base_chunker import BaseChunker
from core.chunking.fixed_chunker import FixedChunker
from core.chunking.semantic_chunker import SemanticChunker

logger = structlog.get_logger(__name__)

_REGISTRY: dict[str, type[BaseChunker]] = {
    "fixed": FixedChunker,
    "semantic": SemanticChunker,
}


class ChunkerFactory:
    @staticmethod
    def get_chunker(strategy: str) -> BaseChunker:
        chunker_cls = _REGISTRY.get(strategy)
        if chunker_cls is None:
            supported = ", ".join(sorted(_REGISTRY))
            raise ValueError(
                f"Unknown chunking strategy '{strategy}'. "
                f"Supported strategies: {supported}"
            )
        logger.debug("chunker_resolved", strategy=strategy, chunker=chunker_cls.__name__)
        return chunker_cls()
