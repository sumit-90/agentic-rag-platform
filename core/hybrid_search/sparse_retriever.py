import time

import structlog
from rank_bm25 import BM25Okapi

from core.chunking.base_chunker import Chunk
from core.vectorstore.base_vectorstore import SearchResult

logger = structlog.get_logger(__name__)


class SparseRetriever:
    """BM25-based sparse retriever.

    :meth:`index` must be called before :meth:`search`.
    Tokenisation: lowercase + whitespace split (no external tokeniser dependency).
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self, chunks: list[Chunk]) -> None:
        """Build a fresh BM25 index from *chunks*, replacing any previous index."""
        self._chunks = chunks
        tokenised_corpus = [self._tokenise(c.content) for c in chunks]
        self._bm25 = BM25Okapi(tokenised_corpus)
        logger.info("sparse_retriever_indexed", corpus_size=len(chunks))

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        if self._bm25 is None:
            raise RuntimeError(
                "SparseRetriever.search() called before index(). "
                "Call index(chunks) first to build the BM25 corpus."
            )

        log = logger.bind(query=query, top_k=top_k)
        t0 = time.perf_counter()

        tokens = self._tokenise(query)
        scores = self._bm25.get_scores(tokens)

        # Pair each chunk with its BM25 score and take the top-k by score
        ranked = sorted(
            ((i, s) for i, s in enumerate(scores) if s > 0.0),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]


        results: list[SearchResult] = []
        for idx, score in ranked:
            chunk = self._chunks[idx]
            results.append(
                SearchResult(
                    id=chunk.id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    score=float(score),
                    source=chunk.source,
                )
            )

        log.info(
            "sparse_retriever_search_complete",
            result_count=len(results),
            duration_s=round(time.perf_counter() - t0, 4),
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return text.lower().split()
