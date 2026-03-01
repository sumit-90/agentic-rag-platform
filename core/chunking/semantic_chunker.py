import time

import structlog
from langchain_experimental.text_splitter import SemanticChunker as LangchainSemanticChunker
from langchain_openai import OpenAIEmbeddings

from config.settings import settings
from core.chunking.base_chunker import BaseChunker, Chunk
from core.parsing.base_parser import ParsedDocument

logger = structlog.get_logger(__name__)

_SLOW_THRESHOLD_S = 10.0


class SemanticChunker(BaseChunker):
    def __init__(self) -> None:
        embeddings = OpenAIEmbeddings(
            api_key=settings.openai.api_key.get_secret_value()
        )
        self._splitter = LangchainSemanticChunker(embeddings=embeddings)

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        log = logger.bind(source=document.source)

        if not document.content.strip():
            log.warning("semantic_chunker_empty_content")
            return []

        start = time.perf_counter()
        texts = self._splitter.split_text(document.content)
        duration = time.perf_counter() - start

        if duration > _SLOW_THRESHOLD_S:
            log.warning(
                "semantic_chunker_slow",
                duration_s=round(duration, 4),
                threshold_s=_SLOW_THRESHOLD_S,
            )

        total = len(texts)
        chunks: list[Chunk] = []
        for idx, text in enumerate(texts):
            chunks.append(
                Chunk(
                    id=Chunk.make_id(document.source, idx),
                    content=text,
                    metadata={
                        **document.metadata,
                        "chunk_index": idx,
                        "strategy": "semantic",
                    },
                    source=document.source,
                    chunk_index=idx,
                    total_chunks=total,
                )
            )

        log.info(
            "semantic_chunker_complete",
            chunk_count=total,
            duration_s=round(duration, 4),
        )
        return chunks
