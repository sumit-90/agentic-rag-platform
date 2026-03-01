import time

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from core.chunking.base_chunker import BaseChunker, Chunk
from core.parsing.base_parser import ParsedDocument

logger = structlog.get_logger(__name__)


class FixedChunker(BaseChunker):
    def __init__(self) -> None:
        self._chunk_size = settings.chunking.chunk_size
        self._chunk_overlap = settings.chunking.chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        log = logger.bind(source=document.source)

        if not document.content.strip():
            log.warning("fixed_chunker_empty_content")
            return []

        start = time.perf_counter()
        texts = self._splitter.split_text(document.content)
        duration = time.perf_counter() - start

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
                        "strategy": "fixed",
                    },
                    source=document.source,
                    chunk_index=idx,
                    total_chunks=total,
                )
            )

        log.info(
            "fixed_chunker_complete",
            chunk_count=total,
            chunk_size=self._chunk_size,
            duration_s=round(duration, 4),
        )
        return chunks
