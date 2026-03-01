import time
from pathlib import Path
from typing import Literal

import structlog
from pydantic import BaseModel

from config.settings import settings
from core.cache.base_cache import BaseCache
from core.chunking.base_chunker import Chunk
from core.chunking.chunker_factory import ChunkerFactory
from core.embeddings.base_embedder import BaseEmbedder
from core.parsing.parser_factory import ParserFactory
from core.vectorstore.base_vectorstore import BaseVectorStore, VectorDocument

logger = structlog.get_logger(__name__)


class IngestionResult(BaseModel):
    source: str
    total_chunks: int
    indexed_chunks: int
    skipped_chunks: int
    duration_s: float
    status: Literal["success", "failed"]


class IngestionService:
    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        cache: BaseCache | None = None,
        chunking_strategy: str | None = None,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._cache = cache
        self._chunking_strategy = chunking_strategy or settings.chunking.strategy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, file_path: Path) -> IngestionResult:
        log = logger.bind(source=str(file_path))
        log.info("ingestion_start")
        pipeline_start = time.perf_counter()

        try:
            return self._run_pipeline(file_path, log, pipeline_start)
        except Exception as exc:
            duration = round(time.perf_counter() - pipeline_start, 4)
            log.error("ingestion_pipeline_error", error=str(exc), duration_s=duration)
            return IngestionResult(
                source=str(file_path),
                total_chunks=0,
                indexed_chunks=0,
                skipped_chunks=0,
                duration_s=duration,
                status="failed",
            )

    # ------------------------------------------------------------------
    # Pipeline — each step isolated so exceptions surface with context
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        file_path: Path,
        log: structlog.BoundLogger,
        pipeline_start: float,
    ) -> IngestionResult:
        source = str(file_path)
        if not file_path.exists():
            log.error("ingestion_file_not_found", source=source)
            return IngestionResult(
                source=source, total_chunks=0, indexed_chunks=0,
                skipped_chunks=0, duration_s=0.0, status="failed"
            )

        

        # Step 1 — Parse
        t0 = time.perf_counter()
        parser = ParserFactory.get_parser(file_path)
        parsed_doc = parser.parse(file_path)
        log.info("ingestion_step_parse", duration_s=round(time.perf_counter() - t0, 4))

        if not parsed_doc.content.strip():
            duration = round(time.perf_counter() - pipeline_start, 4)
            log.warning("ingestion_empty_document", source=source, duration_s=duration)
            return IngestionResult(
                source=source,
                total_chunks=0,
                indexed_chunks=0,
                skipped_chunks=0,
                duration_s=duration,
                status="failed",
            )

        # Step 2 — Chunk
        t0 = time.perf_counter()
        chunker = ChunkerFactory.get_chunker(self._chunking_strategy)
        chunks: list[Chunk] = chunker.chunk(parsed_doc)
        total_chunks = len(chunks)
        log.info(
            "ingestion_step_chunk",
            chunk_count=total_chunks,
            duration_s=round(time.perf_counter() - t0, 4),
        )

        if not chunks:
            duration = round(time.perf_counter() - pipeline_start, 4)
            log.warning("ingestion_no_chunks", source=source, duration_s=duration)
            return IngestionResult(
                source=source,
                total_chunks=0,
                indexed_chunks=0,
                skipped_chunks=0,
                duration_s=duration,
                status="failed",
            )

        # Step 3 — Cache check: separate already-indexed from new chunks
        t0 = time.perf_counter()
        new_chunks, skipped = self._cache_filter(chunks)
        log.info(
            "ingestion_step_cache_check",
            total=total_chunks,
            new=len(new_chunks),
            skipped=skipped,
            duration_s=round(time.perf_counter() - t0, 4),
        )

        if not new_chunks:
            duration = round(time.perf_counter() - pipeline_start, 4)
            log.info("ingestion_all_cached", total_chunks=total_chunks, duration_s=duration)
            return IngestionResult(
                source=source,
                total_chunks=total_chunks,
                indexed_chunks=0,
                skipped_chunks=skipped,
                duration_s=duration,
                status="success",
            )

        # Step 4 — Embed new chunks only
        t0 = time.perf_counter()
        embeddings: list[list[float]] = self._embedder.embed_batch(
            [c.content for c in new_chunks]
        )
        log.info(
            "ingestion_step_embed",
            chunk_count=len(new_chunks),
            duration_s=round(time.perf_counter() - t0, 4),
        )

        if len(embeddings) != len(new_chunks):
            raise ValueError(
                f"Embedding count mismatch: "
                f"expected {len(new_chunks)}, got {len(embeddings)}"
            )

        # Step 5 — Pair chunks with their embeddings into VectorDocuments
        vector_docs: list[VectorDocument] = [
            VectorDocument(
                id=chunk.id,
                embedding=embedding,
                content=chunk.content,
                metadata=chunk.metadata,
            )
            for chunk, embedding in zip(new_chunks, embeddings)
        ]

        # Step 6 — Upsert into vector store
        t0 = time.perf_counter()
        self._vector_store.upsert(vector_docs)
        log.info(
            "ingestion_step_upsert",
            indexed_count=len(vector_docs),
            duration_s=round(time.perf_counter() - t0, 4),
        )

        # Step 7 — Mark successfully indexed chunks in cache
        if self._cache is not None:
            for chunk in new_chunks:
                self._cache.set(chunk.id, True)

        duration = round(time.perf_counter() - pipeline_start, 4)
        log.info(
            "ingestion_complete",
            total_chunks=total_chunks,
            indexed_chunks=len(new_chunks),
            skipped_chunks=skipped,
            duration_s=duration,
        )
        return IngestionResult(
            source=source,
            total_chunks=total_chunks,
            indexed_chunks=len(new_chunks),
            skipped_chunks=skipped,
            duration_s=duration,
            status="success",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_filter(self, chunks: list[Chunk]) -> tuple[list[Chunk], int]:
        """Return (new_chunks, skipped_count).

        If cache is None, all chunks are treated as new.
        """
        if self._cache is None:
            return list(chunks), 0

        new: list[Chunk] = []
        skipped = 0
        for chunk in chunks:
            if self._cache.exists(chunk.id):
                skipped += 1
            else:
                new.append(chunk)
        return new, skipped
