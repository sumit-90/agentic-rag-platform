"""Shared pytest fixtures for unit and integration tests."""

import pytest

from core.cache.in_memory_cache import InMemoryCache
from core.chunking.base_chunker import Chunk
from core.parsing.base_parser import ParsedDocument
from core.vectorstore.base_vectorstore import SearchResult


# ---------------------------------------------------------------------------
# Primitive fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def in_memory_cache() -> InMemoryCache:
    """Fresh InMemoryCache instance for each test."""
    return InMemoryCache()


@pytest.fixture()
def parsed_document() -> ParsedDocument:
    """Minimal ParsedDocument for parser / chunker tests."""
    return ParsedDocument(
        content="This is a test document. It has multiple sentences for chunking.",
        metadata={"page_count": 1},
        source="test_doc.pdf",
    )


@pytest.fixture()
def sample_chunk() -> Chunk:
    """A single Chunk with deterministic ID for cache / vector store tests."""
    return Chunk(
        id="abc123",
        content="Sample chunk content for testing.",
        metadata={"strategy": "fixed", "source": "test_doc.pdf"},
        source="test_doc.pdf",
        chunk_index=0,
        total_chunks=1,
    )


@pytest.fixture()
def sample_search_results() -> list[SearchResult]:
    """Two SearchResults for retrieval / reranking tests."""
    return [
        SearchResult(
            id="chunk_0",
            content="The capital of France is Paris.",
            metadata={"source": "geo.pdf"},
            score=0.92,
            source="geo.pdf",
        ),
        SearchResult(
            id="chunk_1",
            content="Paris is known for the Eiffel Tower.",
            metadata={"source": "geo.pdf"},
            score=0.85,
            source="geo.pdf",
        ),
    ]
