import hashlib
from abc import ABC, abstractmethod
from typing import Any

import structlog
from pydantic import BaseModel, Field

from core.parsing.base_parser import ParsedDocument

logger = structlog.get_logger(__name__)


class Chunk(BaseModel):
    id: str
    content: str
    metadata: dict[str, Any]
    source: str
    chunk_index: int
    total_chunks: int

    @staticmethod
    def make_id(source: str, chunk_index: int) -> str:
        return hashlib.md5(f"{source}_{chunk_index}".encode()).hexdigest()


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Split a ParsedDocument into a list of Chunk objects."""
