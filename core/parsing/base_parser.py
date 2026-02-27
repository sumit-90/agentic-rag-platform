from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ParsedDocument(BaseModel):
    content: str
    metadata: dict[str, Any]
    source: str
    parsed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse the file at file_path and return a ParsedDocument."""
