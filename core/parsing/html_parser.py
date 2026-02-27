import time
from pathlib import Path

import structlog
from bs4 import BeautifulSoup

from core.parsing.base_parser import BaseParser, ParsedDocument

logger = structlog.get_logger(__name__)


class HTMLParser(BaseParser):
    def parse(self, file_path: Path) -> ParsedDocument:
        log = logger.bind(file_path=str(file_path))

        if not file_path.exists():
            log.error("html_file_not_found")
            raise FileNotFoundError(f"HTML file not found: {file_path}")

        file_size_bytes = file_path.stat().st_size
        log = log.bind(file_size_bytes=file_size_bytes)
        log.info("html_parse_start")

        start = time.perf_counter()

        try:
            raw_html = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            log.warning("html_read_failed", error=str(exc))
            raise ValueError(f"Cannot read HTML file '{file_path}': {exc}") from exc

        try:
            soup = BeautifulSoup(raw_html, "lxml")
            title: str | None = soup.title.string.strip() if soup.title and soup.title.string else None
            for tag in soup(["script", "style"]):
                tag.decompose()

            content = soup.get_text(separator="\n", strip=True)
        except Exception as exc:
            log.warning("html_parse_failed", error=str(exc))
            raise ValueError(f"Failed to parse HTML '{file_path}': {exc}") from exc

        duration = time.perf_counter() - start
        log.info("html_parse_complete", title=title, duration_s=round(duration, 4))

        metadata: dict = {
            "file_type": "html",
            "file_size_bytes": file_size_bytes,
        }
        if title is not None:
            metadata["title"] = title

        return ParsedDocument(
            content=content,
            metadata=metadata,
            source=str(file_path),
        )
