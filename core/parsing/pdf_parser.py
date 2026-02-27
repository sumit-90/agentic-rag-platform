import time
from pathlib import Path

import fitz  # PyMuPDF
import structlog

from core.parsing.base_parser import BaseParser, ParsedDocument

logger = structlog.get_logger(__name__)


class PDFParser(BaseParser):
    def parse(self, file_path: Path) -> ParsedDocument:
        log = logger.bind(file_path=str(file_path))

        if not file_path.exists():
            log.error("pdf_file_not_found")
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        file_size_bytes = file_path.stat().st_size
        log = log.bind(file_size_bytes=file_size_bytes)
        log.info("pdf_parse_start")

        start = time.perf_counter()

        try:
            doc = fitz.open(str(file_path))
        except Exception as exc:
            log.warning("pdf_open_failed", error=str(exc))
            raise ValueError(f"Cannot open PDF '{file_path}': {exc}") from exc

        if doc.is_encrypted:
            doc.close()
            log.warning("pdf_encrypted")
            raise ValueError(f"PDF is encrypted and cannot be parsed: {file_path}")

        try:
            pages: list[str] = []
            for page in doc:
                pages.append(page.get_text())
            content = "\n".join(pages)
            page_count = doc.page_count
        except Exception as exc:
            log.warning("pdf_text_extraction_failed", error=str(exc))
            raise ValueError(f"Failed to extract text from PDF '{file_path}': {exc}") from exc
        finally:
            doc.close()

        duration = time.perf_counter() - start
        log.info("pdf_parse_complete", page_count=page_count, duration_s=round(duration, 4))

        return ParsedDocument(
            content=content,
            metadata={
                "file_type": "pdf",
                "file_size_bytes": file_size_bytes,
                "page_count": page_count,
            },
            source=str(file_path),
        )
