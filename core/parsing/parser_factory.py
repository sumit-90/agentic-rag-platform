from pathlib import Path

import structlog

from core.parsing.base_parser import BaseParser
from core.parsing.docx_parser import DOCXParser
from core.parsing.html_parser import HTMLParser
from core.parsing.pdf_parser import PDFParser

logger = structlog.get_logger(__name__)

_REGISTRY: dict[str, type[BaseParser]] = {
    ".pdf": PDFParser,
    ".docx": DOCXParser,
    ".html": HTMLParser,
    ".htm": HTMLParser,
}


class ParserFactory:
    @staticmethod
    def get_parser(file_path: Path) -> BaseParser:
        suffix = file_path.suffix.lower()
        parser_cls = _REGISTRY.get(suffix)
        if parser_cls is None:
            supported = ", ".join(sorted(_REGISTRY))
            raise ValueError(
                f"Unsupported file type '{suffix}' for '{file_path.name}'. "
                f"Supported extensions: {supported}"
            )
        logger.debug("parser_resolved", file=file_path.name, parser=parser_cls.__name__)
        return parser_cls()
