import logging
import sys
from typing import Any

import structlog


def setup_logging() -> None:
    """Configure structlog for the application.

    - ENV=dev  → colored, human-readable console output
    - ENV=prod → JSON with timestamp, level, module, and message

    Call once at startup before any loggers are used. All modules should
    obtain loggers via: logger = structlog.get_logger(__name__)
    """
    from config.settings import settings

    log_level: int = getattr(logging, settings.app.log_level.upper(), logging.INFO)
    is_prod: bool = settings.app.env == "prod"

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.ExceptionRenderer(), 
    ]

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if is_prod
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
    processors=[
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
        renderer,
    ],
    foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)
