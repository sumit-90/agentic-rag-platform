from abc import ABC, abstractmethod
from typing import Any


class BaseCache(ABC):
    """Abstract cache interface.

    All implementations must be drop-in replaceable.
    Cache failures must never propagate — callers degrade gracefully.
    """

    # ------------------------------------------------------------------
    # Key construction
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(*parts: str) -> str:
        """Join *parts* with ':' to build a consistent, namespaced cache key.

        Example:
            BaseCache.make_key("embedding", "doc_abc", "chunk_0")
            # → "embedding:doc_abc:chunk_0"
        """
        return ":".join(parts)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Return the cached value or ``None`` if missing or expired."""

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store *value* under *key*.

        Args:
            key:   Cache key (always a plain string).
            value: Any serialisable value (embeddings, dicts, strings …).
            ttl:   Time-to-live in seconds.  ``None`` means no expiry.
        """

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove a single key (no-op if the key does not exist)."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return ``True`` if *key* exists and has not expired."""

    @abstractmethod
    def clear(self) -> None:
        """Flush all cached entries."""
