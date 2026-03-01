import threading
import time
from typing import Any

import structlog

from config.settings import settings
from core.cache.base_cache import BaseCache

logger = structlog.get_logger(__name__)

_DEFAULT_MAX_SIZE = 1024


class InMemoryCache(BaseCache):
    """Thread-safe in-process cache backed by :class:`cachetools.TTLCache`.

    *max_size* bounds memory usage; items are evicted LRU-style once full.
    *default_ttl* is used when :meth:`set` is called without an explicit TTL.
    A per-item TTL override is implemented on top of TTLCache via a parallel
    ``_expiry`` dict — TTLCache itself uses a single global TTL at construction
    time, so we need the extra layer to honour per-call TTL values.
    """

    def __init__(
        self,
        max_size: int = _DEFAULT_MAX_SIZE,
        default_ttl: int | None = None,
    ) -> None:
        self._default_ttl = (
            default_ttl if default_ttl is not None else settings.redis.ttl_seconds
        )
        # TTLCache requires a global maxsize and ttl; we set ttl to math.inf
        # so the underlying cache never auto-expires — expiry is managed
        # manually via self._expiry to support per-entry TTL overrides.
        self._cache: dict[str, Any] = {}
        self._expiry: dict[str, float] = {}   # key → absolute expiry timestamp
        self._max_size = max_size
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # BaseCache interface
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        with self._lock:
            if not self._is_alive(key):
                if key in self._cache:
                    self._evict(key)
                logger.debug("in_memory_cache_miss", key=key)
                return None
            logger.debug("in_memory_cache_hit", key=key)
            return self._cache[key]

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        with self._lock:
            # Enforce max_size by evicting the oldest-expiring entry
            if key not in self._cache and len(self._cache) >= self._max_size:
                self._evict_oldest()
            self._cache[key] = value
            if effective_ttl is not None:
                self._expiry[key] = time.monotonic() + effective_ttl
            else:
                self._expiry.pop(key, None)  # no expiry

    def delete(self, key: str) -> None:
        with self._lock:
            self._evict(key)

    def exists(self, key: str) -> bool:
        with self._lock:
            alive = self._is_alive(key)
            if not alive and key in self._cache:
                self._evict(key)
            return alive

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._expiry.clear()

    # ------------------------------------------------------------------
    # Internal helpers  (must be called with _lock held)
    # ------------------------------------------------------------------

    def _is_alive(self, key: str) -> bool:
        if key not in self._cache:
            return False
        exp = self._expiry.get(key)
        if exp is None:
            return True  # no TTL → immortal
        return time.monotonic() < exp

    def _evict(self, key: str) -> None:
        self._cache.pop(key, None)
        self._expiry.pop(key, None)

    def _evict_oldest(self) -> None:
        """Remove the entry whose TTL expires soonest (or any key if none have TTLs)."""
        if not self._cache:
            return
        # Prefer evicting entries with the earliest expiry; immortal entries last
        target = min(
            self._cache.keys(),
            key=lambda k: self._expiry.get(k, float("inf")),
        )
        self._evict(target)
