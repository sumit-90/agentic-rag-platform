import pickle
from typing import Any

import structlog
from redis import Redis
from redis.exceptions import RedisError

from config.settings import settings
from core.cache.base_cache import BaseCache

logger = structlog.get_logger(__name__)


class RedisCache(BaseCache):
    """Cache backed by Redis.

    Connection errors are caught and logged at WARNING level — they must
    never propagate to callers.  On a failed ``get`` the caller receives
    ``None`` (cache miss) and the pipeline continues unaffected.

    Values are serialised with :mod:`pickle` because embeddings
    (``list[float]``) are not efficiently JSON-serialisable.
    """

    def __init__(self, default_ttl: int | None = None) -> None:
        self._default_ttl = (
            default_ttl if default_ttl is not None else settings.redis.ttl_seconds
        )
        self._client: Redis = Redis.from_url(
            settings.redis.url,
            decode_responses=False,  # keep raw bytes for pickle round-trips
        )
        try:
            self._client.ping()
            logger.info("redis_cache_connected", url=settings.redis.url)
        except RedisError as exc:
            logger.warning("redis_cache_unavailable", url=settings.redis.url, error=str(exc))

    # ------------------------------------------------------------------
    # BaseCache interface
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        try:
            raw = self._client.get(key)
        except RedisError as exc:
            logger.warning("redis_cache_get_error", key=key, error=str(exc))
            return None

        if raw is None:
            logger.debug("redis_cache_miss", key=key)
            return None

        try:
            value = pickle.loads(raw)  # noqa: S301 — controlled internal cache
        except Exception as exc:
            logger.warning("redis_cache_deserialise_error", key=key, error=str(exc))
            return None

        logger.debug("redis_cache_hit", key=key)
        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        try:
            raw = pickle.dumps(value)
            if effective_ttl is not None:
                self._client.setex(key, effective_ttl, raw)
            else:
                self._client.set(key, raw)
        except RedisError as exc:
            logger.warning("redis_cache_set_error", key=key, error=str(exc))
        except Exception as exc:
            logger.warning("redis_cache_serialise_error", key=key, error=str(exc))

    def delete(self, key: str) -> None:
        try:
            self._client.delete(key)
        except RedisError as exc:
            logger.warning("redis_cache_delete_error", key=key, error=str(exc))

    def exists(self, key: str) -> bool:
        try:
            return bool(self._client.exists(key))
        except RedisError as exc:
            logger.warning("redis_cache_exists_error", key=key, error=str(exc))
            return False

    def clear(self) -> None:
        try:
            logger.warning(
                "redis_cache_flush_all",
                url=settings.redis.url,
                note="This wipes the entire Redis DB — use only in isolated environments",
            )
            self._client.flushdb()
        except RedisError as exc:
            logger.warning("redis_cache_clear_error", error=str(exc))

