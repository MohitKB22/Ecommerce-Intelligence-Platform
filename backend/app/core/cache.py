"""Cache abstraction: Redis when available, in-process LRU+TTL otherwise.

Includes TTL, namespacing, JSON serialisation, single-flight (stampede
protection) and a fixed-window rate limiter that works on both backends.
"""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Protocol

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger("cache")


class CacheBackend(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, ttl: int) -> None: ...
    def delete(self, key: str) -> None: ...
    def delete_prefix(self, prefix: str) -> int: ...
    def incr(self, key: str, ttl: int) -> int: ...
    def ping(self) -> bool: ...
    @property
    def name(self) -> str: ...


class InMemoryCache:
    """Thread-safe LRU cache with per-key TTL. Suitable for single-process dev."""

    def __init__(self, max_entries: int = 20000):
        self._data: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._counters: dict[str, tuple[float, int]] = {}
        self._lock = threading.RLock()
        self._max = max_entries

    @property
    def name(self) -> str:
        return "memory"

    def _purge_locked(self) -> None:
        now = time.time()
        expired = [k for k, (exp, _) in self._data.items() if exp <= now]
        for k in expired:
            self._data.pop(k, None)
        while len(self._data) > self._max:
            self._data.popitem(last=False)

    def get(self, key: str) -> str | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= time.time():
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: str, ttl: int) -> None:
        with self._lock:
            self._data[key] = (time.time() + max(ttl, 1), value)
            self._data.move_to_end(key)
            self._purge_locked()

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def delete_prefix(self, prefix: str) -> int:
        with self._lock:
            keys = [k for k in self._data if k.startswith(prefix)]
            for k in keys:
                self._data.pop(k, None)
            return len(keys)

    def incr(self, key: str, ttl: int) -> int:
        with self._lock:
            now = time.time()
            exp, count = self._counters.get(key, (0.0, 0))
            if exp <= now:
                exp, count = now + ttl, 0
            count += 1
            self._counters[key] = (exp, count)
            if len(self._counters) > self._max:
                for k in [k for k, (e, _) in self._counters.items() if e <= now]:
                    self._counters.pop(k, None)
            return count

    def ping(self) -> bool:
        return True


class RedisCache:
    def __init__(self, url: str):
        import redis  # imported lazily so redis stays optional

        self._client = redis.Redis.from_url(
            url, decode_responses=True, socket_connect_timeout=1.5, socket_timeout=1.5, health_check_interval=30
        )

    @property
    def name(self) -> str:
        return "redis"

    def get(self, key: str) -> str | None:
        try:
            return self._client.get(key)
        except Exception as exc:
            logger.warning("redis_get_failed", key=key, error=str(exc))
            return None

    def set(self, key: str, value: str, ttl: int) -> None:
        try:
            self._client.setex(key, max(ttl, 1), value)
        except Exception as exc:
            logger.warning("redis_set_failed", key=key, error=str(exc))

    def delete(self, key: str) -> None:
        try:
            self._client.delete(key)
        except Exception as exc:
            logger.warning("redis_delete_failed", key=key, error=str(exc))

    def delete_prefix(self, prefix: str) -> int:
        deleted = 0
        try:
            for key in self._client.scan_iter(match=f"{prefix}*", count=500):
                self._client.delete(key)
                deleted += 1
        except Exception as exc:
            logger.warning("redis_delete_prefix_failed", prefix=prefix, error=str(exc))
        return deleted

    def incr(self, key: str, ttl: int) -> int:
        try:
            pipe = self._client.pipeline()
            pipe.incr(key)
            pipe.expire(key, ttl, nx=True)
            return int(pipe.execute()[0])
        except Exception as exc:
            logger.warning("redis_incr_failed", key=key, error=str(exc))
            return 0

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:
            return False


def _build_backend() -> CacheBackend:
    if settings.REDIS_URL:
        try:
            backend = RedisCache(settings.REDIS_URL)
            if backend.ping():
                logger.info("cache_backend_selected", backend="redis")
                return backend
            logger.warning("redis_unreachable_falling_back", url=settings.REDIS_URL)
        except Exception as exc:
            logger.warning("redis_init_failed_falling_back", error=str(exc))
    logger.info("cache_backend_selected", backend="memory")
    return InMemoryCache(settings.CACHE_MAX_ENTRIES)


class CacheService:
    """High-level JSON cache with namespacing and stampede protection."""

    def __init__(self, backend: CacheBackend | None = None, enabled: bool | None = None):
        self._backend = backend or _build_backend()
        self._enabled = settings.CACHE_ENABLED if enabled is None else enabled
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def key(namespace: str, *parts: Any) -> str:
        return "eci:" + namespace + ":" + ":".join(str(p) for p in parts)

    def get_json(self, key: str) -> Any | None:
        if not self._enabled:
            return None
        raw = self._backend.get(key)
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            self._backend.delete(key)
            return None

    def set_json(self, key: str, value: Any, ttl: int | None = None) -> None:
        if not self._enabled:
            return
        try:
            self._backend.set(key, json.dumps(value, default=str), ttl or settings.CACHE_DEFAULT_TTL)
        except (TypeError, ValueError) as exc:
            logger.warning("cache_serialize_failed", key=key, error=str(exc))

    def get_or_set(self, key: str, producer: Callable[[], Any], ttl: int | None = None) -> Any:
        """Cache-aside read with single-flight to avoid a stampede on miss."""
        cached = self.get_json(key)
        if cached is not None:
            return cached
        with self._locks_guard:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            cached = self.get_json(key)  # another thread may have filled it
            if cached is not None:
                return cached
            value = producer()
            self.set_json(key, value, ttl)
            return value
        # lock entries are intentionally retained: bounded by distinct hot keys

    def invalidate(self, key: str) -> None:
        self._backend.delete(key)

    def invalidate_namespace(self, namespace: str) -> int:
        return self._backend.delete_prefix("eci:" + namespace + ":")

    def rate_limit(self, identity: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        """Fixed-window limiter. Returns (allowed, current_count)."""
        bucket = int(time.time() // window_seconds)
        key = self.key("ratelimit", identity, bucket)
        count = self._backend.incr(key, window_seconds)
        if count == 0:  # backend failure -> fail open rather than block traffic
            return True, 0
        return count <= limit, count

    def ping(self) -> bool:
        return self._backend.ping()

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "backend": self.backend_name,
            "enabled": self._enabled,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


_cache: CacheService | None = None
_cache_lock = threading.Lock()


def get_cache() -> CacheService:
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                _cache = CacheService()
    return _cache


def reset_cache() -> None:
    global _cache
    with _cache_lock:
        _cache = None
