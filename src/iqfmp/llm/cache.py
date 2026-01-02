"""LLM caching & chat history backends.

This module provides pluggable backends for:
- LLM request/response caching (to reduce cost/latency)
- Chat session history persistence (for multi-turn workflows)
- Prompt-level persistent caching (RD-Agent style)

Architecture:
- L1: Redis (hot cache, TTL auto-expire, ~1ms latency)
- L2: PostgreSQL (persistent, cross-restart, ~10ms latency)

Design goals:
- Reuse existing infrastructure (Redis + TimescaleDB)
- MD5-based cache key generation for prompt deduplication
- Token savings tracking and statistics
- Two-tier caching for optimal performance
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)

import redis.asyncio as aioredis
from sqlalchemy import select, delete, func, update

from iqfmp.db.database import get_redis_client, get_async_session
from iqfmp.db.models import PromptCacheORM


class CacheBackend(Protocol):
    """Protocol for a simple TTL cache."""

    def get(self, key: str) -> Optional[dict[str, Any]]: ...

    def set(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None: ...

    def delete(self, key: str) -> None: ...


class ChatHistoryBackend(Protocol):
    """Protocol for chat history storage."""

    def get_messages(self, conversation_id: str) -> list[dict[str, str]]: ...

    def set_messages(self, conversation_id: str, messages: list[dict[str, str]]) -> None: ...


@dataclass
class CacheEntry:
    value: dict[str, Any]
    expires_at: float


class MemoryCache(CacheBackend):
    """In-memory TTL cache (process-local)."""

    def __init__(self) -> None:
        self._items: dict[str, CacheEntry] = {}

    def get(self, key: str) -> Optional[dict[str, Any]]:
        entry = self._items.get(key)
        if entry is None:
            return None
        if time.time() >= entry.expires_at:
            self._items.pop(key, None)
            return None
        return entry.value

    def set(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        self._items[key] = CacheEntry(
            value=value,
            expires_at=time.time() + max(0, ttl_seconds),
        )

    def delete(self, key: str) -> None:
        self._items.pop(key, None)


class MemoryChatHistory(ChatHistoryBackend):
    """In-memory chat history (process-local)."""

    def __init__(self) -> None:
        self._messages: dict[str, list[dict[str, str]]] = {}

    def get_messages(self, conversation_id: str) -> list[dict[str, str]]:
        return list(self._messages.get(conversation_id, []))

    def set_messages(self, conversation_id: str, messages: list[dict[str, str]]) -> None:
        self._messages[conversation_id] = list(messages)


@dataclass
class PromptCacheStats:
    """Statistics for prompt cache."""

    total_entries: int
    total_tokens_saved: int
    total_hits: int
    session_hits: int
    session_misses: int
    hit_rate: float
    l1_hits: int  # Redis hits
    l2_hits: int  # PostgreSQL hits


class PromptCache:
    """Two-tier prompt cache using Redis (L1) + PostgreSQL (L2).

    Provides persistent caching for LLM responses with:
    - L1 Redis: Hot cache with auto-TTL (fast, ~1ms)
    - L2 PostgreSQL: Persistent storage (slower, ~10ms)
    - MD5-based cache key generation (messages + model + seed + temperature)
    - Token savings tracking
    - Access statistics

    Example:
        cache = PromptCache()

        # Check cache (async)
        messages = [{"role": "user", "content": "Hello"}]
        cached = await cache.get(messages, model="gpt-4")
        if cached:
            return cached

        # Call LLM and cache result
        response = await llm.complete(...)
        await cache.set(messages, model="gpt-4", response=response, tokens_saved=1000)

        # Get stats
        stats = await cache.get_stats()
        print(f"Cache hit rate: {stats.hit_rate:.2%}")
    """

    # Redis key prefix
    REDIS_PREFIX = "llm:cache:"
    # Default TTL for Redis (1 hour)
    DEFAULT_REDIS_TTL = 3600
    # Maximum age for PostgreSQL entries (30 days)
    MAX_PG_AGE_DAYS = 30
    # Maximum entries to keep in PostgreSQL
    MAX_PG_ENTRIES = 10000

    def __init__(
        self,
        redis_ttl: int = DEFAULT_REDIS_TTL,
        max_pg_entries: int = MAX_PG_ENTRIES,
        max_pg_age_days: int = MAX_PG_AGE_DAYS,
    ):
        """Initialize prompt cache.

        Args:
            redis_ttl: TTL for Redis entries in seconds (default: 1 hour)
            max_pg_entries: Maximum entries to keep in PostgreSQL
            max_pg_age_days: Maximum age of PostgreSQL entries in days
        """
        self.redis_ttl = redis_ttl
        self.max_pg_entries = max_pg_entries
        self.max_pg_age_days = max_pg_age_days

        # Session-level statistics
        self._session_hits = 0
        self._session_misses = 0
        self._l1_hits = 0
        self._l2_hits = 0

    def _generate_key(
        self,
        messages: list[dict[str, Any]],
        model: str,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate cache key from messages and parameters.

        Uses MD5 hash of serialized messages + model + seed + temperature.

        Args:
            messages: Chat messages
            model: Model identifier
            seed: Random seed (affects output)
            temperature: Temperature (affects output)

        Returns:
            MD5 hash key
        """
        content = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        key_parts = [
            content,
            f"model={model}",
            f"seed={seed}",
            f"temp={temperature}",
        ]
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    async def get(
        self,
        messages: list[dict[str, Any]],
        model: str,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        """Get cached response for messages.

        Checks L1 (Redis) first, then L2 (PostgreSQL).
        On L2 hit, promotes entry back to L1.

        Args:
            messages: Chat messages
            model: Model identifier
            seed: Random seed
            temperature: Temperature

        Returns:
            Cached response string or None if not found
        """
        key = self._generate_key(messages, model, seed, temperature)
        redis_key = f"{self.REDIS_PREFIX}{key}"

        # L1: Check Redis first
        try:
            redis_client = get_redis_client()
            cached_value = await redis_client.get(redis_key)
            if cached_value:
                self._session_hits += 1
                self._l1_hits += 1
                return cached_value
        except Exception as e:
            # Redis unavailable, continue to L2
            logger.warning(f"L1 Redis cache get failed for key {key[:8]}...: {e}")

        # L2: Check PostgreSQL
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    select(PromptCacheORM.value).where(PromptCacheORM.key == key)
                )
                row = result.scalar_one_or_none()

                if row:
                    # Update access statistics
                    await session.execute(
                        update(PromptCacheORM)
                        .where(PromptCacheORM.key == key)
                        .values(
                            accessed_at=func.now(),
                            access_count=PromptCacheORM.access_count + 1
                        )
                    )

                    # Promote to L1 Redis
                    try:
                        redis_client = get_redis_client()
                        await redis_client.setex(redis_key, self.redis_ttl, row)
                    except Exception as e:
                        logger.warning(f"Failed to promote cache entry to L1 Redis: {e}")

                    self._session_hits += 1
                    self._l2_hits += 1
                    return row
        except Exception as e:
            logger.warning(f"L2 PostgreSQL cache get failed for key {key[:8]}...: {e}")

        self._session_misses += 1
        return None

    async def set(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response: str,
        tokens_saved: int = 0,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """Cache a response.

        Writes to both L1 (Redis) and L2 (PostgreSQL).

        Args:
            messages: Chat messages
            model: Model identifier
            response: Response to cache
            tokens_saved: Estimated tokens saved by caching
            seed: Random seed
            temperature: Temperature
        """
        key = self._generate_key(messages, model, seed, temperature)
        redis_key = f"{self.REDIS_PREFIX}{key}"

        # L1: Write to Redis
        try:
            redis_client = get_redis_client()
            await redis_client.setex(redis_key, self.redis_ttl, response)
        except Exception as e:
            logger.warning(f"L1 Redis cache set failed for key {key[:8]}...: {e}")

        # L2: Write to PostgreSQL
        try:
            async with get_async_session() as session:
                # Upsert: INSERT or UPDATE on conflict
                existing = await session.get(PromptCacheORM, key)
                if existing:
                    existing.value = response
                    existing.tokens_saved = tokens_saved
                    existing.accessed_at = func.now()
                    existing.access_count += 1
                else:
                    entry = PromptCacheORM(
                        key=key,
                        value=response,
                        model=model,
                        tokens_saved=tokens_saved,
                    )
                    session.add(entry)
        except Exception as e:
            logger.warning(f"L2 PostgreSQL cache set failed for key {key[:8]}...: {e}")

    async def get_stats(self) -> PromptCacheStats:
        """Get cache statistics.

        Returns:
            PromptCacheStats with hit rate, total entries, tokens saved, etc.
        """
        total_entries = 0
        total_tokens_saved = 0
        total_hits = 0

        try:
            async with get_async_session() as session:
                result = await session.execute(
                    select(
                        func.count(PromptCacheORM.key),
                        func.coalesce(func.sum(PromptCacheORM.tokens_saved), 0),
                        func.coalesce(func.sum(PromptCacheORM.access_count), 0),
                    )
                )
                row = result.one()
                total_entries = row[0] or 0
                total_tokens_saved = row[1] or 0
                total_hits = row[2] or 0
        except Exception as e:
            logger.warning(f"Failed to get cache stats from PostgreSQL: {e}")

        # Calculate session hit rate
        total_requests = self._session_hits + self._session_misses
        hit_rate = self._session_hits / total_requests if total_requests > 0 else 0.0

        return PromptCacheStats(
            total_entries=total_entries,
            total_tokens_saved=total_tokens_saved,
            total_hits=total_hits,
            session_hits=self._session_hits,
            session_misses=self._session_misses,
            hit_rate=hit_rate,
            l1_hits=self._l1_hits,
            l2_hits=self._l2_hits,
        )

    async def cleanup(
        self,
        max_age_days: Optional[int] = None,
        max_entries: Optional[int] = None,
    ) -> int:
        """Clean up old cache entries from PostgreSQL.

        Args:
            max_age_days: Remove entries older than this. Defaults to init value.
            max_entries: Keep at most this many entries. Defaults to init value.

        Returns:
            Number of entries removed
        """
        max_age_days = max_age_days or self.max_pg_age_days
        max_entries = max_entries or self.max_pg_entries

        removed = 0
        try:
            async with get_async_session() as session:
                # Count before cleanup
                result = await session.execute(
                    select(func.count(PromptCacheORM.key))
                )
                before_count = result.scalar() or 0

                # Remove old entries (older than max_age_days)
                from datetime import datetime, timedelta, timezone
                cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
                await session.execute(
                    delete(PromptCacheORM).where(PromptCacheORM.accessed_at < cutoff)
                )

                # Keep only max_entries most recent (subquery approach)
                if before_count > max_entries:
                    # Get keys to keep (most recently accessed)
                    subquery = (
                        select(PromptCacheORM.key)
                        .order_by(PromptCacheORM.accessed_at.desc())
                        .limit(max_entries)
                    )
                    await session.execute(
                        delete(PromptCacheORM).where(
                            PromptCacheORM.key.notin_(subquery)
                        )
                    )

                # Count after cleanup
                result = await session.execute(
                    select(func.count(PromptCacheORM.key))
                )
                after_count = result.scalar() or 0
                removed = before_count - after_count
        except Exception as e:
            logger.warning(f"Failed to cleanup cache entries: {e}")

        return removed

    async def clear(self) -> None:
        """Clear all cache entries from both L1 and L2."""
        # Clear Redis (L1)
        try:
            redis_client = get_redis_client()
            # Use SCAN to find all matching keys
            cursor = 0
            while True:
                cursor, keys = await redis_client.scan(
                    cursor, match=f"{self.REDIS_PREFIX}*", count=100
                )
                if keys:
                    await redis_client.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.warning(f"Failed to clear L1 Redis cache: {e}")

        # Clear PostgreSQL (L2)
        try:
            async with get_async_session() as session:
                await session.execute(delete(PromptCacheORM))
        except Exception as e:
            logger.warning(f"Failed to clear L2 PostgreSQL cache: {e}")

        # Reset session stats
        self._session_hits = 0
        self._session_misses = 0
        self._l1_hits = 0
        self._l2_hits = 0

    # Sync wrappers for backward compatibility
    def get_sync(
        self,
        messages: list[dict[str, Any]],
        model: str,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        """Synchronous wrapper for get()."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context, create a new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.get(messages, model, seed, temperature)
                    )
                    return future.result(timeout=5.0)
            else:
                return loop.run_until_complete(
                    self.get(messages, model, seed, temperature)
                )
        except Exception as e:
            logger.warning(f"Sync cache get failed: {e}")
            return None

    def set_sync(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response: str,
        tokens_saved: int = 0,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """Synchronous wrapper for set()."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.set(messages, model, response, tokens_saved, seed, temperature)
                    )
                    future.result(timeout=5.0)
            else:
                loop.run_until_complete(
                    self.set(messages, model, response, tokens_saved, seed, temperature)
                )
        except Exception as e:
            logger.warning(f"Sync cache set failed: {e}")

    def get_stats_sync(self) -> PromptCacheStats:
        """Synchronous wrapper for get_stats()."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.get_stats())
                    return future.result(timeout=5.0)
            else:
                return loop.run_until_complete(self.get_stats())
        except Exception as e:
            logger.warning(f"Sync get_stats failed: {e}")
            return PromptCacheStats(
                total_entries=0,
                total_tokens_saved=0,
                total_hits=0,
                session_hits=self._session_hits,
                session_misses=self._session_misses,
                hit_rate=0.0,
                l1_hits=self._l1_hits,
                l2_hits=self._l2_hits,
            )


# Global prompt cache instance
_prompt_cache: Optional[PromptCache] = None


def get_prompt_cache() -> PromptCache:
    """Get global prompt cache instance (singleton)."""
    global _prompt_cache
    if _prompt_cache is None:
        _prompt_cache = PromptCache()
    return _prompt_cache
