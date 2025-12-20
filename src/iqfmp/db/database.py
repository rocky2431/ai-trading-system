"""Database connection and session management for TimescaleDB + Redis."""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import redis.asyncio as redis
from pydantic_settings import BaseSettings
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool


class DatabaseSettings(BaseSettings):
    """Database configuration from environment variables."""

    # PostgreSQL / TimescaleDB
    PGHOST: str = "localhost"
    PGPORT: int = 5433  # Use 5433 to avoid conflict with local PostgreSQL
    PGUSER: str = "iqfmp"
    PGPASSWORD: str = "iqfmp"  # Matches docker-compose.dev.yml
    PGDATABASE: str = "iqfmp"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0

    # Pool settings
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_ECHO: bool = False

    @property
    def database_url(self) -> str:
        """Get async PostgreSQL connection URL."""
        return (
            f"postgresql+asyncpg://{self.PGUSER}:{self.PGPASSWORD}"
            f"@{self.PGHOST}:{self.PGPORT}/{self.PGDATABASE}"
        )

    @property
    def sync_database_url(self) -> str:
        """Get sync PostgreSQL connection URL (for migrations)."""
        return (
            f"postgresql://{self.PGUSER}:{self.PGPASSWORD}"
            f"@{self.PGHOST}:{self.PGPORT}/{self.PGDATABASE}"
        )

    @property
    def redis_url(self) -> str:
        """Get Redis connection URL."""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    class Config:
        env_file = ".env"
        extra = "ignore"


# Global instances
_settings: Optional[DatabaseSettings] = None
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None
_redis_client: Optional[redis.Redis] = None


def _is_test_env() -> bool:
    """Return True when running under pytest or explicit test env.

    NOTE: Our CI/unit tests should not require a live TimescaleDB/Redis instance.
    """
    return bool(os.environ.get("PYTEST_CURRENT_TEST")) or os.environ.get("IQFMP_ENV") == "test"


def get_settings() -> DatabaseSettings:
    """Get database settings singleton."""
    global _settings
    if _settings is None:
        _settings = DatabaseSettings()
    return _settings


async def init_db() -> None:
    """Initialize database connections."""
    global _engine, _session_factory, _redis_client

    settings = get_settings()

    # Create async engine
    _engine = create_async_engine(
        settings.database_url,
        echo=settings.DB_ECHO,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
    )

    # Create session factory
    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    # Create Redis client
    _redis_client = redis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
    )

    # Test connections
    try:
        async with _engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        print(f"Connected to TimescaleDB: {settings.PGHOST}:{settings.PGPORT}/{settings.PGDATABASE}")
    except Exception as e:
        print(f"Warning: Failed to connect to TimescaleDB: {e}")

    try:
        await _redis_client.ping()
        print(f"Connected to Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    except Exception as e:
        print(f"Warning: Failed to connect to Redis: {e}")


async def close_db() -> None:
    """Close database connections."""
    global _engine, _redis_client

    if _engine:
        await _engine.dispose()
        _engine = None

    if _redis_client:
        await _redis_client.close()
        _redis_client = None


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session."""
    global _session_factory

    if _session_factory is None:
        await init_db()

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_redis_client() -> redis.Redis:
    """Get Redis client."""
    global _redis_client

    if _redis_client is None:
        settings = get_settings()
        _redis_client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )

    return _redis_client


# Dependency for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database session."""
    async with get_async_session() as session:
        yield session


async def get_redis() -> redis.Redis:
    """FastAPI dependency for Redis client."""
    return get_redis_client()


async def get_optional_db() -> AsyncGenerator[AsyncSession | None, None]:
    """FastAPI dependency for optional database session.

    In test/dev environments (or when DB is not reachable), this yields None instead
    of failing the whole request.
    """
    if _is_test_env():
        yield None
        return

    try:
        async with get_async_session() as session:
            yield session
    except Exception as e:
        print(f"Warning: Database unavailable, using in-memory fallback: {e}")
        yield None


async def get_optional_redis() -> Optional[redis.Redis]:
    """FastAPI dependency for optional Redis client."""
    if _is_test_env():
        return None

    try:
        client = get_redis_client()
        await client.ping()
        return client
    except Exception as e:
        print(f"Warning: Redis unavailable, using no-cache mode: {e}")
        return None
