"""FastAPI application entry point."""

import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from iqfmp import __version__
from iqfmp.api.auth.dependencies import get_current_user
from iqfmp.api.auth.router import router as auth_router
from iqfmp.api.backtest.router import router as backtest_router
from iqfmp.api.config.router import router as config_router
from iqfmp.api.data.router import router as data_router
from iqfmp.api.factors.router import router as factors_router
from iqfmp.api.pipeline.router import router as pipeline_router
from iqfmp.api.prompts.router import router as prompts_router
from iqfmp.api.research.router import metrics_router, router as research_router
from iqfmp.api.review.router import router as review_router
from iqfmp.api.rl.router import router as rl_router
from iqfmp.api.strategies.router import router as strategies_router
from iqfmp.api.system.router import router as system_router
from iqfmp.api.trading.router import router as trading_router
from iqfmp.db.database import init_db, close_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup: Initialize ConfigService first to restore API keys from config
    from iqfmp.api.config.service import ConfigService
    config_service = ConfigService()  # This restores env vars from saved config
    logger.info(f"Config initialized from {config_service._config_file}")

    # M2 FIX: Initialize Qlib for factor evaluation
    try:
        from iqfmp.core.qlib_init import ensure_qlib_initialized, is_qlib_initialized
        if ensure_qlib_initialized():
            logger.info("Qlib initialized successfully")
        else:
            logger.warning("Qlib initialization failed, factor evaluation may be limited")
    except Exception as e:
        logger.warning(f"Qlib initialization error: {e}, factor evaluation may be limited")

    # Initialize database connections
    await init_db()
    yield
    # Shutdown: Close database connections
    await close_db()


# =============================================================================
# P0 SECURITY: Rate Limiting Middleware
# Prevents API abuse and DoS attacks per 8.2 API Security requirements
# Uses Redis for distributed rate limiting across multiple instances
# =============================================================================
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Redis-backed rate limiter with in-memory fallback.

    Uses Redis sliding window counters for distributed rate limiting.
    Falls back to in-memory limiting if Redis is unavailable.
    """

    REDIS_PREFIX = "ratelimit:"

    def __init__(
        self,
        app: FastAPI,
        requests_per_minute: int = 60,
        requests_per_second: int = 10,
    ) -> None:
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_second
        # Fallback in-memory storage (used when Redis unavailable)
        self._minute_requests: dict[str, list[float]] = defaultdict(list)
        self._second_requests: dict[str, list[float]] = defaultdict(list)
        self._redis_available: bool | None = None

    async def _get_redis_client(self):
        """Get Redis client, caching availability status."""
        if self._redis_available is False:
            return None
        try:
            from iqfmp.db.database import get_redis_client
            client = get_redis_client()
            await client.ping()
            self._redis_available = True
            return client
        except Exception:
            self._redis_available = False
            logger.warning("Redis unavailable for rate limiting, using in-memory fallback")
            return None

    async def _check_rate_limit_redis(
        self, client_ip: str, redis_client
    ) -> tuple[bool, str, int]:
        """Check rate limits using Redis sliding window.

        Returns:
            (is_limited, message, retry_after_seconds)
        """
        now = int(time.time())

        # Per-second limit using Redis INCR with TTL
        second_key = f"{self.REDIS_PREFIX}sec:{client_ip}:{now}"
        try:
            count = await redis_client.incr(second_key)
            if count == 1:
                await redis_client.expire(second_key, 2)  # Expire after 2 seconds
            if count > self.requests_per_second:
                return True, "Rate limit exceeded. Please slow down.", 1
        except Exception as e:
            logger.warning(f"Redis rate limit check failed (second): {e}")

        # Per-minute limit using sorted set sliding window
        minute_key = f"{self.REDIS_PREFIX}min:{client_ip}"
        window_start = now - 60
        try:
            pipe = redis_client.pipeline()
            # Remove old entries
            pipe.zremrangebyscore(minute_key, 0, window_start)
            # Add current request
            pipe.zadd(minute_key, {str(now): now})
            # Count requests in window
            pipe.zcard(minute_key)
            # Set expiry
            pipe.expire(minute_key, 70)
            results = await pipe.execute()
            request_count = results[2]

            if request_count > self.requests_per_minute:
                return True, "Rate limit exceeded. Try again in a minute.", 60
        except Exception as e:
            logger.warning(f"Redis rate limit check failed (minute): {e}")

        return False, "", 0

    async def _check_rate_limit_memory(self, client_ip: str) -> tuple[bool, str, int]:
        """Fallback in-memory rate limiting."""
        now = time.time()

        # Per-second limit
        self._second_requests[client_ip] = [
            t for t in self._second_requests[client_ip] if now - t < 1
        ]
        if len(self._second_requests[client_ip]) >= self.requests_per_second:
            return True, "Rate limit exceeded. Please slow down.", 1

        # Per-minute limit
        self._minute_requests[client_ip] = [
            t for t in self._minute_requests[client_ip] if now - t < 60
        ]
        if len(self._minute_requests[client_ip]) >= self.requests_per_minute:
            return True, "Rate limit exceeded. Try again in a minute.", 60

        # Record request
        self._second_requests[client_ip].append(now)
        self._minute_requests[client_ip].append(now)

        return False, "", 0

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health check
        if request.url.path == "/health":
            return await call_next(request)

        # Get client IP (use X-Forwarded-For if behind proxy)
        client_ip = request.headers.get(
            "X-Forwarded-For", request.client.host if request.client else "unknown"
        )
        if "," in client_ip:
            client_ip = client_ip.split(",")[0].strip()

        # Try Redis first, fall back to in-memory
        redis_client = await self._get_redis_client()
        if redis_client:
            is_limited, message, retry_after = await self._check_rate_limit_redis(
                client_ip, redis_client
            )
        else:
            is_limited, message, retry_after = await self._check_rate_limit_memory(
                client_ip
            )

        if is_limited:
            return JSONResponse(
                status_code=429,
                content={"detail": message},
                headers={"Retry-After": str(retry_after)},
            )

        response = await call_next(request)
        return response


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="IQFMP API",
        description="Intelligent Quantitative Factor Mining Platform",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS middleware - configure via environment for security
    # CORS_ORIGINS: comma-separated list of allowed origins
    # Default: localhost development origins only
    cors_origins_str = os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
    )
    cors_origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]

    # In production, CORS_ORIGINS must be explicitly set to allowed domains
    # Never use allow_origins=["*"] with allow_credentials=True (security vulnerability)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
    )

    # P0 SECURITY: Rate limiting middleware per 8.2 API Security
    # 60 requests/minute, 10 requests/second per client IP
    app.add_middleware(RateLimitMiddleware, requests_per_minute=60, requests_per_second=10)

    # Include routers
    # Auth router - NO authentication required (login/register endpoints)
    app.include_router(auth_router, prefix="/api/v1/auth")

    # =========================================================================
    # P0-2 SECURITY: All protected routes MUST have JWT authentication
    # This ensures no API endpoint can be accessed without valid credentials
    # =========================================================================
    auth_dependency = [Depends(get_current_user)]

    app.include_router(
        backtest_router, prefix="/api/v1/backtest", dependencies=auth_dependency
    )
    app.include_router(
        config_router, prefix="/api/v1/config", dependencies=auth_dependency
    )
    app.include_router(
        data_router, prefix="/api/v1/data", dependencies=auth_dependency
    )
    app.include_router(
        factors_router, prefix="/api/v1/factors", dependencies=auth_dependency
    )
    app.include_router(
        pipeline_router, prefix="/api/v1/pipeline", dependencies=auth_dependency
    )
    app.include_router(
        prompts_router, prefix="/api/v1/prompts", dependencies=auth_dependency
    )
    app.include_router(
        research_router, prefix="/api/v1/research", dependencies=auth_dependency
    )
    app.include_router(
        review_router, prefix="/api/v1/review", dependencies=auth_dependency
    )
    app.include_router(
        rl_router, prefix="/api/v1/rl", dependencies=auth_dependency
    )
    app.include_router(
        metrics_router, prefix="/api/v1/metrics", dependencies=auth_dependency
    )
    app.include_router(
        strategies_router, prefix="/api/v1/strategies", dependencies=auth_dependency
    )
    app.include_router(
        system_router, prefix="/api/v1/system", dependencies=auth_dependency
    )
    app.include_router(
        trading_router, prefix="/api/v1/trading", dependencies=auth_dependency
    )

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "version": __version__}

    return app


app = create_app()


def main() -> None:
    """Run the application."""
    import uvicorn

    uvicorn.run(
        "iqfmp.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
