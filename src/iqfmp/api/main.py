"""FastAPI application entry point."""

import logging
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
from iqfmp.db.database import init_db, close_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup: Initialize ConfigService first to restore API keys from config
    from iqfmp.api.config.service import ConfigService
    config_service = ConfigService()  # This restores env vars from saved config
    print(f"Config initialized from {config_service._config_file}")

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
# =============================================================================
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter.

    For production, replace with Redis-based rate limiting.
    """

    def __init__(
        self,
        app: FastAPI,
        requests_per_minute: int = 60,
        requests_per_second: int = 10,
    ) -> None:
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_second
        self._minute_requests: dict[str, list[float]] = defaultdict(list)
        self._second_requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health check
        if request.url.path == "/health":
            return await call_next(request)

        # Get client IP (use X-Forwarded-For if behind proxy)
        client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
        if "," in client_ip:
            client_ip = client_ip.split(",")[0].strip()

        now = time.time()

        # Clean old entries and check per-second limit
        self._second_requests[client_ip] = [
            t for t in self._second_requests[client_ip] if now - t < 1
        ]
        if len(self._second_requests[client_ip]) >= self.requests_per_second:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please slow down."},
                headers={"Retry-After": "1"},
            )

        # Clean old entries and check per-minute limit
        self._minute_requests[client_ip] = [
            t for t in self._minute_requests[client_ip] if now - t < 60
        ]
        if len(self._minute_requests[client_ip]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again in a minute."},
                headers={"Retry-After": "60"},
            )

        # Record this request
        self._second_requests[client_ip].append(now)
        self._minute_requests[client_ip].append(now)

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

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
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
