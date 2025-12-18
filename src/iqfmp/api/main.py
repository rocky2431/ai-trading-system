"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from iqfmp import __version__
from iqfmp.api.auth.router import router as auth_router
from iqfmp.api.backtest.router import router as backtest_router
from iqfmp.api.config.router import router as config_router
from iqfmp.api.data.router import router as data_router
from iqfmp.api.factors.router import router as factors_router
from iqfmp.api.pipeline.router import router as pipeline_router
from iqfmp.api.research.router import metrics_router, router as research_router
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

    # Include routers
    app.include_router(auth_router, prefix="/api/v1/auth")
    app.include_router(backtest_router, prefix="/api/v1/backtest")
    app.include_router(config_router, prefix="/api/v1/config")
    app.include_router(data_router, prefix="/api/v1/data")
    app.include_router(factors_router, prefix="/api/v1/factors")
    app.include_router(pipeline_router, prefix="/api/v1/pipeline")
    app.include_router(research_router, prefix="/api/v1/research")
    app.include_router(metrics_router, prefix="/api/v1/metrics")
    app.include_router(strategies_router, prefix="/api/v1/strategies")
    app.include_router(system_router, prefix="/api/v1/system")

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
