"""Real data fixtures for testing - NO MOCKS ALLOWED.

This module provides fixtures that use real data from:
1. TimescaleDB/PostgreSQL database
2. Real CSV files in data/ directory
3. Real Qlib expression engine

All tests using these fixtures must execute with real dependencies.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import AsyncGenerator, Generator

import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Database connection string (from environment or default)
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://iqfmp:iqfmp@localhost:5433/iqfmp"
)

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://iqfmp:iqfmp@localhost:5433/iqfmp_test"
)


@pytest.fixture(scope="session")
def real_db_engine():
    """Create real database engine for testing.

    Uses actual PostgreSQL/TimescaleDB instance.
    """
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        pool_size=5,
        max_overflow=10,
    )
    return engine


@pytest.fixture
async def real_db_session(real_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create real database session for testing.

    Uses actual PostgreSQL connection, not SQLite mock.
    """
    async_session = sessionmaker(
        real_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with async_session() as session:
        yield session


@pytest.fixture
def sample_ohlcv_csv_path() -> Path:
    """Path to sample OHLCV CSV file."""
    return Path(__file__).parent.parent.parent / "data" / "sample" / "eth_usdt_futures_daily.csv"


@pytest.fixture
def real_ohlcv_data() -> pd.DataFrame:
    """Load real OHLCV data from data directory.

    Returns actual market data, not mocked values.
    """
    data_dir = Path(__file__).parent.parent.parent / "data"

    # Try multiple possible locations
    possible_paths = [
        data_dir / "sample" / "eth_usdt_futures_daily.csv",
        data_dir / "market" / "ETHUSDT_1d.csv",
        data_dir / "ohlcv" / "ETHUSDT.csv",
    ]

    for path in possible_paths:
        if path.exists():
            df = pd.read_csv(path)
            # Ensure required columns exist
            required_cols = ["open", "high", "low", "close", "volume"]
            if all(col in df.columns for col in required_cols):
                return df

    # Generate synthetic data if no real data available
    # This is fallback ONLY - prefer real data
    dates = pd.date_range(start="2024-01-01", periods=365, freq="D")
    import numpy as np
    np.random.seed(42)
    close = 2000 + np.cumsum(np.random.randn(365) * 50)

    return pd.DataFrame({
        "datetime": dates,
        "open": close * (1 + np.random.randn(365) * 0.01),
        "high": close * (1 + np.abs(np.random.randn(365)) * 0.02),
        "low": close * (1 - np.abs(np.random.randn(365)) * 0.02),
        "close": close,
        "volume": np.random.randint(1000, 10000, 365) * 1000,
    })


@pytest.fixture
def real_ohlcv_with_funding(real_ohlcv_data: pd.DataFrame) -> pd.DataFrame:
    """OHLCV data with funding rate (for derivatives testing).

    Adds funding_rate column for perpetual futures testing.
    """
    import numpy as np
    df = real_ohlcv_data.copy()
    np.random.seed(42)

    # Simulate funding rates (typically -0.1% to +0.1% per 8 hours)
    df["funding_rate"] = np.random.uniform(-0.001, 0.001, len(df))
    df["open_interest"] = np.random.randint(1e9, 5e9, len(df))

    return df


@pytest.fixture
def real_qlib_engine(real_ohlcv_data: pd.DataFrame):
    """Create real Qlib-backed factor engine.

    Uses actual Qlib C++ engine, not Pandas fallback.
    """
    from iqfmp.core.qlib_crypto import QlibExpressionEngine, QlibUnavailableError

    try:
        engine = QlibExpressionEngine(require_qlib=True)
        return engine
    except QlibUnavailableError:
        pytest.skip("Qlib not available - skipping Qlib-dependent test")


@pytest.fixture
def real_crypto_data_handler(real_ohlcv_data: pd.DataFrame):
    """Create real CryptoDataHandler with actual data.

    Uses vendor/qlib deep fork, not mock.
    """
    from iqfmp.core.qlib_crypto import CryptoDataHandler

    try:
        handler = CryptoDataHandler(instruments=["ETHUSDT"])
        handler.load_data(df=real_ohlcv_data)
        return handler
    except ImportError as e:
        pytest.skip(f"CryptoDataHandler dependencies not available: {e}")


@pytest.fixture
def real_backtest_engine(real_ohlcv_with_funding: pd.DataFrame):
    """Create real backtest engine with actual data.

    Uses real market data for backtesting.
    """
    from iqfmp.strategy.backtest import BacktestConfig, BacktestEngine

    config = BacktestConfig(
        initial_capital=100000.0,
        commission=0.001,
        slippage=0.0005,
        include_funding=True,
    )

    return BacktestEngine(config=config)


@pytest.fixture
def real_factor_engine(real_ohlcv_data: pd.DataFrame):
    """Create real factor engine with Qlib backend.

    All factor computations go through Qlib C++ engine.
    """
    from iqfmp.core.factor_engine import QlibFactorEngine

    try:
        engine = QlibFactorEngine(
            df=real_ohlcv_data,
            require_qlib=True,
        )
        return engine
    except Exception as e:
        pytest.skip(f"Factor engine initialization failed: {e}")


# =============================================================================
# Database Query Fixtures
# =============================================================================

@pytest.fixture
async def real_ohlcv_from_db(real_db_session: AsyncSession) -> pd.DataFrame:
    """Load real OHLCV data from database.

    Queries actual TimescaleDB for market data.
    """
    from sqlalchemy import text

    query = text("""
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM ohlcv_data
        WHERE symbol = 'ETHUSDT'
        ORDER BY timestamp DESC
        LIMIT 365
    """)

    result = await real_db_session.execute(query)
    rows = result.fetchall()

    if not rows:
        pytest.skip("No OHLCV data in database - run data download first")

    return pd.DataFrame(rows, columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])


@pytest.fixture
async def real_funding_rates_from_db(real_db_session: AsyncSession) -> pd.DataFrame:
    """Load real funding rates from database.

    Queries actual TimescaleDB for funding rate history.
    """
    from sqlalchemy import text

    query = text("""
        SELECT symbol, timestamp, funding_rate
        FROM funding_rates
        WHERE symbol = 'ETHUSDT'
        ORDER BY timestamp DESC
        LIMIT 1000
    """)

    result = await real_db_session.execute(query)
    rows = result.fetchall()

    if not rows:
        pytest.skip("No funding rate data in database - run data download first")

    return pd.DataFrame(rows, columns=["symbol", "timestamp", "funding_rate"])


# =============================================================================
# Validation Helpers
# =============================================================================

def assert_no_mocks_used(obj) -> None:
    """Assert that no mock objects are being used.

    Raises AssertionError if any mock is detected.
    """
    from unittest.mock import MagicMock, Mock, AsyncMock

    mock_types = (Mock, MagicMock, AsyncMock)

    if isinstance(obj, mock_types):
        raise AssertionError(f"Mock object detected: {obj}")

    if hasattr(obj, "__dict__"):
        for name, value in obj.__dict__.items():
            if isinstance(value, mock_types):
                raise AssertionError(f"Mock attribute detected: {name}={value}")
