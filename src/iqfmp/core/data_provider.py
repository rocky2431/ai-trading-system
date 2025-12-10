"""Data provider for IQFMP - unified data loading from TimescaleDB.

This module provides a centralized interface for loading OHLCV data
from TimescaleDB with fallback to CSV files.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class DataProvider:
    """Unified data provider for factor computation and backtesting.

    Supports loading from:
    1. TimescaleDB (primary source)
    2. CSV files (fallback)
    3. Pre-loaded DataFrame
    """

    def __init__(
        self,
        session: Optional[AsyncSession] = None,
        fallback_csv_dir: Optional[Path] = None,
    ):
        """Initialize data provider.

        Args:
            session: Async database session for TimescaleDB
            fallback_csv_dir: Directory for CSV fallback files
        """
        self._session = session
        self._fallback_dir = fallback_csv_dir or Path(__file__).parent.parent.parent.parent / "data" / "sample"
        self._cache: dict[str, pd.DataFrame] = {}

    async def load_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        exchange: str = "binance",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Load OHLCV data from TimescaleDB with CSV fallback.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT" or "BTC/USDT")
            timeframe: Data timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            start_date: Start datetime
            end_date: End datetime
            exchange: Exchange name
            use_cache: Whether to use cached data

        Returns:
            DataFrame with timestamp, open, high, low, close, volume columns
        """
        cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key].copy()

        # Normalize symbol format
        symbol_db = symbol.replace("/", "")
        symbol_ccxt = symbol if "/" in symbol else self._format_ccxt_symbol(symbol)

        df = None

        # Try loading from TimescaleDB first
        if self._session:
            try:
                df = await self._load_from_db(
                    symbol=symbol_ccxt,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    exchange=exchange,
                )
                if df is not None and len(df) > 0:
                    logger.info(f"Loaded {len(df)} rows from TimescaleDB for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to load from TimescaleDB: {e}")

        # Fallback to CSV if DB failed or empty
        if df is None or len(df) == 0:
            df = self._load_from_csv(symbol_db, timeframe)
            if df is not None and len(df) > 0:
                logger.info(f"Loaded {len(df)} rows from CSV for {symbol}")

        if df is None or len(df) == 0:
            raise ValueError(f"No data available for {symbol} {timeframe}")

        # Filter by date range
        if start_date and "timestamp" in df.columns:
            df = df[df["timestamp"] >= start_date]
        if end_date and "timestamp" in df.columns:
            df = df[df["timestamp"] <= end_date]

        # Cache result
        if use_cache:
            self._cache[cache_key] = df.copy()

        return df

    async def _load_from_db(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        exchange: str,
    ) -> Optional[pd.DataFrame]:
        """Load data from TimescaleDB."""
        from iqfmp.db.models import OHLCVDataORM

        # Build query
        conditions = [
            OHLCVDataORM.symbol == symbol,
            OHLCVDataORM.timeframe == timeframe,
        ]

        if exchange:
            conditions.append(OHLCVDataORM.exchange == exchange)
        if start_date:
            conditions.append(OHLCVDataORM.timestamp >= start_date)
        if end_date:
            conditions.append(OHLCVDataORM.timestamp <= end_date)

        query = (
            select(OHLCVDataORM)
            .where(and_(*conditions))
            .order_by(OHLCVDataORM.timestamp.asc())
        )

        result = await self._session.execute(query)
        rows = result.scalars().all()

        if not rows:
            return None

        # Convert to DataFrame
        data = []
        for row in rows:
            data.append({
                "timestamp": row.timestamp,
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.volume,
                "symbol": row.symbol,
            })

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    def _load_from_csv(
        self,
        symbol: str,
        timeframe: str,
    ) -> Optional[pd.DataFrame]:
        """Load data from CSV fallback."""
        # Try different file naming conventions
        possible_names = [
            f"{symbol.lower()}_{timeframe}.csv",
            f"{symbol.lower()}_futures_daily.csv",
            f"{symbol.lower()}_daily.csv",
            f"{symbol.lower()}.csv",
            "eth_usdt_futures_daily.csv",  # Default sample
        ]

        for name in possible_names:
            csv_path = self._fallback_dir / name
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)

                    # Normalize column names
                    df.columns = df.columns.str.lower()

                    # Ensure timestamp column
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                    elif "date" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["date"])
                    elif "datetime" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["datetime"])

                    df = df.sort_values("timestamp").reset_index(drop=True)
                    return df

                except Exception as e:
                    logger.warning(f"Failed to load CSV {csv_path}: {e}")
                    continue

        return None

    def _format_ccxt_symbol(self, symbol: str) -> str:
        """Convert symbol to CCXT format (BTC/USDT)."""
        symbol = symbol.upper()
        if symbol.endswith("USDT"):
            return symbol[:-4] + "/USDT"
        elif symbol.endswith("BTC"):
            return symbol[:-3] + "/BTC"
        elif symbol.endswith("ETH"):
            return symbol[:-3] + "/ETH"
        return symbol

    def load_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a pre-loaded DataFrame.

        Args:
            df: Raw DataFrame with OHLCV data

        Returns:
            Processed DataFrame with standardized columns
        """
        df = df.copy()
        df.columns = df.columns.str.lower()

        # Ensure timestamp
        if "timestamp" not in df.columns:
            if "date" in df.columns:
                df["timestamp"] = pd.to_datetime(df["date"])
            elif "datetime" in df.columns:
                df["timestamp"] = pd.to_datetime(df["datetime"])
            elif "time" in df.columns:
                df["timestamp"] = pd.to_datetime(df["time"])
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def clear_cache(self):
        """Clear cached data."""
        self._cache.clear()


# Async helper for loading data without session
async def load_ohlcv_data(
    symbol: str,
    timeframe: str = "1d",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    session: Optional[AsyncSession] = None,
) -> pd.DataFrame:
    """Helper function to load OHLCV data.

    Args:
        symbol: Trading pair
        timeframe: Data timeframe
        start_date: Start datetime
        end_date: End datetime
        session: Optional database session

    Returns:
        DataFrame with OHLCV data
    """
    provider = DataProvider(session=session)
    return await provider.load_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )


def load_ohlcv_sync(
    symbol: str = "ETHUSDT",
    timeframe: str = "1d",
    csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Synchronous helper for loading OHLCV data from CSV.

    For use in non-async contexts (factor computation).

    Args:
        symbol: Trading pair
        timeframe: Data timeframe
        csv_path: Optional specific CSV path

    Returns:
        DataFrame with OHLCV data
    """
    provider = DataProvider()

    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path)
        return provider.load_dataframe(df)

    # Try to load from default locations
    df = provider._load_from_csv(symbol, timeframe)
    if df is not None:
        return df

    raise ValueError(f"No CSV data found for {symbol} {timeframe}")
