"""Real CCXT data downloader for IQFMP.

This module provides actual market data downloading from cryptocurrency exchanges
using the CCXT library. It handles rate limiting, pagination, and database storage.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Callable, Any

import ccxt.async_support as ccxt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from iqfmp.db.models import OHLCVDataORM, DataDownloadTaskORM, SymbolInfoORM

logger = logging.getLogger(__name__)


# Timeframe mapping for CCXT
TIMEFRAME_MAPPING = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

# Milliseconds per timeframe
TIMEFRAME_MS = {
    "1m": 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}


class CCXTDownloader:
    """Real market data downloader using CCXT."""

    def __init__(
        self,
        exchange_id: str = "binance",
        rate_limit_delay: float = 0.5,
    ):
        """Initialize downloader.

        Args:
            exchange_id: Exchange identifier (binance, okx, bybit, gate)
            rate_limit_delay: Delay between API calls in seconds
        """
        self.exchange_id = exchange_id
        self.rate_limit_delay = rate_limit_delay
        self._exchange: Optional[ccxt.Exchange] = None

    async def _get_exchange(self) -> ccxt.Exchange:
        """Get or create exchange instance."""
        if self._exchange is None:
            exchange_class = getattr(ccxt, self.exchange_id)
            self._exchange = exchange_class({
                "enableRateLimit": True,
                "options": {
                    "defaultType": "spot",
                }
            })
        return self._exchange

    async def close(self):
        """Close exchange connection."""
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[int] = None,
        limit: int = 1000,
    ) -> list[list]:
        """Fetch OHLCV data from exchange.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Timeframe (e.g., "1h")
            since: Start timestamp in milliseconds
            limit: Number of candles to fetch

        Returns:
            List of [timestamp, open, high, low, close, volume]
        """
        exchange = await self._get_exchange()

        try:
            ohlcv = await exchange.fetch_ohlcv(
                symbol,
                timeframe=TIMEFRAME_MAPPING.get(timeframe, timeframe),
                since=since,
                limit=limit,
            )
            return ohlcv
        except Exception as e:
            logger.error(f"Error fetching OHLCV: {e}")
            raise

    async def download_historical(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        session: AsyncSession,
        progress_callback: Optional[Callable[[float, int], Any]] = None,
        batch_size: int = 1000,
    ) -> int:
        """Download historical data and store in database.

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            start_date: Start datetime
            end_date: End datetime
            session: Database session
            progress_callback: Optional callback(progress_pct, rows_downloaded)
            batch_size: Number of candles per API request

        Returns:
            Total rows downloaded
        """
        symbol = symbol.upper()
        if "/" not in symbol:
            if symbol.endswith("USDT"):
                symbol = symbol[:-4] + "/USDT"
            elif symbol.endswith("BTC"):
                symbol = symbol[:-3] + "/BTC"

        # Convert to timestamps
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)

        timeframe_ms = TIMEFRAME_MS.get(timeframe, 60 * 60 * 1000)
        total_expected = (end_ts - start_ts) // timeframe_ms

        total_rows = 0
        current_ts = start_ts

        logger.info(f"Downloading {symbol} {timeframe} from {start_date} to {end_date}")

        while current_ts < end_ts:
            try:
                # Fetch OHLCV data
                ohlcv = await self.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_ts,
                    limit=batch_size,
                )

                if not ohlcv:
                    logger.warning(f"No data returned for {symbol} at {current_ts}")
                    # Move forward by batch_size * timeframe_ms
                    current_ts += batch_size * timeframe_ms
                    continue

                # Filter to only include data within range
                ohlcv = [bar for bar in ohlcv if start_ts <= bar[0] <= end_ts]

                if not ohlcv:
                    current_ts += batch_size * timeframe_ms
                    continue

                # Store in database
                for bar in ohlcv:
                    timestamp = datetime.fromtimestamp(bar[0] / 1000, tz=timezone.utc)

                    # Check if exists
                    existing = await session.execute(
                        select(OHLCVDataORM).where(
                            OHLCVDataORM.symbol == symbol,
                            OHLCVDataORM.timeframe == timeframe,
                            OHLCVDataORM.timestamp == timestamp,
                        )
                    )
                    if existing.scalar_one_or_none():
                        continue

                    # Insert new record
                    record = OHLCVDataORM(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=timestamp,
                        open=float(bar[1]),
                        high=float(bar[2]),
                        low=float(bar[3]),
                        close=float(bar[4]),
                        volume=float(bar[5]) if len(bar) > 5 else 0.0,
                        exchange=self.exchange_id,
                    )
                    session.add(record)
                    total_rows += 1

                # Commit batch
                await session.commit()

                # Update progress
                if progress_callback and total_expected > 0:
                    progress = min(100.0, (total_rows / total_expected) * 100)
                    await progress_callback(progress, total_rows)

                # Move to next batch
                last_ts = ohlcv[-1][0]
                current_ts = last_ts + timeframe_ms

                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(f"Error downloading data: {e}")
                await asyncio.sleep(1.0)
                continue

        logger.info(f"Download complete: {total_rows} rows for {symbol} {timeframe}")
        return total_rows


async def execute_download_task(
    task_id: str,
    session: AsyncSession,
) -> bool:
    """Execute a download task with real data fetching.

    Args:
        task_id: Download task ID
        session: Database session

    Returns:
        True if successful
    """
    # Get task
    result = await session.execute(
        select(DataDownloadTaskORM).where(DataDownloadTaskORM.id == task_id)
    )
    task = result.scalar_one_or_none()

    if not task:
        logger.error(f"Task {task_id} not found")
        return False

    if task.status not in ["pending", "running"]:
        logger.warning(f"Task {task_id} status is {task.status}, skipping")
        return False

    # Update task status
    task.status = "running"
    task.started_at = datetime.now(timezone.utc)
    await session.commit()

    try:
        # Create downloader
        downloader = CCXTDownloader(exchange_id=task.exchange)

        # Progress callback
        async def update_progress(progress: float, rows: int):
            task.progress = progress
            task.rows_downloaded = rows
            await session.commit()

        # Execute download
        total_rows = await downloader.download_historical(
            symbol=task.symbol,
            timeframe=task.timeframe,
            start_date=task.start_date,
            end_date=task.end_date,
            session=session,
            progress_callback=update_progress,
        )

        # Update task completion
        task.status = "completed"
        task.progress = 100.0
        task.rows_downloaded = total_rows
        task.completed_at = datetime.now(timezone.utc)
        await session.commit()

        # Update symbol info
        await _update_symbol_info(session, task.symbol, task.timeframe, task.exchange)

        await downloader.close()

        logger.info(f"Task {task_id} completed: {total_rows} rows")
        return True

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        task.status = "failed"
        task.error_message = str(e)
        task.completed_at = datetime.now(timezone.utc)
        await session.commit()
        return False


async def _update_symbol_info(
    session: AsyncSession,
    symbol: str,
    timeframe: str,
    exchange: str,
):
    """Update symbol info after download."""
    # Get or create symbol info
    result = await session.execute(
        select(SymbolInfoORM).where(SymbolInfoORM.symbol == symbol)
    )
    symbol_info = result.scalar_one_or_none()

    if not symbol_info:
        # Parse symbol
        parts = symbol.split("/")
        base = parts[0] if len(parts) > 0 else symbol
        quote = parts[1] if len(parts) > 1 else "USDT"

        symbol_info = SymbolInfoORM(
            symbol=symbol,
            exchange=exchange,
            base_asset=base,
            quote_asset=quote,
            is_active=True,
        )
        session.add(symbol_info)

    # Update timeframe flags
    tf_flags = {
        "1m": "has_1m",
        "5m": "has_5m",
        "15m": "has_15m",
        "1h": "has_1h",
        "4h": "has_4h",
        "1d": "has_1d",
    }
    if timeframe in tf_flags:
        setattr(symbol_info, tf_flags[timeframe], True)

    # Update date range
    from sqlalchemy import func
    result = await session.execute(
        select(
            func.min(OHLCVDataORM.timestamp),
            func.max(OHLCVDataORM.timestamp),
            func.count(),
        ).where(
            OHLCVDataORM.symbol == symbol,
        )
    )
    row = result.one_or_none()
    if row:
        symbol_info.data_start = row[0]
        symbol_info.data_end = row[1]
        symbol_info.total_rows = row[2]

    symbol_info.updated_at = datetime.now(timezone.utc)
    await session.commit()
