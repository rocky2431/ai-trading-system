"""Derivative data downloader for crypto perpetuals/futures.

This module provides data downloading for:
- Funding rates
- Open interest
- Liquidations
- Long/short ratios
- Mark prices
- Taker buy/sell volume
"""

import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

import ccxt.async_support as ccxt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from iqfmp.db.models import (
    FundingRateORM,
    OpenInterestORM,
    LiquidationORM,
    LongShortRatioORM,
    MarkPriceORM,
    TakerBuySellORM,
)

logger = logging.getLogger(__name__)


class DerivativeDataType(str, Enum):
    """Types of derivative data."""
    FUNDING_RATE = "funding_rate"
    OPEN_INTEREST = "open_interest"
    LIQUIDATION = "liquidation"
    LONG_SHORT_RATIO = "long_short_ratio"
    MARK_PRICE = "mark_price"
    TAKER_BUY_SELL = "taker_buy_sell"


class DerivativeDownloader:
    """Downloader for derivative market data from exchanges."""

    def __init__(
        self,
        exchange_id: str = "binance",
        rate_limit_delay: float = 0.5,
    ):
        """Initialize derivative data downloader.

        Args:
            exchange_id: Exchange identifier (binance, okx, bybit)
            rate_limit_delay: Delay between API calls in seconds
        """
        self.exchange_id = exchange_id
        self.rate_limit_delay = rate_limit_delay
        self._exchange: Optional[ccxt.Exchange] = None

    async def _get_exchange(self) -> ccxt.Exchange:
        """Get or create futures exchange instance."""
        if self._exchange is None:
            # Use binanceusdm for USDT-M perpetuals
            if self.exchange_id == "binance":
                exchange_class = getattr(ccxt, "binanceusdm")
            else:
                exchange_class = getattr(ccxt, self.exchange_id)

            self._exchange = exchange_class({
                "enableRateLimit": True,
                "options": {
                    "defaultType": "future",
                }
            })
        return self._exchange

    async def close(self):
        """Close exchange connection."""
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None

    # =========================================================================
    # Funding Rate
    # =========================================================================

    async def fetch_funding_rate_history(
        self,
        symbol: str,
        since: Optional[int] = None,
        limit: int = 500,
    ) -> list[dict]:
        """Fetch historical funding rates.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT:USDT")
            since: Start timestamp in milliseconds
            limit: Number of records to fetch

        Returns:
            List of funding rate records
        """
        exchange = await self._get_exchange()

        try:
            # CCXT unified method for funding rate history
            if hasattr(exchange, 'fetch_funding_rate_history'):
                funding_history = await exchange.fetch_funding_rate_history(
                    symbol,
                    since=since,
                    limit=limit,
                )
                return funding_history
            else:
                logger.warning(f"{self.exchange_id} does not support fetch_funding_rate_history")
                return []
        except Exception as e:
            logger.error(f"Error fetching funding rate history: {e}")
            raise

    async def download_funding_rates(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        session: AsyncSession,
        progress_callback: Optional[Callable[[float, int], Any]] = None,
    ) -> int:
        """Download and store funding rate history.

        Args:
            symbol: Trading pair
            start_date: Start datetime
            end_date: End datetime
            session: Database session
            progress_callback: Optional callback(progress_pct, rows_downloaded)

        Returns:
            Total rows downloaded
        """
        symbol = self._normalize_symbol(symbol)
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)

        total_rows = 0
        current_ts = start_ts

        logger.info(f"Downloading funding rates for {symbol} from {start_date} to {end_date}")

        while current_ts < end_ts:
            try:
                funding_data = await self.fetch_funding_rate_history(
                    symbol=symbol,
                    since=current_ts,
                    limit=500,
                )

                if not funding_data:
                    break

                for record in funding_data:
                    timestamp = datetime.fromtimestamp(
                        record.get('timestamp', 0) / 1000,
                        tz=timezone.utc
                    )

                    if timestamp > end_date:
                        break

                    # Check if exists
                    existing = await session.execute(
                        select(FundingRateORM).where(
                            FundingRateORM.symbol == symbol,
                            FundingRateORM.timestamp == timestamp,
                            FundingRateORM.exchange == self.exchange_id,
                        )
                    )
                    if existing.scalar_one_or_none():
                        continue

                    # Insert new record
                    orm_record = FundingRateORM(
                        symbol=symbol,
                        exchange=self.exchange_id,
                        funding_rate=float(record.get('fundingRate', 0)),
                        mark_price=float(record.get('markPrice', 0)) if record.get('markPrice') else None,
                        index_price=float(record.get('indexPrice', 0)) if record.get('indexPrice') else None,
                        timestamp=timestamp,
                    )
                    session.add(orm_record)
                    total_rows += 1

                await session.commit()

                if progress_callback:
                    progress = min(100.0, ((current_ts - start_ts) / (end_ts - start_ts)) * 100)
                    await progress_callback(progress, total_rows)

                # Move to next batch
                if funding_data:
                    current_ts = funding_data[-1].get('timestamp', current_ts) + 1
                else:
                    break

                await asyncio.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(f"Error downloading funding rates: {e}")
                await asyncio.sleep(1.0)
                break

        logger.info(f"Downloaded {total_rows} funding rate records for {symbol}")
        return total_rows

    # =========================================================================
    # Open Interest
    # =========================================================================

    async def fetch_open_interest(
        self,
        symbol: str,
    ) -> dict:
        """Fetch current open interest.

        Args:
            symbol: Trading pair

        Returns:
            Open interest data
        """
        exchange = await self._get_exchange()

        try:
            if hasattr(exchange, 'fetch_open_interest'):
                oi = await exchange.fetch_open_interest(symbol)
                return oi
            else:
                logger.warning(f"{self.exchange_id} does not support fetch_open_interest")
                return {}
        except Exception as e:
            logger.error(f"Error fetching open interest: {e}")
            raise

    async def fetch_open_interest_history(
        self,
        symbol: str,
        timeframe: str = "5m",
        since: Optional[int] = None,
        limit: int = 500,
    ) -> list[dict]:
        """Fetch historical open interest.

        Args:
            symbol: Trading pair
            timeframe: Data timeframe
            since: Start timestamp in milliseconds
            limit: Number of records

        Returns:
            List of open interest records
        """
        exchange = await self._get_exchange()

        try:
            if hasattr(exchange, 'fetch_open_interest_history'):
                oi_history = await exchange.fetch_open_interest_history(
                    symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit,
                )
                return oi_history
            else:
                logger.warning(f"{self.exchange_id} does not support fetch_open_interest_history")
                return []
        except Exception as e:
            logger.error(f"Error fetching open interest history: {e}")
            raise

    async def download_open_interest(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        session: AsyncSession,
        timeframe: str = "5m",
        progress_callback: Optional[Callable[[float, int], Any]] = None,
    ) -> int:
        """Download and store open interest history.

        Args:
            symbol: Trading pair
            start_date: Start datetime
            end_date: End datetime
            session: Database session
            timeframe: Data timeframe
            progress_callback: Optional callback(progress_pct, rows_downloaded)

        Returns:
            Total rows downloaded
        """
        symbol = self._normalize_symbol(symbol)
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)

        total_rows = 0
        current_ts = start_ts

        logger.info(f"Downloading open interest for {symbol} from {start_date} to {end_date}")

        while current_ts < end_ts:
            try:
                oi_data = await self.fetch_open_interest_history(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_ts,
                    limit=500,
                )

                if not oi_data:
                    break

                for record in oi_data:
                    timestamp = datetime.fromtimestamp(
                        record.get('timestamp', 0) / 1000,
                        tz=timezone.utc
                    )

                    if timestamp > end_date:
                        break

                    # Check if exists
                    existing = await session.execute(
                        select(OpenInterestORM).where(
                            OpenInterestORM.symbol == symbol,
                            OpenInterestORM.timestamp == timestamp,
                            OpenInterestORM.exchange == self.exchange_id,
                        )
                    )
                    if existing.scalar_one_or_none():
                        continue

                    # Insert new record
                    orm_record = OpenInterestORM(
                        symbol=symbol,
                        exchange=self.exchange_id,
                        open_interest=float(record.get('openInterestAmount', 0)),
                        open_interest_value=float(record.get('openInterestValue', 0)) if record.get('openInterestValue') else None,
                        contract_type="perpetual",
                        timestamp=timestamp,
                    )
                    session.add(orm_record)
                    total_rows += 1

                await session.commit()

                if progress_callback:
                    progress = min(100.0, ((current_ts - start_ts) / (end_ts - start_ts)) * 100)
                    await progress_callback(progress, total_rows)

                # Move to next batch
                if oi_data:
                    current_ts = oi_data[-1].get('timestamp', current_ts) + 1
                else:
                    break

                await asyncio.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(f"Error downloading open interest: {e}")
                await asyncio.sleep(1.0)
                break

        logger.info(f"Downloaded {total_rows} open interest records for {symbol}")
        return total_rows

    # =========================================================================
    # Long/Short Ratio
    # =========================================================================

    async def fetch_long_short_ratio(
        self,
        symbol: str,
        period: str = "5m",
    ) -> dict:
        """Fetch current long/short ratio.

        Args:
            symbol: Trading pair
            period: Time period for ratio

        Returns:
            Long/short ratio data
        """
        exchange = await self._get_exchange()

        try:
            # Binance-specific endpoint via CCXT
            if self.exchange_id == "binance":
                # This uses Binance's top trader long/short ratio
                params = {"period": period}
                if hasattr(exchange, 'fapiPublicGetTopLongShortAccountRatio'):
                    response = await exchange.fapiPublicGetTopLongShortAccountRatio({
                        "symbol": symbol.replace("/", "").replace(":USDT", ""),
                        "period": period,
                        "limit": 1,
                    })
                    if response:
                        return response[0] if isinstance(response, list) else response
            return {}
        except Exception as e:
            logger.error(f"Error fetching long/short ratio: {e}")
            return {}

    async def download_long_short_ratios(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        session: AsyncSession,
        ratio_type: str = "top_trader_accounts",
        period: str = "5m",
        progress_callback: Optional[Callable[[float, int], Any]] = None,
    ) -> int:
        """Download and store long/short ratio history.

        Note: Most exchanges only provide recent data, historical data may be limited.

        Args:
            symbol: Trading pair
            start_date: Start datetime
            end_date: End datetime
            session: Database session
            ratio_type: Type of ratio (global, top_trader_accounts, top_trader_positions)
            period: Time period
            progress_callback: Optional callback

        Returns:
            Total rows downloaded
        """
        symbol = self._normalize_symbol(symbol)
        exchange = await self._get_exchange()

        total_rows = 0
        logger.info(f"Downloading long/short ratios for {symbol}")

        try:
            # Binance-specific implementation
            if self.exchange_id == "binance":
                raw_symbol = symbol.replace("/", "").replace(":USDT", "")

                # Map ratio_type to Binance endpoint
                endpoint_map = {
                    "global": "fapiPublicGetGlobalLongShortAccountRatio",
                    "top_trader_accounts": "fapiPublicGetTopLongShortAccountRatio",
                    "top_trader_positions": "fapiPublicGetTopLongShortPositionRatio",
                }

                endpoint = endpoint_map.get(ratio_type)
                if endpoint and hasattr(exchange, endpoint):
                    fetch_func = getattr(exchange, endpoint)
                    response = await fetch_func({
                        "symbol": raw_symbol,
                        "period": period,
                        "limit": 500,
                        "startTime": int(start_date.timestamp() * 1000),
                        "endTime": int(end_date.timestamp() * 1000),
                    })

                    if response:
                        for record in response:
                            timestamp = datetime.fromtimestamp(
                                int(record.get('timestamp', 0)) / 1000,
                                tz=timezone.utc
                            )

                            long_ratio = float(record.get('longAccount', record.get('longPosition', 0.5)))
                            short_ratio = float(record.get('shortAccount', record.get('shortPosition', 0.5)))
                            ls_ratio = float(record.get('longShortRatio', long_ratio / (short_ratio + 1e-10)))

                            orm_record = LongShortRatioORM(
                                symbol=symbol,
                                exchange=self.exchange_id,
                                ratio_type=ratio_type,
                                long_ratio=long_ratio,
                                short_ratio=short_ratio,
                                long_short_ratio=ls_ratio,
                                timestamp=timestamp,
                            )
                            session.add(orm_record)
                            total_rows += 1

                        await session.commit()

        except Exception as e:
            logger.error(f"Error downloading long/short ratios: {e}")

        logger.info(f"Downloaded {total_rows} long/short ratio records for {symbol}")
        return total_rows

    # =========================================================================
    # Mark Price
    # =========================================================================

    async def fetch_mark_price(
        self,
        symbol: str,
    ) -> dict:
        """Fetch current mark price.

        Args:
            symbol: Trading pair

        Returns:
            Mark price data
        """
        exchange = await self._get_exchange()

        try:
            if hasattr(exchange, 'fetch_mark_price'):
                mark_price = await exchange.fetch_mark_price(symbol)
                return mark_price
            else:
                # Fallback to ticker
                ticker = await exchange.fetch_ticker(symbol)
                return {
                    'markPrice': ticker.get('last'),
                    'indexPrice': ticker.get('last'),
                    'timestamp': ticker.get('timestamp'),
                }
        except Exception as e:
            logger.error(f"Error fetching mark price: {e}")
            raise

    async def download_mark_prices(
        self,
        symbol: str,
        session: AsyncSession,
    ) -> int:
        """Download and store current mark price.

        Args:
            symbol: Trading pair
            session: Database session

        Returns:
            1 if successful, 0 otherwise
        """
        symbol = self._normalize_symbol(symbol)

        try:
            data = await self.fetch_mark_price(symbol)

            if not data:
                return 0

            timestamp = datetime.fromtimestamp(
                data.get('timestamp', int(datetime.now().timestamp() * 1000)) / 1000,
                tz=timezone.utc
            )

            mark_price = float(data.get('markPrice', 0))
            index_price = float(data.get('indexPrice', 0)) if data.get('indexPrice') else None
            last_price = float(data.get('lastPrice', 0)) if data.get('lastPrice') else None

            basis = None
            basis_rate = None
            if mark_price and index_price:
                basis = mark_price - index_price
                basis_rate = basis / index_price if index_price else None

            orm_record = MarkPriceORM(
                symbol=symbol,
                exchange=self.exchange_id,
                mark_price=mark_price,
                index_price=index_price,
                last_price=last_price,
                basis=basis,
                basis_rate=basis_rate,
                timestamp=timestamp,
            )
            session.add(orm_record)
            await session.commit()

            return 1

        except Exception as e:
            logger.error(f"Error downloading mark price: {e}")
            return 0

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to perpetual format.

        Args:
            symbol: Input symbol

        Returns:
            Normalized symbol (e.g., "BTC/USDT:USDT")
        """
        symbol = symbol.upper()

        # Already in perpetual format
        if ":USDT" in symbol:
            return symbol

        # Convert spot format to perpetual
        if "/" in symbol:
            base, quote = symbol.split("/")
            return f"{base}/{quote}:{quote}"

        # Convert raw format (BTCUSDT) to perpetual
        if symbol.endswith("USDT"):
            base = symbol[:-4]
            return f"{base}/USDT:USDT"

        return symbol


# =============================================================================
# Factory Functions
# =============================================================================

def get_derivative_downloader(exchange_id: str = "binance") -> DerivativeDownloader:
    """Factory function to create a derivative downloader.

    Args:
        exchange_id: Exchange identifier

    Returns:
        DerivativeDownloader instance
    """
    return DerivativeDownloader(exchange_id=exchange_id)


async def download_all_derivative_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    session: AsyncSession,
    exchange_id: str = "binance",
    data_types: Optional[list[DerivativeDataType]] = None,
) -> dict[str, int]:
    """Download all types of derivative data for a symbol.

    Args:
        symbol: Trading pair
        start_date: Start datetime
        end_date: End datetime
        session: Database session
        exchange_id: Exchange identifier
        data_types: List of data types to download (None = all)

    Returns:
        Dictionary of data type -> rows downloaded
    """
    downloader = DerivativeDownloader(exchange_id=exchange_id)

    if data_types is None:
        data_types = [
            DerivativeDataType.FUNDING_RATE,
            DerivativeDataType.OPEN_INTEREST,
            DerivativeDataType.LONG_SHORT_RATIO,
        ]

    results = {}

    try:
        for data_type in data_types:
            logger.info(f"Downloading {data_type.value} for {symbol}")

            if data_type == DerivativeDataType.FUNDING_RATE:
                rows = await downloader.download_funding_rates(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    session=session,
                )
            elif data_type == DerivativeDataType.OPEN_INTEREST:
                rows = await downloader.download_open_interest(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    session=session,
                )
            elif data_type == DerivativeDataType.LONG_SHORT_RATIO:
                rows = await downloader.download_long_short_ratios(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    session=session,
                )
            elif data_type == DerivativeDataType.MARK_PRICE:
                rows = await downloader.download_mark_prices(
                    symbol=symbol,
                    session=session,
                )
            else:
                logger.warning(f"Unknown data type: {data_type}")
                rows = 0

            results[data_type.value] = rows

    finally:
        await downloader.close()

    return results
