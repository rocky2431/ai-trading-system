"""Unified Market Data Provider for IQFMP.

This module provides a unified interface for loading market data including:
- OHLCV price data
- Derivative data (funding rates, open interest, liquidations, long/short ratios)
- Derived features (funding_zscore, oi_change, etc.)

C1: UnifiedMarketDataProvider - Merges OHLCV + derivatives data
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from iqfmp.db.models import (
    FundingRateORM,
    LiquidationORM,
    LongShortRatioORM,
    OHLCVDataORM,
    OpenInterestORM,
)
from iqfmp.data.alignment import (
    calculate_funding_features,
    merge_derivative_data,
    validate_time_alignment,
)

logger = logging.getLogger(__name__)


class DerivativeType(str, Enum):
    """Available derivative data types."""

    FUNDING_RATE = "funding_rate"
    OPEN_INTEREST = "open_interest"
    LIQUIDATION = "liquidation"
    LONG_SHORT_RATIO = "long_short_ratio"


@dataclass
class DataLoadConfig:
    """Configuration for data loading.

    Attributes:
        include_derivatives: Whether to include derivative data.
        derivative_types: List of derivative types to include (None = all).
        calculate_features: Whether to calculate derived features.
        forward_fill_method: Method for forward-filling derivative data.
        validate_alignment: Whether to validate time alignment.
        market_type: Market type filter (spot, futures, or both).
    """

    include_derivatives: bool = True
    derivative_types: list[DerivativeType] | None = None
    calculate_features: bool = True
    forward_fill_method: Literal["ffill", "last", "next"] = "ffill"
    validate_alignment: bool = True
    market_type: Literal["spot", "futures", "both"] = "futures"


@dataclass
class DataLoadResult:
    """Result of data loading operation.

    Attributes:
        df: The loaded and merged DataFrame.
        columns: List of available columns.
        derivative_columns: List of derivative-specific columns.
        derived_columns: List of derived feature columns.
        validation: Validation results if validation was performed.
        warnings: Any warnings generated during loading.
    """

    df: pd.DataFrame
    columns: list[str] = field(default_factory=list)
    derivative_columns: list[str] = field(default_factory=list)
    derived_columns: list[str] = field(default_factory=list)
    validation: dict | None = None
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize column lists from DataFrame."""
        if not self.columns and not self.df.empty:
            self.columns = list(self.df.columns)


class UnifiedMarketDataProvider:
    """Unified provider for OHLCV and derivative market data.

    This class provides a single interface for loading all types of market data,
    handling time alignment, and calculating derived features.

    Example:
        provider = UnifiedMarketDataProvider(session)
        result = await provider.load_market_data(
            symbol="BTC/USDT:USDT",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            timeframe="1h",
        )
        df = result.df  # Contains OHLCV + funding_rate + open_interest + ...
    """

    # Standard OHLCV columns
    OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

    # Derivative columns added by this provider
    DERIVATIVE_COLUMNS = [
        "funding_rate",
        "open_interest",
        "open_interest_change",
        "long_short_ratio",
        "liquidation_long",
        "liquidation_short",
        "liquidation_total",
    ]

    # Derived feature columns (calculated from derivatives)
    DERIVED_COLUMNS = [
        "funding_ma_8h",
        "funding_ma_24h",
        "funding_momentum",
        "funding_zscore",
        "funding_extreme",
        "funding_annualized",
    ]

    def __init__(
        self,
        session: AsyncSession,
        exchange: str = "binance",
    ) -> None:
        """Initialize the provider.

        Args:
            session: SQLAlchemy async session.
            exchange: Exchange identifier.
        """
        self._session = session
        self._exchange = exchange

    async def load_market_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1h",
        config: DataLoadConfig | None = None,
    ) -> DataLoadResult:
        """Load unified market data with OHLCV and derivatives.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT:USDT" or "BTC/USDT").
            start_date: Start datetime.
            end_date: End datetime.
            timeframe: OHLCV timeframe (1m, 5m, 15m, 1h, 4h, 1d).
            config: Data loading configuration.

        Returns:
            DataLoadResult with merged DataFrame and metadata.
        """
        if config is None:
            config = DataLoadConfig()

        warnings: list[str] = []
        symbol = self._normalize_symbol(symbol)

        # Step 1: Load OHLCV data
        ohlcv_df = await self._load_ohlcv(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            market_type=config.market_type,
        )

        if ohlcv_df.empty:
            warnings.append(f"No OHLCV data found for {symbol}")
            return DataLoadResult(
                df=pd.DataFrame(),
                warnings=warnings,
            )

        logger.info(f"Loaded {len(ohlcv_df)} OHLCV rows for {symbol}")

        # Step 2: Load derivative data if enabled
        derivative_columns: list[str] = []
        if config.include_derivatives:
            derivative_types = config.derivative_types or list(DerivativeType)

            # Load funding rates
            if DerivativeType.FUNDING_RATE in derivative_types:
                funding_df = await self._load_funding_rates(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                )
                if not funding_df.empty:
                    derivative_columns.append("funding_rate")
                else:
                    warnings.append("No funding rate data found")
                    funding_df = None
            else:
                funding_df = None

            # Load open interest
            if DerivativeType.OPEN_INTEREST in derivative_types:
                oi_df = await self._load_open_interest(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                )
                if not oi_df.empty:
                    derivative_columns.extend(["open_interest", "open_interest_change"])
                else:
                    warnings.append("No open interest data found")
                    oi_df = None
            else:
                oi_df = None

            # Load long/short ratio
            if DerivativeType.LONG_SHORT_RATIO in derivative_types:
                ls_df = await self._load_long_short_ratio(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                )
                if not ls_df.empty:
                    derivative_columns.append("long_short_ratio")
                else:
                    warnings.append("No long/short ratio data found")
                    ls_df = None
            else:
                ls_df = None

            # Load liquidations
            if DerivativeType.LIQUIDATION in derivative_types:
                liq_df = await self._load_liquidations(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                )
                if not liq_df.empty:
                    derivative_columns.extend([
                        "liquidation_long",
                        "liquidation_short",
                        "liquidation_total",
                    ])
                else:
                    warnings.append("No liquidation data found")
                    liq_df = None
            else:
                liq_df = None

            # Merge all derivative data with OHLCV
            merged_df = merge_derivative_data(
                ohlcv_df=ohlcv_df,
                funding_df=funding_df,
                oi_df=oi_df,
                ls_ratio_df=ls_df,
                liquidation_df=liq_df,
            )
        else:
            merged_df = ohlcv_df

        # Step 3: Calculate derived features if enabled
        derived_columns: list[str] = []
        if config.calculate_features and "funding_rate" in merged_df.columns:
            merged_df = calculate_funding_features(merged_df, "funding_rate")
            derived_columns = [c for c in self.DERIVED_COLUMNS if c in merged_df.columns]
            logger.info(f"Calculated {len(derived_columns)} derived features")

        # Step 4: Validate time alignment if enabled
        validation_result = None
        if config.validate_alignment:
            is_valid, validation = validate_time_alignment(
                merged_df,
                expected_freq=self._timeframe_to_freq(timeframe),
            )
            validation_result = validation
            if not is_valid:
                warnings.append(f"Time alignment issues detected: {validation}")

        # Sort by timestamp
        if "timestamp" in merged_df.columns:
            merged_df = merged_df.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            f"Unified data loaded: {len(merged_df)} rows, "
            f"{len(derivative_columns)} derivative columns, "
            f"{len(derived_columns)} derived columns"
        )

        return DataLoadResult(
            df=merged_df,
            columns=list(merged_df.columns),
            derivative_columns=derivative_columns,
            derived_columns=derived_columns,
            validation=validation_result,
            warnings=warnings,
        )

    async def _load_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
        market_type: str,
    ) -> pd.DataFrame:
        """Load OHLCV data from database.

        Args:
            symbol: Trading pair.
            start_date: Start datetime.
            end_date: End datetime.
            timeframe: OHLCV timeframe.
            market_type: Market type filter.

        Returns:
            DataFrame with OHLCV data.
        """
        # Ensure timezone-aware
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        query = (
            select(
                OHLCVDataORM.timestamp,
                OHLCVDataORM.open,
                OHLCVDataORM.high,
                OHLCVDataORM.low,
                OHLCVDataORM.close,
                OHLCVDataORM.volume,
            )
            .where(
                OHLCVDataORM.symbol == symbol,
                OHLCVDataORM.timeframe == timeframe,
                OHLCVDataORM.exchange == self._exchange,
                OHLCVDataORM.timestamp >= start_date,
                OHLCVDataORM.timestamp <= end_date,
            )
            .order_by(OHLCVDataORM.timestamp)
        )

        # Add market_type filter
        if market_type != "both":
            query = query.where(OHLCVDataORM.market_type == market_type)

        result = await self._session.execute(query)
        rows = result.all()

        if not rows:
            return pd.DataFrame(columns=self.OHLCV_COLUMNS)

        df = pd.DataFrame(rows, columns=self.OHLCV_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        return df

    async def _load_funding_rates(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Load funding rate data from database.

        Args:
            symbol: Trading pair.
            start_date: Start datetime.
            end_date: End datetime.

        Returns:
            DataFrame with funding rate data.
        """
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        query = (
            select(
                FundingRateORM.timestamp,
                FundingRateORM.funding_rate,
            )
            .where(
                FundingRateORM.symbol == symbol,
                FundingRateORM.exchange == self._exchange,
                FundingRateORM.timestamp >= start_date,
                FundingRateORM.timestamp <= end_date,
            )
            .order_by(FundingRateORM.timestamp)
        )

        result = await self._session.execute(query)
        rows = result.all()

        if not rows:
            return pd.DataFrame(columns=["timestamp", "funding_rate"])

        df = pd.DataFrame(rows, columns=["timestamp", "funding_rate"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        return df

    async def _load_open_interest(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Load open interest data from database.

        Args:
            symbol: Trading pair.
            start_date: Start datetime.
            end_date: End datetime.

        Returns:
            DataFrame with open interest data.
        """
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        query = (
            select(
                OpenInterestORM.timestamp,
                OpenInterestORM.open_interest,
            )
            .where(
                OpenInterestORM.symbol == symbol,
                OpenInterestORM.exchange == self._exchange,
                OpenInterestORM.timestamp >= start_date,
                OpenInterestORM.timestamp <= end_date,
            )
            .order_by(OpenInterestORM.timestamp)
        )

        result = await self._session.execute(query)
        rows = result.all()

        if not rows:
            return pd.DataFrame(columns=["timestamp", "open_interest"])

        df = pd.DataFrame(rows, columns=["timestamp", "open_interest"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        return df

    async def _load_long_short_ratio(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        ratio_type: str = "top_trader_accounts",
    ) -> pd.DataFrame:
        """Load long/short ratio data from database.

        Args:
            symbol: Trading pair.
            start_date: Start datetime.
            end_date: End datetime.
            ratio_type: Type of ratio to load.

        Returns:
            DataFrame with long/short ratio data.
        """
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        query = (
            select(
                LongShortRatioORM.timestamp,
                LongShortRatioORM.long_short_ratio,
            )
            .where(
                LongShortRatioORM.symbol == symbol,
                LongShortRatioORM.exchange == self._exchange,
                LongShortRatioORM.ratio_type == ratio_type,
                LongShortRatioORM.timestamp >= start_date,
                LongShortRatioORM.timestamp <= end_date,
            )
            .order_by(LongShortRatioORM.timestamp)
        )

        result = await self._session.execute(query)
        rows = result.all()

        if not rows:
            return pd.DataFrame(columns=["timestamp", "long_short_ratio"])

        df = pd.DataFrame(rows, columns=["timestamp", "long_short_ratio"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        return df

    async def _load_liquidations(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Load liquidation data from database.

        Args:
            symbol: Trading pair.
            start_date: Start datetime.
            end_date: End datetime.

        Returns:
            DataFrame with liquidation data (aggregated by timestamp).
        """
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        query = (
            select(
                LiquidationORM.timestamp,
                LiquidationORM.side,
                LiquidationORM.value_usd,
            )
            .where(
                LiquidationORM.symbol == symbol,
                LiquidationORM.exchange == self._exchange,
                LiquidationORM.timestamp >= start_date,
                LiquidationORM.timestamp <= end_date,
            )
            .order_by(LiquidationORM.timestamp)
        )

        result = await self._session.execute(query)
        rows = result.all()

        if not rows:
            return pd.DataFrame(
                columns=["timestamp", "liquidation_long", "liquidation_short"]
            )

        df = pd.DataFrame(rows, columns=["timestamp", "side", "value_usd"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Pivot to get long/short columns
        df["liquidation_long"] = np.where(df["side"] == "sell", df["value_usd"], 0)
        df["liquidation_short"] = np.where(df["side"] == "buy", df["value_usd"], 0)

        # Aggregate by timestamp
        agg_df = (
            df.groupby("timestamp")
            .agg({"liquidation_long": "sum", "liquidation_short": "sum"})
            .reset_index()
        )

        return agg_df

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to perpetual format.

        Args:
            symbol: Input symbol.

        Returns:
            Normalized symbol (e.g., "BTC/USDT:USDT").
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

    def _timeframe_to_freq(self, timeframe: str) -> str:
        """Convert timeframe to pandas frequency string.

        Args:
            timeframe: Timeframe string (1m, 5m, 1h, etc.).

        Returns:
            Pandas frequency string.
        """
        mapping = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D",
        }
        return mapping.get(timeframe, "1H")

    async def get_available_columns(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, bool]:
        """Check which data columns are available for a symbol.

        Args:
            symbol: Trading pair.
            start_date: Start datetime.
            end_date: End datetime.

        Returns:
            Dictionary mapping column names to availability status.
        """
        symbol = self._normalize_symbol(symbol)
        availability = {}

        # Check OHLCV
        ohlcv_df = await self._load_ohlcv(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe="1h",
            market_type="futures",
        )
        availability["ohlcv"] = not ohlcv_df.empty

        # Check funding rates
        funding_df = await self._load_funding_rates(symbol, start_date, end_date)
        availability["funding_rate"] = not funding_df.empty

        # Check open interest
        oi_df = await self._load_open_interest(symbol, start_date, end_date)
        availability["open_interest"] = not oi_df.empty

        # Check long/short ratio
        ls_df = await self._load_long_short_ratio(symbol, start_date, end_date)
        availability["long_short_ratio"] = not ls_df.empty

        # Check liquidations
        liq_df = await self._load_liquidations(symbol, start_date, end_date)
        availability["liquidation"] = not liq_df.empty

        return availability


# =============================================================================
# Convenience Functions
# =============================================================================


async def load_unified_data(
    session: AsyncSession,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    timeframe: str = "1h",
    exchange: str = "binance",
    include_derivatives: bool = True,
) -> pd.DataFrame:
    """Convenience function to load unified market data.

    Args:
        session: SQLAlchemy async session.
        symbol: Trading pair.
        start_date: Start datetime.
        end_date: End datetime.
        timeframe: OHLCV timeframe.
        exchange: Exchange identifier.
        include_derivatives: Whether to include derivative data.

    Returns:
        Merged DataFrame with OHLCV and derivative data.
    """
    provider = UnifiedMarketDataProvider(session, exchange=exchange)
    config = DataLoadConfig(include_derivatives=include_derivatives)
    result = await provider.load_market_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        config=config,
    )
    return result.df


async def check_data_availability(
    session: AsyncSession,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    exchange: str = "binance",
) -> dict[str, bool]:
    """Check data availability for a symbol.

    Args:
        session: SQLAlchemy async session.
        symbol: Trading pair.
        start_date: Start datetime.
        end_date: End datetime.
        exchange: Exchange identifier.

    Returns:
        Dictionary mapping data types to availability.
    """
    provider = UnifiedMarketDataProvider(session, exchange=exchange)
    return await provider.get_available_columns(symbol, start_date, end_date)
