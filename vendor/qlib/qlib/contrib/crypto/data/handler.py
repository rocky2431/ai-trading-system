"""Cryptocurrency Data Handler for Qlib Deep Fork.

This module provides CryptoDataHandler, which extends Qlib's DataHandlerLP
with cryptocurrency-specific data fields and multi-exchange support.

Deep Fork Strategy:
- Code lives in forked Qlib repository (vendor/qlib/qlib/contrib/crypto/)
- Conditional inheritance from DataHandlerLP when Qlib is properly installed
- Standalone implementation fallback when Qlib C extensions unavailable
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

from qlib.contrib.crypto.data.validator import (
    DataValidator,
    ValidationError,
    ValidationResult,
)

# Conditional Qlib inheritance
# Qlib requires compiled C extensions; use standalone fallback otherwise
QLIB_AVAILABLE = False
_QlibDataHandlerLP = None

try:
    from qlib.data.dataset.handler import DataHandlerLP as QlibDataHandlerLP
    QLIB_AVAILABLE = True
    _QlibDataHandlerLP = QlibDataHandlerLP
except (ImportError, ModuleNotFoundError, LookupError, Exception):
    # Qlib not available or missing C extensions
    pass


class _StandaloneDataHandler:
    """Standalone base class when Qlib is not available."""
    pass


class CryptoField(str, Enum):
    """Cryptocurrency data fields.

    Standard OHLCV fields plus crypto-specific fields like funding rates,
    open interest, and basis for perpetual futures trading.
    """

    # Standard OHLCV
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"

    # Crypto-specific fields
    FUNDING_RATE = "funding_rate"
    OPEN_INTEREST = "open_interest"
    BASIS = "basis"
    MARK_PRICE = "mark_price"
    INDEX_PRICE = "index_price"

    # Additional fields
    QUOTE_VOLUME = "quote_volume"
    TRADES = "trades"
    TAKER_BUY_VOLUME = "taker_buy_volume"


class Exchange(str, Enum):
    """Supported cryptocurrency exchanges."""

    BINANCE = "binance"
    OKX = "okx"
    BYBIT = "bybit"
    DERIBIT = "deribit"
    FTX = "ftx"  # Historical
    HUOBI = "huobi"


class TimeFrame(str, Enum):
    """Supported timeframes for cryptocurrency data."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    H8 = "8h"
    D1 = "1d"
    W1 = "1w"

    def to_minutes(self) -> int:
        """Convert timeframe to minutes."""
        mapping = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "8h": 480, "1d": 1440, "1w": 10080,
        }
        return mapping[self.value]

    def to_pandas_freq(self) -> str:
        """Convert timeframe to pandas frequency string."""
        mapping = {
            "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
            "1h": "1h", "4h": "4h", "8h": "8h", "1d": "1D", "1w": "1W",
        }
        return mapping[self.value]


@dataclass
class CryptoDataConfig:
    """Configuration for CryptoDataHandler.

    Attributes:
        fields: List of data fields to include
        exchange: Target exchange for data normalization
        timeframe: Data timeframe
        symbols: List of trading symbols (e.g., ["BTCUSDT", "ETHUSDT"])
    """

    fields: list[CryptoField] = field(default_factory=lambda: [
        CryptoField.OPEN, CryptoField.HIGH, CryptoField.LOW,
        CryptoField.CLOSE, CryptoField.VOLUME,
    ])
    exchange: Exchange = Exchange.BINANCE
    timeframe: TimeFrame = TimeFrame.H1
    symbols: list[str] = field(default_factory=list)


# Column mappings for different exchanges
EXCHANGE_COLUMN_MAPPINGS: dict[Exchange, dict[str, str]] = {
    Exchange.BINANCE: {
        # Common in-house format
        "instrument": "symbol",
        # Binance REST
        "open_time": "datetime",
        "close_time": "_close_time",
        "quote_volume": "quote_volume",
        "trades": "trades",
    },
    Exchange.OKX: {
        # Common in-house format
        "instrument": "symbol",
        # OKX REST
        "ts": "datetime",
        "instId": "symbol",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "vol": "volume",
        "volCcy": "quote_volume",
    },
    Exchange.BYBIT: {
        # Common in-house format
        "instrument": "symbol",
        # Bybit REST
        "timestamp": "datetime",
        "turnover": "quote_volume",
    },
}


def _create_crypto_data_handler():
    """Factory function to create CryptoDataHandler with conditional inheritance."""

    # Choose base class based on Qlib availability
    if QLIB_AVAILABLE and _QlibDataHandlerLP is not None:
        BaseClass = _QlibDataHandlerLP
    else:
        BaseClass = _StandaloneDataHandler

    class _CryptoDataHandler(BaseClass):  # type: ignore[misc]
        """Data handler for cryptocurrency market data.

        This class extends Qlib's DataHandlerLP (when available) to support
        cryptocurrency-specific data fields and multi-exchange data normalization.

        Deep Fork Features:
        - Conditional inheritance from DataHandlerLP
        - Crypto-specific fields (funding_rate, open_interest, basis)
        - Multi-exchange column normalization
        - Funding-adjusted returns calculation
        - Timeframe resampling

        Example:
            >>> config = CryptoDataConfig(
            ...     exchange=Exchange.BINANCE,
            ...     timeframe=TimeFrame.H1,
            ...     symbols=["BTCUSDT"]
            ... )
            >>> handler = CryptoDataHandler(config=config)
            >>> handler.load(df, exchange=Exchange.BINANCE, validate=True)
            >>> returns = handler.calculate_funding_adjusted_returns()
        """

        def __init__(
            self,
            config: Optional[CryptoDataConfig] = None,
            **kwargs: Any,
        ) -> None:
            """Initialize the CryptoDataHandler."""
            # NOTE:
            # Qlib's DataHandlerLP asserts `data_loader is not None` during init.
            # Our crypto handler primarily operates on in-memory DataFrames via
            # `load()`, so we only initialize the Qlib base class when a valid
            # data_loader is explicitly provided by the caller.
            if (
                QLIB_AVAILABLE
                and _QlibDataHandlerLP is not None
                and kwargs.get("data_loader") is not None
            ):
                super().__init__(**kwargs)

            self.config = config or CryptoDataConfig()
            self._crypto_data: Optional[pd.DataFrame] = None
            self._validator = DataValidator()

        @property
        def data(self) -> Optional[pd.DataFrame]:
            """Access loaded data (backward compatibility)."""
            return self._crypto_data

        # === Extended Qlib Interface ===

        def fetch(
            self,
            selector: Any = None,
            level: str = "feature",
            col_set: str = "__all",
        ) -> pd.DataFrame:
            """Fetch data with crypto-specific handling."""
            if self._crypto_data is None:
                return pd.DataFrame()

            df = self._crypto_data.copy()

            # Apply selector
            if selector is not None and "symbol" in df.columns:
                if isinstance(selector, str):
                    df = df[df["symbol"] == selector]
                elif isinstance(selector, (list, tuple)):
                    df = df[df["symbol"].isin(selector)]

            # Apply column set filter
            if col_set != "__all":
                cols = self.get_cols(col_set)
                available = [c for c in cols if c in df.columns]
                df = df[available]

            return df

        def get_cols(self, col_set: str = "__all") -> list[str]:
            """Get column names with crypto-specific columns."""
            if self._crypto_data is None:
                return []

            if col_set == "__all":
                return list(self._crypto_data.columns)

            feature_cols = [
                "open", "high", "low", "close", "volume",
                "funding_rate", "open_interest", "basis",
            ]
            label_cols = ["label", "target", "return"]

            if col_set == "feature":
                return [c for c in feature_cols if c in self._crypto_data.columns]
            elif col_set == "label":
                return [c for c in label_cols if c in self._crypto_data.columns]

            return list(self._crypto_data.columns)

        # === Crypto-Specific Methods ===

        def load(
            self,
            data: pd.DataFrame,
            exchange: Optional[Exchange] = None,
            validate: bool = False,
            fill_na: bool = False,
        ) -> None:
            """Load cryptocurrency data into the handler."""
            if data.empty:
                if validate:
                    raise ValidationError("DataFrame is empty")
                self._crypto_data = data
                return

            # Apply exchange-specific column mappings
            if exchange:
                data = self._normalize_columns(data, exchange)
            elif self.config.exchange:
                data = self._normalize_columns(data, self.config.exchange)

            # Normalize column names
            data = data.copy()
            data.columns = [col.lower() for col in data.columns]

            # Validate if requested
            if validate:
                result = self.validate(data)
                if not result.is_valid:
                    raise ValidationError(
                        f"Validation failed: missing required columns: {result.errors}"
                    )

            # Fill NaN values if requested
            if fill_na:
                data = self._fill_na(data)

            self._crypto_data = data

        def validate(self, data: Optional[pd.DataFrame] = None) -> ValidationResult:
            """Validate cryptocurrency data."""
            if data is None:
                data = self._crypto_data
            if data is None:
                return ValidationResult(is_valid=False, errors=["No data to validate"])
            return self._validator.validate(data)

        def get_field(self, field: CryptoField) -> Optional[pd.Series]:
            """Get a specific field from the loaded data."""
            if self._crypto_data is None:
                raise RuntimeError("Data not loaded. Call load() first.")
            field_name = field.value
            if field_name in self._crypto_data.columns:
                return self._crypto_data[field_name]
            return None

        def get_symbols(self) -> list[str]:
            """Get list of symbols in the data."""
            if self._crypto_data is None:
                return []
            if "symbol" in self._crypto_data.columns:
                return list(self._crypto_data["symbol"].unique())
            return []

        def calculate_returns(self) -> Optional[pd.Series]:
            """Calculate simple returns from close prices."""
            if self._crypto_data is None:
                return None
            close = self.get_field(CryptoField.CLOSE)
            if close is None:
                return None
            return close.pct_change().dropna()

        def calculate_funding_adjusted_returns(self) -> Optional[pd.Series]:
            """Calculate funding-adjusted returns for perpetual futures."""
            if self._crypto_data is None:
                return None
            close = self.get_field(CryptoField.CLOSE)
            funding = self.get_field(CryptoField.FUNDING_RATE)
            if close is None:
                return None
            returns = close.pct_change()
            if funding is not None:
                returns = returns - funding
            return returns.dropna()

        def resample(self, timeframe: TimeFrame) -> pd.DataFrame:
            """Resample data to a different timeframe."""
            if self._crypto_data is None:
                raise RuntimeError("Data not loaded. Call load() first.")

            current_minutes = self._estimate_data_frequency()
            target_minutes = timeframe.to_minutes()

            if target_minutes < current_minutes:
                raise ValueError(
                    f"Cannot resample from {current_minutes}min to "
                    f"{target_minutes}min (upsampling not supported)"
                )

            datetime_col = self._find_datetime_column()
            if datetime_col is None:
                raise ValueError("No datetime column found for resampling")

            df = self._crypto_data.copy()
            df = df.set_index(datetime_col)

            agg_rules: dict[str, Any] = {}
            if "open" in df.columns:
                agg_rules["open"] = "first"
            if "high" in df.columns:
                agg_rules["high"] = "max"
            if "low" in df.columns:
                agg_rules["low"] = "min"
            if "close" in df.columns:
                agg_rules["close"] = "last"
            if "volume" in df.columns:
                agg_rules["volume"] = "sum"
            if "funding_rate" in df.columns:
                agg_rules["funding_rate"] = "sum"
            if "open_interest" in df.columns:
                agg_rules["open_interest"] = "last"

            freq = timeframe.to_pandas_freq()
            resampled = df.resample(freq).agg(agg_rules)
            return resampled.reset_index()

        def to_qlib_format(self) -> pd.DataFrame:
            """Convert data to Qlib-compatible format with $ prefixes."""
            if self._crypto_data is None:
                raise RuntimeError("Data not loaded. Call load() first.")

            df = self._crypto_data.copy()
            column_mapping = {
                "open": "$open", "high": "$high", "low": "$low",
                "close": "$close", "volume": "$volume",
                "funding_rate": "$funding_rate",
                "open_interest": "$open_interest", "basis": "$basis",
            }
            rename_map = {k: v for k, v in column_mapping.items() if k in df.columns}
            return df.rename(columns=rename_map)

        # === Funding Rate Methods ===

        def align_funding_rate(self, method: str = "ffill") -> pd.DataFrame:
            """Align sparse funding rate data to match OHLCV frequency.

            Funding rates are typically settled every 8 hours (00:00, 08:00, 16:00 UTC),
            but OHLCV data may be hourly or more granular. This method aligns the
            sparse funding rate data to match the OHLCV frequency.

            Args:
                method: Alignment method:
                    - "ffill": Forward fill (carry last known rate)
                    - "interpolate": Linear interpolation between settlements
                    - "distribute": Distribute rate evenly (rate / 8 for hourly data)

            Returns:
                DataFrame with aligned funding_rate column.
            """
            if self._crypto_data is None:
                raise RuntimeError("Data not loaded. Call load() first.")

            df = self._crypto_data.copy()

            if "funding_rate" not in df.columns:
                return df

            if method == "ffill":
                df["funding_rate"] = df["funding_rate"].ffill()
            elif method == "interpolate":
                df["funding_rate"] = df["funding_rate"].interpolate(method="linear")
            elif method == "distribute":
                # Count periods between funding settlements (typically 8 hours)
                settlement_interval = 8  # hours
                df["funding_rate"] = df["funding_rate"].ffill() / settlement_interval
            else:
                raise ValueError(f"Unknown alignment method: {method}")

            return df

        # === Basis Calculation Methods ===

        def calculate_basis(self, method: str = "absolute") -> Optional[pd.Series]:
            """Calculate basis (spread between mark price and index price).

            Args:
                method: Calculation method:
                    - "absolute": mark_price - index_price
                    - "percentage": (mark_price - index_price) / index_price * 100

            Returns:
                Series with basis values, or None if mark/index prices unavailable.
            """
            if self._crypto_data is None:
                return None

            mark = self.get_field(CryptoField.MARK_PRICE)
            index = self.get_field(CryptoField.INDEX_PRICE)

            if mark is None or index is None:
                return None

            if method == "absolute":
                return mark - index
            elif method == "percentage":
                return (mark - index) / index * 100
            else:
                raise ValueError(f"Unknown basis calculation method: {method}")

        def calculate_annualized_basis(self) -> Optional[pd.Series]:
            """Calculate annualized basis rate.

            Returns:
                Series with annualized basis rate (percentage), or None if unavailable.
            """
            basis_pct = self.calculate_basis(method="percentage")
            if basis_pct is None:
                return None

            # Estimate data frequency to annualize
            freq_minutes = self._estimate_data_frequency()
            periods_per_year = (365 * 24 * 60) / freq_minutes

            return basis_pct * periods_per_year / 100  # Convert to decimal rate

        # === Open Interest Methods ===

        def calculate_oi_change(self, method: str = "absolute") -> Optional[pd.Series]:
            """Calculate open interest change rate.

            Args:
                method: Calculation method:
                    - "absolute": Absolute change (diff)
                    - "percentage": Percentage change

            Returns:
                Series with OI change values, or None if OI unavailable.
            """
            if self._crypto_data is None:
                return None

            oi = self.get_field(CryptoField.OPEN_INTEREST)
            if oi is None:
                return None

            if method == "absolute":
                return oi.diff().dropna()
            elif method == "percentage":
                # Handle division by zero gracefully
                pct_change = oi.pct_change() * 100
                # Replace inf values with NaN, then forward fill or use 0
                pct_change = pct_change.replace([np.inf, -np.inf], np.nan)
                return pct_change.dropna()
            else:
                raise ValueError(f"Unknown OI change method: {method}")

        def normalize_oi(self, method: str = "contract_value") -> Optional[pd.Series]:
            """Normalize open interest to contract value.

            Args:
                method: Normalization method:
                    - "contract_value": OI * close price (notional value)

            Returns:
                Series with normalized OI values, or None if OI unavailable.
            """
            if self._crypto_data is None:
                return None

            oi = self.get_field(CryptoField.OPEN_INTEREST)
            close = self.get_field(CryptoField.CLOSE)

            if oi is None or close is None:
                return None

            if method == "contract_value":
                return oi * close
            else:
                raise ValueError(f"Unknown OI normalization method: {method}")

        # === Exchange Detection Methods ===

        def detect_exchange_format(self, data: pd.DataFrame) -> Exchange:
            """Auto-detect exchange format from column names.

            Args:
                data: DataFrame to analyze.

            Returns:
                Detected Exchange enum value.
            """
            columns = set(col.lower() for col in data.columns)

            # OKX has distinctive short column names
            if {"ts", "instid", "o", "h", "l", "c"}.intersection(columns):
                return Exchange.OKX

            # Bybit uses turnover and openInterest
            if "turnover" in columns or "openinterest" in columns:
                return Exchange.BYBIT

            # Binance uses openTime and camelCase
            if "opentime" in columns or "closetime" in columns:
                return Exchange.BINANCE

            # Default to Binance (most common format)
            return Exchange.BINANCE

        # === Private Helpers ===

        def _normalize_columns(
            self, data: pd.DataFrame, exchange: Exchange
        ) -> pd.DataFrame:
            """Normalize column names based on exchange format."""
            if exchange not in EXCHANGE_COLUMN_MAPPINGS:
                return data
            mapping = EXCHANGE_COLUMN_MAPPINGS[exchange]
            df = data.copy()
            existing = {col.lower(): col for col in df.columns}
            rename_map = {}
            for old_name, new_name in mapping.items():
                if old_name.lower() in existing:
                    rename_map[existing[old_name.lower()]] = new_name
            if rename_map:
                df = df.rename(columns=rename_map)
            return df

        def _fill_na(self, data: pd.DataFrame) -> pd.DataFrame:
            """Fill NaN values in the data."""
            df = data.copy()
            price_cols = ["open", "high", "low", "close"]
            for col in price_cols:
                if col in df.columns:
                    df[col] = df[col].ffill()
            if "volume" in df.columns:
                df["volume"] = df["volume"].fillna(0)
            if "funding_rate" in df.columns:
                df["funding_rate"] = df["funding_rate"].fillna(0)
            return df

        def _find_datetime_column(self) -> Optional[str]:
            """Find the datetime column in loaded data."""
            if self._crypto_data is None:
                return None
            for col in ["datetime", "timestamp", "date", "time", "open_time"]:
                if col in self._crypto_data.columns:
                    return col
            return None

        def _estimate_data_frequency(self) -> int:
            """Estimate the data frequency in minutes."""
            if self._crypto_data is None or len(self._crypto_data) < 2:
                return 60
            datetime_col = self._find_datetime_column()
            if datetime_col is None:
                return 60
            dt = pd.to_datetime(self._crypto_data[datetime_col])
            diff = dt.diff().dropna()
            if len(diff) == 0:
                return 60
            return int(diff.median().total_seconds() / 60)

    return _CryptoDataHandler


# Create the actual class via factory
CryptoDataHandler = _create_crypto_data_handler()
