"""Cryptocurrency Data Handler for Qlib.

This module provides CryptoDataHandler, a Qlib-compatible data handler
that supports cryptocurrency-specific data fields.

Deep Fork Strategy:
- If Qlib is installed: Inherits from DataHandlerLP for full compatibility
- If Qlib is not installed: Uses standalone implementation
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

from iqfmp.qlib_crypto.data.validator import (
    DataValidator,
    ValidationError,
    ValidationResult,
)

# Try to import Qlib components
QLIB_AVAILABLE = False
_QlibDataHandlerLP = None

try:
    from qlib.data.dataset.handler import DataHandlerLP as QlibDataHandlerLP
    QLIB_AVAILABLE = True
    _QlibDataHandlerLP = QlibDataHandlerLP
except (ImportError, LookupError, Exception):
    # Qlib not available or has version detection issues
    pass


class CryptoField(str, Enum):
    """Cryptocurrency data fields."""

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
    """Supported timeframes."""

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
    """Configuration for CryptoDataHandler."""

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
        "open_time": "datetime",
        "close_time": "_close_time",
        "quote_volume": "quote_volume",
        "trades": "trades",
    },
    Exchange.OKX: {
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
        "timestamp": "datetime",
        "turnover": "quote_volume",
    },
}


# === Base class for standalone mode ===
class _StandaloneDataHandler:
    """Standalone base class when Qlib is not available."""
    pass


# === Factory function to create the handler class ===
def _create_crypto_data_handler():
    """Create CryptoDataHandler with conditional Qlib inheritance."""

    # Choose base class
    if QLIB_AVAILABLE and _QlibDataHandlerLP is not None:
        BaseClass = _QlibDataHandlerLP
    else:
        BaseClass = _StandaloneDataHandler

    class _CryptoDataHandler(BaseClass):  # type: ignore[misc]
        """Data handler for cryptocurrency market data.

        This class provides a Qlib-compatible interface for loading,
        validating, and transforming cryptocurrency market data.

        Deep Fork Integration:
        - When Qlib is installed: Inherits from DataHandlerLP
        - When Qlib is not installed: Standalone implementation
        - Always implements Qlib-compatible interface (fetch, get_cols)
        """

        def __init__(
            self,
            config: Optional[CryptoDataConfig] = None,
            **kwargs: Any,
        ) -> None:
            """Initialize the data handler."""
            if QLIB_AVAILABLE and _QlibDataHandlerLP is not None:
                super().__init__(**kwargs)

            self.config = config or CryptoDataConfig()
            self.data: Optional[pd.DataFrame] = None
            self._validator = DataValidator()

        # === Qlib-compatible interface ===

        def fetch(
            self,
            selector: Any = None,
            level: str = "feature",
            col_set: str = "__all",
        ) -> pd.DataFrame:
            """Fetch data (Qlib-compatible interface)."""
            if self.data is None:
                return pd.DataFrame()

            df = self.data.copy()

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
            """Get column names (Qlib-compatible interface)."""
            if self.data is None:
                return []

            if col_set == "__all":
                return list(self.data.columns)

            feature_cols = [
                "open", "high", "low", "close", "volume",
                "funding_rate", "open_interest", "basis",
            ]
            label_cols = ["label", "target", "return"]

            if col_set == "feature":
                return [c for c in feature_cols if c in self.data.columns]
            elif col_set == "label":
                return [c for c in label_cols if c in self.data.columns]

            return list(self.data.columns)

        # === Core methods ===

        def load(
            self,
            data: pd.DataFrame,
            exchange: Optional[Exchange] = None,
            validate: bool = False,
            fill_na: bool = False,
        ) -> None:
            """Load data into the handler."""
            if data.empty:
                if validate:
                    raise ValidationError("DataFrame is empty")
                self.data = data
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

            self.data = data

        def validate(self, data: Optional[pd.DataFrame] = None) -> ValidationResult:
            """Validate data."""
            if data is None:
                data = self.data
            if data is None:
                return ValidationResult(is_valid=False, errors=["No data to validate"])
            return self._validator.validate(data)

        def get_field(self, field: CryptoField) -> Optional[pd.Series]:
            """Get a specific field from the loaded data."""
            if self.data is None:
                raise RuntimeError("Data not loaded. Call load() first.")
            field_name = field.value
            if field_name in self.data.columns:
                return self.data[field_name]
            return None

        def get_symbols(self) -> list[str]:
            """Get list of symbols in the data."""
            if self.data is None:
                return []
            if "symbol" in self.data.columns:
                return list(self.data["symbol"].unique())
            return []

        def calculate_returns(self) -> Optional[pd.Series]:
            """Calculate simple returns from close prices."""
            if self.data is None:
                return None
            close = self.get_field(CryptoField.CLOSE)
            if close is None:
                return None
            return close.pct_change().dropna()

        def calculate_funding_adjusted_returns(self) -> Optional[pd.Series]:
            """Calculate funding-adjusted returns for perpetual futures."""
            if self.data is None:
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
            if self.data is None:
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

            df = self.data.copy()
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
            """Convert data to Qlib-compatible format."""
            if self.data is None:
                raise RuntimeError("Data not loaded. Call load() first.")

            df = self.data.copy()
            column_mapping = {
                "open": "$open", "high": "$high", "low": "$low",
                "close": "$close", "volume": "$volume",
                "funding_rate": "$funding_rate",
                "open_interest": "$open_interest", "basis": "$basis",
            }
            rename_map = {k: v for k, v in column_mapping.items() if k in df.columns}
            return df.rename(columns=rename_map)

        # === Private helpers ===

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
            if self.data is None:
                return None
            for col in ["datetime", "timestamp", "date", "time", "open_time"]:
                if col in self.data.columns:
                    return col
            return None

        def _estimate_data_frequency(self) -> int:
            """Estimate the data frequency in minutes."""
            if self.data is None or len(self.data) < 2:
                return 60
            datetime_col = self._find_datetime_column()
            if datetime_col is None:
                return 60
            dt = pd.to_datetime(self.data[datetime_col])
            diff = dt.diff().dropna()
            if len(diff) == 0:
                return 60
            return int(diff.median().total_seconds() / 60)

    return _CryptoDataHandler


# Create the actual class
CryptoDataHandler = _create_crypto_data_handler()
