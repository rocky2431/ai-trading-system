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

    # P4.3: Order Book / Microstructure Fields
    BID_PRICE = "bid_price"  # Best bid price (L1)
    ASK_PRICE = "ask_price"  # Best ask price (L1)
    BID_SIZE = "bid_size"  # Best bid quantity
    ASK_SIZE = "ask_size"  # Best ask quantity
    MID_PRICE = "mid_price"  # (bid + ask) / 2
    SPREAD = "spread"  # ask - bid
    SPREAD_BPS = "spread_bps"  # Spread in basis points
    ORDER_BOOK_IMBALANCE = "order_book_imbalance"  # (bid_size - ask_size) / (bid_size + ask_size)
    DEPTH_BID_5 = "depth_bid_5"  # Sum of top 5 bid quantities
    DEPTH_ASK_5 = "depth_ask_5"  # Sum of top 5 ask quantities
    DEPTH_IMBALANCE_5 = "depth_imbalance_5"  # Imbalance at 5 levels
    VWAP_BID_5 = "vwap_bid_5"  # Volume-weighted avg bid price (5 levels)
    VWAP_ASK_5 = "vwap_ask_5"  # Volume-weighted avg ask price (5 levels)


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

        # === P4.3: Order Book / Microstructure Methods ===

        def calculate_spread(self, method: str = "absolute") -> Optional[pd.Series]:
            """Calculate bid-ask spread from L1 order book data.

            Args:
                method: Calculation method:
                    - "absolute": ask_price - bid_price
                    - "percentage": (ask - bid) / mid_price * 100
                    - "bps": (ask - bid) / mid_price * 10000 (basis points)

            Returns:
                Series with spread values, or None if bid/ask prices unavailable.
            """
            if self._crypto_data is None:
                return None

            bid = self.get_field(CryptoField.BID_PRICE)
            ask = self.get_field(CryptoField.ASK_PRICE)

            if bid is None or ask is None:
                return None

            spread = ask - bid

            if method == "absolute":
                return spread
            elif method == "percentage":
                mid = (bid + ask) / 2
                return (spread / mid) * 100
            elif method == "bps":
                mid = (bid + ask) / 2
                return (spread / mid) * 10000
            else:
                raise ValueError(f"Unknown spread calculation method: {method}")

        def calculate_mid_price(self) -> Optional[pd.Series]:
            """Calculate mid price from L1 order book data.

            Returns:
                Series with mid prices, or None if bid/ask unavailable.
            """
            if self._crypto_data is None:
                return None

            bid = self.get_field(CryptoField.BID_PRICE)
            ask = self.get_field(CryptoField.ASK_PRICE)

            if bid is None or ask is None:
                return None

            return (bid + ask) / 2

        def calculate_order_book_imbalance(self) -> Optional[pd.Series]:
            """Calculate L1 order book imbalance.

            Formula: (bid_size - ask_size) / (bid_size + ask_size)
            Range: -1 (all ask) to +1 (all bid)
            Positive values indicate buying pressure.

            Returns:
                Series with imbalance values, or None if sizes unavailable.
            """
            if self._crypto_data is None:
                return None

            bid_size = self.get_field(CryptoField.BID_SIZE)
            ask_size = self.get_field(CryptoField.ASK_SIZE)

            if bid_size is None or ask_size is None:
                return None

            total = bid_size + ask_size
            # Handle division by zero
            imbalance = (bid_size - ask_size) / total.replace(0, np.nan)
            return imbalance

        def calculate_depth_imbalance(self, levels: int = 5) -> Optional[pd.Series]:
            """Calculate order book depth imbalance at specified levels.

            Uses pre-aggregated depth fields (depth_bid_5, depth_ask_5).

            Args:
                levels: Number of order book levels (default 5).

            Returns:
                Series with depth imbalance values, or None if unavailable.
            """
            if self._crypto_data is None:
                return None

            if levels == 5:
                bid_depth = self.get_field(CryptoField.DEPTH_BID_5)
                ask_depth = self.get_field(CryptoField.DEPTH_ASK_5)
            else:
                # For other levels, try generic column names
                bid_col = f"depth_bid_{levels}"
                ask_col = f"depth_ask_{levels}"
                bid_depth = self._crypto_data.get(bid_col)
                ask_depth = self._crypto_data.get(ask_col)

            if bid_depth is None or ask_depth is None:
                return None

            total = bid_depth + ask_depth
            imbalance = (bid_depth - ask_depth) / total.replace(0, np.nan)
            return imbalance

        def calculate_microprice(self) -> Optional[pd.Series]:
            """Calculate microprice (volume-weighted mid price).

            Formula: (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)
            More accurate than simple mid price for predicting short-term direction.

            Returns:
                Series with microprice values, or None if L1 data unavailable.
            """
            if self._crypto_data is None:
                return None

            bid = self.get_field(CryptoField.BID_PRICE)
            ask = self.get_field(CryptoField.ASK_PRICE)
            bid_size = self.get_field(CryptoField.BID_SIZE)
            ask_size = self.get_field(CryptoField.ASK_SIZE)

            if any(x is None for x in [bid, ask, bid_size, ask_size]):
                return None

            total_size = bid_size + ask_size
            microprice = (bid * ask_size + ask * bid_size) / total_size.replace(0, np.nan)
            return microprice

        def calculate_effective_spread(
            self,
            trade_price: Optional[pd.Series] = None,
        ) -> Optional[pd.Series]:
            """Calculate effective spread from trade execution prices.

            Formula: 2 * |trade_price - mid_price|
            Measures actual execution cost vs theoretical cost.

            Args:
                trade_price: Series of actual trade prices.
                    If None, uses close price as proxy.

            Returns:
                Series with effective spread values.
            """
            if self._crypto_data is None:
                return None

            mid = self.calculate_mid_price()
            if mid is None:
                return None

            if trade_price is None:
                trade_price = self.get_field(CryptoField.CLOSE)

            if trade_price is None:
                return None

            return 2 * (trade_price - mid).abs()

        def load_orderbook_snapshot(
            self,
            bids: list[tuple[float, float]],
            asks: list[tuple[float, float]],
            timestamp: Optional[pd.Timestamp] = None,
            levels: int = 5,
        ) -> dict[str, float]:
            """Process a single order book snapshot into aggregated features.

            Args:
                bids: List of (price, quantity) tuples for bid side, sorted best first.
                asks: List of (price, quantity) tuples for ask side, sorted best first.
                timestamp: Optional timestamp for this snapshot.
                levels: Number of levels to process (default 5).

            Returns:
                Dictionary with calculated order book features.
            """
            features: dict[str, float] = {}

            if timestamp:
                features["datetime"] = timestamp

            # L1 data
            if bids:
                features["bid_price"] = bids[0][0]
                features["bid_size"] = bids[0][1]
            if asks:
                features["ask_price"] = asks[0][0]
                features["ask_size"] = asks[0][1]

            # Mid price and spread
            if bids and asks:
                mid = (bids[0][0] + asks[0][0]) / 2
                spread = asks[0][0] - bids[0][0]
                features["mid_price"] = mid
                features["spread"] = spread
                features["spread_bps"] = (spread / mid) * 10000 if mid > 0 else 0

                # L1 imbalance
                total_l1 = bids[0][1] + asks[0][1]
                if total_l1 > 0:
                    features["order_book_imbalance"] = (
                        bids[0][1] - asks[0][1]
                    ) / total_l1

            # Depth aggregation
            bid_depth = sum(qty for _, qty in bids[:levels])
            ask_depth = sum(qty for _, qty in asks[:levels])
            features[f"depth_bid_{levels}"] = bid_depth
            features[f"depth_ask_{levels}"] = ask_depth

            total_depth = bid_depth + ask_depth
            if total_depth > 0:
                features[f"depth_imbalance_{levels}"] = (
                    bid_depth - ask_depth
                ) / total_depth

            # VWAP calculation
            if bids:
                bid_value = sum(p * q for p, q in bids[:levels])
                bid_vol = sum(q for _, q in bids[:levels])
                if bid_vol > 0:
                    features[f"vwap_bid_{levels}"] = bid_value / bid_vol

            if asks:
                ask_value = sum(p * q for p, q in asks[:levels])
                ask_vol = sum(q for _, q in asks[:levels])
                if ask_vol > 0:
                    features[f"vwap_ask_{levels}"] = ask_value / ask_vol

            return features

        def process_orderbook_stream(
            self,
            snapshots: list[dict[str, Any]],
            levels: int = 5,
        ) -> pd.DataFrame:
            """Process a stream of order book snapshots into a DataFrame.

            Args:
                snapshots: List of order book snapshots, each with:
                    - "bids": List of (price, qty) tuples
                    - "asks": List of (price, qty) tuples
                    - "timestamp": Optional timestamp
                levels: Number of levels to process.

            Returns:
                DataFrame with aggregated order book features.
            """
            processed = []
            for snapshot in snapshots:
                bids = snapshot.get("bids", [])
                asks = snapshot.get("asks", [])
                timestamp = snapshot.get("timestamp")
                if isinstance(timestamp, str):
                    timestamp = pd.Timestamp(timestamp)
                features = self.load_orderbook_snapshot(bids, asks, timestamp, levels)
                processed.append(features)

            return pd.DataFrame(processed)

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
