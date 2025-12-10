"""Crypto market data handler for Qlib.

Provides CryptoDataHandler and CryptoField classes for handling
cryptocurrency OHLCV data within Qlib's framework.

This module extends Qlib's data handling capabilities to support:
- 24/7 trading (no market close)
- High-frequency data (1m, 5m, 15m, 1h, 4h, 1d)
- Crypto-specific features (funding rate, open interest, liquidations)
"""

import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CryptoField(Enum):
    """Standard fields for crypto market data."""

    # OHLCV fields
    OPEN = "$open"
    HIGH = "$high"
    LOW = "$low"
    CLOSE = "$close"
    VOLUME = "$volume"

    # Derived fields
    VWAP = "$vwap"
    TYPICAL_PRICE = "$typical"
    DOLLAR_VOLUME = "$dollar_volume"

    # Crypto-specific fields
    FUNDING_RATE = "$funding_rate"
    OPEN_INTEREST = "$open_interest"
    LONG_RATIO = "$long_ratio"
    SHORT_RATIO = "$short_ratio"
    TAKER_BUY_VOLUME = "$taker_buy_vol"
    TAKER_SELL_VOLUME = "$taker_sell_vol"

    # Technical indicators (pre-computed for efficiency)
    RSI_14 = "$rsi_14"
    MACD = "$macd"
    MACD_SIGNAL = "$macd_signal"
    MACD_HIST = "$macd_hist"
    BOLLINGER_UPPER = "$bb_upper"
    BOLLINGER_LOWER = "$bb_lower"
    ATR_14 = "$atr_14"


class CryptoDataHandler:
    """Data handler for cryptocurrency market data.

    Compatible with Qlib's DataHandlerLP interface but optimized for
    crypto markets with 24/7 trading and high-frequency data.

    Features:
    - Load data from CSV, Parquet, or database
    - Compute derived fields (VWAP, typical price, etc.)
    - Handle multiple timeframes
    - Support resampling and alignment
    """

    def __init__(
        self,
        instruments: Optional[list[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        data_dir: Optional[Path] = None,
        timeframe: str = "1d",
    ):
        """Initialize crypto data handler.

        Args:
            instruments: List of crypto pairs (e.g., ["BTCUSDT", "ETHUSDT"])
            start_time: Start datetime string
            end_time: End datetime string
            data_dir: Directory containing data files
            timeframe: Data timeframe (1m, 5m, 15m, 1h, 4h, 1d)
        """
        self.instruments = instruments or []
        self.start_time = start_time
        self.end_time = end_time
        self.data_dir = data_dir
        self.timeframe = timeframe

        self._data: dict[str, pd.DataFrame] = {}
        self._combined_data: Optional[pd.DataFrame] = None

    def load_data(
        self,
        path: Optional[Union[str, Path]] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Load data from file or DataFrame.

        Args:
            path: Path to data file (CSV or Parquet)
            df: Pre-loaded DataFrame
        """
        if df is not None:
            self._process_dataframe(df)
            return

        if path is None:
            raise ValueError("Either path or df must be provided")

        path = Path(path)

        if path.suffix == ".csv":
            df = pd.read_csv(path)
        elif path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        self._process_dataframe(df)

    def _process_dataframe(self, df: pd.DataFrame) -> None:
        """Process and standardize DataFrame.

        Args:
            df: Raw DataFrame with OHLCV data
        """
        # Standardize column names
        column_mapping = {
            "timestamp": "datetime",
            "time": "datetime",
            "date": "datetime",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "vol": "volume",
            "symbol": "symbol",
            "instrument": "symbol",
        }

        df = df.rename(columns={
            k: v for k, v in column_mapping.items()
            if k in df.columns
        })

        # Ensure datetime column
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)

        # Add Qlib-style fields
        df["$open"] = df["open"]
        df["$high"] = df["high"]
        df["$low"] = df["low"]
        df["$close"] = df["close"]
        df["$volume"] = df["volume"]

        # Compute derived fields
        df["$vwap"] = self._compute_vwap(df)
        df["$typical"] = (df["high"] + df["low"] + df["close"]) / 3
        df["$dollar_volume"] = df["volume"] * df["close"]

        # Store data
        symbol = df.get("symbol", pd.Series(["UNKNOWN"] * len(df))).iloc[0]
        if isinstance(symbol, (pd.Series, np.ndarray)):
            symbol = str(symbol)
        self._data[str(symbol)] = df
        self._combined_data = df

    def _compute_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Compute Volume Weighted Average Price."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cum_tp_vol = (typical_price * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum()
        return cum_tp_vol / (cum_vol + 1e-10)

    def get_data(
        self,
        instruments: Optional[list[str]] = None,
        fields: Optional[list[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get data for specified instruments and fields.

        Args:
            instruments: List of instruments (None for all)
            fields: List of field names (None for all)
            start_time: Start datetime
            end_time: End datetime

        Returns:
            DataFrame with requested data
        """
        if self._combined_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        df = self._combined_data.copy()

        # Filter by time
        if start_time and "datetime" in df.columns:
            df = df[df["datetime"] >= pd.to_datetime(start_time)]
        if end_time and "datetime" in df.columns:
            df = df[df["datetime"] <= pd.to_datetime(end_time)]

        # Filter by fields
        if fields:
            available_cols = ["datetime", "symbol"] + [
                c for c in fields if c in df.columns
            ]
            df = df[available_cols]

        return df

    def compute_factor(
        self,
        expression: str,
        factor_name: str = "factor",
    ) -> pd.Series:
        """Compute factor from expression.

        Supports Qlib expression syntax with $ field references.

        Args:
            expression: Factor expression
            factor_name: Name for the result series

        Returns:
            Series of factor values
        """
        if self._combined_data is None:
            raise ValueError("No data loaded")

        df = self._combined_data

        # Simple expression evaluation using pandas eval
        try:
            # Replace $ with column names
            expr = expression
            for field in CryptoField:
                expr = expr.replace(field.value, f"`{field.value}`")

            result = df.eval(expr)
            return pd.Series(result, name=factor_name, index=df.index)
        except Exception:
            # Fall back to manual parsing
            return self._evaluate_expression(expression, df, factor_name)

    def _evaluate_expression(
        self,
        expression: str,
        df: pd.DataFrame,
        factor_name: str,
    ) -> pd.Series:
        """Evaluate expression manually."""
        # Handle direct field references
        if expression.startswith("$"):
            if expression in df.columns:
                return df[expression].rename(factor_name)
            raise ValueError(f"Unknown field: {expression}")

        # Handle basic operations
        # This is a simplified parser - complex expressions should use QlibFactorEngine
        return pd.Series(np.nan, index=df.index, name=factor_name)

    def add_technical_indicators(self) -> None:
        """Add pre-computed technical indicators to data."""
        if self._combined_data is None:
            return

        df = self._combined_data

        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df["$rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        df["$macd"] = ema_12 - ema_26
        df["$macd_signal"] = df["$macd"].ewm(span=9).mean()
        df["$macd_hist"] = df["$macd"] - df["$macd_signal"]

        # Bollinger Bands
        sma_20 = df["close"].rolling(20).mean()
        std_20 = df["close"].rolling(20).std()
        df["$bb_upper"] = sma_20 + 2 * std_20
        df["$bb_lower"] = sma_20 - 2 * std_20

        # ATR
        tr = pd.DataFrame({
            "hl": df["high"] - df["low"],
            "hc": (df["high"] - df["close"].shift()).abs(),
            "lc": (df["low"] - df["close"].shift()).abs(),
        }).max(axis=1)
        df["$atr_14"] = tr.rolling(14).mean()

        self._combined_data = df

    def resample(self, timeframe: str) -> "CryptoDataHandler":
        """Resample data to a different timeframe.

        Args:
            timeframe: Target timeframe (1m, 5m, 15m, 1h, 4h, 1d)

        Returns:
            New CryptoDataHandler with resampled data
        """
        if self._combined_data is None:
            raise ValueError("No data to resample")

        # Map timeframe strings to pandas resample rules
        tf_map = {
            "1m": "1T",
            "5m": "5T",
            "15m": "15T",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D",
        }

        rule = tf_map.get(timeframe, "1D")
        df = self._combined_data.copy()

        if "datetime" in df.columns:
            df = df.set_index("datetime")

        resampled = df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        resampled = resampled.reset_index()

        new_handler = CryptoDataHandler(
            instruments=self.instruments,
            timeframe=timeframe,
        )
        new_handler._process_dataframe(resampled)

        return new_handler

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Get combined data DataFrame."""
        return self._combined_data

    @property
    def instruments_list(self) -> list[str]:
        """Get list of loaded instruments."""
        return list(self._data.keys())


class CryptoDataLoader:
    """Utility class for loading crypto data from various sources."""

    @staticmethod
    def from_csv(
        path: Union[str, Path],
        instruments: Optional[list[str]] = None,
    ) -> CryptoDataHandler:
        """Load data from CSV file.

        Args:
            path: Path to CSV file
            instruments: List of instruments to filter

        Returns:
            CryptoDataHandler with loaded data
        """
        handler = CryptoDataHandler(instruments=instruments)
        handler.load_data(path=path)
        return handler

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        instruments: Optional[list[str]] = None,
    ) -> CryptoDataHandler:
        """Create handler from DataFrame.

        Args:
            df: DataFrame with OHLCV data
            instruments: List of instruments

        Returns:
            CryptoDataHandler with loaded data
        """
        handler = CryptoDataHandler(instruments=instruments)
        handler.load_data(df=df)
        return handler

    @staticmethod
    def from_exchange(
        exchange: str,
        symbol: str,
        timeframe: str = "1d",
        limit: int = 1000,
    ) -> CryptoDataHandler:
        """Load data from crypto exchange API.

        Args:
            exchange: Exchange name (binance, okx, etc.)
            symbol: Trading pair (BTCUSDT)
            timeframe: Data timeframe
            limit: Number of candles to fetch

        Returns:
            CryptoDataHandler with loaded data
        """
        try:
            import ccxt

            exchange_class = getattr(ccxt, exchange.lower())
            ex = exchange_class()
            ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=limit)

            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["symbol"] = symbol

            handler = CryptoDataHandler(instruments=[symbol], timeframe=timeframe)
            handler.load_data(df=df)
            return handler

        except ImportError:
            raise ImportError("ccxt library required for exchange data loading")
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {exchange}: {e}")
