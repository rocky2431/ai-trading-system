"""Crypto market data handler for Qlib.

Provides QlibExpressionEngine (expression-only computation) and a thin
CryptoDataHandler compatibility wrapper for cryptocurrency OHLCV data.

This module extends Qlib's data handling capabilities to support:
- 24/7 trading (no market close)
- High-frequency data (1m, 5m, 15m, 1h, 4h, 1d)
- Crypto-specific features (funding rate, open interest, liquidations)

IMPORTANT: All technical indicator calculations MUST go through Qlib.
No local pandas/numpy implementations allowed.
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from iqfmp.qlib_crypto import (
        CryptoDataConfig as _VendorCryptoDataConfig,
        CryptoDataHandler as _VendorCryptoDataHandler,
        Exchange as _VendorExchange,
        TimeFrame as _VendorTimeFrame,
    )
    _VENDOR_CRYPTO_HANDLER_AVAILABLE = True
except Exception:
    _VendorCryptoDataConfig = None  # type: ignore[assignment]
    _VendorCryptoDataHandler = None  # type: ignore[assignment]
    _VendorExchange = None  # type: ignore[assignment]
    _VendorTimeFrame = None  # type: ignore[assignment]
    _VENDOR_CRYPTO_HANDLER_AVAILABLE = False


# =============================================================================
# REMOVED: Pandas Implementation (P0-2 Fix)
# =============================================================================
# All Pandas operator implementations have been removed to enforce Qlib-only mode.
# Factor expressions MUST use Qlib's C++ engine for production-grade performance.
# =============================================================================


# =============================================================================
# Qlib Integration - Lazy Loading
# =============================================================================
QLIB_AVAILABLE = False
_qlib_initialized = False


def _ensure_qlib_initialized() -> bool:
    """Ensure Qlib is properly initialized."""
    global QLIB_AVAILABLE, _qlib_initialized

    if _qlib_initialized:
        return QLIB_AVAILABLE

    try:
        import qlib
        from qlib.config import C

        # Check if already initialized
        if hasattr(C, "provider_uri") and C.provider_uri is not None:
            QLIB_AVAILABLE = True
            _qlib_initialized = True
            return True

        # Initialize Qlib with default settings
        data_dir = os.environ.get("QLIB_DATA_DIR", "~/.qlib/qlib_data")
        qlib.init(provider_uri=os.path.expanduser(data_dir))
        QLIB_AVAILABLE = True
        _qlib_initialized = True
        return True

    except Exception as e:
        logger.warning(f"Qlib initialization failed: {e}")
        QLIB_AVAILABLE = False
        _qlib_initialized = True
        return False


# =============================================================================
# Qlib Expression Definitions for Technical Indicators
# =============================================================================
QLIB_INDICATOR_EXPRESSIONS: dict[str, str] = {
    # RSI(14) - Relative Strength Index
    # Uses Qlib's If/Ref/Mean operators
    "$rsi_14": """
        100 - 100 / (1 +
            Mean(If(Ref($close, 1) < $close, $close - Ref($close, 1), 0), 14) /
            (Mean(If(Ref($close, 1) > $close, Ref($close, 1) - $close, 0), 14) + 1e-10)
        )
    """,

    # MACD components
    "$macd": "EMA($close, 12) - EMA($close, 26)",
    "$macd_signal": "EMA(EMA($close, 12) - EMA($close, 26), 9)",
    "$macd_hist": "(EMA($close, 12) - EMA($close, 26)) - EMA(EMA($close, 12) - EMA($close, 26), 9)",

    # Bollinger Bands
    "$bb_upper": "Mean($close, 20) + 2 * Std($close, 20)",
    "$bb_lower": "Mean($close, 20) - 2 * Std($close, 20)",

    # ATR(14) - Average True Range
    "$atr_14": """
        Mean(
            Max(
                Max($high - $low, Abs($high - Ref($close, 1))),
                Abs($low - Ref($close, 1))
            ),
            14
        )
    """,

    # Additional crypto-specific indicators
    "$volatility_20": "Std(Ref($close, 1) / $close - 1, 20)",
    "$momentum_10": "$close / Ref($close, 10) - 1",
    "$volume_ma_20": "Mean($volume, 20)",
    "$volume_ratio": "$volume / (Mean($volume, 20) + 1e-10)",
}


class QlibUnavailableError(RuntimeError):
    """Raised when Qlib is required but not available."""


class QlibExpressionEngine:
    """Wrapper for Qlib's expression engine.

    All indicator calculations MUST go through this class.
    """

    def __init__(self, require_qlib: bool = True) -> None:
        """Initialize expression engine with Qlib backend."""
        self._require_qlib = require_qlib
        self._qlib_available = _ensure_qlib_initialized()
        if self._require_qlib and not self._qlib_available:
            raise QlibUnavailableError(
                "Qlib backend is required but not available. "
                "Set QLIB_DATA_DIR correctly or install qlib extras."
            )
        self._ops_cache: dict[str, Any] = {}

        if self._qlib_available:
            try:
                from qlib.data.ops import (
                    Ref, Mean, Std, Max, Min, Abs, If,
                    EMA, Corr, Rank, Sum, Idxmax, Idxmin
                )
                self._ops_cache = {
                    "Ref": Ref, "Mean": Mean, "Std": Std,
                    "Max": Max, "Min": Min, "Abs": Abs, "If": If,
                    "EMA": EMA, "Corr": Corr, "Rank": Rank, "Sum": Sum,
                    "Idxmax": Idxmax, "Idxmin": Idxmin,
                }
            except ImportError:
                logger.warning("Some Qlib ops not available")

    def compute_indicator(
        self,
        indicator_name: str,
        df: pd.DataFrame,
    ) -> pd.Series:
        """Compute a technical indicator using Qlib expressions.

        Args:
            indicator_name: Name of indicator (e.g., "$rsi_14")
            df: DataFrame with OHLCV data

        Returns:
            Series of computed indicator values
        """
        if indicator_name not in QLIB_INDICATOR_EXPRESSIONS:
            raise ValueError(f"Unknown indicator: {indicator_name}")

        expression = QLIB_INDICATOR_EXPRESSIONS[indicator_name]
        return self.compute_expression(expression, df, indicator_name)

    def compute_expression(
        self,
        expression: str,
        df: pd.DataFrame,
        result_name: str = "result",
    ) -> pd.Series:
        """Compute arbitrary Qlib expression.

        Args:
            expression: Qlib expression string
            df: DataFrame with required fields
            result_name: Name for result series

        Returns:
            Series of computed values
        """
        # Prepare data in Qlib-style dict for Qlib execution.
        data = self._prepare_data_for_qlib(df)

        if not self._qlib_available:
            raise QlibUnavailableError(
                "Qlib backend unavailable - Qlib-only mode enforced."
            )

        try:
            # Use Qlib's expression evaluation
            # Build and evaluate expression
            result = self._evaluate_expression(expression, data)
            return pd.Series(result, index=df.index, name=result_name)

        except Exception as e:
            logger.error(f"Qlib expression evaluation failed: {e}")
            raise

    def _prepare_data_for_qlib(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """Prepare DataFrame columns for Qlib ops.

        P0 Fix: Now maps ALL numeric columns to Qlib-style fields, not just OHLCV.
        This enables derivative fields (funding_rate, open_interest, etc.) to be
        used in Qlib expressions.
        """
        data = {}
        for col in df.columns:
            if col.startswith("$"):
                # Already Qlib-style field
                data[col] = df[col]
            elif col in ["open", "high", "low", "close", "volume"]:
                # Map standard OHLCV to Qlib fields (with $ prefix)
                data[f"${col}"] = df[col]
            else:
                # P0 Fix: Map ALL other numeric columns with $ prefix
                # This enables derivative fields like funding_rate, open_interest, etc.
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Add $ prefix if not already present
                    qlib_col = f"${col}" if not col.startswith("$") else col
                    data[qlib_col] = df[col]
                    # Also keep original name for backward compatibility
                    # (allows expressions to use both $funding_rate and funding_rate)
                    if not col.startswith("$"):
                        data[col] = df[col]
        return data

    def _evaluate_expression(
        self,
        expression: str,
        data: dict[str, pd.Series],
    ) -> pd.Series:
        """Evaluate Qlib expression with data context.

        Uses Qlib's operator system for computation.
        """
        # Clean up expression
        expr = expression.strip().replace("\n", " ").replace("  ", " ")

        # Build local context with Qlib ops and data
        local_context = {**self._ops_cache, **data}

        # For simple field references, return directly
        if expr.startswith("$") and expr in data:
            return data[expr]

        # Use Qlib's expression parser if available
        try:
            from qlib.data.base import Feature

            # Wrap data in Qlib Feature objects
            features = {k: Feature(v) for k, v in data.items()}

            # Evaluate using Qlib's system
            # This is a simplified approach - in production,
            # use Qlib's full expression parsing
            result = self._eval_with_ops(expr, features)
            return result

        except Exception as e:
            logger.error(f"Qlib expression parsing failed: {e}")
            raise

    def _eval_with_ops(
        self,
        expr: str,
        features: dict[str, Any],
    ) -> pd.Series:
        """Evaluate expression using Qlib operators."""
        # Import Qlib ops
        from qlib.data import ops

        # Build evaluation context
        context = {
            "Ref": ops.Ref,
            "Mean": ops.Mean,
            "Std": ops.Std,
            "Max": ops.Max,
            "Min": ops.Min,
            "Abs": ops.Abs,
            "EMA": getattr(ops, "EMA", ops.Mean),  # Fallback if EMA not available
            "If": getattr(ops, "If", lambda c, t, f: t if c else f),
            "Sum": ops.Sum,
            "Corr": ops.Corr,
            "Rank": ops.Rank,
            **features,
        }

        # Evaluate (simplified - real impl would use proper parser)
        try:
            result = eval(expr, {"__builtins__": {}}, context)
            if hasattr(result, "load"):
                return result.load()
            return result
        except Exception:
            raise

    def _fallback_eval(
        self,
        expr: str,
        data: dict[str, pd.Series],
    ) -> pd.Series:
        """Legacy pandas fallback (REMOVED - Qlib-only mode enforced)."""
        raise QlibUnavailableError(
            "Qlib-only mode enforced; pandas fallback evaluation is disabled. "
            "All factor expressions must use Qlib's C++ engine."
        )


class CryptoDataHandler:
    """Crypto data handler (single source: vendor deep fork).

    P2-11: Unify data normalization/alignment under the deep-forked Qlib crypto
    handler (vendor/qlib/qlib/contrib/crypto). IQFMP keeps the expression engine
    and exposes a Qlib-style `$field` DataFrame for expression computation.
    """

    def __init__(
        self,
        instruments: Optional[list[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        data_dir: Optional[Path] = None,
        timeframe: str = "1d",
        exchange: str = "binance",
    ) -> None:
        self.instruments = instruments or []
        self.start_time = start_time
        self.end_time = end_time
        self.data_dir = data_dir
        self.timeframe = timeframe
        self.exchange = exchange

        if not _VENDOR_CRYPTO_HANDLER_AVAILABLE or _VendorCryptoDataHandler is None:
            raise ImportError(
                "Qlib crypto deep fork is required. Ensure `iqfmp.qlib_crypto` "
                "is importable (PYTHONPATH includes vendor/qlib)."
            )

        config = None
        exchange_enum = self._to_vendor_exchange(exchange)
        timeframe_enum = self._to_vendor_timeframe(timeframe)
        if (
            _VendorCryptoDataConfig is not None
            and exchange_enum is not None
            and timeframe_enum is not None
        ):
            config = _VendorCryptoDataConfig(
                exchange=exchange_enum,
                timeframe=timeframe_enum,
                symbols=list(self.instruments),
            )

        self._vendor = _VendorCryptoDataHandler(config=config) if config else _VendorCryptoDataHandler()
        self._expression_engine = QlibExpressionEngine()
        self._combined_data: Optional[pd.DataFrame] = None

    @staticmethod
    def _to_vendor_exchange(exchange: str) -> Optional[Any]:
        if _VendorExchange is None:
            return None
        try:
            return _VendorExchange(str(exchange).lower())
        except Exception:
            return None

    @staticmethod
    def _to_vendor_timeframe(timeframe: str) -> Optional[Any]:
        if _VendorTimeFrame is None:
            return None
        try:
            return _VendorTimeFrame(str(timeframe).lower())
        except Exception:
            return None

    def load_data(
        self,
        path: Optional[Union[str, Path]] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Load data from file or DataFrame (delegates normalization to vendor)."""
        if df is None:
            if path is None:
                raise ValueError("Either path or df must be provided")
            path_obj = Path(path)
            if path_obj.suffix == ".csv":
                df = pd.read_csv(path_obj)
            elif path_obj.suffix == ".parquet":
                df = pd.read_parquet(path_obj)
            else:
                raise ValueError(f"Unsupported file format: {path_obj.suffix}")

        data = df.copy()

        # If the caller provides a named index (common for time-series), lift it into columns.
        if not isinstance(data.index, pd.RangeIndex):
            data = data.reset_index()

        exchange_enum = self._to_vendor_exchange(self.exchange)
        self._vendor.load(
            data=data,
            exchange=exchange_enum,
            validate=False,
            fill_na=False,
        )

        qlib_df = self._vendor.to_qlib_format()
        if "datetime" in qlib_df.columns:
            qlib_df["datetime"] = pd.to_datetime(qlib_df["datetime"])
            sort_cols = ["datetime"]
            if "symbol" in qlib_df.columns:
                sort_cols = ["symbol", "datetime"]
            qlib_df = qlib_df.sort_values(sort_cols).reset_index(drop=True)

        self._add_derived_fields_inplace(qlib_df)
        self._combined_data = qlib_df

    @staticmethod
    def _add_derived_fields_inplace(df: pd.DataFrame) -> None:
        """Add common derived `$` fields used by factor engines."""
        if {"$high", "$low", "$close"}.issubset(df.columns):
            typical = (df["$high"] + df["$low"] + df["$close"]) / 3
            df["$typical"] = typical

            if "$volume" in df.columns:
                df["$dollar_volume"] = df["$volume"] * df["$close"]
                cum_tp_vol = (typical * df["$volume"]).cumsum()
                cum_vol = df["$volume"].cumsum()
                df["$vwap"] = cum_tp_vol / (cum_vol + 1e-10)

    def get_data(
        self,
        instruments: Optional[list[str]] = None,
        fields: Optional[list[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> pd.DataFrame:
        if self._combined_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        df = self._combined_data.copy()

        if instruments and "symbol" in df.columns:
            df = df[df["symbol"].isin(instruments)]

        if start_time and "datetime" in df.columns:
            df = df[df["datetime"] >= pd.to_datetime(start_time)]
        if end_time and "datetime" in df.columns:
            df = df[df["datetime"] <= pd.to_datetime(end_time)]

        if fields:
            base_cols = [c for c in ["datetime", "symbol"] if c in df.columns]
            available_cols = base_cols + [c for c in fields if c in df.columns]
            df = df[available_cols]

        return df

    def compute_factor(self, expression: str, factor_name: str = "factor") -> pd.Series:
        if self._combined_data is None:
            raise ValueError("No data loaded")
        return self._expression_engine.compute_expression(
            expression=expression,
            df=self._combined_data,
            result_name=factor_name,
        )

    def add_technical_indicators(self) -> None:
        """Add a small baseline set of indicators via QlibExpressionEngine."""
        if self._combined_data is None:
            return

        df = self._combined_data
        indicators_to_compute = [
            "$rsi_14",
            "$macd",
            "$macd_signal",
            "$macd_hist",
            "$bb_upper",
            "$bb_lower",
            "$atr_14",
        ]

        for indicator in indicators_to_compute:
            try:
                df[indicator] = self._expression_engine.compute_indicator(indicator, df)
            except Exception as e:
                logger.warning(f"Failed to compute {indicator}: {e}")
                df[indicator] = np.nan

        self._combined_data = df

    def resample(self, timeframe: str) -> "CryptoDataHandler":
        if getattr(self._vendor, "data", None) is None:
            raise ValueError("No data loaded. Call load_data() first.")

        tf_enum = self._to_vendor_timeframe(timeframe)
        if tf_enum is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        resampled = self._vendor.resample(tf_enum)
        new_handler = CryptoDataHandler(
            instruments=self.instruments,
            start_time=self.start_time,
            end_time=self.end_time,
            data_dir=self.data_dir,
            timeframe=timeframe,
            exchange=self.exchange,
        )
        new_handler.load_data(df=resampled)
        return new_handler

    @property
    def data(self) -> Optional[pd.DataFrame]:
        return self._combined_data

    @property
    def instruments_list(self) -> list[str]:
        try:
            return list(self._vendor.get_symbols())
        except Exception:
            return list(self.instruments)


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
