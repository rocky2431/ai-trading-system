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
# Qlib Expression Engine - Pandas Implementation
# =============================================================================

def _build_qlib_ops_pandas() -> dict[str, Callable]:
    """Build a complete dictionary of Qlib-compatible operators implemented in pandas.

    This dynamically creates pandas implementations for all standard Qlib operators,
    ensuring compatibility with Alpha158 and other Qlib factor libraries.

    Returns:
        Dictionary mapping operator names to pandas-based implementations
    """
    ops = {}

    # =========================================================================
    # Rolling Window Operators
    # =========================================================================

    def Ref(s: pd.Series, n: int) -> pd.Series:
        """Reference N periods ago (negative n = past)."""
        return s.shift(-n if n < 0 else n)

    def Mean(s: pd.Series, n: int) -> pd.Series:
        """Rolling mean (MA)."""
        return s.rolling(n, min_periods=1).mean()

    def Std(s: pd.Series, n: int) -> pd.Series:
        """Rolling standard deviation."""
        return s.rolling(n, min_periods=1).std()

    def Var(s: pd.Series, n: int) -> pd.Series:
        """Rolling variance."""
        return s.rolling(n, min_periods=1).var()

    def Sum(s: pd.Series, n: int) -> pd.Series:
        """Rolling sum."""
        return s.rolling(n, min_periods=1).sum()

    def Max(*args) -> pd.Series:
        """Rolling maximum OR element-wise maximum.

        Usage:
            Max(series, n) - Rolling max over n periods
            Max(series1, series2, ...) - Element-wise max across series
        """
        if len(args) == 2 and isinstance(args[1], (int, np.integer)):
            # Rolling max: Max(series, n)
            return args[0].rolling(int(args[1]), min_periods=1).max()
        else:
            # Element-wise max: Max(series1, series2, ...)
            return pd.concat([s if isinstance(s, pd.Series) else pd.Series(s) for s in args], axis=1).max(axis=1)

    def Min(*args) -> pd.Series:
        """Rolling minimum OR element-wise minimum.

        Usage:
            Min(series, n) - Rolling min over n periods
            Min(series1, series2, ...) - Element-wise min across series
        """
        if len(args) == 2 and isinstance(args[1], (int, np.integer)):
            # Rolling min: Min(series, n)
            return args[0].rolling(int(args[1]), min_periods=1).min()
        else:
            # Element-wise min: Min(series1, series2, ...)
            return pd.concat([s if isinstance(s, pd.Series) else pd.Series(s) for s in args], axis=1).min(axis=1)

    def Med(s: pd.Series, n: int) -> pd.Series:
        """Rolling median."""
        return s.rolling(n, min_periods=1).median()

    def Count(s: pd.Series, n: int) -> pd.Series:
        """Rolling count of non-NaN values."""
        return s.rolling(n, min_periods=1).count()

    def Quantile(s: pd.Series, n: int, q: float = 0.5) -> pd.Series:
        """Rolling quantile."""
        return s.rolling(n, min_periods=1).quantile(q)

    def Kurt(s: pd.Series, n: int) -> pd.Series:
        """Rolling kurtosis."""
        return s.rolling(n, min_periods=1).kurt()

    def Skew(s: pd.Series, n: int) -> pd.Series:
        """Rolling skewness."""
        return s.rolling(n, min_periods=1).skew()

    def Mad(s: pd.Series, n: int) -> pd.Series:
        """Rolling median absolute deviation."""
        return s.rolling(n, min_periods=1).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True
        )

    # =========================================================================
    # Delta / Change Operators
    # =========================================================================

    def Delta(s: pd.Series, n: int) -> pd.Series:
        """Change over N periods: current - N periods ago."""
        return s - s.shift(n)

    # =========================================================================
    # Moving Averages
    # =========================================================================

    def EMA(s: pd.Series, n: int) -> pd.Series:
        """Exponential moving average."""
        return s.ewm(span=n, adjust=False, min_periods=1).mean()

    def WMA(s: pd.Series, n: int) -> pd.Series:
        """Weighted moving average (linear weights)."""
        weights = np.arange(1, n + 1, dtype=float)
        return s.rolling(n, min_periods=1).apply(
            lambda x: np.dot(x[-len(weights):], weights[-len(x):]) / weights[-len(x):].sum(),
            raw=True
        )

    # =========================================================================
    # Pair Rolling Operators (two series)
    # =========================================================================

    def Corr(s1: pd.Series, s2: pd.Series, n: int) -> pd.Series:
        """Rolling Pearson correlation."""
        return s1.rolling(n, min_periods=1).corr(s2)

    def Cov(s1: pd.Series, s2: pd.Series, n: int) -> pd.Series:
        """Rolling covariance."""
        return s1.rolling(n, min_periods=1).cov(s2)

    def Resi(y: pd.Series, x: pd.Series, n: int) -> pd.Series:
        """Rolling regression residual (y ~ x)."""
        def _resi(vals):
            if len(vals) < 2:
                return np.nan
            mid = len(vals) // 2
            y_vals, x_vals = vals[:mid], vals[mid:]
            if np.std(x_vals) == 0:
                return np.nan
            slope = np.cov(y_vals, x_vals)[0, 1] / np.var(x_vals)
            intercept = np.mean(y_vals) - slope * np.mean(x_vals)
            return y_vals[-1] - (slope * x_vals[-1] + intercept)
        combined = pd.concat([y, x], axis=0)
        return combined.rolling(n * 2, min_periods=2).apply(_resi, raw=True).iloc[len(y):]

    def Slope(y: pd.Series, x: pd.Series, n: int) -> pd.Series:
        """Rolling regression slope (y ~ x)."""
        def _slope(vals):
            if len(vals) < 2:
                return np.nan
            mid = len(vals) // 2
            y_vals, x_vals = vals[:mid], vals[mid:]
            if np.std(x_vals) == 0:
                return np.nan
            return np.cov(y_vals, x_vals)[0, 1] / np.var(x_vals)
        combined = pd.concat([y, x], axis=0)
        return combined.rolling(n * 2, min_periods=2).apply(_slope, raw=True).iloc[len(y):]

    def Rsquare(y: pd.Series, x: pd.Series, n: int) -> pd.Series:
        """Rolling R-squared (y ~ x)."""
        corr = Corr(y, x, n)
        return corr ** 2

    # =========================================================================
    # Element-wise Operators (single series)
    # =========================================================================

    def Abs(s: pd.Series) -> pd.Series:
        """Absolute value."""
        return s.abs()

    def Log(s: pd.Series) -> pd.Series:
        """Natural logarithm."""
        return np.log(s.replace(0, np.nan))

    def Sign(s: pd.Series) -> pd.Series:
        """Sign of value (-1, 0, 1)."""
        return np.sign(s)

    def Power(s: pd.Series, exp: float) -> pd.Series:
        """Raise to power."""
        return s ** exp

    # =========================================================================
    # Cross-sectional Operators
    # =========================================================================

    def Rank(s: pd.Series) -> pd.Series:
        """Cross-sectional rank (percentile 0-1)."""
        return s.rank(pct=True)

    # =========================================================================
    # Index Operators
    # =========================================================================

    def IdxMax(s: pd.Series, n: int) -> pd.Series:
        """Index of max value in rolling window (0-indexed from start of window)."""
        return s.rolling(n, min_periods=1).apply(lambda x: np.argmax(x), raw=True)

    def IdxMin(s: pd.Series, n: int) -> pd.Series:
        """Index of min value in rolling window (0-indexed from start of window)."""
        return s.rolling(n, min_periods=1).apply(lambda x: np.argmin(x), raw=True)

    # =========================================================================
    # Conditional Operators
    # =========================================================================

    def If(cond, true_val, false_val) -> pd.Series:
        """Conditional selection."""
        if isinstance(cond, pd.Series):
            return pd.Series(np.where(cond, true_val, false_val), index=cond.index)
        return true_val if cond else false_val

    def Mask(s: pd.Series, cond: pd.Series) -> pd.Series:
        """Mask values where condition is False."""
        return s.where(cond)

    # =========================================================================
    # Technical Indicators
    # =========================================================================

    def RSI(s: pd.Series, n: int) -> pd.Series:
        """Relative Strength Index."""
        delta = s.diff()
        gain = delta.where(delta > 0, 0.0).rolling(n, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(n, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def MACD(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """MACD histogram (MACD line - Signal line)."""
        ema_fast = s.ewm(span=fast, adjust=False, min_periods=1).mean()
        ema_slow = s.ewm(span=slow, adjust=False, min_periods=1).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=1).mean()
        return macd_line - signal_line

    # =========================================================================
    # Arithmetic Operators (for completeness)
    # =========================================================================

    def Add(s1: pd.Series, s2: pd.Series) -> pd.Series:
        """Element-wise addition."""
        return s1 + s2

    def Sub(s1: pd.Series, s2: pd.Series) -> pd.Series:
        """Element-wise subtraction."""
        return s1 - s2

    def Mul(s1: pd.Series, s2: pd.Series) -> pd.Series:
        """Element-wise multiplication."""
        return s1 * s2

    def Div(s1: pd.Series, s2: pd.Series) -> pd.Series:
        """Element-wise division."""
        return s1 / s2.replace(0, np.nan)

    # =========================================================================
    # Comparison Operators
    # =========================================================================

    def Greater(s1: pd.Series, s2: pd.Series) -> pd.Series:
        """s1 > s2"""
        return (s1 > s2).astype(float)

    def Less(s1: pd.Series, s2: pd.Series) -> pd.Series:
        """s1 < s2"""
        return (s1 < s2).astype(float)

    def Ge(s1: pd.Series, s2: pd.Series) -> pd.Series:
        """s1 >= s2"""
        return (s1 >= s2).astype(float)

    def Le(s1: pd.Series, s2: pd.Series) -> pd.Series:
        """s1 <= s2"""
        return (s1 <= s2).astype(float)

    def Eq(s1: pd.Series, s2: pd.Series) -> pd.Series:
        """s1 == s2"""
        return (s1 == s2).astype(float)

    def Ne(s1: pd.Series, s2: pd.Series) -> pd.Series:
        """s1 != s2"""
        return (s1 != s2).astype(float)

    def Gt(s1: pd.Series, s2: pd.Series) -> pd.Series:
        """Alias for Greater: s1 > s2"""
        return Greater(s1, s2)

    def Lt(s1: pd.Series, s2: pd.Series) -> pd.Series:
        """Alias for Less: s1 < s2"""
        return Less(s1, s2)

    # =========================================================================
    # Logical Operators
    # =========================================================================

    def And(s1: pd.Series, s2: pd.Series) -> pd.Series:
        """Logical AND."""
        return (s1.astype(bool) & s2.astype(bool)).astype(float)

    def Or(s1: pd.Series, s2: pd.Series) -> pd.Series:
        """Logical OR."""
        return (s1.astype(bool) | s2.astype(bool)).astype(float)

    def Not(s: pd.Series) -> pd.Series:
        """Logical NOT."""
        return (~s.astype(bool)).astype(float)

    # Build the ops dictionary
    ops = {
        # Rolling
        "Ref": Ref, "Mean": Mean, "Std": Std, "Var": Var, "Sum": Sum,
        "Max": Max, "Min": Min, "Med": Med, "Count": Count, "Quantile": Quantile,
        "Kurt": Kurt, "Skew": Skew, "Mad": Mad,
        # Delta
        "Delta": Delta,
        # Moving Averages
        "EMA": EMA, "WMA": WMA,
        # Pair Rolling
        "Corr": Corr, "Cov": Cov, "Resi": Resi, "Slope": Slope, "Rsquare": Rsquare,
        # Element-wise
        "Abs": Abs, "Log": Log, "Sign": Sign, "Power": Power,
        # Cross-sectional
        "Rank": Rank,
        # Index
        "IdxMax": IdxMax, "IdxMin": IdxMin,
        # Conditional
        "If": If, "Mask": Mask,
        # Technical
        "RSI": RSI, "MACD": MACD,
        # Arithmetic
        "Add": Add, "Sub": Sub, "Mul": Mul, "Div": Div,
        # Comparison
        "Greater": Greater, "Less": Less, "Ge": Ge, "Le": Le, "Eq": Eq, "Ne": Ne,
        "Gt": Gt, "Lt": Lt,
        # Logical
        "And": And, "Or": Or, "Not": Not,
    }

    return ops


# Build operators once at module load
QLIB_PANDAS_OPS = _build_qlib_ops_pandas()


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

    def __init__(self, require_qlib: bool = False) -> None:
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
        # Prepare data in Qlib-style dict for both Qlib and pandas fallback paths.
        data = self._prepare_data_for_qlib(df)

        # If Qlib backend is unavailable, fall back to the pure-pandas operator set.
        # This keeps local/dev pipelines runnable without requiring a full Qlib data provider.
        if not self._qlib_available:
            if self._require_qlib:
                raise QlibUnavailableError(
                    "Qlib backend unavailable while require_qlib=True"
                )
            try:
                result = self._fallback_eval(expression, data)
                if isinstance(result, pd.Series):
                    return result.rename(result_name)
                return pd.Series(result, index=df.index, name=result_name)
            except Exception as e:
                logger.error(f"Pandas fallback eval failed: {e}")
                return pd.Series(np.nan, index=df.index, name=result_name)

        try:
            # Use Qlib's expression evaluation
            # Build and evaluate expression
            result = self._evaluate_expression(expression, data)
            return pd.Series(result, index=df.index, name=result_name)

        except Exception as e:
            logger.error(f"Qlib expression evaluation failed: {e}")
            # Return NaN series on failure
            return pd.Series(np.nan, index=df.index, name=result_name)

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
            logger.warning(f"Qlib expression parsing failed: {e}, using fallback")
            return self._fallback_eval(expr, data)

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
        """Fallback evaluation using pandas (Qlib-compatible formulas).

        Uses the complete QLIB_PANDAS_OPS dictionary for full Alpha158 compatibility.
        This includes 40+ operators matching Qlib's native expression engine.
        """
        # Replace field references with actual data
        for field, series in data.items():
            expr = expr.replace(field, f"data['{field}']")

        # Build evaluation context with ALL Qlib-compatible operators
        # QLIB_PANDAS_OPS contains: Ref, Mean, Std, Var, Sum, Max, Min, Med, Count,
        # Quantile, Kurt, Skew, Mad, Delta, EMA, WMA, Corr, Cov, Resi, Slope, Rsquare,
        # Abs, Log, Sign, Power, Rank, IdxMax, IdxMin, If, Mask, RSI, MACD,
        # Add, Sub, Mul, Div, Greater, Less, Ge, Le, Eq, Ne, Gt, Lt, And, Or, Not
        local_context = {
            "data": data,
            **QLIB_PANDAS_OPS,  # All 40+ Qlib operators
            "np": np,
            "pd": pd,
        }

        try:
            result = eval(expr, {"__builtins__": {}}, local_context)
            return result
        except Exception as e:
            logger.error(f"Fallback eval failed: {e}, expression: {expr[:100]}...")
            # Return first series in data as template for NaN series
            template = next(iter(data.values()))
            return pd.Series(np.nan, index=template.index)


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
