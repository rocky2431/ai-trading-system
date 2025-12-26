"""Unified Qlib Factor Library.

This module serves as the SINGLE ENTRY POINT for all factor computations.
All factors MUST be defined as Qlib expressions.

Architecture Principle:
======================
Qlib is the SOLE computational core. All factor calculations go through
Qlib's expression engine. No local pandas/numpy calculations are allowed.

Migration Status:
=================
- alpha_benchmark.py: ✅ MIGRATED - Uses ALPHA158_EXPRESSIONS (Qlib)
- alpha101.py: ✅ DELETED - Migrated to ADDITIONAL_EXPRESSIONS
- alpha158.py: ✅ DELETED - Now uses ALPHA158_EXPRESSIONS from alpha_benchmark
- alpha360.py: ✅ DELETED - Migrated to ADDITIONAL_EXPRESSIONS

P0-2 Cleanup Complete: ALL factors now use Qlib expression engine.

Usage:
======
```python
from iqfmp.evaluation.qlib_factor_library import (
    QlibFactorLibrary,
    get_factor_expression,
    compute_factor,
)

# Get expression for a factor
expr = get_factor_expression("KMID")

# Compute factor value
library = QlibFactorLibrary()
result = library.compute("ROC5", data)
```
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

# P0 SECURITY: Import ASTSecurityChecker for mandatory expression validation
from iqfmp.core.security import ASTSecurityChecker

logger = logging.getLogger(__name__)

# Module-level security checker instance (reused for performance)
_security_checker = ASTSecurityChecker()


# =============================================================================
# Qlib Integration
# =============================================================================
QLIB_AVAILABLE = False
_qlib_initialized = False


class QlibUnavailableError(RuntimeError):
    """Raised when Qlib backend is required but unavailable."""


def _ensure_qlib_initialized() -> bool:
    """Ensure Qlib is properly initialized."""
    global QLIB_AVAILABLE, _qlib_initialized

    if _qlib_initialized:
        return QLIB_AVAILABLE

    try:
        import qlib
        from qlib.config import C

        if hasattr(C, "provider_uri") and C.provider_uri is not None:
            QLIB_AVAILABLE = True
            _qlib_initialized = True
            return True

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
# Unified Factor Expression Library
# =============================================================================

# Import expressions from alpha_benchmark (the canonical source)
from iqfmp.evaluation.alpha_benchmark import ALPHA158_EXPRESSIONS

# Additional Qlib expressions not in alpha_benchmark
ADDITIONAL_EXPRESSIONS: dict[str, str] = {
    # Alpha101 style factors (Qlib expression format)
    "ALPHA001": "-1 * Corr(Rank($close), Rank($volume), 6)",
    "ALPHA002": "-1 * Delta(Log($close), 2)",
    "ALPHA003": "-1 * Corr(Rank($open), Rank($volume), 10)",
    "ALPHA004": "-1 * Rank(Ref($low, 5))",
    "ALPHA005": "Rank($open - Mean($open, 10)) * (-1 * Abs(Rank($close - $vwap)))",
    "ALPHA006": "-1 * Corr($open, $volume, 10)",

    # Alpha360 style factors (Qlib expression format)
    "CLOSE_ROC5": "Ref($close, 5) / $close - 1",
    "CLOSE_ROC10": "Ref($close, 10) / $close - 1",
    "CLOSE_ROC20": "Ref($close, 20) / $close - 1",
    "CLOSE_ROC30": "Ref($close, 30) / $close - 1",
    "CLOSE_ROC60": "Ref($close, 60) / $close - 1",

    "CLOSE_MA5_RATIO": "$close / Mean($close, 5) - 1",
    "CLOSE_MA10_RATIO": "$close / Mean($close, 10) - 1",
    "CLOSE_MA20_RATIO": "$close / Mean($close, 20) - 1",
    "CLOSE_MA30_RATIO": "$close / Mean($close, 30) - 1",
    "CLOSE_MA60_RATIO": "$close / Mean($close, 60) - 1",

    "CLOSE_STD5": "Std($close, 5) / (Mean($close, 5) + 1e-10)",
    "CLOSE_STD10": "Std($close, 10) / (Mean($close, 10) + 1e-10)",
    "CLOSE_STD20": "Std($close, 20) / (Mean($close, 20) + 1e-10)",

    "VOLUME_ROC5": "Ref($volume, 5) / ($volume + 1e-10) - 1",
    "VOLUME_ROC10": "Ref($volume, 10) / ($volume + 1e-10) - 1",
    "VOLUME_ROC20": "Ref($volume, 20) / ($volume + 1e-10) - 1",

    "VOLUME_MA5_RATIO": "$volume / (Mean($volume, 5) + 1e-10) - 1",
    "VOLUME_MA10_RATIO": "$volume / (Mean($volume, 10) + 1e-10) - 1",
    "VOLUME_MA20_RATIO": "$volume / (Mean($volume, 20) + 1e-10) - 1",

    # Technical indicators (Qlib expression format)
    "RSI_14": """
        100 - 100 / (1 +
            Mean(If(Ref($close, 1) < $close, $close - Ref($close, 1), 0), 14) /
            (Mean(If(Ref($close, 1) > $close, Ref($close, 1) - $close, 0), 14) + 1e-10)
        )
    """,
    "MACD": "EMA($close, 12) - EMA($close, 26)",
    "MACD_SIGNAL": "EMA(EMA($close, 12) - EMA($close, 26), 9)",
    "BOLLINGER_UPPER": "Mean($close, 20) + 2 * Std($close, 20)",
    "BOLLINGER_LOWER": "Mean($close, 20) - 2 * Std($close, 20)",
    "ATR_14": """
        Mean(
            Max(
                Max($high - $low, Abs($high - Ref($close, 1))),
                Abs($low - Ref($close, 1))
            ),
            14
        )
    """,

    # Crypto-specific factors
    "CRYPTO_MOMENTUM": "$close / Ref($close, 24) - 1",  # 24h momentum
    "CRYPTO_VOLATILITY": "Std(Ref($close, 1) / $close - 1, 24)",  # 24h volatility
    "VOLUME_SPIKE": "$volume / (Mean($volume, 24) + 1e-10)",  # Volume spike detection
    "PRICE_RANGE": "($high - $low) / ($close + 1e-10)",  # Intraday range
    "CRYPTO_FUNDING_ZSCORE": "($funding_rate - Mean($funding_rate, 24)) / (Std($funding_rate, 24) + 1e-10)",
    "CRYPTO_FUNDING_MOMENTUM": "Ref($funding_rate, 8) - $funding_rate",
    "CRYPTO_PREMIUM_ZSCORE": "($premium - Mean($premium, 24)) / (Std($premium, 24) + 1e-10)",
    "CRYPTO_OI_CHANGE_1": "Ref($open_interest, 1) / ($open_interest + 1e-10) - 1",
    "CRYPTO_OI_ZSCORE": "($open_interest - Mean($open_interest, 24)) / (Std($open_interest, 24) + 1e-10)",
    "CRYPTO_OI_PRICE_DIVERGENCE": "(Ref($open_interest, 1) / ($open_interest + 1e-10) - 1) - (Ref($close, 1) / ($close + 1e-10) - 1)",
    "CRYPTO_LIQUIDATION_SPIKE": "$liquidation_volume / (Mean($liquidation_volume, 24) + 1e-10)",
    "CRYPTO_BASIS_PCT": "($mark_price - $index_price) / ($index_price + 1e-10)",
    "CRYPTO_ORDERBOOK_IMBALANCE": "$bid_ask_imbalance",
    "CRYPTO_SPREAD_NORM": "$spread / ($close + 1e-10)",
    "CRYPTO_LONG_SHORT_SKEW": "$long_ratio - $short_ratio",
}


# Combine all expressions into unified library
QLIB_FACTOR_EXPRESSIONS: dict[str, str] = {
    **ALPHA158_EXPRESSIONS,
    **ADDITIONAL_EXPRESSIONS,
}


def get_factor_expression(factor_name: str) -> str:
    """Get Qlib expression for a factor.

    Args:
        factor_name: Name of the factor

    Returns:
        Qlib expression string

    Raises:
        KeyError: If factor not found
    """
    if factor_name not in QLIB_FACTOR_EXPRESSIONS:
        raise KeyError(
            f"Factor '{factor_name}' not found. "
            f"Available factors: {list(QLIB_FACTOR_EXPRESSIONS.keys())[:10]}..."
        )
    return QLIB_FACTOR_EXPRESSIONS[factor_name]


def list_factors() -> list[str]:
    """List all available factor names."""
    return sorted(QLIB_FACTOR_EXPRESSIONS.keys())


def list_factors_by_category() -> dict[str, list[str]]:
    """List factors organized by category."""
    categories: dict[str, list[str]] = {
        "kline": [],
        "momentum": [],
        "moving_average": [],
        "volatility": [],
        "volume": [],
        "technical": [],
        "correlation": [],
        "alpha101": [],
        "crypto": [],
        "other": [],
    }

    for name in QLIB_FACTOR_EXPRESSIONS.keys():
        name_upper = name.upper()
        if name_upper.startswith(("KMID", "KLEN", "KUP", "KLOW")):
            categories["kline"].append(name)
        elif name_upper.startswith(("ROC", "MOMENTUM")):
            categories["momentum"].append(name)
        elif "MA" in name_upper or "MEAN" in name_upper:
            categories["moving_average"].append(name)
        elif "STD" in name_upper or "VOLATILITY" in name_upper:
            categories["volatility"].append(name)
        elif "VOLUME" in name_upper or "VOL" in name_upper:
            categories["volume"].append(name)
        elif name_upper.startswith(("RSI", "MACD", "BOLLINGER", "ATR")):
            categories["technical"].append(name)
        elif "CORR" in name_upper or "BETA" in name_upper:
            categories["correlation"].append(name)
        elif name_upper.startswith("ALPHA"):
            categories["alpha101"].append(name)
        elif name_upper.startswith("CRYPTO"):
            categories["crypto"].append(name)
        else:
            categories["other"].append(name)

    return {k: v for k, v in categories.items() if v}


class QlibFactorLibrary:
    """Unified factor library using Qlib as computational backend.

    This is the ONLY allowed way to compute factors in IQFMP.
    All factor computations go through Qlib's expression engine.
    """

    def __init__(self) -> None:
        """Initialize factor library with Qlib backend."""
        self._qlib_available = _ensure_qlib_initialized()
        self._ops_cache: dict[str, Any] = {}

        if self._qlib_available:
            try:
                from qlib.data.ops import (
                    Ref, Mean, Std, Max, Min, Abs, If,
                    EMA, Corr, Rank, Sum, Delta, Log
                )
                self._ops_cache = {
                    "Ref": Ref, "Mean": Mean, "Std": Std,
                    "Max": Max, "Min": Min, "Abs": Abs, "If": If,
                    "EMA": EMA, "Corr": Corr, "Rank": Rank,
                    "Sum": Sum, "Delta": Delta, "Log": Log,
                }
            except ImportError as e:
                logger.warning(f"Some Qlib ops not available: {e}")

    @property
    def available_factors(self) -> list[str]:
        """List of available factor names."""
        return list_factors()

    @property
    def factor_count(self) -> int:
        """Number of available factors."""
        return len(QLIB_FACTOR_EXPRESSIONS)

    def get_expression(self, factor_name: str) -> str:
        """Get Qlib expression for a factor."""
        return get_factor_expression(factor_name)

    def compute(
        self,
        factor_name: str,
        data: pd.DataFrame,
    ) -> pd.Series:
        """Compute a factor value using Qlib.

        Args:
            factor_name: Name of the factor to compute
            data: DataFrame with OHLCV data

        Returns:
            Series of factor values
        """
        expression = get_factor_expression(factor_name)
        return self.compute_expression(expression, data, factor_name)

    def compute_expression(
        self,
        expression: str,
        data: pd.DataFrame,
        result_name: str = "factor",
    ) -> pd.Series:
        """Compute arbitrary Qlib expression.

        Args:
            expression: Qlib expression string
            data: DataFrame with required fields
            result_name: Name for result series

        Returns:
            Series of computed values
        """
        if not self._qlib_available:
            raise QlibUnavailableError(
                "Qlib backend unavailable - Qlib-only mode enforced."
            )

        try:
            # Prepare data for Qlib
            qlib_data = self._prepare_data(data)

            # Evaluate expression
            result = self._evaluate(expression, qlib_data)
            return pd.Series(result, index=data.index, name=result_name)

        except Exception as e:
            logger.error(f"Qlib computation failed: {e}")
            raise

    def compute_batch(
        self,
        factor_names: list[str],
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute multiple factors at once.

        Args:
            factor_names: List of factor names to compute
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with all computed factors
        """
        results = {}
        for name in factor_names:
            try:
                results[name] = self.compute(name, data)
            except QlibUnavailableError:
                raise
            except Exception as e:
                logger.warning(f"Failed to compute {name}: {e}")
                results[name] = pd.Series(np.nan, index=data.index)

        return pd.DataFrame(results)

    def _prepare_data(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """Prepare DataFrame for Qlib ops."""
        data = {}
        for col in df.columns:
            if col.startswith("$"):
                data[col] = df[col]
            elif col in ["open", "high", "low", "close", "volume"]:
                data[f"${col}"] = df[col]
            elif pd.api.types.is_numeric_dtype(df[col]):
                qlib_col = f"${col}"
                data[qlib_col] = df[col]
                data[col] = df[col]

        # Add computed fields if not present
        if "$vwap" not in data and all(f"${c}" in data for c in ["high", "low", "close"]):
            data["$vwap"] = (data["$high"] + data["$low"] + data["$close"]) / 3

        return data

    def _evaluate(
        self,
        expression: str,
        data: dict[str, pd.Series],
    ) -> pd.Series:
        """Evaluate Qlib expression."""
        # Clean expression
        expr = expression.strip().replace("\n", " ").replace("  ", " ")

        # Simple field reference
        if expr.startswith("$") and expr in data:
            return data[expr]

        # =====================================================================
        # P0 SECURITY: Mandatory AST security check before any eval
        # This prevents code injection through malicious expressions
        # =====================================================================
        is_safe, violations = _security_checker.check(expr)
        if not is_safe:
            violation_details = "; ".join(violations[:5])
            raise ValueError(
                f"SECURITY VIOLATION: Expression failed security check. "
                f"Expression: {expr[:100]}... Violations: {violation_details}"
            )

        # Build context with Qlib ops and data
        context = {**self._ops_cache, **data}

        try:
            result = eval(expr, {"__builtins__": {}}, context)
            if hasattr(result, "load"):
                return result.load()
            return result
        except Exception as e:
            logger.error(f"Qlib expression evaluation failed: {e}")
            raise


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_factor(
    factor_name: str,
    data: pd.DataFrame,
) -> pd.Series:
    """Convenience function to compute a single factor.

    Args:
        factor_name: Name of the factor
        data: DataFrame with OHLCV data

    Returns:
        Series of factor values
    """
    library = QlibFactorLibrary()
    return library.compute(factor_name, data)


def compute_factors(
    factor_names: list[str],
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Convenience function to compute multiple factors.

    Args:
        factor_names: List of factor names
        data: DataFrame with OHLCV data

    Returns:
        DataFrame with all computed factors
    """
    library = QlibFactorLibrary()
    return library.compute_batch(factor_names, data)
