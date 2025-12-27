"""Alpha158 Benchmark for factor comparison - Qlib Expression Engine Integration.

This module provides Alpha158 factor implementations using Qlib's expression engine,
ensuring all factor calculations go through Qlib's core capabilities.

Based on Qlib's Alpha158 factor set, adapted for cryptocurrency markets.

ARCHITECTURE NOTE:
- All factors are defined as Qlib expressions, NOT pandas functions
- Factor computation MUST go through Qlib expression engine
- This ensures Qlib is the SOLE computational core
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import logging

import numpy as np
import pandas as pd

# Use Qlib-native statistical functions instead of scipy
from iqfmp.evaluation.qlib_stats import spearman_rank_correlation

logger = logging.getLogger(__name__)

# =============================================================================
# Lazy Import: QlibExpressionEngine (avoid circular import with core/)
# =============================================================================
# Import is delayed to avoid circular dependency:
# core/__init__ → rd_loop → alpha_benchmark → qlib_crypto

QlibExpressionEngine = None
QLIB_AVAILABLE = False


def _get_qlib_expression_engine():
    """Lazy load QlibExpressionEngine to avoid circular imports."""
    global QlibExpressionEngine, QLIB_AVAILABLE
    if QlibExpressionEngine is None:
        try:
            from iqfmp.core.qlib_crypto import (
                QlibExpressionEngine as _Engine,
                QLIB_AVAILABLE as _Available
            )
            QlibExpressionEngine = _Engine
            QLIB_AVAILABLE = _Available
        except ImportError:
            logger.warning("QlibExpressionEngine not available")
    return QlibExpressionEngine

# =============================================================================
# Qlib Integration (REMOVED: Pandas fallback - Qlib-only mode enforced)
# =============================================================================

try:
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.contrib.data.handler import Alpha158 as QlibAlpha158
except ImportError:
    DataHandlerLP = None
    QlibAlpha158 = None
    logger.warning("Qlib handlers not available")


@dataclass
class BenchmarkResult:
    """Result of benchmark comparison."""

    factor_name: str
    factor_ic: float
    factor_ir: float
    factor_sharpe: float
    benchmark_ic: float
    benchmark_ir: float
    benchmark_sharpe: float
    ic_improvement: float
    ir_improvement: float
    sharpe_improvement: float
    rank_in_benchmark: int
    total_benchmark_factors: int
    beats_benchmark_avg: bool
    correlation_with_best: float
    is_novel: bool  # Low correlation with existing factors

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "factor_name": self.factor_name,
            "factor_metrics": {
                "ic": self.factor_ic,
                "ir": self.factor_ir,
                "sharpe": self.factor_sharpe,
            },
            "benchmark_metrics": {
                "avg_ic": self.benchmark_ic,
                "avg_ir": self.benchmark_ir,
                "avg_sharpe": self.benchmark_sharpe,
            },
            "improvements": {
                "ic": self.ic_improvement,
                "ir": self.ir_improvement,
                "sharpe": self.sharpe_improvement,
            },
            "ranking": {
                "rank": self.rank_in_benchmark,
                "total": self.total_benchmark_factors,
                "percentile": 1 - (self.rank_in_benchmark / self.total_benchmark_factors),
            },
            "beats_benchmark_avg": self.beats_benchmark_avg,
            "correlation_with_best": self.correlation_with_best,
            "is_novel": self.is_novel,
        }

    def get_summary(self) -> str:
        """Generate text summary."""
        percentile = 100 * (1 - self.rank_in_benchmark / self.total_benchmark_factors)
        lines = [
            f"Alpha158 Benchmark Result: {self.factor_name}",
            "=" * 50,
            "",
            f"Factor IC:  {self.factor_ic:.4f}  (Benchmark avg: {self.benchmark_ic:.4f}, +{self.ic_improvement:+.1%})",
            f"Factor IR:  {self.factor_ir:.2f}  (Benchmark avg: {self.benchmark_ir:.2f}, +{self.ir_improvement:+.1%})",
            f"Factor Sharpe: {self.factor_sharpe:.2f}  (Benchmark avg: {self.benchmark_sharpe:.2f})",
            "",
            f"Rank: {self.rank_in_benchmark}/{self.total_benchmark_factors} (Top {percentile:.0f}%)",
            f"Beats Benchmark Average: {'Yes' if self.beats_benchmark_avg else 'No'}",
            f"Is Novel (low correlation): {'Yes' if self.is_novel else 'No'}",
        ]
        return "\n".join(lines)


# =============================================================================
# Alpha158 Factor Definitions as Qlib Expressions
# P2.2 FIX: Complete Alpha158 factor set aligned with Qlib official benchmark
# These are passed to Qlib's expression engine for computation
#
# Reference: vendor/qlib/qlib/contrib/data/loader.py (Alpha158DL)
# Total factors: ~158 (9 kbar + rolling operators across 5 windows)
# =============================================================================

# Default rolling windows matching Qlib Alpha158
ALPHA158_WINDOWS = [5, 10, 20, 30, 60]


def _generate_alpha158_expressions() -> dict[str, str]:
    """Generate complete Alpha158 factor expressions matching Qlib official.

    Based on: qlib/contrib/data/loader.py::Alpha158DL.get_feature_config()
    """
    expressions: dict[str, str] = {}

    # =========================================================================
    # 1. K-bar factors (9 factors) - Price pattern features
    # =========================================================================
    expressions.update({
        "KMID": "($close - $open) / $open",
        "KLEN": "($high - $low) / $open",
        "KMID2": "($close - $open) / ($high - $low + 1e-12)",
        "KUP": "($high - Greater($open, $close)) / $open",
        "KUP2": "($high - Greater($open, $close)) / ($high - $low + 1e-12)",
        "KLOW": "(Less($open, $close) - $low) / $open",
        "KLOW2": "(Less($open, $close) - $low) / ($high - $low + 1e-12)",
        "KSFT": "(2 * $close - $high - $low) / $open",
        "KSFT2": "(2 * $close - $high - $low) / ($high - $low + 1e-12)",
    })

    # =========================================================================
    # 2. Price features (normalized by current close)
    # =========================================================================
    for field in ["OPEN", "HIGH", "LOW", "VWAP"]:
        field_lower = field.lower()
        expressions[f"{field}0"] = f"${field_lower} / $close"

    # =========================================================================
    # 3. Rolling factors across all windows [5, 10, 20, 30, 60]
    # =========================================================================
    for d in ALPHA158_WINDOWS:
        # ROC: Rate of Change - price change over d days
        expressions[f"ROC{d}"] = f"Ref($close, {d}) / $close"

        # MA: Simple Moving Average ratio
        expressions[f"MA{d}"] = f"Mean($close, {d}) / $close"

        # STD: Standard deviation of close price
        expressions[f"STD{d}"] = f"Std($close, {d}) / $close"

        # BETA: Slope of price trend (regression coefficient)
        expressions[f"BETA{d}"] = f"Slope($close, {d}) / $close"

        # RSQR: R-squared of linear regression (trend linearity)
        expressions[f"RSQR{d}"] = f"Rsquare($close, {d})"

        # RESI: Residual of linear regression
        expressions[f"RESI{d}"] = f"Resi($close, {d}) / $close"

        # MAX: Maximum high price over d days
        expressions[f"MAX{d}"] = f"Max($high, {d}) / $close"

        # MIN: Minimum low price over d days
        expressions[f"MIN{d}"] = f"Min($low, {d}) / $close"

        # QTLU: 80% quantile of close price
        expressions[f"QTLU{d}"] = f"Quantile($close, {d}, 0.8) / $close"

        # QTLD: 20% quantile of close price
        expressions[f"QTLD{d}"] = f"Quantile($close, {d}, 0.2) / $close"

        # RANK: Percentile rank of current price in past d days
        expressions[f"RANK{d}"] = f"Rank($close, {d})"

        # RSV: Relative Strength Value (stochastic oscillator)
        expressions[f"RSV{d}"] = (
            f"($close - Min($low, {d})) / (Max($high, {d}) - Min($low, {d}) + 1e-12)"
        )

        # IMAX: Days since maximum high (Aroon indicator component)
        expressions[f"IMAX{d}"] = f"IdxMax($high, {d}) / {d}"

        # IMIN: Days since minimum low (Aroon indicator component)
        expressions[f"IMIN{d}"] = f"IdxMin($low, {d}) / {d}"

        # IMXD: Difference between IMAX and IMIN (momentum indicator)
        expressions[f"IMXD{d}"] = f"(IdxMax($high, {d}) - IdxMin($low, {d})) / {d}"

        # CORR: Correlation between price and log volume
        expressions[f"CORR{d}"] = f"Corr($close, Log($volume + 1), {d})"

        # CORD: Correlation between price change ratio and volume change ratio
        expressions[f"CORD{d}"] = (
            f"Corr($close / Ref($close, 1), Log($volume / Ref($volume, 1) + 1), {d})"
        )

        # CNTP: Percentage of up days in past d days
        expressions[f"CNTP{d}"] = f"Mean($close > Ref($close, 1), {d})"

        # CNTN: Percentage of down days in past d days
        expressions[f"CNTN{d}"] = f"Mean($close < Ref($close, 1), {d})"

        # CNTD: Difference between up and down day percentages
        expressions[f"CNTD{d}"] = (
            f"Mean($close > Ref($close, 1), {d}) - Mean($close < Ref($close, 1), {d})"
        )

        # SUMP: RSI-like - total gain ratio (similar to RSI)
        expressions[f"SUMP{d}"] = (
            f"Sum(Greater($close - Ref($close, 1), 0), {d}) / "
            f"(Sum(Abs($close - Ref($close, 1)), {d}) + 1e-12)"
        )

        # SUMN: RSI-like - total loss ratio
        expressions[f"SUMN{d}"] = (
            f"Sum(Greater(Ref($close, 1) - $close, 0), {d}) / "
            f"(Sum(Abs($close - Ref($close, 1)), {d}) + 1e-12)"
        )

        # SUMD: RSI-like - difference between gain and loss ratio
        expressions[f"SUMD{d}"] = (
            f"(Sum(Greater($close - Ref($close, 1), 0), {d}) - "
            f"Sum(Greater(Ref($close, 1) - $close, 0), {d})) / "
            f"(Sum(Abs($close - Ref($close, 1)), {d}) + 1e-12)"
        )

        # VMA: Volume moving average ratio
        expressions[f"VMA{d}"] = f"Mean($volume, {d}) / ($volume + 1e-12)"

        # VSTD: Volume standard deviation
        expressions[f"VSTD{d}"] = f"Std($volume, {d}) / ($volume + 1e-12)"

        # WVMA: Weighted volume moving average (price-weighted volatility)
        expressions[f"WVMA{d}"] = (
            f"Std(Abs($close / Ref($close, 1) - 1) * $volume, {d}) / "
            f"(Mean(Abs($close / Ref($close, 1) - 1) * $volume, {d}) + 1e-12)"
        )

        # VSUMP: Volume RSI-like - volume increase ratio
        expressions[f"VSUMP{d}"] = (
            f"Sum(Greater($volume - Ref($volume, 1), 0), {d}) / "
            f"(Sum(Abs($volume - Ref($volume, 1)), {d}) + 1e-12)"
        )

        # VSUMN: Volume RSI-like - volume decrease ratio
        expressions[f"VSUMN{d}"] = (
            f"Sum(Greater(Ref($volume, 1) - $volume, 0), {d}) / "
            f"(Sum(Abs($volume - Ref($volume, 1)), {d}) + 1e-12)"
        )

        # VSUMD: Volume RSI-like - difference between increase and decrease
        expressions[f"VSUMD{d}"] = (
            f"(Sum(Greater($volume - Ref($volume, 1), 0), {d}) - "
            f"Sum(Greater(Ref($volume, 1) - $volume, 0), {d})) / "
            f"(Sum(Abs($volume - Ref($volume, 1)), {d}) + 1e-12)"
        )

    return expressions


# Generate complete Alpha158 expressions
ALPHA158_EXPRESSIONS: dict[str, str] = _generate_alpha158_expressions()


# =============================================================================
# Alpha360 Factor Definitions
# P2.2 FIX: Add Alpha360 factor set for comprehensive benchmark coverage
#
# Reference: vendor/qlib/qlib/contrib/data/loader.py (Alpha360DL)
# Total factors: 360 = 6 fields × 60 lookback days
# Fields: CLOSE, OPEN, HIGH, LOW, VWAP, VOLUME (all normalized)
# =============================================================================

def _generate_alpha360_expressions() -> dict[str, str]:
    """Generate complete Alpha360 factor expressions matching Qlib official.

    Based on: qlib/contrib/data/loader.py::Alpha360DL.get_feature_config()

    Alpha360 provides raw price data normalized by current close/volume,
    capturing price patterns over the last 60 days.
    """
    expressions: dict[str, str] = {}

    # Price fields normalized by current close
    price_fields = ["close", "open", "high", "low", "vwap"]

    for field in price_fields:
        field_upper = field.upper()
        # Current day (day 0)
        expressions[f"{field_upper}0"] = f"${field} / $close"

        # Historical days (day 1 to day 59)
        for i in range(1, 60):
            expressions[f"{field_upper}{i}"] = f"Ref(${field}, {i}) / $close"

    # Volume normalized by current volume
    expressions["VOLUME0"] = "$volume / ($volume + 1e-12)"
    for i in range(1, 60):
        expressions[f"VOLUME{i}"] = f"Ref($volume, {i}) / ($volume + 1e-12)"

    return expressions


# Generate complete Alpha360 expressions
ALPHA360_EXPRESSIONS: dict[str, str] = _generate_alpha360_expressions()

# Backward compatibility alias (deprecated - use ALPHA158_EXPRESSIONS)
ALPHA158_FACTORS = ALPHA158_EXPRESSIONS


# =============================================================================
# REMOVED: Local Pandas Expression Engine (P0-2 Fix)
# =============================================================================
# The legacy Pandas-based QlibExpressionEngine has been REMOVED.
# All factor computations now go through the REAL Qlib C++ engine
# imported from iqfmp.core.qlib_crypto.QlibExpressionEngine.
#
# This ensures:
# - Production-grade performance (~100x faster than Pandas)
# - Consistency with Qlib's official operator implementations
# - No fallback paths that could introduce bugs
# =============================================================================


# =============================================================================
# Alpha Benchmarker using Qlib Expression Engine
# =============================================================================

class AlphaBenchmarker:
    """Benchmark new factors against Alpha158 standard factors.

    Uses Qlib expression engine for ALL factor computations.
    """

    def __init__(
        self,
        expressions: Optional[dict[str, str]] = None,
        novelty_threshold: float = 0.7,
    ) -> None:
        """Initialize benchmarker.

        Args:
            expressions: Custom expression dictionary, defaults to ALPHA158_EXPRESSIONS
            novelty_threshold: Max correlation with existing factors to be "novel"
        """
        self.expressions = expressions or ALPHA158_EXPRESSIONS
        self.novelty_threshold = novelty_threshold
        # Use lazy loader to get QlibExpressionEngine (avoid circular import)
        engine_class = _get_qlib_expression_engine()
        if engine_class is None:
            raise RuntimeError("QlibExpressionEngine not available - Qlib must be installed")
        self._engine = engine_class()
        self._benchmark_cache: dict[str, pd.Series] = {}
        self._benchmark_metrics: dict[str, dict[str, float]] = {}

    def compute_benchmark_factors(
        self,
        df: pd.DataFrame,
        factor_names: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Compute all benchmark factors via Qlib expression engine.

        Args:
            df: OHLCV DataFrame
            factor_names: Optional list of factors to compute

        Returns:
            DataFrame with all factor values
        """
        factors_to_compute = factor_names or list(self.expressions.keys())
        result = {}

        for name in factors_to_compute:
            if name not in self.expressions:
                continue

            try:
                expression = self.expressions[name]
                values = self._engine.compute_expression(expression, df)
                result[name] = values
                self._benchmark_cache[name] = values
            except Exception as e:
                logger.warning(f"Failed to compute {name}: {e}")
                continue

        return pd.DataFrame(result, index=df.index)

    def evaluate_benchmark_factors(
        self,
        df: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> dict[str, dict[str, float]]:
        """Evaluate all benchmark factors.

        Args:
            df: OHLCV DataFrame
            forward_returns: Forward returns for IC calculation

        Returns:
            Dictionary of factor metrics
        """
        # Compute benchmark factors if not cached
        if not self._benchmark_cache:
            self.compute_benchmark_factors(df)

        results = {}

        for name, factor_values in self._benchmark_cache.items():
            try:
                # Calculate IC
                valid_mask = ~(factor_values.isna() | forward_returns.isna())
                if valid_mask.sum() < 20:
                    continue

                ic, _ = spearman_rank_correlation(
                    factor_values[valid_mask],
                    forward_returns[valid_mask],
                )

                # Calculate IC stability (IR)
                ic_series = self._compute_rolling_ic(
                    factor_values, forward_returns, window=20
                )
                ic_mean = ic_series.mean() if len(ic_series) > 0 else 0.0
                ic_std = ic_series.std() if len(ic_series) > 0 else 1.0
                ir = ic_mean / ic_std if ic_std > 0 else 0.0

                # Sharpe from long-short backtest
                sharpe = self._backtest_sharpe(factor_values, df)

                results[name] = {
                    "ic": float(ic) if not np.isnan(ic) else 0.0,
                    "ir": float(ir),
                    "sharpe": float(sharpe),
                }

            except Exception:
                continue

        self._benchmark_metrics = results
        return results

    def benchmark(
        self,
        factor_name: str,
        factor_values: pd.Series,
        df: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> BenchmarkResult:
        """Benchmark a new factor against Alpha158.

        Args:
            factor_name: Name of the factor being benchmarked
            factor_values: Factor values to benchmark
            df: OHLCV DataFrame
            forward_returns: Forward returns

        Returns:
            BenchmarkResult with comparison metrics
        """
        # Ensure benchmark is computed
        if not self._benchmark_metrics:
            self.evaluate_benchmark_factors(df, forward_returns)

        # Calculate new factor metrics
        valid_mask = ~(factor_values.isna() | forward_returns.isna())

        if valid_mask.sum() < 20:
            raise ValueError("Insufficient valid data points for benchmarking")

        factor_ic, _ = spearman_rank_correlation(
            factor_values[valid_mask],
            forward_returns[valid_mask],
        )
        factor_ic = float(factor_ic) if not np.isnan(factor_ic) else 0.0

        ic_series = self._compute_rolling_ic(factor_values, forward_returns, window=20)
        ic_mean = ic_series.mean() if len(ic_series) > 0 else 0.0
        ic_std = ic_series.std() if len(ic_series) > 0 else 1.0
        factor_ir = ic_mean / ic_std if ic_std > 0 else 0.0

        factor_sharpe = self._backtest_sharpe(factor_values, df)

        # Calculate benchmark averages
        if not self._benchmark_metrics:
            benchmark_ic = benchmark_ir = benchmark_sharpe = 0.0
        else:
            ics = [m["ic"] for m in self._benchmark_metrics.values()]
            irs = [m["ir"] for m in self._benchmark_metrics.values()]
            sharpes = [m["sharpe"] for m in self._benchmark_metrics.values()]

            benchmark_ic = np.mean(ics) if ics else 0.0
            benchmark_ir = np.mean(irs) if irs else 0.0
            benchmark_sharpe = np.mean(sharpes) if sharpes else 0.0

        # Calculate improvements
        ic_improvement = (factor_ic - benchmark_ic) / (abs(benchmark_ic) + 1e-10)
        ir_improvement = (factor_ir - benchmark_ir) / (abs(benchmark_ir) + 1e-10)
        sharpe_improvement = (factor_sharpe - benchmark_sharpe) / (abs(benchmark_sharpe) + 1e-10)

        # Calculate rank among benchmark factors
        all_ics = [abs(m["ic"]) for m in self._benchmark_metrics.values()]
        all_ics.append(abs(factor_ic))
        all_ics.sort(reverse=True)
        rank = all_ics.index(abs(factor_ic)) + 1

        # Check if beats average
        beats_avg = (
            abs(factor_ic) > abs(benchmark_ic) and
            factor_ir > benchmark_ir
        )

        # Check novelty (low correlation with existing factors)
        max_corr = 0.0

        for name, bench_values in self._benchmark_cache.items():
            try:
                valid = ~(factor_values.isna() | bench_values.isna())
                if valid.sum() > 20:
                    corr, _ = spearman_rank_correlation(
                        factor_values[valid],
                        bench_values[valid],
                    )
                    corr = abs(corr) if not np.isnan(corr) else 0.0
                    if corr > max_corr:
                        max_corr = corr
            except Exception:
                continue

        is_novel = max_corr < self.novelty_threshold

        return BenchmarkResult(
            factor_name=factor_name,
            factor_ic=factor_ic,
            factor_ir=factor_ir,
            factor_sharpe=factor_sharpe,
            benchmark_ic=benchmark_ic,
            benchmark_ir=benchmark_ir,
            benchmark_sharpe=benchmark_sharpe,
            ic_improvement=ic_improvement,
            ir_improvement=ir_improvement,
            sharpe_improvement=sharpe_improvement,
            rank_in_benchmark=rank,
            total_benchmark_factors=len(self._benchmark_metrics) + 1,
            beats_benchmark_avg=beats_avg,
            correlation_with_best=max_corr,
            is_novel=is_novel,
        )

    def get_top_factors(
        self,
        n: int = 10,
        sort_by: str = "ic",
    ) -> list[tuple[str, dict[str, float]]]:
        """Get top N benchmark factors by specified metric.

        Args:
            n: Number of factors to return
            sort_by: Metric to sort by ("ic", "ir", "sharpe")

        Returns:
            List of (name, metrics) tuples
        """
        if not self._benchmark_metrics:
            return []

        sorted_factors = sorted(
            self._benchmark_metrics.items(),
            key=lambda x: abs(x[1].get(sort_by, 0)),
            reverse=True,
        )

        return sorted_factors[:n]

    def _compute_rolling_ic(
        self,
        factor: pd.Series,
        returns: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """Compute rolling IC."""
        ic_values = []

        for i in range(window, len(factor)):
            f_window = factor.iloc[i - window : i]
            r_window = returns.iloc[i - window : i]

            valid_mask = ~(f_window.isna() | r_window.isna())
            if valid_mask.sum() >= 5:
                ic, _ = spearman_rank_correlation(
                    f_window[valid_mask],
                    r_window[valid_mask],
                )
                if not np.isnan(ic):
                    ic_values.append(ic)

        return pd.Series(ic_values)

    def _backtest_sharpe(
        self,
        factor: pd.Series,
        df: pd.DataFrame,
    ) -> float:
        """Calculate Sharpe ratio from factor backtest."""
        # Normalize factor to z-score
        factor_zscore = (factor - factor.rolling(20).mean()) / (factor.rolling(20).std() + 1e-10)
        factor_zscore = factor_zscore.clip(-3, 3)

        # Position: sign of z-score
        position = np.sign(factor_zscore).fillna(0)

        # Strategy returns
        close_col = df.get("close", df.get("$close"))
        if close_col is None:
            return 0.0

        returns = close_col.pct_change().fillna(0)
        strategy_returns = position.shift(1) * returns

        # Sharpe ratio
        if strategy_returns.std() > 0:
            sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        return float(sharpe) if not np.isnan(sharpe) else 0.0


# =============================================================================
# Factory functions for easy integration
# P2.2 FIX: Support both Alpha158 and Alpha360 benchmarks
# =============================================================================

def create_alpha_benchmarker(
    benchmark_type: str = "alpha158",
    include_volume_factors: bool = True,
) -> AlphaBenchmarker:
    """Create an Alpha benchmarker using Qlib expressions.

    Args:
        benchmark_type: "alpha158" (default) or "alpha360"
        include_volume_factors: Whether to include volume-based factors

    Returns:
        Configured AlphaBenchmarker instance
    """
    if benchmark_type == "alpha360":
        expressions = ALPHA360_EXPRESSIONS
    else:
        expressions = ALPHA158_EXPRESSIONS

    if include_volume_factors:
        return AlphaBenchmarker(expressions=expressions)

    # Filter out volume factors
    non_volume_expressions = {
        k: v for k, v in expressions.items()
        if "$volume" not in v.lower() and "volume" not in k.lower()
    }
    return AlphaBenchmarker(expressions=non_volume_expressions)


def create_alpha158_benchmarker(
    include_volume_factors: bool = True,
) -> AlphaBenchmarker:
    """Create an Alpha158 benchmarker (158 technical indicators).

    Args:
        include_volume_factors: Whether to include volume-based factors

    Returns:
        Configured AlphaBenchmarker instance
    """
    return create_alpha_benchmarker("alpha158", include_volume_factors)


def create_alpha360_benchmarker(
    include_volume_factors: bool = True,
) -> AlphaBenchmarker:
    """Create an Alpha360 benchmarker (360 raw price features).

    Alpha360 uses 60-day lookback for 6 price fields (CLOSE, OPEN, HIGH, LOW, VWAP, VOLUME),
    providing raw normalized price data for ML models.

    Args:
        include_volume_factors: Whether to include volume-based factors

    Returns:
        Configured AlphaBenchmarker instance
    """
    return create_alpha_benchmarker("alpha360", include_volume_factors)


def get_available_factors(benchmark_type: str = "alpha158") -> list[str]:
    """Get list of available factor names.

    Args:
        benchmark_type: "alpha158" (default) or "alpha360"

    Returns:
        List of factor names
    """
    if benchmark_type == "alpha360":
        return list(ALPHA360_EXPRESSIONS.keys())
    return list(ALPHA158_EXPRESSIONS.keys())


def get_factor_expression(
    factor_name: str,
    benchmark_type: str = "alpha158",
) -> Optional[str]:
    """Get Qlib expression for a factor.

    Args:
        factor_name: Name of the factor
        benchmark_type: "alpha158" (default) or "alpha360"

    Returns:
        Qlib expression string or None if not found
    """
    if benchmark_type == "alpha360":
        return ALPHA360_EXPRESSIONS.get(factor_name)
    return ALPHA158_EXPRESSIONS.get(factor_name)


def get_factor_count() -> dict[str, int]:
    """Get count of factors in each benchmark set.

    Returns:
        Dictionary with factor counts
    """
    return {
        "alpha158": len(ALPHA158_EXPRESSIONS),
        "alpha360": len(ALPHA360_EXPRESSIONS),
        "total": len(ALPHA158_EXPRESSIONS) + len(ALPHA360_EXPRESSIONS),
    }


def get_combined_expressions() -> dict[str, str]:
    """Get combined Alpha158 + Alpha360 expressions.

    Note: Some factor names may overlap (e.g., CLOSE0, OPEN0).
    Alpha360 versions take precedence in the combined set.

    Returns:
        Combined dictionary of all expressions
    """
    combined = dict(ALPHA158_EXPRESSIONS)
    combined.update(ALPHA360_EXPRESSIONS)  # Alpha360 overwrites overlapping names
    return combined


# =============================================================================
# P3.5 FIX: Alpha Benchmark Runner with Research Trial Recording
# =============================================================================

def run_alpha_benchmark_workflow(
    df: pd.DataFrame,
    forward_returns: pd.Series,
    benchmark_type: str = "alpha158",
    record_to_ledger: bool = True,
    factor_family: str = "alpha_benchmark",
) -> dict[str, Any]:
    """Run Alpha158/360 benchmark workflow and record results to ResearchLedger.

    P3.5 FIX: Complete benchmark workflow that:
    1. Computes all Alpha158 or Alpha360 factors
    2. Evaluates each factor against forward returns
    3. Records trials to ResearchLedger (PostgresStorage)
    4. Returns comprehensive benchmark statistics

    Args:
        df: OHLCV DataFrame with columns [open, high, low, close, volume]
        forward_returns: Forward returns for IC calculation
        benchmark_type: "alpha158" or "alpha360"
        record_to_ledger: Whether to record trials to ResearchLedger
        factor_family: Factor family name for trial records

    Returns:
        Dictionary with:
            - total_factors: Total number of factors evaluated
            - passed_factors: Factors that pass IC threshold
            - top_factors: Top 10 factors by IC
            - summary_stats: Aggregate statistics
            - trial_ids: List of recorded trial IDs (if record_to_ledger=True)
    """
    from iqfmp.evaluation.research_ledger import ResearchLedger, TrialRecord

    logger.info(f"Starting {benchmark_type} benchmark workflow")

    # Create benchmarker
    benchmarker = create_alpha_benchmarker(benchmark_type=benchmark_type)

    # Compute and evaluate all benchmark factors
    benchmarker.compute_benchmark_factors(df)
    metrics = benchmarker.evaluate_benchmark_factors(df, forward_returns)

    logger.info(f"Evaluated {len(metrics)} factors")

    # Initialize ledger for recording (if enabled)
    ledger: Optional[ResearchLedger] = None
    trial_ids: list[str] = []

    if record_to_ledger:
        ledger = ResearchLedger()  # Uses PostgresStorage by default
        logger.info("Recording benchmark trials to ResearchLedger")

    # Process each factor
    passed_count = 0
    ic_threshold = 0.03  # 3% IC threshold for "passing"

    for factor_name, factor_metrics in metrics.items():
        ic = factor_metrics.get("ic", 0.0)
        ir = factor_metrics.get("ir", 0.0)
        sharpe = factor_metrics.get("sharpe", 0.0)

        passes = abs(ic) >= ic_threshold
        if passes:
            passed_count += 1

        # Record trial to ledger
        if ledger is not None:
            try:
                trial = TrialRecord(
                    factor_name=factor_name,
                    factor_family=factor_family,
                    sharpe_ratio=sharpe,
                    ic_mean=ic,
                    ir=ir,
                    metadata={
                        "benchmark_type": benchmark_type,
                        "passed_threshold": passes,
                        "ic_threshold": ic_threshold,
                    },
                )
                trial_id = ledger.record(trial)
                trial_ids.append(trial_id)
            except Exception as e:
                logger.warning(f"Failed to record trial for {factor_name}: {e}")

    # Get top factors
    top_factors = benchmarker.get_top_factors(n=10, sort_by="ic")

    # Calculate summary statistics
    all_ics = [abs(m.get("ic", 0)) for m in metrics.values()]
    all_irs = [m.get("ir", 0) for m in metrics.values()]
    all_sharpes = [m.get("sharpe", 0) for m in metrics.values()]

    summary_stats = {
        "mean_ic": float(np.mean(all_ics)) if all_ics else 0.0,
        "median_ic": float(np.median(all_ics)) if all_ics else 0.0,
        "max_ic": float(max(all_ics)) if all_ics else 0.0,
        "mean_ir": float(np.mean(all_irs)) if all_irs else 0.0,
        "mean_sharpe": float(np.mean(all_sharpes)) if all_sharpes else 0.0,
        "pass_rate": passed_count / len(metrics) if metrics else 0.0,
    }

    result = {
        "benchmark_type": benchmark_type,
        "total_factors": len(metrics),
        "passed_factors": passed_count,
        "top_factors": [
            {"name": name, **m} for name, m in top_factors
        ],
        "summary_stats": summary_stats,
        "trial_ids": trial_ids,
        "recorded_trials": len(trial_ids),
    }

    logger.info(
        f"Benchmark complete: {len(metrics)} factors, "
        f"{passed_count} passed ({summary_stats['pass_rate']:.1%}), "
        f"mean IC={summary_stats['mean_ic']:.4f}"
    )

    return result


async def run_alpha_benchmark_async(
    df: pd.DataFrame,
    forward_returns: pd.Series,
    benchmark_type: str = "alpha158",
) -> dict[str, Any]:
    """Async wrapper for alpha benchmark workflow.

    Args:
        df: OHLCV DataFrame
        forward_returns: Forward returns for IC calculation
        benchmark_type: "alpha158" or "alpha360"

    Returns:
        Benchmark result dictionary
    """
    import asyncio
    return await asyncio.to_thread(
        run_alpha_benchmark_workflow,
        df,
        forward_returns,
        benchmark_type,
    )
