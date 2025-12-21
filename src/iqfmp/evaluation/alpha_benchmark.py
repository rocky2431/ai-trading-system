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
# These are passed to Qlib's expression engine for computation
# =============================================================================

ALPHA158_EXPRESSIONS: dict[str, str] = {
    # K-line shape factors
    "KMID": "($close - $low) / ($high - $low + 1e-10)",
    "KLEN": "($high - $low) / $close",
    "KMID2": "($close - $open) / ($high - $low + 1e-10)",
    "KUP": "($high - Greater($open, $close)) / ($high - $low + 1e-10)",
    "KUP2": "($high - $open) / ($high - $low + 1e-10)",
    "KLOW": "(Less($open, $close) - $low) / ($high - $low + 1e-10)",
    "KLOW2": "($open - $low) / ($high - $low + 1e-10)",
    "KSFT": "(2 * $close - $high - $low) / ($high - $low + 1e-10)",
    "KSFT2": "(2 * $close - $high - $low) / $close",

    # Rate of Change (ROC) factors
    "ROC5": "Ref($close, 5) / $close - 1",
    "ROC10": "Ref($close, 10) / $close - 1",
    "ROC20": "Ref($close, 20) / $close - 1",

    # Moving Average deviation factors
    "MA5_RATIO": "$close / Mean($close, 5) - 1",
    "MA10_RATIO": "$close / Mean($close, 10) - 1",
    "MA20_RATIO": "$close / Mean($close, 20) - 1",

    # Volatility factors
    "STD5": "Std(Ref($close, 1) / $close - 1, 5)",
    "STD10": "Std(Ref($close, 1) / $close - 1, 10)",
    "STD20": "Std(Ref($close, 1) / $close - 1, 20)",

    # Beta and regression factors
    "BETA5": "Slope($close, 5) / $close",
    "RSQR5": "Rsquare($close, 5)",
    "RESI5": "Resi($close, 5) / $close",

    # Extreme value factors
    "MAX5": "Max(Ref($close, 1) / $close - 1, 5)",
    "MIN5": "Min(Ref($close, 1) / $close - 1, 5)",
    "QTLU5": "Quantile(Ref($close, 1) / $close - 1, 5, 0.8)",
    "QTLD5": "Quantile(Ref($close, 1) / $close - 1, 5, 0.2)",
    "RANK5": "Rank($close, 5)",

    # Stochastic factors
    # RSV (Relative Strength Value) - Use Min/Max instead of TsMin/TsMax
    "RSV5": "($close - Min($low, 5)) / (Max($high, 5) - Min($low, 5) + 1e-10)",
    "RSV10": "($close - Min($low, 10)) / (Max($high, 10) - Min($low, 10) + 1e-10)",
    "RSV20": "($close - Min($low, 20)) / (Max($high, 20) - Min($low, 20) + 1e-10)",

    # Index of max/min factors
    "IMAX5": "IdxMax($high, 5) / 5",
    "IMIN5": "IdxMin($low, 5) / 5",
    "IMXD5": "(IdxMax($high, 5) - IdxMin($low, 5)) / 5",

    # Volume factors
    "VMA5": "Mean($volume, 5) / (Mean($volume, 20) + 1e-10) - 1",
    "VMA10": "Mean($volume, 10) / (Mean($volume, 20) + 1e-10) - 1",
    "VSTD5": "Std($volume, 5) / (Mean($volume, 20) + 1e-10)",
    "VSTD10": "Std($volume, 10) / (Mean($volume, 20) + 1e-10)",
    "WVMA5": "Std(Abs(Ref($close, 1) / $close - 1) * $volume, 5) / (Mean(Abs(Ref($close, 1) / $close - 1) * $volume, 20) + 1e-10)",
    "TURN5": "Mean($volume, 5) / (Mean($volume, 20) + 1e-10) - 1",

    # RSI factors
    "RSI6": "100 - 100 / (1 + Mean(Greater(Ref($close, 1) / $close - 1, 0), 6) / (Mean(Abs(Less(Ref($close, 1) / $close - 1, 0)), 6) + 1e-10))",
    "RSI14": "100 - 100 / (1 + Mean(Greater(Ref($close, 1) / $close - 1, 0), 14) / (Mean(Abs(Less(Ref($close, 1) / $close - 1, 0)), 14) + 1e-10))",

    # Correlation factors
    "CORR5": "Corr(Ref($close, 1) / $close - 1, Ref($volume, 1) / $volume - 1, 5)",
    "CORR10": "Corr(Ref($close, 1) / $close - 1, Ref($volume, 1) / $volume - 1, 10)",
    "CORD5": "Corr(Rank(Ref($close, 1) / $close - 1, 5), Rank($volume, 5), 5)",

    # Price level factors
    "SUMP5": "Sum(Greater(Ref($close, 1) / $close - 1, 0), 5) / (Sum(Abs(Ref($close, 1) / $close - 1), 5) + 1e-10)",
    "SUMN5": "Sum(Less(Ref($close, 1) / $close - 1, 0), 5) / (Sum(Abs(Ref($close, 1) / $close - 1), 5) + 1e-10)",
    "SUMD5": "(Sum(Greater(Ref($close, 1) / $close - 1, 0), 5) - Sum(Less(Ref($close, 1) / $close - 1, 0), 5)) / (Sum(Abs(Ref($close, 1) / $close - 1), 5) + 1e-10)",
}

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
        self._engine = QlibExpressionEngine()
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
# Factory function for easy integration
# =============================================================================

def create_alpha_benchmarker(
    include_volume_factors: bool = True,
) -> AlphaBenchmarker:
    """Create an Alpha158 benchmarker using Qlib expressions.

    Args:
        include_volume_factors: Whether to include volume-based factors

    Returns:
        Configured AlphaBenchmarker instance
    """
    if include_volume_factors:
        return AlphaBenchmarker(expressions=ALPHA158_EXPRESSIONS)

    # Filter out volume factors
    non_volume_expressions = {
        k: v for k, v in ALPHA158_EXPRESSIONS.items()
        if "$volume" not in v.lower() and "volume" not in k.lower()
    }
    return AlphaBenchmarker(expressions=non_volume_expressions)


def get_available_factors() -> list[str]:
    """Get list of available Alpha158 factor names."""
    return list(ALPHA158_EXPRESSIONS.keys())


def get_factor_expression(factor_name: str) -> Optional[str]:
    """Get Qlib expression for a factor.

    Args:
        factor_name: Name of the factor

    Returns:
        Qlib expression string or None if not found
    """
    return ALPHA158_EXPRESSIONS.get(factor_name)
