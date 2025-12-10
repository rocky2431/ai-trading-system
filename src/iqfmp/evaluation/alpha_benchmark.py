"""Alpha158 Benchmark for factor comparison.

This module provides Alpha158 (and Alpha101/Alpha360) factor implementations
for benchmarking new factors against established standards.

Based on Qlib's Alpha158 factor set but adapted for cryptocurrency markets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import numpy as np
import pandas as pd
from scipy import stats


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
# Alpha158 Core Factor Definitions (18 key factors from Qlib)
# =============================================================================

ALPHA158_FACTORS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {}


def _register_factor(name: str):
    """Decorator to register an Alpha158 factor."""
    def decorator(func: Callable[[pd.DataFrame], pd.Series]):
        ALPHA158_FACTORS[name] = func
        return func
    return decorator


@_register_factor("KMID")
def kmid(df: pd.DataFrame) -> pd.Series:
    """K-line middle price relative position."""
    return (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-10)


@_register_factor("KLEN")
def klen(df: pd.DataFrame) -> pd.Series:
    """K-line length normalized by close."""
    return (df["high"] - df["low"]) / df["close"]


@_register_factor("KMID2")
def kmid2(df: pd.DataFrame) -> pd.Series:
    """K-line middle price: (close - open) / (high - low)."""
    return (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-10)


@_register_factor("KUP")
def kup(df: pd.DataFrame) -> pd.Series:
    """Upper shadow ratio."""
    return (df["high"] - df[["open", "close"]].max(axis=1)) / (df["high"] - df["low"] + 1e-10)


@_register_factor("KUP2")
def kup2(df: pd.DataFrame) -> pd.Series:
    """Extended upper shadow ratio."""
    return (df["high"] - df["open"]) / (df["high"] - df["low"] + 1e-10)


@_register_factor("KLOW")
def klow(df: pd.DataFrame) -> pd.Series:
    """Lower shadow ratio."""
    return (df[["open", "close"]].min(axis=1) - df["low"]) / (df["high"] - df["low"] + 1e-10)


@_register_factor("KLOW2")
def klow2(df: pd.DataFrame) -> pd.Series:
    """Extended lower shadow ratio."""
    return (df["open"] - df["low"]) / (df["high"] - df["low"] + 1e-10)


@_register_factor("KSFT")
def ksft(df: pd.DataFrame) -> pd.Series:
    """K-line shift: (2*close - high - low) / (high - low)."""
    return (2 * df["close"] - df["high"] - df["low"]) / (df["high"] - df["low"] + 1e-10)


@_register_factor("KSFT2")
def ksft2(df: pd.DataFrame) -> pd.Series:
    """K-line shift v2: (2*close - high - low) / close."""
    return (2 * df["close"] - df["high"] - df["low"]) / df["close"]


@_register_factor("ROC5")
def roc5(df: pd.DataFrame) -> pd.Series:
    """5-day rate of change."""
    return df["close"].pct_change(5)


@_register_factor("ROC10")
def roc10(df: pd.DataFrame) -> pd.Series:
    """10-day rate of change."""
    return df["close"].pct_change(10)


@_register_factor("ROC20")
def roc20(df: pd.DataFrame) -> pd.Series:
    """20-day rate of change."""
    return df["close"].pct_change(20)


@_register_factor("MA5_RATIO")
def ma5_ratio(df: pd.DataFrame) -> pd.Series:
    """Close / MA5 - 1."""
    ma5 = df["close"].rolling(5).mean()
    return df["close"] / ma5 - 1


@_register_factor("MA10_RATIO")
def ma10_ratio(df: pd.DataFrame) -> pd.Series:
    """Close / MA10 - 1."""
    ma10 = df["close"].rolling(10).mean()
    return df["close"] / ma10 - 1


@_register_factor("MA20_RATIO")
def ma20_ratio(df: pd.DataFrame) -> pd.Series:
    """Close / MA20 - 1."""
    ma20 = df["close"].rolling(20).mean()
    return df["close"] / ma20 - 1


@_register_factor("STD5")
def std5(df: pd.DataFrame) -> pd.Series:
    """5-day standard deviation of returns."""
    return df["close"].pct_change().rolling(5).std()


@_register_factor("STD10")
def std10(df: pd.DataFrame) -> pd.Series:
    """10-day standard deviation of returns."""
    return df["close"].pct_change().rolling(10).std()


@_register_factor("STD20")
def std20(df: pd.DataFrame) -> pd.Series:
    """20-day standard deviation of returns."""
    return df["close"].pct_change().rolling(20).std()


@_register_factor("BETA5")
def beta5(df: pd.DataFrame) -> pd.Series:
    """5-day beta approximation using rolling correlation with market."""
    returns = df["close"].pct_change()
    market_returns = returns.rolling(20).mean()  # Proxy for market
    cov = returns.rolling(5).cov(market_returns)
    var = market_returns.rolling(5).var()
    return cov / (var + 1e-10)


@_register_factor("RSQR5")
def rsqr5(df: pd.DataFrame) -> pd.Series:
    """5-day R-squared of returns vs market proxy."""
    returns = df["close"].pct_change()
    ma = returns.rolling(5).mean()
    return returns.rolling(5).corr(ma) ** 2


@_register_factor("RESI5")
def resi5(df: pd.DataFrame) -> pd.Series:
    """5-day residual volatility."""
    returns = df["close"].pct_change()
    ma = returns.rolling(5).mean()
    residual = returns - ma
    return residual.rolling(5).std()


@_register_factor("MAX5")
def max5(df: pd.DataFrame) -> pd.Series:
    """5-day max return."""
    return df["close"].pct_change().rolling(5).max()


@_register_factor("MIN5")
def min5(df: pd.DataFrame) -> pd.Series:
    """5-day min return."""
    return df["close"].pct_change().rolling(5).min()


@_register_factor("QTLU5")
def qtlu5(df: pd.DataFrame) -> pd.Series:
    """5-day 80th percentile return."""
    return df["close"].pct_change().rolling(5).quantile(0.8)


@_register_factor("QTLD5")
def qtld5(df: pd.DataFrame) -> pd.Series:
    """5-day 20th percentile return."""
    return df["close"].pct_change().rolling(5).quantile(0.2)


@_register_factor("RANK5")
def rank5(df: pd.DataFrame) -> pd.Series:
    """Rank of current close within 5-day window."""
    return df["close"].rolling(5).apply(
        lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100,
        raw=False,
    )


@_register_factor("RSV5")
def rsv5(df: pd.DataFrame) -> pd.Series:
    """5-day Raw Stochastic Value."""
    lowest = df["low"].rolling(5).min()
    highest = df["high"].rolling(5).max()
    return (df["close"] - lowest) / (highest - lowest + 1e-10)


@_register_factor("IMAX5")
def imax5(df: pd.DataFrame) -> pd.Series:
    """Days since 5-day high."""
    return df["high"].rolling(5).apply(
        lambda x: 5 - x.argmax() - 1,
        raw=True,
    )


@_register_factor("IMIN5")
def imin5(df: pd.DataFrame) -> pd.Series:
    """Days since 5-day low."""
    return df["low"].rolling(5).apply(
        lambda x: 5 - x.argmin() - 1,
        raw=True,
    )


@_register_factor("IMXD5")
def imxd5(df: pd.DataFrame) -> pd.Series:
    """Days between 5-day high and low."""
    imax = df["high"].rolling(5).apply(lambda x: 5 - x.argmax() - 1, raw=True)
    imin = df["low"].rolling(5).apply(lambda x: 5 - x.argmin() - 1, raw=True)
    return imax - imin


@_register_factor("VMA5")
def vma5(df: pd.DataFrame) -> pd.Series:
    """5-day volume moving average ratio."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    vma5 = df["volume"].rolling(5).mean()
    vma20 = df["volume"].rolling(20).mean()
    return vma5 / (vma20 + 1e-10) - 1


@_register_factor("VSTD5")
def vstd5(df: pd.DataFrame) -> pd.Series:
    """5-day volume standard deviation."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"].rolling(5).std() / (df["volume"].rolling(20).mean() + 1e-10)


@_register_factor("WVMA5")
def wvma5(df: pd.DataFrame) -> pd.Series:
    """5-day weighted volume moving average."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    weighted = df["volume"] * df["close"].pct_change().abs()
    return weighted.rolling(5).mean() / (weighted.rolling(20).mean() + 1e-10)


@_register_factor("TURN5")
def turn5(df: pd.DataFrame) -> pd.Series:
    """5-day turnover ratio change."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    turn = df["volume"] / (df["volume"].rolling(20).mean() + 1e-10)
    return turn.rolling(5).mean() - 1


@_register_factor("RSI6")
def rsi6(df: pd.DataFrame) -> pd.Series:
    """6-period RSI."""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


@_register_factor("RSI14")
def rsi14(df: pd.DataFrame) -> pd.Series:
    """14-period RSI."""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


@_register_factor("CORR5")
def corr5(df: pd.DataFrame) -> pd.Series:
    """5-day correlation between returns and volume change."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    returns = df["close"].pct_change()
    vol_change = df["volume"].pct_change()
    return returns.rolling(5).corr(vol_change)


@_register_factor("CORD5")
def cord5(df: pd.DataFrame) -> pd.Series:
    """5-day correlation between returns rank and volume rank."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    returns = df["close"].pct_change()
    vol = df["volume"]

    def rolling_rank_corr(data):
        r = data[:5]
        v = data[5:]
        if len(r) < 5:
            return np.nan
        return stats.spearmanr(r, v)[0]

    combined = pd.concat([returns, vol], axis=1)
    return combined.rolling(10).apply(
        lambda x: stats.spearmanr(x[:5], x[5:])[0] if len(x) >= 10 else np.nan,
        raw=True,
    )


class AlphaBenchmarker:
    """Benchmark new factors against Alpha158 standard factors."""

    def __init__(
        self,
        factors: Optional[dict[str, Callable[[pd.DataFrame], pd.Series]]] = None,
        novelty_threshold: float = 0.7,  # Correlation threshold for novelty
    ) -> None:
        """Initialize benchmarker.

        Args:
            factors: Custom factor dictionary, defaults to ALPHA158_FACTORS
            novelty_threshold: Max correlation with existing factors to be "novel"
        """
        self.factors = factors or ALPHA158_FACTORS
        self.novelty_threshold = novelty_threshold
        self._benchmark_cache: dict[str, pd.Series] = {}
        self._benchmark_metrics: dict[str, dict[str, float]] = {}

    def compute_benchmark_factors(
        self,
        df: pd.DataFrame,
        factor_names: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Compute all benchmark factors.

        Args:
            df: OHLCV DataFrame
            factor_names: Optional list of factors to compute

        Returns:
            DataFrame with all factor values
        """
        factors_to_compute = factor_names or list(self.factors.keys())
        result = {}

        for name in factors_to_compute:
            if name not in self.factors:
                continue

            try:
                factor_func = self.factors[name]
                values = factor_func(df)
                result[name] = values
                self._benchmark_cache[name] = values
            except Exception as e:
                # Skip factors that fail (e.g., missing volume)
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

                ic, _ = stats.spearmanr(
                    factor_values[valid_mask].values,
                    forward_returns[valid_mask].values,
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

        factor_ic, _ = stats.spearmanr(
            factor_values[valid_mask].values,
            forward_returns[valid_mask].values,
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
        best_corr_factor = ""

        for name, bench_values in self._benchmark_cache.items():
            try:
                valid = ~(factor_values.isna() | bench_values.isna())
                if valid.sum() > 20:
                    corr, _ = stats.spearmanr(
                        factor_values[valid].values,
                        bench_values[valid].values,
                    )
                    corr = abs(corr) if not np.isnan(corr) else 0.0
                    if corr > max_corr:
                        max_corr = corr
                        best_corr_factor = name
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
                ic, _ = stats.spearmanr(
                    f_window[valid_mask].values,
                    r_window[valid_mask].values,
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
        returns = df["close"].pct_change().fillna(0)
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
    """Create an Alpha158 benchmarker.

    Args:
        include_volume_factors: Whether to include volume-based factors

    Returns:
        Configured AlphaBenchmarker instance
    """
    if include_volume_factors:
        return AlphaBenchmarker(factors=ALPHA158_FACTORS)

    # Filter out volume factors
    non_volume_factors = {
        k: v for k, v in ALPHA158_FACTORS.items()
        if not k.startswith("V") and not k.startswith("TURN") and k not in ("WVMA5", "CORR5", "CORD5")
    }
    return AlphaBenchmarker(factors=non_volume_factors)


def get_available_factors() -> list[str]:
    """Get list of available Alpha158 factor names."""
    return list(ALPHA158_FACTORS.keys())
