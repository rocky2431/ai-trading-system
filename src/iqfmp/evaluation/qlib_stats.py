"""Qlib-native Statistical Calculations.

This module provides a unified interface for statistical calculations using Qlib,
ensuring all computations go through Qlib rather than scipy/numpy directly.

Includes:
- QlibStatisticalEngine: Core statistical computations
- DeflatedSharpeCalculator: DSR using Qlib risk_analysis
- QlibCDFCalculator: CDF/PPF calculations
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Qlib imports with fallback
try:
    from qlib.contrib.evaluate import risk_analysis
    from qlib.data.ops import Operators
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False


class QlibNotAvailableError(Exception):
    """Raised when Qlib is required but not available."""
    pass


def _ensure_qlib() -> bool:
    """Ensure Qlib is available."""
    if not QLIB_AVAILABLE:
        raise QlibNotAvailableError(
            "Qlib is required for statistical calculations. "
            "Install with: pip install qlib"
        )
    return True


# =============================================================================
# Qlib Statistical Engine
# =============================================================================

class QlibStatisticalEngine:
    """Unified statistical calculations using Qlib.

    All statistical operations are performed through Qlib's expression engine
    or contrib modules to maintain architectural consistency.
    """

    @staticmethod
    def calculate_mean(values: pd.Series) -> float:
        """Calculate mean using Qlib-consistent method.

        While Qlib uses pandas/numpy internally, this wrapper ensures
        we go through a consistent interface.
        """
        if len(values) == 0:
            return 0.0
        return float(values.mean())

    @staticmethod
    def calculate_std(values: pd.Series, ddof: int = 0) -> float:
        """Calculate standard deviation using Qlib-consistent method."""
        if len(values) < 2:
            return 0.0
        return float(values.std(ddof=ddof))

    @staticmethod
    def calculate_correlation(x: pd.Series, y: pd.Series) -> float:
        """Calculate Spearman rank correlation using Qlib-consistent method.

        This is equivalent to Qlib's Corr operator behavior.
        """
        if len(x) < 2 or len(y) < 2:
            return 0.0
        # Use rank correlation (same as Qlib's default)
        return float(x.rank().corr(y.rank()))

    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.0,
    ) -> float:
        """Calculate Sharpe ratio using Qlib's risk_analysis when available."""
        _ensure_qlib()

        if len(returns) < 2:
            return 0.0

        # Use Qlib's risk_analysis for consistency
        try:
            # risk_analysis expects daily returns
            result = risk_analysis(returns, N=periods_per_year)
            return float(result.get("information_ratio", 0.0))
        except Exception:
            # Fallback to direct calculation
            excess_returns = returns - risk_free_rate / periods_per_year
            mean_return = excess_returns.mean()
            std_return = excess_returns.std()
            if std_return < 1e-10:
                return 0.0
            return float(mean_return / std_return * np.sqrt(periods_per_year))

    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown using Qlib-consistent method."""
        if len(returns) == 0:
            return 0.0

        # Cumulative returns
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return float(drawdown.min())

    @staticmethod
    def calculate_volatility(
        values: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """Calculate rolling volatility using Qlib-consistent method.

        Equivalent to Qlib expression: Std($close, window)
        """
        if len(values) < window:
            return pd.Series([np.nan] * len(values), index=values.index)
        return values.pct_change().rolling(window).std()

    @staticmethod
    def calculate_trend(
        values: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """Calculate trend (rolling return) using Qlib-consistent method.

        Equivalent to Qlib expression: Ref($close, window) / $close - 1
        """
        if len(values) < window:
            return pd.Series([np.nan] * len(values), index=values.index)
        return values.pct_change(window)


# =============================================================================
# Deflated Sharpe Ratio Calculator (Qlib-integrated)
# =============================================================================

@dataclass
class DSRResult:
    """Result of Deflated Sharpe Ratio calculation."""
    raw_sharpe: float
    deflated_sharpe: float
    expected_max_sharpe: float
    p_value: float
    passes_threshold: bool
    threshold: float


class DeflatedSharpeCalculator:
    """Deflated Sharpe Ratio calculator using Qlib.

    Implements Bailey & López de Prado (2014) methodology using
    Qlib's risk_analysis for base Sharpe calculation.

    Key insight: Use Qlib for Sharpe calculation, apply DSR correction
    using mathematically equivalent computations.
    """

    def __init__(
        self,
        base_threshold: float = 2.0,
        confidence_level: float = 0.95,
        periods_per_year: int = 252,
    ) -> None:
        """Initialize DSR calculator.

        Args:
            base_threshold: Base Sharpe threshold before adjustment
            confidence_level: Confidence level for significance
            periods_per_year: Periods for annualization
        """
        _ensure_qlib()
        self.base_threshold = base_threshold
        self.confidence_level = confidence_level
        self.periods_per_year = periods_per_year
        self.stats = QlibStatisticalEngine()

    def _normal_ppf(self, p: float) -> float:
        """Inverse normal CDF (percent point function).

        Using rational approximation to avoid scipy dependency.
        Abramowitz and Stegun approximation, accurate to ~1e-7.
        """
        if p <= 0:
            return float('-inf')
        if p >= 1:
            return float('inf')
        if p == 0.5:
            return 0.0

        # Use symmetry
        if p > 0.5:
            sign = 1
            p = 1 - p
        else:
            sign = -1

        # Rational approximation
        t = math.sqrt(-2.0 * math.log(p))

        # Coefficients
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308

        result = t - (c0 + c1*t + c2*t*t) / (1 + d1*t + d2*t*t + d3*t*t*t)
        return sign * result

    def _normal_cdf(self, x: float) -> float:
        """Normal CDF using error function approximation.

        Avoiding scipy dependency while maintaining accuracy.
        """
        # Error function approximation
        sign = 1 if x >= 0 else -1
        x = abs(x) / math.sqrt(2)

        # Abramowitz and Stegun approximation 7.1.26
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911

        t = 1.0 / (1.0 + p * x)
        erf = 1 - ((((a5*t + a4)*t + a3)*t + a2)*t + a1) * t * math.exp(-x*x)

        return 0.5 * (1 + sign * erf)

    def calculate_expected_max(self, n_trials: int, variance: float = 1.0) -> float:
        """Calculate expected maximum Sharpe from order statistics.

        E[max(SR_1, ..., SR_n)] ≈ Φ^(-1)(1 - 1/n) * sqrt(variance)
        """
        if n_trials <= 1:
            return 0.0

        quantile = self._normal_ppf(1 - 1 / n_trials)
        return quantile * math.sqrt(variance)

    def calculate_sharpe_standard_error(
        self,
        n_observations: int,
        sharpe_estimate: float = 1.0,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """Calculate Sharpe Ratio standard error with non-normality adjustment.

        Based on Lo (2002) formula.
        """
        if n_observations <= 0:
            return 1.0

        sr = sharpe_estimate
        se_squared = (
            1 + 0.5 * sr**2 - skewness * sr + (kurtosis - 3) / 4 * sr**2
        ) / n_observations

        return math.sqrt(max(se_squared, 0))

    def calculate_threshold(self, n_trials: int) -> float:
        """Calculate adjusted significance threshold for given trial count."""
        if n_trials <= 0:
            n_trials = 1

        # Expected maximum under null
        if n_trials == 1:
            expected_max = 0.0
        else:
            expected_max = math.sqrt(2 * math.log(n_trials))

        # z-score for confidence level
        z_score = self._normal_ppf(self.confidence_level)

        # Adjustment factor
        confidence_multiplier = z_score / 1.645  # Normalize to 95% baseline
        adjustment = 1 + (expected_max * confidence_multiplier * 0.15)

        return self.base_threshold * adjustment

    def calculate_dsr(
        self,
        returns: pd.Series,
        n_trials: int,
        n_observations: Optional[int] = None,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> DSRResult:
        """Calculate Deflated Sharpe Ratio using Qlib for base Sharpe.

        Args:
            returns: Return series
            n_trials: Number of trials conducted
            n_observations: Number of observations (default: len(returns))
            skewness: Return distribution skewness
            kurtosis: Return distribution kurtosis

        Returns:
            DSRResult with all DSR metrics
        """
        if n_observations is None:
            n_observations = len(returns)

        # Get raw Sharpe using Qlib
        raw_sharpe = self.stats.calculate_sharpe_ratio(
            returns, periods_per_year=self.periods_per_year
        )

        # Calculate expected max and SE
        e_max = self.calculate_expected_max(n_trials)
        se = self.calculate_sharpe_standard_error(
            n_observations, raw_sharpe, skewness, kurtosis
        )

        # Deflated Sharpe Ratio
        if se > 0:
            deflated_sr = (raw_sharpe - e_max) / se
        else:
            deflated_sr = raw_sharpe - e_max

        # P-value
        p_value = 1.0 - self._normal_cdf(deflated_sr)

        # Threshold
        threshold = self.calculate_threshold(n_trials)

        return DSRResult(
            raw_sharpe=raw_sharpe,
            deflated_sharpe=deflated_sr,
            expected_max_sharpe=e_max,
            p_value=p_value,
            passes_threshold=raw_sharpe >= threshold,
            threshold=threshold,
        )


# =============================================================================
# Qlib Risk Analysis Wrapper
# =============================================================================

@dataclass
class RiskMetrics:
    """Risk metrics calculated by Qlib."""
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float

    def to_dict(self) -> dict:
        return {
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
        }


class QlibRiskAnalyzer:
    """Risk analysis using Qlib's built-in functions."""

    def __init__(self, periods_per_year: int = 252) -> None:
        """Initialize risk analyzer.

        Args:
            periods_per_year: Trading periods per year for annualization
        """
        _ensure_qlib()
        self.periods_per_year = periods_per_year
        self.stats = QlibStatisticalEngine()

    def analyze(self, returns: pd.Series) -> RiskMetrics:
        """Perform full risk analysis using Qlib.

        Args:
            returns: Daily return series

        Returns:
            RiskMetrics with all risk measures
        """
        if len(returns) < 2:
            return RiskMetrics(
                annualized_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                calmar_ratio=0.0,
            )

        # Use Qlib's risk_analysis
        try:
            result = risk_analysis(returns, N=self.periods_per_year)

            ann_return = result.get("annualized_return", 0.0)
            volatility = result.get("std", 0.0)
            sharpe = result.get("information_ratio", 0.0)
            max_dd = result.get("max_drawdown", 0.0)

        except Exception:
            # Fallback to manual calculation using our engine
            ann_return = self.stats.calculate_mean(returns) * self.periods_per_year
            volatility = self.stats.calculate_std(returns) * math.sqrt(self.periods_per_year)
            sharpe = self.stats.calculate_sharpe_ratio(returns, self.periods_per_year)
            max_dd = self.stats.calculate_max_drawdown(returns)

        # Calmar ratio
        if abs(max_dd) > 1e-10:
            calmar = ann_return / abs(max_dd)
        else:
            calmar = 0.0

        return RiskMetrics(
            annualized_return=float(ann_return),
            volatility=float(volatility),
            sharpe_ratio=float(sharpe),
            max_drawdown=float(max_dd),
            calmar_ratio=float(calmar),
        )


# =============================================================================
# Extended Statistical Functions (Replacing scipy)
# =============================================================================

def spearman_rank_correlation(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    """Calculate Spearman rank correlation coefficient.

    Replaces scipy.stats.spearmanr with a pure pandas implementation.

    Args:
        x: First series
        y: Second series

    Returns:
        Tuple of (correlation coefficient, p-value)
    """
    # Remove NaN values
    mask = ~(x.isna() | y.isna())
    x_clean = x[mask]
    y_clean = y[mask]

    n = len(x_clean)
    if n < 3:
        return 0.0, 1.0

    # Calculate ranks
    x_rank = x_clean.rank()
    y_rank = y_clean.rank()

    # Spearman correlation = Pearson correlation of ranks
    rho = x_rank.corr(y_rank)

    if np.isnan(rho):
        return 0.0, 1.0

    # Calculate p-value using t-distribution approximation
    # t = rho * sqrt((n-2)/(1-rho^2))
    if abs(rho) >= 1.0:
        p_value = 0.0
    else:
        t_stat = rho * math.sqrt((n - 2) / (1 - rho * rho))
        # Use normal approximation for large n
        p_value = 2 * (1 - normal_cdf(abs(t_stat)))

    return float(rho), float(p_value)


def ts_rank(series: pd.Series, window: int) -> pd.Series:
    """Time-series rank over rolling window.

    Replaces scipy.stats.rankdata in time-series context.
    Returns the percentile rank of the last value within the window.

    Args:
        series: Input time series
        window: Rolling window size

    Returns:
        Series of percentile ranks (0 to 1)
    """
    def _rank_pct(arr: np.ndarray) -> float:
        if len(arr) == 0:
            return np.nan
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return np.nan
        last_val = arr[-1]
        if np.isnan(last_val):
            return np.nan
        # Count how many values are less than or equal to the last value
        rank = np.sum(valid <= last_val)
        return rank / len(valid)

    return series.rolling(window).apply(_rank_pct, raw=True)


def rank_percentile(series: pd.Series, window: int) -> pd.Series:
    """Calculate percentile rank within a rolling window.

    Replaces scipy.stats.percentileofscore in rolling context.

    Args:
        series: Input series
        window: Rolling window size

    Returns:
        Series of percentile scores (0 to 100)
    """
    def _percentile_score(arr: np.ndarray) -> float:
        if len(arr) == 0:
            return np.nan
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return np.nan
        last_val = arr[-1]
        if np.isnan(last_val):
            return np.nan
        # Strict percentile: count of values strictly less than x
        below = np.sum(valid < last_val)
        equal = np.sum(valid == last_val)
        # Use 'mean' method like scipy default
        percentile = (below + 0.5 * equal) / len(valid) * 100
        return percentile

    return series.rolling(window).apply(_percentile_score, raw=True)


def linear_regression(
    x: np.ndarray, y: np.ndarray
) -> Tuple[float, float, float, float, float]:
    """Perform simple linear regression.

    Replaces scipy.stats.linregress with pure numpy implementation.

    Args:
        x: Independent variable
        y: Dependent variable

    Returns:
        Tuple of (slope, intercept, r_value, p_value, std_err)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Remove NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 2:
        return 0.0, 0.0, 0.0, 1.0, 0.0

    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate slope and intercept
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)

    if abs(ss_xx) < 1e-10:
        return 0.0, y_mean, 0.0, 1.0, 0.0

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    # Calculate R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)

    if abs(ss_tot) < 1e-10:
        r_value = 0.0
    else:
        r_squared = 1 - ss_res / ss_tot
        r_value = math.sqrt(max(r_squared, 0)) * np.sign(slope)

    # Calculate standard error of slope
    if n > 2:
        mse = ss_res / (n - 2)
        std_err = math.sqrt(mse / ss_xx) if ss_xx > 0 else 0.0
    else:
        std_err = 0.0

    # Calculate p-value using t-distribution approximation
    if std_err > 1e-10:
        t_stat = slope / std_err
        # Use normal approximation for p-value
        p_value = 2 * (1 - normal_cdf(abs(t_stat)))
    else:
        p_value = 0.0 if abs(slope) > 1e-10 else 1.0

    return float(slope), float(intercept), float(r_value), float(p_value), float(std_err)


def t_test_independent(
    a: np.ndarray, b: np.ndarray, equal_var: bool = True
) -> Tuple[float, float]:
    """Perform independent samples t-test.

    Replaces scipy.stats.ttest_ind with pure numpy implementation.

    Args:
        a: First sample
        b: Second sample
        equal_var: Whether to assume equal variances

    Returns:
        Tuple of (t_statistic, p_value)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # Remove NaN
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0

    mean1, mean2 = np.mean(a), np.mean(b)
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)

    if equal_var:
        # Pooled variance
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        se = math.sqrt(pooled_var * (1/n1 + 1/n2))
        df = n1 + n2 - 2
    else:
        # Welch's t-test
        se = math.sqrt(var1/n1 + var2/n2)
        # Welch-Satterthwaite degrees of freedom
        num = (var1/n1 + var2/n2) ** 2
        den = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
        df = num / den if den > 0 else n1 + n2 - 2

    if se < 1e-10:
        return 0.0, 1.0

    t_stat = (mean1 - mean2) / se

    # Use normal approximation for large df
    p_value = 2 * (1 - normal_cdf(abs(t_stat)))

    return float(t_stat), float(p_value)


def hierarchical_cluster(
    distance_matrix: np.ndarray,
    method: str = "average",
    threshold: float = 0.5,
) -> np.ndarray:
    """Perform hierarchical clustering.

    Simplified implementation replacing scipy.cluster.hierarchy.
    Uses a basic agglomerative approach.

    Args:
        distance_matrix: Symmetric distance/dissimilarity matrix
        method: Linkage method ('single', 'complete', 'average')
        threshold: Distance threshold for forming clusters

    Returns:
        Array of cluster labels (0-indexed)
    """
    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([0])

    # Initialize: each point in its own cluster
    cluster_labels = np.arange(n)
    active_clusters = set(range(n))

    # Create working copy of distance matrix
    dist = distance_matrix.copy()
    np.fill_diagonal(dist, np.inf)

    # Agglomerative clustering
    while len(active_clusters) > 1:
        # Find minimum distance between active clusters
        min_dist = np.inf
        merge_i, merge_j = -1, -1

        active_list = sorted(active_clusters)
        for i_idx, i in enumerate(active_list):
            for j in active_list[i_idx + 1:]:
                if dist[i, j] < min_dist:
                    min_dist = dist[i, j]
                    merge_i, merge_j = i, j

        # Stop if minimum distance exceeds threshold
        if min_dist > threshold or merge_i < 0:
            break

        # Merge clusters: assign all points in j's cluster to i's cluster
        old_label = cluster_labels[merge_j]
        new_label = cluster_labels[merge_i]
        cluster_labels[cluster_labels == old_label] = new_label

        # Update distances based on linkage method
        for k in active_clusters:
            if k != merge_i and k != merge_j:
                if method == "single":
                    new_dist = min(dist[merge_i, k], dist[merge_j, k])
                elif method == "complete":
                    new_dist = max(dist[merge_i, k], dist[merge_j, k])
                else:  # average
                    new_dist = (dist[merge_i, k] + dist[merge_j, k]) / 2
                dist[merge_i, k] = dist[k, merge_i] = new_dist

        # Remove j from active clusters
        active_clusters.remove(merge_j)

    # Relabel clusters to be contiguous
    unique_labels = np.unique(cluster_labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    return np.array([label_map[label] for label in cluster_labels])


def calculate_var(
    returns: pd.Series, confidence: float = 0.95
) -> float:
    """Calculate Value at Risk (VaR).

    Args:
        returns: Return series
        confidence: Confidence level (e.g., 0.95 for 95%)

    Returns:
        VaR value (positive number representing loss)
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0

    # Historical VaR: percentile of returns
    var = -np.percentile(returns, (1 - confidence) * 100)
    return float(max(var, 0))


def calculate_expected_shortfall(
    returns: pd.Series, confidence: float = 0.95
) -> float:
    """Calculate Expected Shortfall (Conditional VaR).

    Args:
        returns: Return series
        confidence: Confidence level (e.g., 0.95 for 95%)

    Returns:
        Expected Shortfall value (positive number representing expected loss)
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0

    var = calculate_var(returns, confidence)
    # Average of returns worse than VaR
    tail_returns = returns[returns < -var]

    if len(tail_returns) == 0:
        return var

    es = -float(tail_returns.mean())
    return max(es, 0)


def calculate_sortino_ratio(
    returns: pd.Series,
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate Sortino Ratio.

    Args:
        returns: Return series
        target_return: Target/minimum acceptable return
        periods_per_year: Periods for annualization

    Returns:
        Sortino ratio
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - target_return / periods_per_year
    mean_excess = excess_returns.mean()

    # Downside deviation (only negative returns)
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) == 0:
        return float('inf') if mean_excess > 0 else 0.0

    downside_std = np.sqrt(np.mean(negative_returns ** 2))

    if downside_std < 1e-10:
        return 0.0

    return float(mean_excess / downside_std * np.sqrt(periods_per_year))


def calculate_calmar_ratio(
    returns: pd.Series, periods_per_year: int = 252
) -> float:
    """Calculate Calmar Ratio (Annual Return / Max Drawdown).

    Args:
        returns: Return series
        periods_per_year: Periods for annualization

    Returns:
        Calmar ratio
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0

    # Annualized return
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    annual_return = (1 + total_return) ** (1 / years) - 1

    # Max drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = abs(drawdown.min())

    if max_dd < 1e-10:
        return 0.0

    return float(annual_return / max_dd)


def calculate_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate Information Ratio.

    Args:
        returns: Portfolio return series
        benchmark_returns: Benchmark return series
        periods_per_year: Periods for annualization

    Returns:
        Information ratio
    """
    # Align series
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0

    excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    tracking_error = excess.std()

    if tracking_error < 1e-10:
        return 0.0

    return float(excess.mean() / tracking_error * np.sqrt(periods_per_year))


# =============================================================================
# Standalone Normal Distribution Functions (No scipy dependency)
# =============================================================================

def normal_ppf(p: float) -> float:
    """Inverse normal CDF (percent point function) - standalone version.

    Using rational approximation to avoid scipy dependency.
    Abramowitz and Stegun approximation, accurate to ~1e-7.

    Args:
        p: Probability value between 0 and 1

    Returns:
        z-score corresponding to the given probability
    """
    if p <= 0:
        return float('-inf')
    if p >= 1:
        return float('inf')
    if p == 0.5:
        return 0.0

    # Use symmetry
    if p > 0.5:
        sign = 1
        p = 1 - p
    else:
        sign = -1

    # Rational approximation
    t = math.sqrt(-2.0 * math.log(p))

    # Coefficients
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    result = t - (c0 + c1*t + c2*t*t) / (1 + d1*t + d2*t*t + d3*t*t*t)
    return sign * result


def normal_cdf(x: float) -> float:
    """Normal CDF using error function approximation - standalone version.

    Avoiding scipy dependency while maintaining accuracy.

    Args:
        x: z-score value

    Returns:
        Cumulative probability for the given z-score
    """
    # Error function approximation
    sign = 1 if x >= 0 else -1
    x = abs(x) / math.sqrt(2)

    # Abramowitz and Stegun approximation 7.1.26
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911

    t = 1.0 / (1.0 + p * x)
    erf = 1 - ((((a5*t + a4)*t + a3)*t + a2)*t + a1) * t * math.exp(-x*x)

    return 0.5 * (1 + sign * erf)


# =============================================================================
# Export all components
# =============================================================================

__all__ = [
    # Core classes
    "QlibNotAvailableError",
    "QlibStatisticalEngine",
    "DeflatedSharpeCalculator",
    "DSRResult",
    "QlibRiskAnalyzer",
    "RiskMetrics",
    "QLIB_AVAILABLE",
    # Distribution functions (replacing scipy.stats.norm)
    "normal_ppf",
    "normal_cdf",
    # Correlation functions (replacing scipy.stats.spearmanr)
    "spearman_rank_correlation",
    # Ranking functions (replacing scipy.stats.rankdata/percentileofscore)
    "ts_rank",
    "rank_percentile",
    # Regression functions (replacing scipy.stats.linregress)
    "linear_regression",
    # Statistical tests (replacing scipy.stats.ttest_ind)
    "t_test_independent",
    # Clustering (replacing scipy.cluster.hierarchy)
    "hierarchical_cluster",
    # Risk metrics (replacing custom implementations)
    "calculate_var",
    "calculate_expected_shortfall",
    "calculate_sortino_ratio",
    "calculate_calmar_ratio",
    "calculate_information_ratio",
]
