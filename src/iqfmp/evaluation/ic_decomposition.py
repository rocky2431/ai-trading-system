"""IC Decomposition Analyzer for Factor Diagnostics.

This module provides comprehensive IC analysis by decomposing factor performance
across multiple dimensions:
- Time-based: Monthly, quarterly, yearly IC trends
- Market cap: Large/Mid/Small cap performance
- Volatility regime: High/Low volatility environment performance
- IC decay: Half-life estimation and decay detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import numpy as np
import pandas as pd

# Use Qlib-native statistical functions instead of scipy
from iqfmp.evaluation.qlib_stats import (
    linear_regression,
    t_test_independent,
    spearman_rank_correlation,
)

from iqfmp.evaluation.factor_evaluator import MetricsCalculator


class InsufficientDataError(Exception):
    """Raised when there's not enough data for analysis."""

    pass


@dataclass
class ICDecompositionConfig:
    """Configuration for IC decomposition analysis."""

    # Column names
    date_column: str = "date"
    symbol_column: str = "symbol"
    factor_column: str = "factor_value"
    return_column: str = "forward_return"
    market_cap_column: str = "market_cap"
    volatility_column: str = "volatility"

    # Market cap thresholds (percentiles)
    large_cap_threshold: float = 0.7  # Top 30%
    mid_cap_threshold: float = 0.3  # Middle 40%
    # Below mid_cap_threshold = Small cap (Bottom 30%)

    # Volatility thresholds (percentiles)
    high_vol_threshold: float = 0.7  # Top 30%
    # Below high_vol_threshold = Low volatility

    # Minimum samples for analysis
    min_samples_per_period: int = 10
    min_periods_for_trend: int = 3

    # IC decay detection
    detect_decay: bool = True
    decay_significance_level: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "date_column": self.date_column,
            "symbol_column": self.symbol_column,
            "factor_column": self.factor_column,
            "return_column": self.return_column,
            "market_cap_column": self.market_cap_column,
            "volatility_column": self.volatility_column,
            "large_cap_threshold": self.large_cap_threshold,
            "mid_cap_threshold": self.mid_cap_threshold,
            "high_vol_threshold": self.high_vol_threshold,
            "min_samples_per_period": self.min_samples_per_period,
            "min_periods_for_trend": self.min_periods_for_trend,
            "detect_decay": self.detect_decay,
            "decay_significance_level": self.decay_significance_level,
        }


@dataclass
class ICDecompositionResult:
    """Result of IC decomposition analysis."""

    # Overall IC
    total_ic: float = 0.0

    # Time decomposition
    ic_by_month: dict[str, float] = field(default_factory=dict)
    ic_by_quarter: dict[str, float] = field(default_factory=dict)
    ic_by_year: dict[str, float] = field(default_factory=dict)

    # Time trend analysis
    ic_trend_slope: float = 0.0  # Positive = improving, negative = decaying
    ic_trend_pvalue: float = 1.0
    has_significant_trend: bool = False

    # Market cap decomposition
    large_cap_ic: float = 0.0
    mid_cap_ic: float = 0.0
    small_cap_ic: float = 0.0

    # Volatility regime decomposition
    high_vol_ic: float = 0.0
    low_vol_ic: float = 0.0

    # IC consistency metrics
    ic_hit_rate: float = 0.0  # % of periods with positive IC
    ic_stability: float = 0.0  # 1 - CV(IC)

    # IC decay analysis
    regime_shift_detected: bool = False
    ic_decay_rate: float = 0.0
    predicted_half_life: int = 999

    # Diagnostics
    diagnosis: str = ""
    recommendations: list[str] = field(default_factory=list)

    # Metadata
    n_periods: int = 0
    analysis_date: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for API response."""
        return {
            "total_ic": round(self.total_ic, 4),
            "ic_by_month": {k: round(v, 4) for k, v in self.ic_by_month.items()},
            "ic_by_quarter": {k: round(v, 4) for k, v in self.ic_by_quarter.items()},
            "ic_by_year": {k: round(v, 4) for k, v in self.ic_by_year.items()},
            "ic_trend_slope": round(self.ic_trend_slope, 6),
            "ic_trend_pvalue": round(self.ic_trend_pvalue, 4),
            "has_significant_trend": self.has_significant_trend,
            "large_cap_ic": round(self.large_cap_ic, 4),
            "mid_cap_ic": round(self.mid_cap_ic, 4),
            "small_cap_ic": round(self.small_cap_ic, 4),
            "high_vol_ic": round(self.high_vol_ic, 4),
            "low_vol_ic": round(self.low_vol_ic, 4),
            "ic_hit_rate": round(self.ic_hit_rate, 4),
            "ic_stability": round(self.ic_stability, 4),
            "regime_shift_detected": self.regime_shift_detected,
            "ic_decay_rate": round(self.ic_decay_rate, 6),
            "predicted_half_life": self.predicted_half_life,
            "diagnosis": self.diagnosis,
            "recommendations": self.recommendations,
            "n_periods": self.n_periods,
            "analysis_date": self.analysis_date,
        }


class ICDecompositionAnalyzer:
    """Analyzer for decomposing IC across multiple dimensions.

    Provides insights into:
    - When the factor works best (time periods)
    - Where the factor works best (market segments)
    - How stable the factor is over time
    - Whether IC is decaying
    """

    def __init__(self, config: Optional[ICDecompositionConfig] = None) -> None:
        """Initialize analyzer.

        Args:
            config: Analysis configuration
        """
        self.config = config or ICDecompositionConfig()
        self.calculator = MetricsCalculator()

    def analyze(
        self,
        data: pd.DataFrame,
        factor_column: Optional[str] = None,
        return_column: Optional[str] = None,
    ) -> ICDecompositionResult:
        """Run full IC decomposition analysis.

        Args:
            data: DataFrame with factor values, returns, and metadata
            factor_column: Override factor column name
            return_column: Override return column name

        Returns:
            ICDecompositionResult with all analysis
        """
        factor_col = factor_column or self.config.factor_column
        return_col = return_column or self.config.return_column
        date_col = self.config.date_column

        # Validate and prepare data
        df = self._validate_and_prepare(data, factor_col, return_col, date_col)

        # Calculate total IC
        total_ic = self.calculator.calculate_rank_ic(
            df[factor_col], df[return_col]
        )

        # Time decomposition
        ic_by_month, ic_by_quarter, ic_by_year = self._decompose_by_time(
            df, factor_col, return_col
        )

        # Trend analysis
        trend_slope, trend_pvalue = self._analyze_trend(ic_by_month)
        has_significant_trend = trend_pvalue < self.config.decay_significance_level

        # Market cap decomposition
        large_cap_ic, mid_cap_ic, small_cap_ic = self._decompose_by_market_cap(
            df, factor_col, return_col
        )

        # Volatility regime decomposition
        high_vol_ic, low_vol_ic = self._decompose_by_volatility(
            df, factor_col, return_col
        )

        # IC consistency metrics
        monthly_ics = list(ic_by_month.values())
        ic_hit_rate = self._calculate_hit_rate(monthly_ics)
        ic_stability = self._calculate_stability(monthly_ics)

        # IC decay analysis
        decay_rate, half_life, regime_shift = self._analyze_decay(monthly_ics)

        # Generate diagnosis and recommendations
        diagnosis, recommendations = self._generate_diagnosis(
            total_ic=total_ic,
            trend_slope=trend_slope,
            has_trend=has_significant_trend,
            large_cap_ic=large_cap_ic,
            mid_cap_ic=mid_cap_ic,
            small_cap_ic=small_cap_ic,
            high_vol_ic=high_vol_ic,
            low_vol_ic=low_vol_ic,
            ic_hit_rate=ic_hit_rate,
            ic_stability=ic_stability,
            half_life=half_life,
        )

        return ICDecompositionResult(
            total_ic=total_ic,
            ic_by_month=ic_by_month,
            ic_by_quarter=ic_by_quarter,
            ic_by_year=ic_by_year,
            ic_trend_slope=trend_slope,
            ic_trend_pvalue=trend_pvalue,
            has_significant_trend=has_significant_trend,
            large_cap_ic=large_cap_ic,
            mid_cap_ic=mid_cap_ic,
            small_cap_ic=small_cap_ic,
            high_vol_ic=high_vol_ic,
            low_vol_ic=low_vol_ic,
            ic_hit_rate=ic_hit_rate,
            ic_stability=ic_stability,
            regime_shift_detected=regime_shift,
            ic_decay_rate=decay_rate,
            predicted_half_life=half_life,
            diagnosis=diagnosis,
            recommendations=recommendations,
            n_periods=len(ic_by_month),
        )

    def _validate_and_prepare(
        self,
        data: pd.DataFrame,
        factor_col: str,
        return_col: str,
        date_col: str,
    ) -> pd.DataFrame:
        """Validate and prepare data for analysis."""
        if data.empty:
            raise InsufficientDataError("Input data is empty")

        required_cols = [factor_col, return_col]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        df = data.copy()

        # Ensure date column
        if date_col in df.columns:
            df["_date"] = pd.to_datetime(df[date_col])
        elif isinstance(df.index, pd.DatetimeIndex):
            df["_date"] = df.index
        else:
            raise ValueError(f"Missing date column '{date_col}'")

        # Add time period columns
        df["_month"] = df["_date"].dt.to_period("M")
        df["_quarter"] = df["_date"].dt.to_period("Q")
        df["_year"] = df["_date"].dt.to_period("Y")

        return df

    def _decompose_by_time(
        self,
        df: pd.DataFrame,
        factor_col: str,
        return_col: str,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Decompose IC by time periods."""
        ic_by_month = {}
        ic_by_quarter = {}
        ic_by_year = {}

        # Monthly IC
        for period, group in df.groupby("_month"):
            if len(group) >= self.config.min_samples_per_period:
                ic = self.calculator.calculate_rank_ic(
                    group[factor_col], group[return_col]
                )
                if not np.isnan(ic):
                    ic_by_month[str(period)] = ic

        # Quarterly IC
        for period, group in df.groupby("_quarter"):
            if len(group) >= self.config.min_samples_per_period:
                ic = self.calculator.calculate_rank_ic(
                    group[factor_col], group[return_col]
                )
                if not np.isnan(ic):
                    ic_by_quarter[str(period)] = ic

        # Yearly IC
        for period, group in df.groupby("_year"):
            if len(group) >= self.config.min_samples_per_period:
                ic = self.calculator.calculate_rank_ic(
                    group[factor_col], group[return_col]
                )
                if not np.isnan(ic):
                    ic_by_year[str(period)] = ic

        return ic_by_month, ic_by_quarter, ic_by_year

    def _analyze_trend(
        self, ic_by_month: dict[str, float]
    ) -> tuple[float, float]:
        """Analyze IC trend over time.

        Returns:
            (slope, p-value) from linear regression
        """
        if len(ic_by_month) < self.config.min_periods_for_trend:
            return 0.0, 1.0

        # Sort by month
        sorted_periods = sorted(ic_by_month.keys())
        ics = [ic_by_month[p] for p in sorted_periods]

        # Linear regression using Qlib-native function
        x = np.arange(len(ics))
        y = np.array(ics)

        try:
            slope, intercept, r_value, p_value, std_err = linear_regression(x, y)
            return float(slope), float(p_value)
        except Exception:
            return 0.0, 1.0

    def _decompose_by_market_cap(
        self,
        df: pd.DataFrame,
        factor_col: str,
        return_col: str,
    ) -> tuple[float, float, float]:
        """Decompose IC by market capitalization.

        Returns:
            (large_cap_ic, mid_cap_ic, small_cap_ic)
        """
        market_cap_col = self.config.market_cap_column

        # If no market cap column, return zeros
        if market_cap_col not in df.columns:
            return 0.0, 0.0, 0.0

        # Calculate percentile ranks for market cap
        df["_mcap_pct"] = df[market_cap_col].rank(pct=True)

        # Split by market cap
        large_mask = df["_mcap_pct"] >= self.config.large_cap_threshold
        small_mask = df["_mcap_pct"] < self.config.mid_cap_threshold
        mid_mask = ~large_mask & ~small_mask

        large_cap_ic = 0.0
        mid_cap_ic = 0.0
        small_cap_ic = 0.0

        # Calculate IC for each segment
        if large_mask.sum() >= self.config.min_samples_per_period:
            large_cap_ic = self.calculator.calculate_rank_ic(
                df.loc[large_mask, factor_col],
                df.loc[large_mask, return_col],
            )

        if mid_mask.sum() >= self.config.min_samples_per_period:
            mid_cap_ic = self.calculator.calculate_rank_ic(
                df.loc[mid_mask, factor_col],
                df.loc[mid_mask, return_col],
            )

        if small_mask.sum() >= self.config.min_samples_per_period:
            small_cap_ic = self.calculator.calculate_rank_ic(
                df.loc[small_mask, factor_col],
                df.loc[small_mask, return_col],
            )

        return (
            large_cap_ic if not np.isnan(large_cap_ic) else 0.0,
            mid_cap_ic if not np.isnan(mid_cap_ic) else 0.0,
            small_cap_ic if not np.isnan(small_cap_ic) else 0.0,
        )

    def _decompose_by_volatility(
        self,
        df: pd.DataFrame,
        factor_col: str,
        return_col: str,
    ) -> tuple[float, float]:
        """Decompose IC by volatility regime.

        Returns:
            (high_vol_ic, low_vol_ic)
        """
        vol_col = self.config.volatility_column

        # If no volatility column, try to calculate from returns
        if vol_col not in df.columns:
            if return_col in df.columns:
                # Use rolling volatility of returns
                df["_vol"] = df[return_col].rolling(20, min_periods=5).std()
            else:
                return 0.0, 0.0
        else:
            df["_vol"] = df[vol_col]

        # Remove NaN volatility
        df_valid = df.dropna(subset=["_vol"])
        if len(df_valid) < self.config.min_samples_per_period * 2:
            return 0.0, 0.0

        # Calculate percentile ranks for volatility
        df_valid["_vol_pct"] = df_valid["_vol"].rank(pct=True)

        # Split by volatility
        high_vol_mask = df_valid["_vol_pct"] >= self.config.high_vol_threshold
        low_vol_mask = ~high_vol_mask

        high_vol_ic = 0.0
        low_vol_ic = 0.0

        if high_vol_mask.sum() >= self.config.min_samples_per_period:
            high_vol_ic = self.calculator.calculate_rank_ic(
                df_valid.loc[high_vol_mask, factor_col],
                df_valid.loc[high_vol_mask, return_col],
            )

        if low_vol_mask.sum() >= self.config.min_samples_per_period:
            low_vol_ic = self.calculator.calculate_rank_ic(
                df_valid.loc[low_vol_mask, factor_col],
                df_valid.loc[low_vol_mask, return_col],
            )

        return (
            high_vol_ic if not np.isnan(high_vol_ic) else 0.0,
            low_vol_ic if not np.isnan(low_vol_ic) else 0.0,
        )

    def _calculate_hit_rate(self, ic_series: list[float]) -> float:
        """Calculate IC hit rate (% of positive IC periods)."""
        if not ic_series:
            return 0.0
        positive_count = sum(1 for ic in ic_series if ic > 0)
        return positive_count / len(ic_series)

    def _calculate_stability(self, ic_series: list[float]) -> float:
        """Calculate IC stability score.

        Stability = 1 - CV(IC), where CV = std/mean
        Higher is better (more stable).
        """
        if len(ic_series) < 2:
            return 0.0

        mean_ic = np.mean(ic_series)
        std_ic = np.std(ic_series)

        if abs(mean_ic) < 1e-6:
            return 0.0

        cv = abs(std_ic / mean_ic)
        # Cap CV at 2 for reasonable stability score
        stability = max(0, 1 - min(cv, 2) / 2)
        return stability

    def _analyze_decay(
        self, monthly_ics: list[float]
    ) -> tuple[float, int, bool]:
        """Analyze IC decay pattern.

        Returns:
            (decay_rate, half_life, regime_shift_detected)
        """
        if len(monthly_ics) < 6:
            return 0.0, 999, False

        # Use exponential decay model for absolute IC
        abs_ics = [abs(ic) for ic in monthly_ics]

        # Filter out zeros
        valid_ics = [(i, ic) for i, ic in enumerate(abs_ics) if ic > 1e-6]
        if len(valid_ics) < 3:
            return 0.0, 999, False

        indices, ics = zip(*valid_ics)
        t = np.array(indices)
        log_ics = np.log(np.array(ics))

        try:
            slope, intercept, r_value, p_value, std_err = linear_regression(t, log_ics)

            # Decay rate (negative slope means decay)
            decay_rate = -slope if slope < 0 else 0.0

            # Half-life
            if decay_rate > 1e-6:
                half_life = int(np.log(2) / decay_rate)
                half_life = min(half_life, 999)
            else:
                half_life = 999

            # Regime shift detection: check if recent IC is significantly different
            n = len(monthly_ics)
            if n >= 12:
                first_half = monthly_ics[: n // 2]
                second_half = monthly_ics[n // 2:]
                t_stat, p_val = t_test_independent(first_half, second_half)
                regime_shift = p_val < 0.05
            else:
                regime_shift = False

            return decay_rate, half_life, regime_shift

        except Exception:
            return 0.0, 999, False

    def _generate_diagnosis(
        self,
        total_ic: float,
        trend_slope: float,
        has_trend: bool,
        large_cap_ic: float,
        mid_cap_ic: float,
        small_cap_ic: float,
        high_vol_ic: float,
        low_vol_ic: float,
        ic_hit_rate: float,
        ic_stability: float,
        half_life: int,
    ) -> tuple[str, list[str]]:
        """Generate diagnosis and recommendations."""
        issues = []
        recommendations = []

        # IC strength diagnosis
        if abs(total_ic) < 0.02:
            issues.append("Low overall IC (<2%)")
            recommendations.append("Consider combining with other factors or improving signal")
        elif abs(total_ic) >= 0.05:
            issues.append("Strong IC (>=5%)")

        # Trend diagnosis
        if has_trend and trend_slope < 0:
            issues.append("Significant IC decay detected")
            recommendations.append("Factor may be losing predictive power over time")
        elif has_trend and trend_slope > 0:
            issues.append("IC improving over time")

        # Market cap diagnosis
        cap_ics = [
            ("large_cap", large_cap_ic),
            ("mid_cap", mid_cap_ic),
            ("small_cap", small_cap_ic),
        ]
        best_cap = max(cap_ics, key=lambda x: abs(x[1]))
        if abs(best_cap[1]) > abs(total_ic) * 1.5:
            issues.append(f"Factor works best in {best_cap[0]} stocks")
            recommendations.append(f"Consider focusing on {best_cap[0]} universe")

        # Volatility regime diagnosis
        if abs(high_vol_ic) > abs(low_vol_ic) * 2:
            issues.append("Factor performs better in high volatility regimes")
            recommendations.append("May underperform in calm markets")
        elif abs(low_vol_ic) > abs(high_vol_ic) * 2:
            issues.append("Factor performs better in low volatility regimes")
            recommendations.append("May underperform in volatile markets")

        # Stability diagnosis
        if ic_hit_rate < 0.5:
            issues.append(f"Low IC hit rate ({ic_hit_rate:.0%})")
            recommendations.append("Factor has inconsistent sign across periods")

        if ic_stability < 0.3:
            issues.append("Low IC stability")
            recommendations.append("Factor variance is high relative to mean")

        # Half-life diagnosis
        if half_life < 30:
            issues.append(f"Short IC half-life ({half_life} periods)")
            recommendations.append("Factor alpha decays rapidly")
        elif half_life < 60:
            issues.append(f"Moderate IC half-life ({half_life} periods)")

        # Generate summary diagnosis
        if not issues:
            diagnosis = "Factor shows stable performance across all dimensions"
        else:
            diagnosis = "; ".join(issues)

        return diagnosis, recommendations


class QuickICAnalyzer:
    """Lightweight IC analyzer for quick diagnostics.

    Use when full decomposition is not needed.
    """

    def __init__(self) -> None:
        """Initialize analyzer."""
        self.calculator = MetricsCalculator()

    def quick_analyze(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        dates: Optional[pd.Series] = None,
    ) -> dict[str, Any]:
        """Run quick IC analysis.

        Args:
            factor_values: Factor value series
            returns: Return series
            dates: Optional date series for time decomposition

        Returns:
            Dictionary with key metrics
        """
        # Overall IC
        total_ic = self.calculator.calculate_rank_ic(factor_values, returns)
        pearson_ic = self.calculator.calculate_ic(factor_values, returns)

        result = {
            "rank_ic": total_ic,
            "pearson_ic": pearson_ic,
            "n_samples": len(factor_values),
        }

        # If dates provided, calculate monthly ICs
        if dates is not None:
            df = pd.DataFrame({
                "factor": factor_values.values,
                "return": returns.values,
                "date": pd.to_datetime(dates.values),
            })
            df["month"] = df["date"].dt.to_period("M")

            monthly_ics = []
            for period, group in df.groupby("month"):
                if len(group) >= 5:
                    ic = self.calculator.calculate_rank_ic(
                        group["factor"], group["return"]
                    )
                    if not np.isnan(ic):
                        monthly_ics.append(ic)

            if monthly_ics:
                result["monthly_ic_mean"] = np.mean(monthly_ics)
                result["monthly_ic_std"] = np.std(monthly_ics)
                result["ic_ir"] = (
                    result["monthly_ic_mean"] / result["monthly_ic_std"]
                    if result["monthly_ic_std"] > 0 else 0
                )
                result["ic_hit_rate"] = sum(1 for ic in monthly_ics if ic > 0) / len(monthly_ics)

        return result
