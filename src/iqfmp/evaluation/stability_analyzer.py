"""Stability Analyzer for factor robustness evaluation.

This module provides multi-dimensional stability analysis:
- TimeStabilityAnalyzer: IC over time (monthly/quarterly)
- MarketStabilityAnalyzer: IC by market cap groups
- RegimeStabilityAnalyzer: IC by market regime
- StabilityAnalyzer: Combined analysis with overall score
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import numpy as np
import pandas as pd
from scipy import stats


class InvalidDataError(Exception):
    """Raised when input data is invalid."""

    pass


class InsufficientDataError(Exception):
    """Raised when data is insufficient for analysis."""

    pass


class MarketRegime(Enum):
    """Market regime classifications."""

    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


@dataclass
class StabilityConfig:
    """Configuration for stability analysis."""

    # Column names
    date_column: str = "date"
    symbol_column: str = "symbol"
    factor_column: str = "factor_value"
    return_column: str = "forward_return"
    market_cap_column: str = "market_cap"

    # Time analysis
    time_frequency: str = "monthly"  # monthly or quarterly
    min_periods: int = 20

    # Market cap thresholds (in USD)
    large_cap_threshold: float = 1e11  # 100B
    small_cap_threshold: float = 1e10  # 10B

    # Regime detection
    regime_method: str = "volatility"  # volatility or trend
    volatility_lookback: int = 30
    volatility_threshold: float = 0.5  # percentile for high/low

    # Scoring weights
    time_weight: float = 0.4
    market_weight: float = 0.3
    regime_weight: float = 0.3

    def __post_init__(self) -> None:
        """Validate configuration."""
        total_weight = self.time_weight + self.market_weight + self.regime_weight
        if not np.isclose(total_weight, 1.0):
            raise ValueError(
                f"Stability weights must sum to 1.0, got {total_weight}"
            )


@dataclass
class DecayResult:
    """Result of IC decay analysis."""

    has_decay: bool
    decay_rate: float
    p_value: float
    trend_strength: float


@dataclass
class TimeStabilityResult:
    """Result of time stability analysis."""

    monthly_ic: pd.Series
    quarterly_ic: pd.Series
    ic_mean: float
    ic_std: float
    ir: float  # Information ratio
    rolling_ic_std: float
    decay_result: Optional[DecayResult]
    stability_score: float


@dataclass
class MarketStabilityResult:
    """Result of market stability analysis."""

    group_ic: dict[str, float]
    group_counts: dict[str, int]
    consistency_score: float
    max_ic_difference: float
    stability_score: float


@dataclass
class RegimeStabilityResult:
    """Result of regime stability analysis."""

    regime_ic: dict[MarketRegime, float]
    regime_periods: dict[MarketRegime, int]
    sensitivity_score: float
    stability_score: float


@dataclass
class StabilityScore:
    """Overall stability score."""

    value: float
    time_component: float
    market_component: float
    regime_component: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "time_component": self.time_component,
            "market_component": self.market_component,
            "regime_component": self.regime_component,
        }


@dataclass
class StabilityReport:
    """Complete stability analysis report."""

    time_stability: TimeStabilityResult
    market_stability: MarketStabilityResult
    regime_stability: RegimeStabilityResult
    overall_score: StabilityScore
    grade: str
    recommendations: list[str]
    created_at: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "time_stability": {
                "ic_mean": self.time_stability.ic_mean,
                "ic_std": self.time_stability.ic_std,
                "ir": self.time_stability.ir,
                "stability_score": self.time_stability.stability_score,
            },
            "market_stability": {
                "group_ic": self.market_stability.group_ic,
                "consistency_score": self.market_stability.consistency_score,
                "stability_score": self.market_stability.stability_score,
            },
            "regime_stability": {
                "regime_ic": {k.value: v for k, v in self.regime_stability.regime_ic.items()},
                "sensitivity_score": self.regime_stability.sensitivity_score,
                "stability_score": self.regime_stability.stability_score,
            },
            "overall_score": self.overall_score.to_dict(),
            "grade": self.grade,
            "recommendations": self.recommendations,
        }

    def get_summary(self) -> str:
        """Generate text summary of the report."""
        lines = [
            f"Stability Analysis Report",
            f"=" * 40,
            f"Overall Grade: {self.grade}",
            f"Overall Score: {self.overall_score.value:.2f}",
            f"",
            f"Time Stability: {self.time_stability.stability_score:.2f}",
            f"  - IC Mean: {self.time_stability.ic_mean:.4f}",
            f"  - IC IR: {self.time_stability.ir:.2f}",
            f"",
            f"Market Stability: {self.market_stability.stability_score:.2f}",
            f"  - Consistency: {self.market_stability.consistency_score:.2f}",
            f"",
            f"Regime Stability: {self.regime_stability.stability_score:.2f}",
            f"  - Sensitivity: {self.regime_stability.sensitivity_score:.2f}",
            f"",
            f"Recommendations:",
        ]
        for rec in self.recommendations:
            lines.append(f"  - {rec}")

        return "\n".join(lines)

    def get_visualization_data(self) -> dict[str, Any]:
        """Get data for visualization."""
        return {
            "time_series": {
                "monthly_ic": self.time_stability.monthly_ic.to_dict(),
                "quarterly_ic": self.time_stability.quarterly_ic.to_dict(),
            },
            "market_groups": self.market_stability.group_ic,
            "regime_breakdown": {
                k.value: v for k, v in self.regime_stability.regime_ic.items()
            },
            "scores": {
                "time": self.time_stability.stability_score,
                "market": self.market_stability.stability_score,
                "regime": self.regime_stability.stability_score,
                "overall": self.overall_score.value,
            },
        }


class TimeStabilityAnalyzer:
    """Analyzer for time-based stability."""

    def __init__(self, config: Optional[StabilityConfig] = None) -> None:
        """Initialize with configuration."""
        self.config = config or StabilityConfig()

    def analyze(self, data: pd.DataFrame) -> TimeStabilityResult:
        """Analyze time stability of factor.

        Args:
            data: DataFrame with factor values and returns

        Returns:
            TimeStabilityResult with IC statistics
        """
        df = self._prepare_data(data)

        # Calculate monthly IC
        monthly_ic = self._calculate_periodic_ic(df, "M")

        # Calculate quarterly IC
        quarterly_ic = self._calculate_periodic_ic(df, "Q")

        # Calculate statistics
        ic_mean = float(monthly_ic.mean()) if len(monthly_ic) > 0 else 0.0
        ic_std = float(monthly_ic.std()) if len(monthly_ic) > 1 else 0.0
        ir = ic_mean / ic_std if ic_std > 0 else 0.0

        # Rolling IC std
        rolling_ic_std = self._calculate_rolling_ic_std(monthly_ic)

        # Decay detection
        decay_result = self.detect_decay(monthly_ic) if len(monthly_ic) >= 3 else None

        # Calculate stability score
        stability_score = self._calculate_stability_score(
            ic_mean, ic_std, ir, decay_result
        )

        return TimeStabilityResult(
            monthly_ic=monthly_ic,
            quarterly_ic=quarterly_ic,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ir=ir,
            rolling_ic_std=rolling_ic_std,
            decay_result=decay_result,
            stability_score=stability_score,
        )

    def detect_decay(self, ic_series: pd.Series) -> DecayResult:
        """Detect IC decay trend.

        Args:
            ic_series: Time series of IC values

        Returns:
            DecayResult with trend analysis
        """
        if len(ic_series) < 3:
            return DecayResult(
                has_decay=False, decay_rate=0.0, p_value=1.0, trend_strength=0.0
            )

        # Linear regression on IC over time
        x = np.arange(len(ic_series))
        y = ic_series.values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        has_decay = bool(slope < 0 and p_value < 0.1)
        trend_strength = abs(r_value)

        return DecayResult(
            has_decay=has_decay,
            decay_rate=float(slope),
            p_value=float(p_value),
            trend_strength=float(trend_strength),
        )

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for analysis."""
        df = data.copy()

        # Handle date column or index
        if self.config.date_column in df.columns:
            df["_date"] = pd.to_datetime(df[self.config.date_column])
        elif isinstance(df.index, pd.DatetimeIndex):
            df["_date"] = df.index
        else:
            raise InvalidDataError("No date column or datetime index found")

        return df

    def _calculate_periodic_ic(self, df: pd.DataFrame, freq: str) -> pd.Series:
        """Calculate IC for each period."""
        df["_period"] = df["_date"].dt.to_period(freq)

        ic_values = []
        periods = []

        for period, group in df.groupby("_period"):
            if len(group) >= 5:
                factor_col = self.config.factor_column
                return_col = self.config.return_column

                ic = group[factor_col].corr(group[return_col])
                if not np.isnan(ic):
                    ic_values.append(ic)
                    periods.append(period.to_timestamp())

        return pd.Series(ic_values, index=pd.DatetimeIndex(periods), name="IC")

    def _calculate_rolling_ic_std(self, ic_series: pd.Series) -> float:
        """Calculate rolling IC standard deviation."""
        if len(ic_series) < 3:
            return 0.0

        rolling_std = ic_series.rolling(window=min(3, len(ic_series))).std()
        return float(rolling_std.mean()) if not rolling_std.isna().all() else 0.0

    def _calculate_stability_score(
        self,
        ic_mean: float,
        ic_std: float,
        ir: float,
        decay_result: Optional[DecayResult],
    ) -> float:
        """Calculate time stability score (0-1)."""
        # Base score from IR
        ir_score = min(1.0, abs(ir) / 2.0) if ir > 0 else 0.0

        # Penalty for decay
        decay_penalty = 0.0
        if decay_result and decay_result.has_decay:
            decay_penalty = decay_result.trend_strength * 0.3

        # Bonus for consistency (low std relative to mean)
        consistency_score = 0.0
        if ic_std > 0 and ic_mean != 0:
            cv = abs(ic_std / ic_mean)  # Coefficient of variation
            consistency_score = max(0, 1.0 - cv) * 0.3

        score = ir_score * 0.7 + consistency_score - decay_penalty
        return max(0.0, min(1.0, score))


class MarketStabilityAnalyzer:
    """Analyzer for market-based stability."""

    def __init__(self, config: Optional[StabilityConfig] = None) -> None:
        """Initialize with configuration."""
        self.config = config or StabilityConfig()

    def analyze(self, data: pd.DataFrame) -> MarketStabilityResult:
        """Analyze market stability of factor.

        Args:
            data: DataFrame with factor values, returns, and market cap

        Returns:
            MarketStabilityResult with group IC analysis
        """
        df = data.copy()

        # Classify by market cap
        df["_market_group"] = df[self.config.market_cap_column].apply(
            self._classify_market_cap
        )

        # Calculate IC per group
        group_ic: dict[str, float] = {}
        group_counts: dict[str, int] = {}

        for group in ["large", "mid", "small"]:
            group_data = df[df["_market_group"] == group]
            group_counts[group] = len(group_data)

            if len(group_data) >= 10:
                ic = group_data[self.config.factor_column].corr(
                    group_data[self.config.return_column]
                )
                group_ic[group] = float(ic) if not np.isnan(ic) else 0.0
            else:
                group_ic[group] = 0.0

        # Calculate consistency and max difference
        ic_values = [v for v in group_ic.values() if v != 0]
        consistency_score = self._calculate_consistency(ic_values)
        max_ic_difference = max(ic_values) - min(ic_values) if ic_values else 0.0

        # Calculate stability score
        stability_score = self._calculate_stability_score(
            group_ic, consistency_score, max_ic_difference
        )

        return MarketStabilityResult(
            group_ic=group_ic,
            group_counts=group_counts,
            consistency_score=consistency_score,
            max_ic_difference=max_ic_difference,
            stability_score=stability_score,
        )

    def _classify_market_cap(self, market_cap: float) -> str:
        """Classify market cap into groups."""
        if market_cap >= self.config.large_cap_threshold:
            return "large"
        elif market_cap >= self.config.small_cap_threshold:
            return "mid"
        else:
            return "small"

    def _calculate_consistency(self, ic_values: list[float]) -> float:
        """Calculate consistency score across groups."""
        if len(ic_values) < 2:
            return 1.0

        # Check if all ICs have same sign
        signs = [1 if v > 0 else -1 if v < 0 else 0 for v in ic_values]
        same_sign = len(set(signs)) == 1 or 0 in signs

        # Calculate standard deviation
        std = np.std(ic_values) if len(ic_values) > 1 else 0.0

        # Score based on sign consistency and spread
        sign_score = 1.0 if same_sign else 0.5
        spread_score = max(0, 1.0 - std * 5)

        return sign_score * 0.5 + spread_score * 0.5

    def _calculate_stability_score(
        self,
        group_ic: dict[str, float],
        consistency_score: float,
        max_difference: float,
    ) -> float:
        """Calculate market stability score (0-1)."""
        # Penalty for large IC differences
        diff_penalty = min(0.5, max_difference * 2)

        score = consistency_score - diff_penalty
        return max(0.0, min(1.0, score))


class RegimeStabilityAnalyzer:
    """Analyzer for regime-based stability."""

    def __init__(self, config: Optional[StabilityConfig] = None) -> None:
        """Initialize with configuration."""
        self.config = config or StabilityConfig()

    def analyze(self, data: pd.DataFrame) -> RegimeStabilityResult:
        """Analyze regime stability of factor.

        Args:
            data: DataFrame with factor values and returns

        Returns:
            RegimeStabilityResult with regime IC analysis
        """
        df = self._prepare_data(data)

        # Detect regimes
        df["_regime"] = self._detect_regimes(df)

        # Calculate IC per regime
        regime_ic: dict[MarketRegime, float] = {}
        regime_periods: dict[MarketRegime, int] = {}

        for regime in df["_regime"].unique():
            if pd.isna(regime):
                continue

            regime_data = df[df["_regime"] == regime]
            regime_periods[regime] = len(regime_data)

            if len(regime_data) >= 10:
                ic = regime_data[self.config.factor_column].corr(
                    regime_data[self.config.return_column]
                )
                regime_ic[regime] = float(ic) if not np.isnan(ic) else 0.0
            else:
                regime_ic[regime] = 0.0

        # Calculate sensitivity
        sensitivity_score = self._calculate_sensitivity(regime_ic)

        # Calculate stability score
        stability_score = self._calculate_stability_score(regime_ic, sensitivity_score)

        return RegimeStabilityResult(
            regime_ic=regime_ic,
            regime_periods=regime_periods,
            sensitivity_score=sensitivity_score,
            stability_score=stability_score,
        )

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for analysis."""
        df = data.copy()

        # Handle date column or index
        if self.config.date_column in df.columns:
            df["_date"] = pd.to_datetime(df[self.config.date_column])
        elif isinstance(df.index, pd.DatetimeIndex):
            df["_date"] = df.index
        else:
            # Create synthetic date
            df["_date"] = pd.date_range(start="2022-01-01", periods=len(df), freq="D")

        return df

    def _detect_regimes(self, df: pd.DataFrame) -> pd.Series:
        """Detect market regimes."""
        if self.config.regime_method == "volatility":
            return self._detect_volatility_regimes(df)
        else:
            return self._detect_trend_regimes(df)

    def _detect_volatility_regimes(self, df: pd.DataFrame) -> pd.Series:
        """Detect regimes based on return volatility."""
        returns = df[self.config.return_column]

        # Calculate rolling volatility
        rolling_vol = returns.rolling(
            window=min(self.config.volatility_lookback, len(returns) // 2),
            min_periods=5,
        ).std()

        # Classify by volatility percentile
        vol_median = rolling_vol.median()

        regimes = rolling_vol.apply(
            lambda x: MarketRegime.HIGH_VOLATILITY
            if x > vol_median
            else MarketRegime.LOW_VOLATILITY
        )

        return regimes

    def _detect_trend_regimes(self, df: pd.DataFrame) -> pd.Series:
        """Detect regimes based on return trend."""
        returns = df[self.config.return_column]

        # Calculate rolling return
        lookback = min(self.config.volatility_lookback, len(returns) // 2)
        rolling_ret = returns.rolling(window=lookback, min_periods=5).mean()

        # Classify by trend direction
        def classify_trend(x: float) -> MarketRegime:
            if pd.isna(x):
                return MarketRegime.SIDEWAYS
            if x > 0.001:
                return MarketRegime.BULL
            elif x < -0.001:
                return MarketRegime.BEAR
            else:
                return MarketRegime.SIDEWAYS

        return rolling_ret.apply(classify_trend)

    def _calculate_sensitivity(self, regime_ic: dict[MarketRegime, float]) -> float:
        """Calculate regime sensitivity (higher = more sensitive)."""
        if len(regime_ic) < 2:
            return 0.0

        ic_values = list(regime_ic.values())
        return float(np.std(ic_values)) if len(ic_values) > 1 else 0.0

    def _calculate_stability_score(
        self, regime_ic: dict[MarketRegime, float], sensitivity: float
    ) -> float:
        """Calculate regime stability score (0-1)."""
        # Lower sensitivity = more stable
        base_score = max(0, 1.0 - sensitivity * 5)

        # Check sign consistency
        ic_values = [v for v in regime_ic.values() if v != 0]
        if ic_values:
            signs = [v > 0 for v in ic_values]
            sign_consistency = len(set(signs)) == 1

            if not sign_consistency:
                base_score *= 0.7

        return max(0.0, min(1.0, base_score))


class StabilityAnalyzer:
    """Combined stability analyzer."""

    def __init__(self, config: Optional[StabilityConfig] = None) -> None:
        """Initialize with configuration."""
        self.config = config or StabilityConfig()
        self.time_analyzer = TimeStabilityAnalyzer(self.config)
        self.market_analyzer = MarketStabilityAnalyzer(self.config)
        self.regime_analyzer = RegimeStabilityAnalyzer(self.config)

    def analyze(self, data: pd.DataFrame) -> StabilityReport:
        """Perform complete stability analysis.

        Args:
            data: DataFrame with factor values, returns, and market cap

        Returns:
            StabilityReport with all analysis results
        """
        # Validate data
        self._validate_data(data)

        # Run individual analyses
        time_result = self.time_analyzer.analyze(data)
        market_result = self.market_analyzer.analyze(data)
        regime_result = self.regime_analyzer.analyze(data)

        # Calculate overall score
        overall_score = self._calculate_overall_score(
            time_result, market_result, regime_result
        )

        # Determine grade
        grade = self._determine_grade(overall_score.value)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            time_result, market_result, regime_result, overall_score
        )

        return StabilityReport(
            time_stability=time_result,
            market_stability=market_result,
            regime_stability=regime_result,
            overall_score=overall_score,
            grade=grade,
            recommendations=recommendations,
        )

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data."""
        if data.empty:
            raise InvalidDataError("Input data is empty")

        # Check for required columns
        required = [
            self.config.factor_column,
            self.config.return_column,
        ]

        for col in required:
            if col not in data.columns:
                raise InvalidDataError(f"Missing required column: {col}")

        # Check for minimum data
        if len(data) < self.config.min_periods:
            raise InsufficientDataError(
                f"Need at least {self.config.min_periods} data points, got {len(data)}"
            )

    def _calculate_overall_score(
        self,
        time_result: TimeStabilityResult,
        market_result: MarketStabilityResult,
        regime_result: RegimeStabilityResult,
    ) -> StabilityScore:
        """Calculate weighted overall stability score."""
        time_component = time_result.stability_score * self.config.time_weight
        market_component = market_result.stability_score * self.config.market_weight
        regime_component = regime_result.stability_score * self.config.regime_weight

        overall = time_component + market_component + regime_component

        return StabilityScore(
            value=overall,
            time_component=time_component,
            market_component=market_component,
            regime_component=regime_component,
        )

    def _determine_grade(self, score: float) -> str:
        """Determine letter grade from score."""
        if score >= 0.9:
            return "A"
        elif score >= 0.75:
            return "B"
        elif score >= 0.6:
            return "C"
        elif score >= 0.4:
            return "D"
        else:
            return "F"

    def _generate_recommendations(
        self,
        time_result: TimeStabilityResult,
        market_result: MarketStabilityResult,
        regime_result: RegimeStabilityResult,
        overall_score: StabilityScore,
    ) -> list[str]:
        """Generate improvement recommendations."""
        recommendations = []

        # Time stability recommendations
        if time_result.stability_score < 0.5:
            recommendations.append(
                "Consider extending the evaluation period for more robust time stability"
            )

        if time_result.decay_result and time_result.decay_result.has_decay:
            recommendations.append(
                "IC shows decay trend - factor may be losing predictive power"
            )

        if time_result.ir < 1.0:
            recommendations.append(
                "Low Information Ratio - consider combining with other factors"
            )

        # Market stability recommendations
        if market_result.consistency_score < 0.5:
            recommendations.append(
                "Factor shows inconsistent behavior across market cap groups"
            )

        if market_result.max_ic_difference > 0.1:
            recommendations.append(
                "Large IC variation between market segments - consider market-specific models"
            )

        # Regime stability recommendations
        if regime_result.sensitivity_score > 0.1:
            recommendations.append(
                "Factor is sensitive to market regime - consider regime-conditional usage"
            )

        # Overall recommendations
        if overall_score.value < 0.4:
            recommendations.append(
                "Overall stability is low - exercise caution when deploying this factor"
            )

        if not recommendations:
            recommendations.append(
                "Factor shows good stability across all dimensions"
            )

        return recommendations
