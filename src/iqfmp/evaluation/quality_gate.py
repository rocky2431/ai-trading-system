"""Quality Gate Module for Factor Evaluation (Phase 4).

This module implements advanced quality controls for factor mining:
- D1: Regime-aware CV with crypto-specific regime detection
- D2: Funding rate and fee integration for realistic backtesting
- D6: DSR (Deflated Sharpe Ratio) for multiple testing correction
- B9: Anti p-hacking controls and trial limits

All computations follow Qlib-native architecture.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

from iqfmp.evaluation.qlib_stats import QlibStatisticalEngine

logger = logging.getLogger(__name__)


# =============================================================================
# D1: Crypto Regime Detection (Enhanced)
# =============================================================================


class CryptoRegime(Enum):
    """Crypto-specific market regimes."""

    # Volatility regimes
    LOW_VOL = "low_vol"
    MEDIUM_VOL = "medium_vol"
    HIGH_VOL = "high_vol"
    EXTREME_VOL = "extreme_vol"

    # Funding regimes (crypto-specific)
    FUNDING_NORMAL = "funding_normal"
    FUNDING_EXTREME_POSITIVE = "funding_extreme_positive"  # Longs pay high
    FUNDING_EXTREME_NEGATIVE = "funding_extreme_negative"  # Shorts pay high

    # Trend regimes
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"

    # Liquidation regimes
    HIGH_LIQUIDATION = "high_liquidation"
    LOW_LIQUIDATION = "low_liquidation"


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""

    # Volatility settings
    volatility_window: int = 24  # hours
    volatility_bins: list[float] = field(
        default_factory=lambda: [0.0, 0.015, 0.03, 0.06, float("inf")]
    )

    # Funding rate settings (crypto-specific)
    funding_window: int = 24  # hours for rolling mean
    funding_zscore_threshold: float = 2.0  # Z-score for extreme funding

    # Trend settings
    trend_window: int = 24  # hours
    trend_threshold: float = 0.02  # 2% for trend detection

    # Liquidation settings
    liquidation_window: int = 24
    liquidation_percentile: float = 0.8  # Above 80th percentile = high


class CryptoRegimeDetector:
    """Crypto-specific regime detector.

    Extends base regime detection with:
    - Funding rate extreme detection
    - Liquidation regime detection
    - Combined crypto regime classification
    """

    def __init__(self, config: Optional[RegimeConfig] = None) -> None:
        """Initialize detector."""
        self.config = config or RegimeConfig()
        self._stats = QlibStatisticalEngine()

    def detect_volatility_regime(
        self,
        data: pd.DataFrame,
        price_column: str = "close",
    ) -> pd.Series:
        """Detect volatility regime with crypto-specific bins.

        Args:
            data: DataFrame with price data
            price_column: Name of price column

        Returns:
            Series with volatility regime labels
        """
        if price_column not in data.columns:
            raise ValueError(f"Price column '{price_column}' not found")

        # Calculate rolling volatility (returns std)
        returns = data[price_column].pct_change()
        volatility = returns.rolling(window=self.config.volatility_window).std()

        # Classify into regimes
        labels = ["low_vol", "medium_vol", "high_vol", "extreme_vol"]
        regime = pd.cut(
            volatility,
            bins=self.config.volatility_bins,
            labels=labels,
            include_lowest=True,
        )

        return regime

    def detect_funding_regime(
        self,
        data: pd.DataFrame,
        funding_column: str = "funding_rate",
    ) -> pd.Series:
        """Detect funding rate regime (crypto-specific).

        Extreme positive funding = market is overleveraged long
        Extreme negative funding = market is overleveraged short

        Args:
            data: DataFrame with funding rate data
            funding_column: Name of funding rate column

        Returns:
            Series with funding regime labels
        """
        if funding_column not in data.columns:
            # No funding data - return normal for all
            return pd.Series(
                CryptoRegime.FUNDING_NORMAL.value,
                index=data.index,
            )

        funding = data[funding_column]

        # Calculate rolling mean and std for z-score
        rolling_mean = funding.rolling(
            window=self.config.funding_window,
            min_periods=1,
        ).mean()
        rolling_std = funding.rolling(
            window=self.config.funding_window,
            min_periods=1,
        ).std()

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)
        zscore = (funding - rolling_mean) / rolling_std

        # Classify
        conditions = [
            zscore > self.config.funding_zscore_threshold,
            zscore < -self.config.funding_zscore_threshold,
        ]
        choices = [
            CryptoRegime.FUNDING_EXTREME_POSITIVE.value,
            CryptoRegime.FUNDING_EXTREME_NEGATIVE.value,
        ]

        regime = pd.Series(
            np.select(conditions, choices, default=CryptoRegime.FUNDING_NORMAL.value),
            index=data.index,
        )

        return regime

    def detect_liquidation_regime(
        self,
        data: pd.DataFrame,
        liquidation_column: str = "liquidation_total",
    ) -> pd.Series:
        """Detect liquidation regime.

        Args:
            data: DataFrame with liquidation data
            liquidation_column: Name of liquidation column

        Returns:
            Series with liquidation regime labels
        """
        if liquidation_column not in data.columns:
            return pd.Series(
                CryptoRegime.LOW_LIQUIDATION.value,
                index=data.index,
            )

        liquidation = data[liquidation_column]

        # Calculate rolling percentile threshold
        threshold = liquidation.rolling(
            window=self.config.liquidation_window,
            min_periods=1,
        ).quantile(self.config.liquidation_percentile)

        # Classify
        regime = pd.Series(
            np.where(
                liquidation >= threshold,
                CryptoRegime.HIGH_LIQUIDATION.value,
                CryptoRegime.LOW_LIQUIDATION.value,
            ),
            index=data.index,
        )

        return regime

    def detect_all_regimes(
        self,
        data: pd.DataFrame,
        price_column: str = "close",
        funding_column: str = "funding_rate",
        liquidation_column: str = "liquidation_total",
    ) -> pd.DataFrame:
        """Detect all regime types and return as DataFrame.

        Args:
            data: DataFrame with market data
            price_column: Price column name
            funding_column: Funding rate column name
            liquidation_column: Liquidation column name

        Returns:
            DataFrame with regime columns added
        """
        result = data.copy()

        # Volatility regime
        result["regime_volatility"] = self.detect_volatility_regime(
            data, price_column
        )

        # Funding regime (crypto-specific)
        result["regime_funding"] = self.detect_funding_regime(
            data, funding_column
        )

        # Liquidation regime
        result["regime_liquidation"] = self.detect_liquidation_regime(
            data, liquidation_column
        )

        # Combined regime label
        result["regime_combined"] = (
            result["regime_volatility"].astype(str)
            + "_"
            + result["regime_funding"].astype(str)
        )

        return result


# =============================================================================
# D2: Funding Rate and Fee Integration for Backtesting
# =============================================================================


@dataclass
class BacktestCostConfig:
    """Configuration for realistic backtest costs."""

    # Trading fees (maker/taker)
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0004  # 0.04%

    # Funding rate settings
    funding_settlement_hours: list[int] = field(
        default_factory=lambda: [0, 8, 16]  # UTC hours
    )
    include_funding: bool = True

    # Slippage model
    slippage_bps: float = 5.0  # basis points
    slippage_dynamic: bool = True  # Use volatility-based slippage

    # Position constraints
    max_position_size: float = 1.0  # Max position as fraction of portfolio
    max_leverage: float = 3.0  # Maximum leverage


class RealisticBacktestEngine:
    """Backtest engine with realistic crypto costs.

    Includes:
    - Perpetual funding rate payments (8h settlement)
    - Maker/taker fees
    - Dynamic slippage based on volatility
    - Position and leverage constraints
    """

    def __init__(self, config: Optional[BacktestCostConfig] = None) -> None:
        """Initialize engine."""
        self.config = config or BacktestCostConfig()

    def calculate_funding_cost(
        self,
        data: pd.DataFrame,
        positions: pd.Series,
        funding_column: str = "funding_rate",
    ) -> pd.Series:
        """Calculate funding costs for positions.

        Funding is paid/received every 8 hours.
        Long positions pay positive funding, receive negative.
        Short positions receive positive funding, pay negative.

        Args:
            data: DataFrame with funding rate data
            positions: Series of position sizes (-1 to 1)
            funding_column: Column name for funding rate

        Returns:
            Series of funding costs (negative = cost, positive = income)
        """
        if funding_column not in data.columns:
            return pd.Series(0.0, index=data.index)

        funding_rate = data[funding_column]

        # Funding cost = -position * funding_rate
        # Long (positive position) + positive funding = cost (negative)
        # Short (negative position) + positive funding = income (positive)
        funding_cost = -positions * funding_rate

        # Only apply at settlement hours
        if "timestamp" in data.columns or "datetime" in data.columns:
            ts_col = "timestamp" if "timestamp" in data.columns else "datetime"
            hours = pd.to_datetime(data[ts_col]).dt.hour
            is_settlement = hours.isin(self.config.funding_settlement_hours)
            funding_cost = funding_cost.where(is_settlement, 0)

        return funding_cost

    def calculate_trading_cost(
        self,
        trade_sizes: pd.Series,
        prices: pd.Series,
        is_maker: bool = False,
    ) -> pd.Series:
        """Calculate trading fees.

        Args:
            trade_sizes: Absolute size of trades
            prices: Trade prices
            is_maker: Whether trades are maker (limit) or taker (market)

        Returns:
            Series of trading costs (always negative)
        """
        fee_rate = self.config.maker_fee if is_maker else self.config.taker_fee
        cost = -trade_sizes.abs() * prices * fee_rate
        return cost

    def calculate_slippage(
        self,
        trade_sizes: pd.Series,
        volatility: pd.Series,
    ) -> pd.Series:
        """Calculate slippage cost.

        Args:
            trade_sizes: Absolute size of trades
            volatility: Rolling volatility of prices

        Returns:
            Series of slippage costs (always negative)
        """
        base_slippage = self.config.slippage_bps / 10000  # Convert to decimal

        if self.config.slippage_dynamic:
            # Dynamic slippage based on volatility
            # Higher volatility = higher slippage
            vol_multiplier = 1 + volatility * 10  # Scale factor
            slippage = base_slippage * vol_multiplier
        else:
            slippage = base_slippage

        cost = -trade_sizes.abs() * slippage
        return cost

    def calculate_net_returns(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        price_column: str = "close",
        funding_column: str = "funding_rate",
    ) -> pd.DataFrame:
        """Calculate net returns including all costs.

        Args:
            data: Market data DataFrame
            signals: Trading signals (-1 to 1)
            price_column: Price column name
            funding_column: Funding rate column name

        Returns:
            DataFrame with gross and net returns, cost breakdown
        """
        # Align signals with data
        signals = signals.reindex(data.index).fillna(0)

        # Calculate positions (signals with position limits)
        positions = signals.clip(
            -self.config.max_position_size,
            self.config.max_position_size,
        )

        # Calculate price returns
        price_returns = data[price_column].pct_change()

        # Gross returns (before costs)
        gross_returns = positions.shift(1) * price_returns

        # Calculate trades (position changes)
        trades = positions.diff().fillna(0)

        # Calculate volatility for slippage
        volatility = price_returns.rolling(24).std().fillna(0)

        # Calculate costs
        funding_cost = self.calculate_funding_cost(data, positions, funding_column)
        trading_cost = self.calculate_trading_cost(
            trades, data[price_column], is_maker=False
        )
        slippage_cost = self.calculate_slippage(trades, volatility)

        # Total costs
        total_cost = funding_cost + trading_cost + slippage_cost

        # Net returns
        net_returns = gross_returns + total_cost

        return pd.DataFrame({
            "gross_return": gross_returns,
            "funding_cost": funding_cost,
            "trading_cost": trading_cost,
            "slippage_cost": slippage_cost,
            "total_cost": total_cost,
            "net_return": net_returns,
            "position": positions,
        })


# =============================================================================
# D6: DSR (Deflated Sharpe Ratio) for Multiple Testing
# =============================================================================


@dataclass
class DSRResult:
    """Result of DSR calculation."""

    raw_sharpe: float
    deflated_sharpe: float
    haircut_pct: float  # Percentage reduction
    n_trials: int
    passes_threshold: bool
    threshold: float
    p_value: float  # Probability of observing Sharpe by chance
    confidence: str  # Low/Medium/High

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "raw_sharpe": round(self.raw_sharpe, 4),
            "deflated_sharpe": round(self.deflated_sharpe, 4),
            "haircut_pct": round(self.haircut_pct, 2),
            "n_trials": self.n_trials,
            "passes_threshold": self.passes_threshold,
            "threshold": round(self.threshold, 4),
            "p_value": round(self.p_value, 4),
            "confidence": self.confidence,
        }


class DeflatedSharpeCalculator:
    """Calculate Deflated Sharpe Ratio for multiple testing correction.

    Implements Bailey and Lopez de Prado (2014) DSR methodology
    to account for multiple testing and overfitting risk.

    Reference:
    - "The Deflated Sharpe Ratio: Correcting for Selection Bias,
       Backtest Overfitting and Non-Normality"
    """

    def __init__(
        self,
        threshold: float = 1.0,  # Minimum DSR to pass
        min_trials: int = 1,  # Minimum trials before applying correction
    ) -> None:
        """Initialize calculator."""
        self.threshold = threshold
        self.min_trials = min_trials

    def calculate(
        self,
        sharpe_ratio: float,
        n_trials: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,  # Normal = 3
        t_observations: int = 252,  # Trading days
    ) -> DSRResult:
        """Calculate Deflated Sharpe Ratio.

        Args:
            sharpe_ratio: Raw Sharpe ratio of the strategy
            n_trials: Number of trials/backtests performed
            skewness: Skewness of returns
            kurtosis: Kurtosis of returns (excess = kurtosis - 3)
            t_observations: Number of observations

        Returns:
            DSRResult with deflated metrics
        """
        # If few trials, no correction needed
        if n_trials <= self.min_trials:
            return DSRResult(
                raw_sharpe=sharpe_ratio,
                deflated_sharpe=sharpe_ratio,
                haircut_pct=0.0,
                n_trials=n_trials,
                passes_threshold=sharpe_ratio >= self.threshold,
                threshold=self.threshold,
                p_value=0.0,
                confidence="N/A (insufficient trials)",
            )

        # Expected maximum Sharpe under null hypothesis
        # E[max(SR)] = sqrt(2 * log(N)) approximately
        expected_max_sr = self._expected_max_sharpe(n_trials)

        # Standard deviation of Sharpe estimate
        sr_std = self._sharpe_std(sharpe_ratio, skewness, kurtosis, t_observations)

        # Calculate p-value: P(observing this Sharpe by chance)
        # Under null, Sharpe ~ N(0, sr_std)
        from scipy import stats
        z_score = sharpe_ratio / sr_std if sr_std > 0 else 0
        p_value = 1 - stats.norm.cdf(z_score)

        # Deflated Sharpe = Sharpe - Expected Maximum Sharpe * adjustment
        # More conservative: scale by how many trials
        trial_penalty = math.log(n_trials) / math.log(max(n_trials, 10))
        deflated_sharpe = sharpe_ratio - (expected_max_sr * trial_penalty * 0.5)
        deflated_sharpe = max(0, deflated_sharpe)

        # Calculate haircut percentage
        haircut_pct = (
            ((sharpe_ratio - deflated_sharpe) / sharpe_ratio * 100)
            if sharpe_ratio > 0
            else 0.0
        )

        # Confidence level based on p-value and trials
        if p_value < 0.01 and n_trials >= 20:
            confidence = "High"
        elif p_value < 0.05 and n_trials >= 10:
            confidence = "Medium"
        else:
            confidence = "Low"

        return DSRResult(
            raw_sharpe=sharpe_ratio,
            deflated_sharpe=deflated_sharpe,
            haircut_pct=haircut_pct,
            n_trials=n_trials,
            passes_threshold=deflated_sharpe >= self.threshold,
            threshold=self.threshold,
            p_value=p_value,
            confidence=confidence,
        )

    def _expected_max_sharpe(self, n_trials: int) -> float:
        """Calculate expected maximum Sharpe under null hypothesis.

        E[max(Z_1, ..., Z_n)] where Z_i ~ N(0,1)
        Approximation: sqrt(2 * log(n)) - (log(log(n)) + log(4*pi)) / (2 * sqrt(2 * log(n)))
        """
        if n_trials <= 1:
            return 0.0

        a = math.sqrt(2 * math.log(n_trials))
        b = (math.log(math.log(n_trials)) + math.log(4 * math.pi)) / (2 * a)
        return a - b

    def _sharpe_std(
        self,
        sharpe: float,
        skewness: float,
        kurtosis: float,
        t: int,
    ) -> float:
        """Calculate standard deviation of Sharpe ratio estimate.

        Based on Lo (2002) "The Statistics of Sharpe Ratios"
        """
        excess_kurtosis = kurtosis - 3

        # Variance of Sharpe ratio
        var_sr = (
            1
            + 0.5 * sharpe**2
            - skewness * sharpe
            + 0.25 * excess_kurtosis * sharpe**2
        ) / t

        return math.sqrt(max(0, var_sr))


# =============================================================================
# B9: Anti P-Hacking Controls
# =============================================================================


@dataclass
class AntiPHackingConfig:
    """Configuration for anti p-hacking controls."""

    # Trial limits
    max_trials_per_hypothesis: int = 10  # Max trials per hypothesis family
    max_total_trials: int = 1000  # Max total trials in session
    max_candidates_per_round: int = 5  # Max factor candidates per round

    # Complexity penalties
    complexity_penalty_weight: float = 0.1  # Penalty per unit complexity
    max_expression_depth: int = 5  # Max nesting depth
    max_operators: int = 8  # Max operators in expression

    # Performance thresholds (post-correction)
    min_deflated_sharpe: float = 0.5  # Minimum DSR to accept
    min_ic: float = 0.02  # Minimum IC
    min_ir: float = 0.5  # Minimum IR

    # Cool-down
    hypothesis_cooldown_minutes: int = 5  # Wait before retrying hypothesis


@dataclass
class TrialBudget:
    """Track trial budget for anti p-hacking."""

    total_trials: int = 0
    trials_by_hypothesis: dict[str, int] = field(default_factory=dict)
    last_trial_time: dict[str, datetime] = field(default_factory=dict)

    def can_try(
        self,
        hypothesis_id: str,
        config: AntiPHackingConfig,
    ) -> tuple[bool, str]:
        """Check if a trial is allowed.

        Args:
            hypothesis_id: Identifier for the hypothesis family
            config: Anti p-hacking configuration

        Returns:
            (allowed, reason) tuple
        """
        # Check total trials
        if self.total_trials >= config.max_total_trials:
            return False, f"Total trial limit reached ({config.max_total_trials})"

        # Check hypothesis-specific trials
        hyp_trials = self.trials_by_hypothesis.get(hypothesis_id, 0)
        if hyp_trials >= config.max_trials_per_hypothesis:
            return False, f"Hypothesis trial limit reached ({config.max_trials_per_hypothesis})"

        # Check cooldown
        last_time = self.last_trial_time.get(hypothesis_id)
        if last_time:
            elapsed = (datetime.now() - last_time).total_seconds() / 60
            if elapsed < config.hypothesis_cooldown_minutes:
                remaining = config.hypothesis_cooldown_minutes - elapsed
                return False, f"Cooldown in effect ({remaining:.1f} minutes remaining)"

        return True, "OK"

    def record_trial(self, hypothesis_id: str) -> None:
        """Record a trial attempt."""
        self.total_trials += 1
        self.trials_by_hypothesis[hypothesis_id] = (
            self.trials_by_hypothesis.get(hypothesis_id, 0) + 1
        )
        self.last_trial_time[hypothesis_id] = datetime.now()


class AntiPHackingGate:
    """Gate for preventing p-hacking in factor mining.

    Implements multiple safeguards:
    - Trial budget limits per hypothesis family
    - Complexity penalties for overly complex factors
    - DSR-based acceptance criteria
    - Cooldown periods between hypothesis attempts
    """

    def __init__(self, config: Optional[AntiPHackingConfig] = None) -> None:
        """Initialize gate."""
        self.config = config or AntiPHackingConfig()
        self.dsr_calculator = DeflatedSharpeCalculator(
            threshold=self.config.min_deflated_sharpe
        )
        self.budget = TrialBudget()

    def get_hypothesis_id(self, hypothesis: str, family: str) -> str:
        """Generate stable ID for a hypothesis family.

        Args:
            hypothesis: Hypothesis text
            family: Factor family

        Returns:
            Stable hash-based ID
        """
        content = f"{family}:{hypothesis[:100]}".lower()
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def check_trial_allowed(
        self,
        hypothesis: str,
        family: str,
    ) -> tuple[bool, str]:
        """Check if a trial is allowed under budget.

        Args:
            hypothesis: Hypothesis text
            family: Factor family

        Returns:
            (allowed, reason) tuple
        """
        hyp_id = self.get_hypothesis_id(hypothesis, family)
        return self.budget.can_try(hyp_id, self.config)

    def calculate_complexity_penalty(
        self,
        expression: str,
    ) -> tuple[float, dict[str, Any]]:
        """Calculate complexity penalty for an expression.

        Args:
            expression: Qlib expression string

        Returns:
            (penalty, details) tuple
        """
        import re

        # Count operators
        operators = re.findall(r"([A-Z][a-zA-Z]*)\s*\(", expression)
        n_operators = len(operators)

        # Calculate nesting depth
        max_depth = 0
        current_depth = 0
        for char in expression:
            if char == "(":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ")":
                current_depth -= 1

        # Calculate penalty
        operator_penalty = max(0, n_operators - 3) * self.config.complexity_penalty_weight
        depth_penalty = max(0, max_depth - 2) * self.config.complexity_penalty_weight

        total_penalty = operator_penalty + depth_penalty

        details = {
            "n_operators": n_operators,
            "max_depth": max_depth,
            "operator_penalty": operator_penalty,
            "depth_penalty": depth_penalty,
            "total_penalty": total_penalty,
            "exceeds_limits": (
                n_operators > self.config.max_operators
                or max_depth > self.config.max_expression_depth
            ),
        }

        return total_penalty, details

    def evaluate_factor_quality(
        self,
        sharpe_ratio: float,
        ic: float,
        ir: float,
        expression: str,
        hypothesis: str,
        family: str,
    ) -> dict[str, Any]:
        """Evaluate overall factor quality with anti p-hacking controls.

        Args:
            sharpe_ratio: Raw Sharpe ratio
            ic: Information coefficient
            ir: Information ratio
            expression: Qlib expression
            hypothesis: Original hypothesis
            family: Factor family

        Returns:
            Dictionary with evaluation results
        """
        # Record trial
        hyp_id = self.get_hypothesis_id(hypothesis, family)
        self.budget.record_trial(hyp_id)

        # Get trial count for DSR
        n_trials = self.budget.trials_by_hypothesis.get(hyp_id, 1)

        # Calculate DSR
        dsr_result = self.dsr_calculator.calculate(
            sharpe_ratio=sharpe_ratio,
            n_trials=n_trials,
        )

        # Calculate complexity penalty
        complexity_penalty, complexity_details = self.calculate_complexity_penalty(
            expression
        )

        # Adjust metrics with complexity penalty
        adjusted_sharpe = sharpe_ratio - complexity_penalty
        adjusted_dsr = dsr_result.deflated_sharpe - complexity_penalty

        # Check thresholds
        passes_sharpe = adjusted_dsr >= self.config.min_deflated_sharpe
        passes_ic = abs(ic) >= self.config.min_ic
        passes_ir = ir >= self.config.min_ir
        passes_all = passes_sharpe and passes_ic and passes_ir

        # Determine grade
        if passes_all and dsr_result.confidence == "High":
            grade = "A"
        elif passes_all and dsr_result.confidence == "Medium":
            grade = "B"
        elif passes_sharpe and passes_ic:
            grade = "C"
        elif passes_sharpe or passes_ic:
            grade = "D"
        else:
            grade = "F"

        return {
            "hypothesis_id": hyp_id,
            "n_trials": n_trials,
            "raw_sharpe": sharpe_ratio,
            "deflated_sharpe": dsr_result.deflated_sharpe,
            "adjusted_sharpe": adjusted_sharpe,
            "adjusted_dsr": adjusted_dsr,
            "dsr_result": dsr_result.to_dict(),
            "complexity": complexity_details,
            "passes_threshold": passes_all,
            "grade": grade,
            "recommendations": self._generate_recommendations(
                passes_sharpe, passes_ic, passes_ir, complexity_details, dsr_result
            ),
        }

    def _generate_recommendations(
        self,
        passes_sharpe: bool,
        passes_ic: bool,
        passes_ir: bool,
        complexity: dict,
        dsr: DSRResult,
    ) -> list[str]:
        """Generate recommendations for improving factor."""
        recommendations = []

        if not passes_sharpe:
            recommendations.append(
                f"DSR ({dsr.deflated_sharpe:.2f}) below threshold ({self.config.min_deflated_sharpe}). "
                "Consider simplifying or using different hypothesis."
            )

        if not passes_ic:
            recommendations.append(
                f"IC below threshold ({self.config.min_ic}). "
                "Factor may lack predictive power."
            )

        if not passes_ir:
            recommendations.append(
                f"IR below threshold ({self.config.min_ir}). "
                "Factor performance is inconsistent."
            )

        if complexity.get("exceeds_limits"):
            recommendations.append(
                f"Expression too complex (depth={complexity['max_depth']}, "
                f"ops={complexity['n_operators']}). Simplify to reduce overfitting risk."
            )

        if dsr.haircut_pct > 30:
            recommendations.append(
                f"High DSR haircut ({dsr.haircut_pct:.1f}%). "
                "Many trials detected - consider new hypothesis direction."
            )

        if dsr.confidence == "Low":
            recommendations.append(
                "Low confidence in results. Need more independent validation."
            )

        if not recommendations:
            recommendations.append(
                "Factor meets all quality criteria. Consider additional regime testing."
            )

        return recommendations


# =============================================================================
# Unified Quality Gate
# =============================================================================


class FactorQualityGate:
    """Unified quality gate combining all Phase 4 components.

    Integrates:
    - D1: Regime-aware CV with crypto regimes
    - D2: Realistic backtesting with funding/fees
    - D6: DSR for multiple testing correction
    - B9: Anti p-hacking controls
    """

    def __init__(
        self,
        regime_config: Optional[RegimeConfig] = None,
        cost_config: Optional[BacktestCostConfig] = None,
        anti_phacking_config: Optional[AntiPHackingConfig] = None,
    ) -> None:
        """Initialize quality gate."""
        self.regime_detector = CryptoRegimeDetector(regime_config)
        self.backtest_engine = RealisticBacktestEngine(cost_config)
        self.anti_phacking = AntiPHackingGate(anti_phacking_config)
        self.dsr_calculator = DeflatedSharpeCalculator()

    def evaluate_with_regime_split(
        self,
        data: pd.DataFrame,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        price_column: str = "close",
        funding_column: str = "funding_rate",
    ) -> dict[str, Any]:
        """Evaluate factor performance split by regime.

        Args:
            data: Market data DataFrame
            factor_values: Factor values
            forward_returns: Forward returns for IC calculation
            price_column: Price column name
            funding_column: Funding rate column name

        Returns:
            Dictionary with regime-stratified evaluation results
        """
        # Detect regimes
        data_with_regimes = self.regime_detector.detect_all_regimes(
            data, price_column, funding_column
        )

        results = {"overall": {}, "by_regime": {}}

        # Overall IC
        overall_ic = factor_values.corr(forward_returns, method="spearman")
        results["overall"]["ic"] = float(overall_ic) if not pd.isna(overall_ic) else 0.0

        # Evaluate by volatility regime
        for regime in data_with_regimes["regime_volatility"].dropna().unique():
            mask = data_with_regimes["regime_volatility"] == regime
            if mask.sum() < 30:
                continue

            regime_factor = factor_values[mask]
            regime_returns = forward_returns[mask]

            ic = regime_factor.corr(regime_returns, method="spearman")
            results["by_regime"][f"vol_{regime}"] = {
                "ic": float(ic) if not pd.isna(ic) else 0.0,
                "n_samples": int(mask.sum()),
            }

        # Evaluate by funding regime
        for regime in data_with_regimes["regime_funding"].dropna().unique():
            mask = data_with_regimes["regime_funding"] == regime
            if mask.sum() < 30:
                continue

            regime_factor = factor_values[mask]
            regime_returns = forward_returns[mask]

            ic = regime_factor.corr(regime_returns, method="spearman")
            results["by_regime"][f"funding_{regime}"] = {
                "ic": float(ic) if not pd.isna(ic) else 0.0,
                "n_samples": int(mask.sum()),
            }

        # Calculate stability across regimes
        ic_values = [
            r["ic"] for r in results["by_regime"].values() if r["ic"] != 0
        ]
        if ic_values:
            results["regime_stability"] = {
                "mean_ic": float(np.mean(ic_values)),
                "std_ic": float(np.std(ic_values)),
                "min_ic": float(min(ic_values)),
                "max_ic": float(max(ic_values)),
                "ic_range": float(max(ic_values) - min(ic_values)),
            }

        return results

    def full_evaluation(
        self,
        data: pd.DataFrame,
        factor_values: pd.Series,
        signals: pd.Series,
        expression: str,
        hypothesis: str,
        family: str,
        price_column: str = "close",
        funding_column: str = "funding_rate",
    ) -> dict[str, Any]:
        """Run full quality evaluation with all checks.

        Args:
            data: Market data DataFrame
            factor_values: Factor values
            signals: Trading signals from factor
            expression: Qlib expression
            hypothesis: Original hypothesis
            family: Factor family
            price_column: Price column name
            funding_column: Funding rate column name

        Returns:
            Comprehensive evaluation results
        """
        # Check trial budget
        allowed, reason = self.anti_phacking.check_trial_allowed(hypothesis, family)
        if not allowed:
            return {
                "status": "blocked",
                "reason": reason,
                "recommendation": "Try a different hypothesis or wait for cooldown",
            }

        # Calculate forward returns for IC
        forward_returns = data[price_column].pct_change().shift(-1)

        # Regime-stratified evaluation
        regime_results = self.evaluate_with_regime_split(
            data, factor_values, forward_returns, price_column, funding_column
        )

        # Realistic backtest with costs
        backtest_results = self.backtest_engine.calculate_net_returns(
            data, signals, price_column, funding_column
        )

        # Calculate metrics from backtest
        net_returns = backtest_results["net_return"].dropna()
        if len(net_returns) > 0:
            sharpe = (
                net_returns.mean()
                / net_returns.std()
                * np.sqrt(252 * 24)  # Annualized for hourly data
            ) if net_returns.std() > 0 else 0
        else:
            sharpe = 0.0

        # Calculate IC and IR
        ic = regime_results["overall"]["ic"]
        # Simple IR approximation
        ic_series = factor_values.rolling(24).corr(forward_returns).dropna()
        ir = ic_series.mean() / ic_series.std() if len(ic_series) > 0 and ic_series.std() > 0 else 0

        # Anti p-hacking evaluation
        quality_eval = self.anti_phacking.evaluate_factor_quality(
            sharpe_ratio=float(sharpe),
            ic=float(ic),
            ir=float(ir),
            expression=expression,
            hypothesis=hypothesis,
            family=family,
        )

        # Cost analysis
        total_funding_cost = backtest_results["funding_cost"].sum()
        total_trading_cost = backtest_results["trading_cost"].sum()
        total_slippage = backtest_results["slippage_cost"].sum()

        return {
            "status": "evaluated",
            "regime_analysis": regime_results,
            "backtest": {
                "gross_sharpe": float(
                    backtest_results["gross_return"].mean()
                    / backtest_results["gross_return"].std()
                    * np.sqrt(252 * 24)
                ) if backtest_results["gross_return"].std() > 0 else 0,
                "net_sharpe": float(sharpe),
                "total_funding_cost": float(total_funding_cost),
                "total_trading_cost": float(total_trading_cost),
                "total_slippage": float(total_slippage),
                "total_cost": float(total_funding_cost + total_trading_cost + total_slippage),
            },
            "quality": quality_eval,
            "final_verdict": {
                "passes": quality_eval["passes_threshold"],
                "grade": quality_eval["grade"],
                "confidence": quality_eval["dsr_result"]["confidence"],
            },
        }
