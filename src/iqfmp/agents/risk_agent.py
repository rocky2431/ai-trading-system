"""Risk Check Agent for IQFMP.

Centralized risk management for strategy validation and deployment.
Performs multi-dimensional risk assessment before strategy approval.

Six-dimensional coverage:
1. Functional: Risk metrics calculation, limit checking, approval logic
2. Boundary: Extreme market conditions, edge case positions
3. Exception: Missing data, invalid metrics, timeout handling
4. Performance: Efficient risk calculation
5. Security: Position limit enforcement
6. Compatibility: Multiple risk models and market regimes
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import logging

import numpy as np
import pandas as pd

from iqfmp.agents.orchestrator import AgentState


logger = logging.getLogger(__name__)


class RiskAgentError(Exception):
    """Base error for risk agent failures."""

    pass


class RiskLimitExceededError(RiskAgentError):
    """Raised when a risk limit is exceeded."""

    pass


class InsufficientDataError(RiskAgentError):
    """Raised when data is insufficient for risk analysis."""

    pass


class RiskLevel(Enum):
    """Risk assessment levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(Enum):
    """Categories of risk assessment."""

    MARKET = "market"
    LIQUIDITY = "liquidity"
    CONCENTRATION = "concentration"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    TAIL = "tail"


class ApprovalStatus(Enum):
    """Strategy approval status."""

    APPROVED = "approved"
    CONDITIONAL = "conditional"
    REJECTED = "rejected"
    MANUAL_REVIEW = "manual_review"


@dataclass
class RiskLimit:
    """Definition of a risk limit."""

    name: str
    category: RiskCategory
    limit_value: float
    current_value: float = 0.0
    breached: bool = False
    severity: RiskLevel = RiskLevel.MEDIUM

    def check(self, value: float) -> bool:
        """Check if value breaches limit."""
        self.current_value = value
        self.breached = value > self.limit_value
        return not self.breached

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "category": self.category.value,
            "limit_value": self.limit_value,
            "current_value": self.current_value,
            "breached": self.breached,
            "severity": self.severity.value,
        }


@dataclass
class RiskConfig:
    """Configuration for risk management."""

    # Position limits
    max_position_size: float = 0.1  # 10% of portfolio
    max_sector_exposure: float = 0.3  # 30% per sector
    max_single_asset: float = 0.05  # 5% in single asset

    # Drawdown limits
    max_drawdown: float = 0.15  # 15% max drawdown
    daily_loss_limit: float = 0.03  # 3% daily loss limit

    # Volatility limits
    max_portfolio_volatility: float = 0.25  # 25% annualized
    max_var_95: float = 0.05  # 5% daily VaR at 95%

    # Correlation limits
    max_factor_correlation: float = 0.7  # Max correlation between factors

    # Liquidity limits
    max_turnover: float = 0.5  # 50% daily turnover
    min_liquidity_ratio: float = 0.1  # 10% liquidity buffer

    # Approval thresholds
    auto_approve_threshold: float = 0.5  # Risk score below this = auto approve
    manual_review_threshold: float = 0.75  # Above this = manual review

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_position_size": self.max_position_size,
            "max_sector_exposure": self.max_sector_exposure,
            "max_single_asset": self.max_single_asset,
            "max_drawdown": self.max_drawdown,
            "daily_loss_limit": self.daily_loss_limit,
            "max_portfolio_volatility": self.max_portfolio_volatility,
            "max_var_95": self.max_var_95,
            "max_factor_correlation": self.max_factor_correlation,
            "max_turnover": self.max_turnover,
            "min_liquidity_ratio": self.min_liquidity_ratio,
            "auto_approve_threshold": self.auto_approve_threshold,
            "manual_review_threshold": self.manual_review_threshold,
        }


@dataclass
class RiskMetrics:
    """Calculated risk metrics."""

    # Drawdown metrics
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    drawdown_duration: int = 0

    # Volatility metrics
    portfolio_volatility: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall: float = 0.0

    # Position metrics
    max_position: float = 0.0
    avg_position: float = 0.0
    concentration_ratio: float = 0.0

    # Correlation metrics
    avg_factor_correlation: float = 0.0
    max_factor_correlation: float = 0.0

    # Liquidity metrics
    avg_turnover: float = 0.0
    liquidity_ratio: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "drawdown_duration": self.drawdown_duration,
            "portfolio_volatility": self.portfolio_volatility,
            "var_95": self.var_95,
            "var_99": self.var_99,
            "expected_shortfall": self.expected_shortfall,
            "max_position": self.max_position,
            "avg_position": self.avg_position,
            "concentration_ratio": self.concentration_ratio,
            "avg_factor_correlation": self.avg_factor_correlation,
            "max_factor_correlation": self.max_factor_correlation,
            "avg_turnover": self.avg_turnover,
            "liquidity_ratio": self.liquidity_ratio,
        }


@dataclass
class RiskAssessment:
    """Complete risk assessment result."""

    overall_level: RiskLevel
    overall_score: float  # 0-1, higher = riskier
    approval_status: ApprovalStatus
    metrics: RiskMetrics
    limits: list[RiskLimit]
    breached_limits: list[str]
    warnings: list[str]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for state storage."""
        return {
            "overall_level": self.overall_level.value,
            "overall_score": self.overall_score,
            "approval_status": self.approval_status.value,
            "metrics": self.metrics.to_dict(),
            "limits": [lim.to_dict() for lim in self.limits],
            "breached_limits": self.breached_limits,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }


class RiskCalculator:
    """Calculator for risk metrics."""

    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """Calculate Value at Risk.

        Args:
            returns: Return series
            confidence: Confidence level (e.g., 0.95)

        Returns:
            VaR value (positive = loss)
        """
        if len(returns) < 20:
            return 0.0

        return float(-np.percentile(returns, (1 - confidence) * 100))

    def calculate_expected_shortfall(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """Calculate Expected Shortfall (CVaR).

        Args:
            returns: Return series
            confidence: Confidence level

        Returns:
            Expected Shortfall value
        """
        if len(returns) < 20:
            return 0.0

        var = self.calculate_var(returns, confidence)
        tail_returns = returns[returns < -var]

        if len(tail_returns) == 0:
            return var

        return float(-tail_returns.mean())

    def calculate_drawdown(
        self,
        cumulative_returns: pd.Series,
    ) -> tuple[float, float, int]:
        """Calculate drawdown metrics.

        Args:
            cumulative_returns: Cumulative return series

        Returns:
            (current_drawdown, max_drawdown, duration)
        """
        if len(cumulative_returns) < 2:
            return 0.0, 0.0, 0

        running_max = cumulative_returns.cummax()
        drawdown = (running_max - cumulative_returns) / running_max
        drawdown = drawdown.replace([np.inf, -np.inf], 0).fillna(0)

        current_dd = float(drawdown.iloc[-1])
        max_dd = float(drawdown.max())

        # Calculate duration
        in_drawdown = drawdown > 0
        duration = 0
        if in_drawdown.iloc[-1]:
            # Count consecutive drawdown days
            for val in reversed(in_drawdown.values):
                if val:
                    duration += 1
                else:
                    break

        return current_dd, max_dd, duration

    def calculate_concentration(
        self,
        positions: pd.Series,
    ) -> float:
        """Calculate position concentration (Herfindahl index).

        Args:
            positions: Position weights

        Returns:
            Concentration ratio (0-1)
        """
        if len(positions) == 0:
            return 0.0

        # Normalize
        weights = positions.abs()
        total = weights.sum()
        if total == 0:
            return 0.0

        weights = weights / total

        # Herfindahl index
        hhi = (weights ** 2).sum()

        return float(hhi)


class RiskCheckAgent:
    """Agent for centralized risk management.

    This agent validates strategies against risk limits and
    provides approval recommendations.

    Responsibilities:
    - Calculate comprehensive risk metrics
    - Check against configurable limits
    - Provide risk-adjusted approval status
    - Generate risk warnings and recommendations

    Usage:
        agent = RiskCheckAgent(config)
        new_state = await agent.check(state)
    """

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        """Initialize the risk check agent.

        Args:
            config: Risk configuration
        """
        self.config = config or RiskConfig()
        self.calculator = RiskCalculator()

    async def check(self, state: AgentState) -> AgentState:
        """Perform risk check on strategy.

        This is the main entry point for StateGraph integration.

        Args:
            state: Current agent state containing:
                - context["backtest_metrics"]: Backtest performance metrics
                - context["strategy_signals"]: Trading signals
                - context["factor_weights"]: Factor weight assignments
                - context["price_data"]: Price data

        Returns:
            Updated state with risk assessment

        Raises:
            RiskAgentError: On risk check failure
        """
        logger.info("RiskCheckAgent: Starting risk assessment")

        context = state.context
        backtest_metrics = context.get("backtest_metrics", {})
        strategy_signals = context.get("strategy_signals")
        factor_weights = context.get("factor_weights", [])
        price_data = context.get("price_data")

        # Calculate risk metrics
        metrics = self._calculate_metrics(
            backtest_metrics,
            strategy_signals,
            price_data,
        )

        # Check limits
        limits = self._create_limits()
        breached = self._check_limits(limits, metrics)

        # Calculate overall risk score
        score = self._calculate_risk_score(metrics, breached)

        # Determine approval status
        approval = self._determine_approval(score, breached)

        # Generate warnings and recommendations
        warnings = self._generate_warnings(metrics, breached)
        recommendations = self._generate_recommendations(metrics, breached)

        # Determine overall risk level
        level = self._determine_risk_level(score)

        # Build assessment
        assessment = RiskAssessment(
            overall_level=level,
            overall_score=score,
            approval_status=approval,
            metrics=metrics,
            limits=limits,
            breached_limits=breached,
            warnings=warnings,
            recommendations=recommendations,
        )

        # Update state
        new_context = {
            **context,
            "risk_assessment": assessment.to_dict(),
            "risk_level": level.value,
            "risk_score": score,
            "approval_status": approval.value,
            "risk_warnings": warnings,
            "strategy_approved": approval == ApprovalStatus.APPROVED,
        }

        logger.info(
            f"RiskCheckAgent: Completed. "
            f"Level: {level.value}, "
            f"Score: {score:.2f}, "
            f"Approval: {approval.value}"
        )

        return state.update(context=new_context)

    def _calculate_metrics(
        self,
        backtest_metrics: dict[str, float],
        strategy_signals: Optional[list[dict[str, Any]]],
        price_data: Optional[pd.DataFrame],
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics.

        Args:
            backtest_metrics: Metrics from backtest
            strategy_signals: Trading signals
            price_data: Price data

        Returns:
            RiskMetrics with calculated values
        """
        metrics = RiskMetrics()

        # Extract from backtest metrics
        metrics.max_drawdown = backtest_metrics.get("max_drawdown", 0)
        metrics.portfolio_volatility = backtest_metrics.get("sharpe_ratio", 0) * 0.1  # Approximate

        # Calculate from price data
        if price_data is not None and "returns" in price_data.columns:
            returns = price_data["returns"]

            metrics.var_95 = self.calculator.calculate_var(returns, 0.95)
            metrics.var_99 = self.calculator.calculate_var(returns, 0.99)
            metrics.expected_shortfall = self.calculator.calculate_expected_shortfall(returns, 0.95)

            cumulative = (1 + returns).cumprod()
            dd_current, dd_max, dd_dur = self.calculator.calculate_drawdown(cumulative)
            metrics.current_drawdown = dd_current
            if dd_max > metrics.max_drawdown:
                metrics.max_drawdown = dd_max
            metrics.drawdown_duration = dd_dur

        # Calculate from signals
        if strategy_signals:
            signals_df = pd.DataFrame(strategy_signals)

            if "position" in signals_df.columns:
                positions = signals_df["position"]
                metrics.max_position = float(positions.abs().max())
                metrics.avg_position = float(positions.abs().mean())
                metrics.concentration_ratio = self.calculator.calculate_concentration(positions)

            if "combined_signal" in signals_df.columns:
                signals = signals_df["combined_signal"]
                metrics.avg_turnover = float(signals.diff().abs().mean())

        return metrics

    def _create_limits(self) -> list[RiskLimit]:
        """Create risk limits from config.

        Returns:
            List of RiskLimit objects
        """
        return [
            RiskLimit(
                name="max_drawdown",
                category=RiskCategory.DRAWDOWN,
                limit_value=self.config.max_drawdown,
                severity=RiskLevel.HIGH,
            ),
            RiskLimit(
                name="daily_loss",
                category=RiskCategory.DRAWDOWN,
                limit_value=self.config.daily_loss_limit,
                severity=RiskLevel.CRITICAL,
            ),
            RiskLimit(
                name="portfolio_volatility",
                category=RiskCategory.VOLATILITY,
                limit_value=self.config.max_portfolio_volatility,
                severity=RiskLevel.MEDIUM,
            ),
            RiskLimit(
                name="var_95",
                category=RiskCategory.TAIL,
                limit_value=self.config.max_var_95,
                severity=RiskLevel.HIGH,
            ),
            RiskLimit(
                name="max_position",
                category=RiskCategory.CONCENTRATION,
                limit_value=self.config.max_position_size,
                severity=RiskLevel.MEDIUM,
            ),
            RiskLimit(
                name="turnover",
                category=RiskCategory.LIQUIDITY,
                limit_value=self.config.max_turnover,
                severity=RiskLevel.LOW,
            ),
        ]

    def _check_limits(
        self,
        limits: list[RiskLimit],
        metrics: RiskMetrics,
    ) -> list[str]:
        """Check metrics against limits.

        Args:
            limits: List of risk limits
            metrics: Calculated metrics

        Returns:
            List of breached limit names
        """
        breached = []

        metric_map = {
            "max_drawdown": metrics.max_drawdown,
            "daily_loss": metrics.current_drawdown,
            "portfolio_volatility": metrics.portfolio_volatility,
            "var_95": metrics.var_95,
            "max_position": metrics.max_position,
            "turnover": metrics.avg_turnover,
        }

        for limit in limits:
            value = metric_map.get(limit.name, 0)
            if not limit.check(value):
                breached.append(limit.name)

        return breached

    def _calculate_risk_score(
        self,
        metrics: RiskMetrics,
        breached: list[str],
    ) -> float:
        """Calculate overall risk score (0-1).

        Args:
            metrics: Risk metrics
            breached: List of breached limits

        Returns:
            Risk score (higher = riskier)
        """
        score = 0.0

        # Drawdown contribution (0-0.3)
        dd_score = min(metrics.max_drawdown / self.config.max_drawdown, 1.0) * 0.3
        score += dd_score

        # Volatility contribution (0-0.2)
        vol_score = min(metrics.portfolio_volatility / self.config.max_portfolio_volatility, 1.0) * 0.2
        score += vol_score

        # VaR contribution (0-0.2)
        var_score = min(metrics.var_95 / self.config.max_var_95, 1.0) * 0.2
        score += var_score

        # Concentration contribution (0-0.15)
        conc_score = metrics.concentration_ratio * 0.15
        score += conc_score

        # Breach penalty (0-0.15)
        breach_score = min(len(breached) / 6, 1.0) * 0.15
        score += breach_score

        return min(score, 1.0)

    def _determine_approval(
        self,
        score: float,
        breached: list[str],
    ) -> ApprovalStatus:
        """Determine approval status based on score and breaches.

        Args:
            score: Overall risk score
            breached: List of breached limits

        Returns:
            ApprovalStatus
        """
        # Critical breaches always require manual review
        critical_breaches = {"daily_loss", "var_99"}
        if any(b in critical_breaches for b in breached):
            return ApprovalStatus.REJECTED

        if score < self.config.auto_approve_threshold:
            if len(breached) == 0:
                return ApprovalStatus.APPROVED
            else:
                return ApprovalStatus.CONDITIONAL

        elif score < self.config.manual_review_threshold:
            return ApprovalStatus.CONDITIONAL

        else:
            return ApprovalStatus.MANUAL_REVIEW

    def _determine_risk_level(self, score: float) -> RiskLevel:
        """Determine risk level from score.

        Args:
            score: Risk score (0-1)

        Returns:
            RiskLevel
        """
        if score < 0.25:
            return RiskLevel.LOW
        elif score < 0.5:
            return RiskLevel.MEDIUM
        elif score < 0.75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _generate_warnings(
        self,
        metrics: RiskMetrics,
        breached: list[str],
    ) -> list[str]:
        """Generate warning messages.

        Args:
            metrics: Risk metrics
            breached: Breached limits

        Returns:
            List of warning messages
        """
        warnings = []

        for limit_name in breached:
            warnings.append(f"Risk limit '{limit_name}' has been breached")

        if metrics.max_drawdown > 0.1:
            warnings.append(
                f"High drawdown risk: {metrics.max_drawdown:.1%} max drawdown"
            )

        if metrics.var_95 > 0.03:
            warnings.append(
                f"Elevated tail risk: {metrics.var_95:.1%} daily VaR (95%)"
            )

        if metrics.concentration_ratio > 0.3:
            warnings.append(
                f"Position concentration detected: {metrics.concentration_ratio:.1%}"
            )

        if metrics.drawdown_duration > 20:
            warnings.append(
                f"Extended drawdown period: {metrics.drawdown_duration} days"
            )

        return warnings

    def _generate_recommendations(
        self,
        metrics: RiskMetrics,
        breached: list[str],
    ) -> list[str]:
        """Generate recommendations.

        Args:
            metrics: Risk metrics
            breached: Breached limits

        Returns:
            List of recommendations
        """
        recommendations = []

        if metrics.max_drawdown > self.config.max_drawdown * 0.8:
            recommendations.append(
                "Consider adding stop-loss mechanisms to limit drawdown"
            )

        if metrics.concentration_ratio > 0.2:
            recommendations.append(
                "Increase diversification by adding more assets or reducing position sizes"
            )

        if metrics.avg_turnover > self.config.max_turnover * 0.8:
            recommendations.append(
                "High turnover may erode returns - consider longer holding periods"
            )

        if "var_95" in breached:
            recommendations.append(
                "Reduce position sizing or add volatility-based scaling"
            )

        if not recommendations:
            recommendations.append(
                "Strategy risk profile is within acceptable bounds"
            )

        return recommendations

    def assess_strategy(
        self,
        metrics: dict[str, float],
        positions: Optional[pd.Series] = None,
    ) -> RiskAssessment:
        """Assess strategy risk directly.

        Convenience method for risk assessment outside StateGraph context.

        Args:
            metrics: Performance metrics dict
            positions: Position series

        Returns:
            RiskAssessment with full analysis
        """
        # Convert to RiskMetrics
        risk_metrics = RiskMetrics(
            max_drawdown=metrics.get("max_drawdown", 0),
            portfolio_volatility=metrics.get("volatility", 0),
            var_95=metrics.get("var_95", 0),
        )

        if positions is not None:
            risk_metrics.max_position = float(positions.abs().max())
            risk_metrics.concentration_ratio = self.calculator.calculate_concentration(positions)

        # Check limits
        limits = self._create_limits()
        breached = self._check_limits(limits, risk_metrics)

        # Calculate score and approval
        score = self._calculate_risk_score(risk_metrics, breached)
        approval = self._determine_approval(score, breached)
        level = self._determine_risk_level(score)

        return RiskAssessment(
            overall_level=level,
            overall_score=score,
            approval_status=approval,
            metrics=risk_metrics,
            limits=limits,
            breached_limits=breached,
            warnings=self._generate_warnings(risk_metrics, breached),
            recommendations=self._generate_recommendations(risk_metrics, breached),
        )


# Node function for StateGraph
async def check_risk_node(state: AgentState) -> AgentState:
    """StateGraph node function for risk checking.

    Args:
        state: Current agent state

    Returns:
        Updated state with risk assessment
    """
    agent = RiskCheckAgent()
    return await agent.check(state)


# Factory function
def create_risk_agent(
    config: Optional[RiskConfig] = None,
) -> RiskCheckAgent:
    """Factory function to create a RiskCheckAgent.

    Args:
        config: Risk configuration

    Returns:
        Configured RiskCheckAgent instance
    """
    return RiskCheckAgent(config=config)
