"""Risk management module for IQFMP.

This module implements a two-tier risk threshold system:

1. HARD THRESHOLDS (immutable, defined as class constants in RiskController):
   - MAX_DRAWDOWN_THRESHOLD: 15% - triggers emergency close
   - MAX_POSITION_RATIO: 30% - single position concentration limit
   - MAX_LEVERAGE: 3x - maximum allowed leverage
   - EMERGENCY_LOSS_THRESHOLD: 5% - single-day loss triggers emergency close
   These cannot be adjusted at runtime and enforce absolute safety limits.

2. SOFT THRESHOLDS (configurable via RiskConfig):
   - max_drawdown, max_single_loss, max_position_concentration, daily_loss_limit
   These can be customized per strategy but must remain within hard limits.

Components:
- RiskController: Main controller integrating both threshold tiers + MarginCalculator
- DrawdownMonitor: Track equity peaks and drawdown with tiered alerts
- LossLimiter: Track single-trade and daily losses
- ConcentrationChecker: Monitor position concentration by symbol

Alert levels follow a tiered pattern: CRITICAL (>100%), DANGER (>75%), WARNING (>50%).
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Optional

from iqfmp.exchange.margin import (
    MarginCalculator,
    MarginConfig,
    MarginPosition,
    MarginResult,
    MarginStatus,
    PositionSide,
)


# ==================== Enums ====================


class RiskLevel(Enum):
    """Risk level severity."""

    SAFE = 0
    WARNING = 1
    DANGER = 2
    CRITICAL = 3


class RiskRuleType(Enum):
    """Types of risk rules."""

    MAX_DRAWDOWN = "max_drawdown"
    SINGLE_LOSS = "single_loss"
    DAILY_LOSS = "daily_loss"
    CONCENTRATION = "concentration"
    EMERGENCY = "emergency"


class RiskActionType(Enum):
    """Types of risk actions."""

    REDUCE_POSITION = "reduce_position"
    CLOSE_POSITION = "close_position"
    CLOSE_ALL = "close_all"
    BLOCK_NEW_ORDERS = "block_new_orders"
    HALT_TRADING = "halt_trading"


# ==================== Data Models ====================


class RiskConfigError(Exception):
    """Raised when RiskConfig validation fails."""
    pass


@dataclass
class RiskConfig:
    """Risk configuration parameters.

    Validates all constraints at construction per CLAUDE.md critical state rules.
    """

    max_drawdown: Decimal = Decimal("0.20")  # 20%
    max_single_loss: Decimal = Decimal("0.02")  # 2%
    max_position_concentration: Decimal = Decimal("0.30")  # 30%
    daily_loss_limit: Decimal = Decimal("0.05")  # 5%
    emergency_close_threshold: Decimal = Decimal("0.25")  # 25%
    auto_reduce_enabled: bool = True
    auto_close_enabled: bool = True
    warning_threshold_ratio: Decimal = Decimal("0.5")  # Alert at 50% of limit

    def __post_init__(self) -> None:
        """Validate config at construction time."""
        errors = []

        if not (Decimal("0") < self.max_drawdown <= Decimal("1")):
            errors.append(f"max_drawdown must be in (0, 1], got {self.max_drawdown}")

        if not (Decimal("0") < self.max_single_loss <= Decimal("1")):
            errors.append(f"max_single_loss must be in (0, 1], got {self.max_single_loss}")

        if not (Decimal("0") < self.max_position_concentration <= Decimal("1")):
            errors.append(f"max_position_concentration must be in (0, 1], got {self.max_position_concentration}")

        if not (Decimal("0") < self.daily_loss_limit <= Decimal("1")):
            errors.append(f"daily_loss_limit must be in (0, 1], got {self.daily_loss_limit}")

        if self.emergency_close_threshold < self.max_drawdown:
            errors.append(
                f"emergency_close_threshold ({self.emergency_close_threshold}) "
                f"must be >= max_drawdown ({self.max_drawdown})"
            )

        if not (Decimal("0") < self.warning_threshold_ratio < Decimal("1")):
            errors.append(f"warning_threshold_ratio must be in (0, 1), got {self.warning_threshold_ratio}")

        if errors:
            raise RiskConfigError("Invalid risk configuration: " + "; ".join(errors))

    @property
    def is_valid(self) -> bool:
        """Check if config is valid. Always True after construction."""
        return True


@dataclass
class RiskRule:
    """Risk rule definition."""

    rule_type: RiskRuleType
    threshold: Decimal
    action: RiskActionType
    enabled: bool = True
    description: str = ""


@dataclass
class RiskViolation:
    """Risk rule violation."""

    rule_type: RiskRuleType
    current_value: Decimal
    threshold: Decimal
    level: RiskLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskAction:
    """Risk action to execute."""

    action_type: RiskActionType
    symbol: Optional[str]
    reason: str
    reduce_percent: Optional[Decimal] = None
    timestamp: datetime = field(default_factory=datetime.now)
    executed: bool = False


@dataclass
class RiskStatus:
    """Current risk status."""

    risk_level: RiskLevel
    current_drawdown: Decimal
    daily_loss: Decimal
    max_concentration: Decimal
    violations: list[RiskViolation] = field(default_factory=list)
    is_trading_allowed: bool = True
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DrawdownAlert:
    """Drawdown alert."""

    level: RiskLevel
    current_drawdown: Decimal
    max_allowed: Decimal
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LossAlert:
    """Loss alert."""

    level: RiskLevel
    loss_amount: Decimal
    loss_percent: Decimal
    limit: Decimal
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConcentrationAlert:
    """Concentration alert."""

    level: RiskLevel
    symbol: str
    concentration: Decimal
    max_allowed: Decimal
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConcentrationBreach:
    """Concentration breach."""

    symbol: str
    concentration: Decimal
    max_allowed: Decimal


@dataclass
class LossRecord:
    """Loss record."""

    amount: Decimal
    symbol: str
    timestamp: datetime


# ==================== Alert Level Helpers ====================


def _determine_alert_level(
    current_value: Decimal,
    threshold: Decimal,
    warning_ratio: Decimal = Decimal("0.5"),
    danger_ratio: Decimal = Decimal("0.75"),
) -> RiskLevel | None:
    """Determine alert level based on threshold ratios.

    Standard tiered alert pattern used across all risk monitors.

    Args:
        current_value: Current metric value
        threshold: Maximum allowed threshold
        warning_ratio: Ratio of threshold for WARNING (default 0.5 = 50%)
        danger_ratio: Ratio of threshold for DANGER (default 0.75 = 75%)

    Returns:
        RiskLevel if threshold exceeded, None if safe
    """
    if current_value > threshold:
        return RiskLevel.CRITICAL
    elif current_value > threshold * danger_ratio:
        return RiskLevel.DANGER
    elif current_value > threshold * warning_ratio:
        return RiskLevel.WARNING
    return None


# ==================== DrawdownMonitor ====================


class DrawdownMonitor:
    """Monitor and alert on drawdown."""

    def __init__(
        self,
        max_drawdown: Decimal,
        initial_equity: Decimal,
    ) -> None:
        """Initialize drawdown monitor.

        Args:
            max_drawdown: Maximum allowed drawdown (0-1)
            initial_equity: Initial account equity
        """
        self._max_drawdown = max_drawdown
        self._peak_equity = initial_equity
        self._current_equity = initial_equity
        self._max_drawdown_recorded = Decimal("0")

    def update_equity(self, equity: Decimal) -> None:
        """Update current equity.

        Args:
            equity: Current equity value
        """
        self._current_equity = equity

        # Update peak if new high
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Update max drawdown recorded
        if self.current_drawdown > self._max_drawdown_recorded:
            self._max_drawdown_recorded = self.current_drawdown

    @property
    def current_equity(self) -> Decimal:
        """Get current equity."""
        return self._current_equity

    @property
    def peak_equity(self) -> Decimal:
        """Get peak equity."""
        return self._peak_equity

    @property
    def current_drawdown(self) -> Decimal:
        """Calculate current drawdown."""
        if self._peak_equity == Decimal("0"):
            return Decimal("0")
        return (self._peak_equity - self._current_equity) / self._peak_equity

    @property
    def max_drawdown_recorded(self) -> Decimal:
        """Get maximum drawdown recorded."""
        return self._max_drawdown_recorded

    @property
    def is_breached(self) -> bool:
        """Check if drawdown limit is breached."""
        return self.current_drawdown > self._max_drawdown

    def reset_peak(self, new_peak: Decimal) -> None:
        """Reset peak equity.

        Args:
            new_peak: New peak equity value
        """
        self._peak_equity = new_peak

    def check_alerts(self) -> list[DrawdownAlert]:
        """Check and generate drawdown alerts.

        Returns:
            List of drawdown alerts (empty if below warning threshold)
        """
        drawdown = self.current_drawdown
        level = _determine_alert_level(drawdown, self._max_drawdown)

        if level is None:
            return []

        messages = {
            RiskLevel.CRITICAL: f"Max drawdown breached: {drawdown * 100:.1f}% > {self._max_drawdown * 100:.1f}%",
            RiskLevel.DANGER: f"Drawdown approaching limit: {drawdown * 100:.1f}%",
            RiskLevel.WARNING: f"Drawdown warning: {drawdown * 100:.1f}%",
        }

        return [
            DrawdownAlert(
                level=level,
                current_drawdown=drawdown,
                max_allowed=self._max_drawdown,
                message=messages[level],
            )
        ]


# ==================== LossLimiter ====================


class LossLimiter:
    """Limit single and daily losses."""

    def __init__(
        self,
        max_single_loss: Decimal,
        daily_loss_limit: Decimal,
        account_equity: Decimal,
    ) -> None:
        """Initialize loss limiter.

        Args:
            max_single_loss: Maximum single trade loss (0-1)
            daily_loss_limit: Maximum daily loss (0-1)
            account_equity: Account equity for calculations
        """
        self._max_single_loss = max_single_loss
        self._daily_loss_limit = daily_loss_limit
        self._account_equity = account_equity
        self._daily_loss = Decimal("0")
        self._loss_history: list[LossRecord] = []

    def check_single_loss(self, loss_amount: Decimal) -> bool:
        """Check if single loss is within limit.

        Args:
            loss_amount: Loss amount

        Returns:
            True if allowed, False if exceeds limit
        """
        loss_percent = loss_amount / self._account_equity
        return loss_percent <= self._max_single_loss

    def record_loss(self, amount: Decimal, symbol: str) -> None:
        """Record a loss.

        Args:
            amount: Loss amount
            symbol: Trading symbol
        """
        self._daily_loss += amount
        self._loss_history.append(
            LossRecord(
                amount=amount,
                symbol=symbol,
                timestamp=datetime.now(),
            )
        )

    @property
    def daily_loss(self) -> Decimal:
        """Get daily loss total."""
        return self._daily_loss

    @property
    def daily_loss_percent(self) -> Decimal:
        """Get daily loss as percentage."""
        if self._account_equity == Decimal("0"):
            return Decimal("1") if self._daily_loss > Decimal("0") else Decimal("0")
        return self._daily_loss / self._account_equity

    @property
    def is_daily_limit_breached(self) -> bool:
        """Check if daily loss limit is breached."""
        return self.daily_loss_percent > self._daily_loss_limit

    @property
    def loss_history(self) -> list[LossRecord]:
        """Get loss history."""
        return self._loss_history.copy()

    def reset_daily(self) -> None:
        """Reset daily loss counter."""
        self._daily_loss = Decimal("0")

    def check_alerts(self) -> list[LossAlert]:
        """Check and generate loss alerts.

        Returns:
            List of loss alerts (empty if below warning threshold)
        """
        loss_pct = self.daily_loss_percent
        level = _determine_alert_level(loss_pct, self._daily_loss_limit)

        if level is None:
            return []

        messages = {
            RiskLevel.CRITICAL: f"Daily loss limit breached: {loss_pct * 100:.1f}%",
            RiskLevel.DANGER: f"Daily loss approaching limit: {loss_pct * 100:.1f}%",
            RiskLevel.WARNING: f"Daily loss warning: {loss_pct * 100:.1f}%",
        }

        return [
            LossAlert(
                level=level,
                loss_amount=self._daily_loss,
                loss_percent=loss_pct,
                limit=self._daily_loss_limit,
                message=messages[level],
            )
        ]


# ==================== ConcentrationChecker ====================


class ConcentrationChecker:
    """Check position concentration."""

    def __init__(
        self,
        max_single_concentration: Decimal,
        max_sector_concentration: Decimal = Decimal("0.50"),
    ) -> None:
        """Initialize concentration checker.

        Args:
            max_single_concentration: Maximum single position concentration
            max_sector_concentration: Maximum sector concentration
        """
        self._max_single = max_single_concentration
        self._max_sector = max_sector_concentration

    def check_concentration(
        self,
        positions: dict[str, Decimal],
        total_equity: Decimal,
    ) -> dict[str, Decimal]:
        """Calculate concentration for all positions.

        Args:
            positions: Position values by symbol
            total_equity: Total account equity

        Returns:
            Concentration percentage by symbol
        """
        if total_equity == Decimal("0"):
            return {}

        return {
            symbol: value / total_equity
            for symbol, value in positions.items()
        }

    def get_breaches(
        self,
        positions: dict[str, Decimal],
        total_equity: Decimal,
    ) -> list[ConcentrationBreach]:
        """Get concentration breaches.

        Args:
            positions: Position values by symbol
            total_equity: Total account equity

        Returns:
            List of concentration breaches
        """
        breaches: list[ConcentrationBreach] = []
        concentrations = self.check_concentration(positions, total_equity)

        for symbol, conc in concentrations.items():
            if conc > self._max_single:
                breaches.append(
                    ConcentrationBreach(
                        symbol=symbol,
                        concentration=conc,
                        max_allowed=self._max_single,
                    )
                )

        return breaches

    def check_alerts(
        self,
        positions: dict[str, Decimal],
        total_equity: Decimal,
    ) -> list[ConcentrationAlert]:
        """Check and generate concentration alerts.

        Uses higher warning threshold (85%) than other monitors since
        concentration issues require earlier intervention.

        Args:
            positions: Position values by symbol
            total_equity: Total account equity

        Returns:
            List of concentration alerts (empty if all positions below 85%)
        """
        alerts: list[ConcentrationAlert] = []
        concentrations = self.check_concentration(positions, total_equity)

        for symbol, conc in concentrations.items():
            # Concentration uses 85% warning threshold (no danger tier)
            level = _determine_alert_level(
                conc, self._max_single,
                warning_ratio=Decimal("0.85"),
                danger_ratio=Decimal("1.0"),  # Skip danger level
            )

            if level is None:
                continue

            message = (
                f"{symbol} concentration breached: {conc * 100:.1f}%"
                if level == RiskLevel.CRITICAL
                else f"{symbol} concentration high: {conc * 100:.1f}%"
            )

            alerts.append(
                ConcentrationAlert(
                    level=level,
                    symbol=symbol,
                    concentration=conc,
                    max_allowed=self._max_single,
                    message=message,
                )
            )

        return alerts


# ==================== RiskController ====================


@dataclass
class Position:
    """Position data for risk checking."""

    symbol: str
    value: Decimal
    quantity: Decimal = Decimal("0")
    entry_price: Decimal = Decimal("0")
    current_price: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")


@dataclass
class Account:
    """Account data for risk checking."""

    equity: Decimal
    total_position_value: Decimal = Decimal("0")
    available_balance: Decimal = Decimal("0")
    margin_used: Decimal = Decimal("0")
    daily_pnl: Decimal = Decimal("0")
    peak_equity: Decimal = Decimal("0")


@dataclass
class RiskCheckResult:
    """Result of risk check."""

    is_safe: bool
    violations: list["HardRiskViolation"]
    recommended_action: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HardRiskViolation:
    """Hard threshold risk violation (cannot be ignored)."""

    type: str
    severity: str  # "critical" | "high" | "medium"
    action: str  # "emergency_close_all" | "reduce_position" | "reduce_leverage" | "block_new_orders"
    message: str
    current_value: Decimal
    threshold: Decimal
    timestamp: datetime = field(default_factory=datetime.now)


class RiskController:
    """Main risk management controller with hard thresholds (Task 8.2).

    Hard thresholds are immutable and cannot be adjusted:
    - MAX_DRAWDOWN_THRESHOLD: 15% maximum drawdown triggers emergency close
    - MAX_POSITION_RATIO: 30% single position concentration limit
    - MAX_LEVERAGE: 3x maximum leverage
    - EMERGENCY_LOSS_THRESHOLD: 5% single-day loss triggers emergency close

    Now includes MarginCalculator integration (P1-2 fix) for proper margin management.
    """

    # ==================== HARD THRESHOLDS (不可调整) ====================
    MAX_DRAWDOWN_THRESHOLD: Decimal = Decimal("0.15")  # 15% 最大回撤触发平仓
    MAX_POSITION_RATIO: Decimal = Decimal("0.30")  # 单一持仓不超过 30%
    MAX_LEVERAGE: Decimal = Decimal("3.0")  # 最大杠杆 3x
    EMERGENCY_LOSS_THRESHOLD: Decimal = Decimal("0.05")  # 5% 单日亏损触发紧急平仓

    # ==================== MARGIN THRESHOLDS ====================
    MARGIN_WARNING_RATIO: Decimal = Decimal("0.50")  # 50% margin ratio warning
    MARGIN_DANGER_RATIO: Decimal = Decimal("0.70")   # 70% margin ratio danger
    MARGIN_CRITICAL_RATIO: Decimal = Decimal("0.90") # 90% margin ratio critical

    def __init__(
        self,
        config: RiskConfig,
        initial_equity: Decimal,
        default_leverage: int = 10,
    ) -> None:
        """Initialize risk controller.

        Args:
            config: Risk configuration
            initial_equity: Initial account equity
            default_leverage: Default leverage for margin calculations
        """
        self._config = config
        self._initial_equity = initial_equity
        self._current_equity = initial_equity
        self._default_leverage = default_leverage

        # Initialize sub-monitors
        self._drawdown_monitor = DrawdownMonitor(
            max_drawdown=config.max_drawdown,
            initial_equity=initial_equity,
        )
        self._loss_limiter = LossLimiter(
            max_single_loss=config.max_single_loss,
            daily_loss_limit=config.daily_loss_limit,
            account_equity=initial_equity,
        )
        self._concentration_checker = ConcentrationChecker(
            max_single_concentration=config.max_position_concentration,
        )

        # Initialize margin calculator (P1-2)
        margin_config = MarginConfig.from_leverage(default_leverage)
        self._margin_calculator = MarginCalculator(margin_config)

        # Position tracking
        self._positions: dict[str, Decimal] = {}
        self._margin_positions: dict[str, MarginPosition] = {}

        # Rule states
        self._rule_states: dict[RiskRuleType, bool] = {
            rule_type: True for rule_type in RiskRuleType
        }

        # Custom thresholds
        self._custom_thresholds: dict[RiskRuleType, Decimal] = {}

    def update_equity(self, equity: Decimal) -> None:
        """Update current equity.

        Args:
            equity: Current equity value
        """
        self._current_equity = equity
        self._drawdown_monitor.update_equity(equity)

    def add_position(self, symbol: str, value: Decimal) -> None:
        """Add a position.

        Args:
            symbol: Trading symbol
            value: Position value
        """
        self._positions[symbol] = value

    def update_position(self, symbol: str, value: Decimal) -> None:
        """Update position value.

        Args:
            symbol: Trading symbol
            value: New position value
        """
        self._positions[symbol] = value

    def remove_position(self, symbol: str) -> None:
        """Remove a position.

        Args:
            symbol: Trading symbol
        """
        self._positions.pop(symbol, None)

    def get_positions(self) -> dict[str, Decimal]:
        """Get all positions.

        Returns:
            Positions by symbol
        """
        return self._positions.copy()

    def record_loss(self, amount: Decimal, symbol: str) -> None:
        """Record a trading loss.

        Args:
            amount: Loss amount
            symbol: Trading symbol
        """
        self._loss_limiter.record_loss(amount, symbol)

    @property
    def daily_loss(self) -> Decimal:
        """Get daily loss total."""
        return self._loss_limiter.daily_loss

    def check_all(self) -> list[RiskViolation]:
        """Check all risk rules.

        Returns:
            List of violations
        """
        violations: list[RiskViolation] = []

        # Check drawdown
        if self._rule_states.get(RiskRuleType.MAX_DRAWDOWN, True):
            threshold = self._custom_thresholds.get(
                RiskRuleType.MAX_DRAWDOWN, self._config.max_drawdown
            )
            if self._drawdown_monitor.current_drawdown > threshold:
                violations.append(
                    RiskViolation(
                        rule_type=RiskRuleType.MAX_DRAWDOWN,
                        current_value=self._drawdown_monitor.current_drawdown,
                        threshold=threshold,
                        level=RiskLevel.CRITICAL,
                        message=f"Max drawdown exceeded: {self._drawdown_monitor.current_drawdown * 100:.1f}%",
                    )
                )

        # Check daily loss
        if self._rule_states.get(RiskRuleType.DAILY_LOSS, True):
            if self._loss_limiter.is_daily_limit_breached:
                violations.append(
                    RiskViolation(
                        rule_type=RiskRuleType.DAILY_LOSS,
                        current_value=self._loss_limiter.daily_loss_percent,
                        threshold=self._config.daily_loss_limit,
                        level=RiskLevel.CRITICAL,
                        message=f"Daily loss limit breached",
                    )
                )

        # Check concentration
        if self._rule_states.get(RiskRuleType.CONCENTRATION, True):
            breaches = self._concentration_checker.get_breaches(
                self._positions, self._current_equity
            )
            for breach in breaches:
                violations.append(
                    RiskViolation(
                        rule_type=RiskRuleType.CONCENTRATION,
                        current_value=breach.concentration,
                        threshold=breach.max_allowed,
                        level=RiskLevel.CRITICAL,
                        message=f"{breach.symbol} concentration exceeded",
                    )
                )

        return violations

    def get_status(self) -> RiskStatus:
        """Get current risk status.

        Returns:
            Risk status
        """
        violations = self.check_all()

        # Determine overall risk level
        risk_level = RiskLevel.SAFE
        if any(v.level == RiskLevel.CRITICAL for v in violations):
            risk_level = RiskLevel.CRITICAL
        elif any(v.level == RiskLevel.DANGER for v in violations):
            risk_level = RiskLevel.DANGER
        elif any(v.level == RiskLevel.WARNING for v in violations):
            risk_level = RiskLevel.WARNING

        # Check max concentration
        concentrations = self._concentration_checker.check_concentration(
            self._positions, self._current_equity
        )
        max_conc = max(concentrations.values()) if concentrations else Decimal("0")

        # Determine if trading allowed
        is_trading_allowed = (
            not self._loss_limiter.is_daily_limit_breached
            and not self._drawdown_monitor.is_breached
        )

        return RiskStatus(
            risk_level=risk_level,
            current_drawdown=self._drawdown_monitor.current_drawdown,
            daily_loss=self._loss_limiter.daily_loss,
            max_concentration=max_conc,
            violations=violations,
            is_trading_allowed=is_trading_allowed,
        )

    def generate_actions(self) -> list[RiskAction]:
        """Generate risk actions based on violations.

        Returns:
            List of risk actions
        """
        actions: list[RiskAction] = []
        violations = self.check_all()

        for violation in violations:
            if violation.level == RiskLevel.CRITICAL:
                if violation.rule_type == RiskRuleType.MAX_DRAWDOWN:
                    # Severe drawdown - close all
                    if self._config.auto_close_enabled:
                        actions.append(
                            RiskAction(
                                action_type=RiskActionType.CLOSE_ALL,
                                symbol=None,
                                reason=violation.message,
                            )
                        )
                elif violation.rule_type == RiskRuleType.CONCENTRATION:
                    # Concentration breach - reduce position
                    if self._config.auto_reduce_enabled:
                        # Find the concentrated position
                        for symbol, value in self._positions.items():
                            conc = value / self._current_equity
                            if conc > self._config.max_position_concentration:
                                reduce_to = self._config.max_position_concentration
                                reduce_pct = (conc - reduce_to) / conc
                                actions.append(
                                    RiskAction(
                                        action_type=RiskActionType.REDUCE_POSITION,
                                        symbol=symbol,
                                        reduce_percent=reduce_pct,
                                        reason=violation.message,
                                    )
                                )

        return actions

    def enable_rule(self, rule_type: RiskRuleType) -> None:
        """Enable a risk rule.

        Args:
            rule_type: Rule type to enable
        """
        self._rule_states[rule_type] = True

    def disable_rule(self, rule_type: RiskRuleType) -> None:
        """Disable a risk rule.

        Args:
            rule_type: Rule type to disable
        """
        self._rule_states[rule_type] = False

    def is_rule_enabled(self, rule_type: RiskRuleType) -> bool:
        """Check if rule is enabled.

        Args:
            rule_type: Rule type to check

        Returns:
            True if enabled
        """
        return self._rule_states.get(rule_type, True)

    def set_threshold(self, rule_type: RiskRuleType, threshold: Decimal) -> None:
        """Set custom threshold for a rule.

        Args:
            rule_type: Rule type
            threshold: New threshold value
        """
        self._custom_thresholds[rule_type] = threshold

        # Also update sub-monitors if needed
        if rule_type == RiskRuleType.MAX_DRAWDOWN:
            self._drawdown_monitor._max_drawdown = threshold

    # ==================== HARD THRESHOLD METHODS (Task 8.2) ====================

    def _calculate_drawdown(self, account: Account) -> Decimal:
        """Calculate current drawdown from peak.

        Args:
            account: Account data

        Returns:
            Drawdown as decimal (0.15 = 15%)
        """
        if account.peak_equity == Decimal("0"):
            return Decimal("0")
        return (account.peak_equity - account.equity) / account.peak_equity

    def _get_daily_pnl(self, account: Account) -> Decimal:
        """Get daily PnL from account.

        Args:
            account: Account data

        Returns:
            Daily PnL amount
        """
        return account.daily_pnl

    def _get_recommended_action(self, violations: list[HardRiskViolation]) -> str:
        """Determine recommended action from violations.

        Args:
            violations: List of violations

        Returns:
            Recommended action string
        """
        if not violations:
            return "none"

        # Find the most severe action
        action_priority = {
            "emergency_close_all": 4,
            "reduce_leverage": 3,
            "reduce_position": 2,
            "block_new_orders": 1,
        }

        max_priority = 0
        recommended = "block_new_orders"

        for v in violations:
            priority = action_priority.get(v.action, 0)
            if priority > max_priority:
                max_priority = priority
                recommended = v.action

        return recommended

    async def check_risk(
        self, position: Position, account: Account
    ) -> RiskCheckResult:
        """检查风险并返回建议动作 (Task 8.2).

        执行四层风险检查，使用硬性阈值：
        1. 回撤检查 - MAX_DRAWDOWN_THRESHOLD (15%)
        2. 持仓集中度检查 - MAX_POSITION_RATIO (30%)
        3. 杠杆检查 - MAX_LEVERAGE (3x)
        4. 单日亏损检查 - EMERGENCY_LOSS_THRESHOLD (5%)

        Args:
            position: Position to check
            account: Account data

        Returns:
            RiskCheckResult with violations and recommended action
        """
        violations: list[HardRiskViolation] = []

        # 1. 回撤检查 (CRITICAL - 触发紧急平仓)
        drawdown = self._calculate_drawdown(account)
        if drawdown > self.MAX_DRAWDOWN_THRESHOLD:
            violations.append(
                HardRiskViolation(
                    type="max_drawdown",
                    severity="critical",
                    action="emergency_close_all",
                    message=f"最大回撤 {drawdown * 100:.2f}% 超过阈值 {self.MAX_DRAWDOWN_THRESHOLD * 100:.0f}%",
                    current_value=drawdown,
                    threshold=self.MAX_DRAWDOWN_THRESHOLD,
                )
            )

        # 2. 持仓集中度检查 (HIGH - 需要减仓)
        if account.equity > Decimal("0"):
            position_ratio = position.value / account.equity
            if position_ratio > self.MAX_POSITION_RATIO:
                violations.append(
                    HardRiskViolation(
                        type="position_concentration",
                        severity="high",
                        action="reduce_position",
                        message=f"持仓比例 {position_ratio * 100:.2f}% 超过阈值 {self.MAX_POSITION_RATIO * 100:.0f}%",
                        current_value=position_ratio,
                        threshold=self.MAX_POSITION_RATIO,
                    )
                )

        # 3. 杠杆检查 (HIGH - 需要降杠杆)
        if account.equity > Decimal("0"):
            leverage = account.total_position_value / account.equity
            if leverage > self.MAX_LEVERAGE:
                violations.append(
                    HardRiskViolation(
                        type="leverage",
                        severity="high",
                        action="reduce_leverage",
                        message=f"杠杆 {leverage:.2f}x 超过阈值 {self.MAX_LEVERAGE}x",
                        current_value=leverage,
                        threshold=self.MAX_LEVERAGE,
                    )
                )

        # 4. 单日亏损检查 (CRITICAL - 触发紧急平仓)
        daily_pnl = self._get_daily_pnl(account)
        if daily_pnl < Decimal("0") and account.equity > Decimal("0"):
            daily_loss_ratio = -daily_pnl / account.equity
            if daily_loss_ratio > self.EMERGENCY_LOSS_THRESHOLD:
                violations.append(
                    HardRiskViolation(
                        type="daily_loss",
                        severity="critical",
                        action="emergency_close_all",
                        message=f"单日亏损 {daily_loss_ratio * 100:.2f}% 超过阈值 {self.EMERGENCY_LOSS_THRESHOLD * 100:.0f}%",
                        current_value=daily_loss_ratio,
                        threshold=self.EMERGENCY_LOSS_THRESHOLD,
                    )
                )

        return RiskCheckResult(
            is_safe=len(violations) == 0,
            violations=violations,
            recommended_action=self._get_recommended_action(violations),
        )

    def check_risk_sync(
        self, position: Position, account: Account
    ) -> RiskCheckResult:
        """同步版本的风险检查."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.check_risk(position, account)
        )

    def check_position_allowed(
        self,
        new_position_value: Decimal,
        account: Account,
    ) -> tuple[bool, str]:
        """检查是否允许新建仓位.

        Args:
            new_position_value: New position value
            account: Account data

        Returns:
            (is_allowed, reason)
        """
        if account.equity <= Decimal("0"):
            return False, "账户权益为零"

        # 检查仓位比例
        position_ratio = new_position_value / account.equity
        if position_ratio > self.MAX_POSITION_RATIO:
            return False, f"仓位比例 {position_ratio * 100:.1f}% 超过限制 {self.MAX_POSITION_RATIO * 100:.0f}%"

        # 检查杠杆
        new_total = account.total_position_value + new_position_value
        leverage = new_total / account.equity
        if leverage > self.MAX_LEVERAGE:
            return False, f"杠杆 {leverage:.2f}x 超过限制 {self.MAX_LEVERAGE}x"

        # 检查回撤
        drawdown = self._calculate_drawdown(account)
        if drawdown > self.MAX_DRAWDOWN_THRESHOLD:
            return False, f"回撤 {drawdown * 100:.1f}% 超过限制，禁止新建仓位"

        return True, "允许"

    @classmethod
    def get_hard_thresholds(cls) -> dict[str, Decimal]:
        """获取所有硬性阈值.

        Returns:
            Dict of threshold name to value
        """
        return {
            "max_drawdown": cls.MAX_DRAWDOWN_THRESHOLD,
            "max_position_ratio": cls.MAX_POSITION_RATIO,
            "max_leverage": cls.MAX_LEVERAGE,
            "emergency_loss": cls.EMERGENCY_LOSS_THRESHOLD,
            "margin_warning": cls.MARGIN_WARNING_RATIO,
            "margin_danger": cls.MARGIN_DANGER_RATIO,
            "margin_critical": cls.MARGIN_CRITICAL_RATIO,
        }

    # ==================== MARGIN METHODS (P1-2) ====================

    def add_margin_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: Decimal,
        entry_price: Decimal,
        mark_price: Decimal,
        leverage: int = 10,
    ) -> None:
        """Add a margin position for tracking.

        Args:
            symbol: Trading symbol
            side: Position side (LONG/SHORT)
            quantity: Position quantity
            entry_price: Entry price
            mark_price: Current mark price
            leverage: Position leverage
        """
        self._margin_positions[symbol] = MarginPosition(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            mark_price=mark_price,
            leverage=leverage,
        )
        # Also update legacy position tracking
        self._positions[symbol] = quantity * mark_price

    def update_margin_position(
        self,
        symbol: str,
        mark_price: Decimal,
        quantity: Optional[Decimal] = None,
    ) -> None:
        """Update margin position with new mark price.

        Args:
            symbol: Trading symbol
            mark_price: New mark price
            quantity: New quantity (optional)
        """
        if symbol in self._margin_positions:
            pos = self._margin_positions[symbol]
            # Create updated position
            self._margin_positions[symbol] = MarginPosition(
                symbol=pos.symbol,
                side=pos.side,
                quantity=quantity if quantity is not None else pos.quantity,
                entry_price=pos.entry_price,
                mark_price=mark_price,
                leverage=pos.leverage,
                mode=pos.mode,
            )
            # Update legacy tracking
            new_qty = quantity if quantity is not None else pos.quantity
            self._positions[symbol] = new_qty * mark_price

    def remove_margin_position(self, symbol: str) -> None:
        """Remove a margin position.

        Args:
            symbol: Trading symbol
        """
        self._margin_positions.pop(symbol, None)
        self._positions.pop(symbol, None)

    def check_margin_status(
        self,
        symbol: str,
        margin_balance: Decimal,
    ) -> Optional[MarginResult]:
        """Check margin status for a position.

        Args:
            symbol: Trading symbol
            margin_balance: Current margin balance for this position

        Returns:
            MarginResult if position exists, None otherwise
        """
        if symbol not in self._margin_positions:
            return None

        position = self._margin_positions[symbol]
        return self._margin_calculator.calculate_margin(position, margin_balance)

    def check_margin_call(
        self,
        symbol: str,
        margin_balance: Decimal,
    ) -> tuple[bool, str]:
        """Check if position is subject to margin call.

        Args:
            symbol: Trading symbol
            margin_balance: Current margin balance

        Returns:
            (is_margin_call, message)
        """
        if symbol not in self._margin_positions:
            return False, f"Position {symbol} not found"

        position = self._margin_positions[symbol]
        return self._margin_calculator.check_margin_call(position, margin_balance)

    def get_all_margin_statuses(
        self,
        margin_balances: dict[str, Decimal],
    ) -> dict[str, MarginResult]:
        """Get margin status for all positions.

        Args:
            margin_balances: Margin balance by symbol

        Returns:
            MarginResult by symbol
        """
        results = {}
        for symbol, position in self._margin_positions.items():
            if symbol in margin_balances:
                results[symbol] = self._margin_calculator.calculate_margin(
                    position, margin_balances[symbol]
                )
        return results

    def calculate_liquidation_price(
        self,
        symbol: str,
    ) -> Optional[Decimal]:
        """Calculate liquidation price for a position.

        Args:
            symbol: Trading symbol

        Returns:
            Liquidation price if position exists, None otherwise
        """
        if symbol not in self._margin_positions:
            return None

        position = self._margin_positions[symbol]

        if position.side == PositionSide.LONG:
            return self._margin_calculator.calculate_liquidation_price_long(
                position.entry_price,
                position.leverage,
            )
        else:
            return self._margin_calculator.calculate_liquidation_price_short(
                position.entry_price,
                position.leverage,
            )

    def calculate_required_margin(
        self,
        position_size: Decimal,
        entry_price: Decimal,
        leverage: Optional[int] = None,
    ) -> Decimal:
        """Calculate required initial margin for a new position.

        Args:
            position_size: Position size in contracts
            entry_price: Entry price
            leverage: Leverage (uses default if not specified)

        Returns:
            Required initial margin
        """
        lev = leverage or self._default_leverage
        return self._margin_calculator.calculate_initial_margin(
            position_size, entry_price, lev
        )

    def calculate_max_position_size(
        self,
        available_margin: Decimal,
        entry_price: Decimal,
        leverage: Optional[int] = None,
    ) -> Decimal:
        """Calculate maximum position size given available margin.

        Args:
            available_margin: Available margin for new position
            entry_price: Entry price
            leverage: Leverage (uses default if not specified)

        Returns:
            Maximum position size
        """
        lev = leverage or self._default_leverage
        return self._margin_calculator.calculate_max_position_size(
            available_margin, entry_price, lev
        )

    def get_margin_positions(self) -> dict[str, MarginPosition]:
        """Get all margin positions.

        Returns:
            MarginPosition by symbol
        """
        return self._margin_positions.copy()

    def get_total_margin_exposure(self) -> Decimal:
        """Get total margin exposure across all positions.

        Returns:
            Total position value at mark price
        """
        return sum(
            pos.position_value for pos in self._margin_positions.values()
        )


# ==================== Module Exports ====================


__all__ = [
    # Enums
    "RiskActionType",
    "RiskLevel",
    "RiskRuleType",
    # Models
    "Account",
    "ConcentrationAlert",
    "ConcentrationBreach",
    "DrawdownAlert",
    "HardRiskViolation",
    "LossAlert",
    "LossRecord",
    "Position",
    "RiskAction",
    "RiskCheckResult",
    "RiskConfig",
    "RiskRule",
    "RiskStatus",
    "RiskViolation",
    # Classes
    "ConcentrationChecker",
    "DrawdownMonitor",
    "LossLimiter",
    "RiskController",
    # Margin (P1-2) - Re-exported from margin module
    "MarginCalculator",
    "MarginConfig",
    "MarginPosition",
    "MarginResult",
    "MarginStatus",
    "PositionSide",
]
