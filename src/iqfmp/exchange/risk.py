"""Risk management module for IQFMP.

Provides risk control and monitoring:
- RiskController: Main risk management controller
- DrawdownMonitor: Monitor and alert on drawdown
- LossLimiter: Limit single and daily losses
- ConcentrationChecker: Check position concentration
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Optional


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


@dataclass
class RiskConfig:
    """Risk configuration parameters."""

    max_drawdown: Decimal = Decimal("0.20")  # 20%
    max_single_loss: Decimal = Decimal("0.02")  # 2%
    max_position_concentration: Decimal = Decimal("0.30")  # 30%
    daily_loss_limit: Decimal = Decimal("0.05")  # 5%
    emergency_close_threshold: Decimal = Decimal("0.25")  # 25%
    auto_reduce_enabled: bool = True
    auto_close_enabled: bool = True
    warning_threshold_ratio: Decimal = Decimal("0.5")  # Alert at 50% of limit

    @property
    def is_valid(self) -> bool:
        """Check if config is valid."""
        return (
            Decimal("0") < self.max_drawdown <= Decimal("1")
            and Decimal("0") < self.max_single_loss <= Decimal("1")
            and Decimal("0") < self.max_position_concentration <= Decimal("1")
            and Decimal("0") < self.daily_loss_limit <= Decimal("1")
        )


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
            List of drawdown alerts
        """
        alerts: list[DrawdownAlert] = []
        drawdown = self.current_drawdown

        if drawdown > self._max_drawdown:
            alerts.append(
                DrawdownAlert(
                    level=RiskLevel.CRITICAL,
                    current_drawdown=drawdown,
                    max_allowed=self._max_drawdown,
                    message=f"Max drawdown breached: {drawdown * 100:.1f}% > {self._max_drawdown * 100:.1f}%",
                )
            )
        elif drawdown > self._max_drawdown * Decimal("0.75"):
            alerts.append(
                DrawdownAlert(
                    level=RiskLevel.DANGER,
                    current_drawdown=drawdown,
                    max_allowed=self._max_drawdown,
                    message=f"Drawdown approaching limit: {drawdown * 100:.1f}%",
                )
            )
        elif drawdown > self._max_drawdown * Decimal("0.5"):
            alerts.append(
                DrawdownAlert(
                    level=RiskLevel.WARNING,
                    current_drawdown=drawdown,
                    max_allowed=self._max_drawdown,
                    message=f"Drawdown warning: {drawdown * 100:.1f}%",
                )
            )

        return alerts


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
            List of loss alerts
        """
        alerts: list[LossAlert] = []
        loss_pct = self.daily_loss_percent

        if loss_pct > self._daily_loss_limit:
            alerts.append(
                LossAlert(
                    level=RiskLevel.CRITICAL,
                    loss_amount=self._daily_loss,
                    loss_percent=loss_pct,
                    limit=self._daily_loss_limit,
                    message=f"Daily loss limit breached: {loss_pct * 100:.1f}%",
                )
            )
        elif loss_pct > self._daily_loss_limit * Decimal("0.75"):
            alerts.append(
                LossAlert(
                    level=RiskLevel.DANGER,
                    loss_amount=self._daily_loss,
                    loss_percent=loss_pct,
                    limit=self._daily_loss_limit,
                    message=f"Daily loss approaching limit: {loss_pct * 100:.1f}%",
                )
            )
        elif loss_pct > self._daily_loss_limit * Decimal("0.5"):
            alerts.append(
                LossAlert(
                    level=RiskLevel.WARNING,
                    loss_amount=self._daily_loss,
                    loss_percent=loss_pct,
                    limit=self._daily_loss_limit,
                    message=f"Daily loss warning: {loss_pct * 100:.1f}%",
                )
            )

        return alerts


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

        Args:
            positions: Position values by symbol
            total_equity: Total account equity

        Returns:
            List of concentration alerts
        """
        alerts: list[ConcentrationAlert] = []
        concentrations = self.check_concentration(positions, total_equity)

        for symbol, conc in concentrations.items():
            if conc > self._max_single:
                alerts.append(
                    ConcentrationAlert(
                        level=RiskLevel.CRITICAL,
                        symbol=symbol,
                        concentration=conc,
                        max_allowed=self._max_single,
                        message=f"{symbol} concentration breached: {conc * 100:.1f}%",
                    )
                )
            elif conc > self._max_single * Decimal("0.85"):
                alerts.append(
                    ConcentrationAlert(
                        level=RiskLevel.WARNING,
                        symbol=symbol,
                        concentration=conc,
                        max_allowed=self._max_single,
                        message=f"{symbol} concentration high: {conc * 100:.1f}%",
                    )
                )

        return alerts


# ==================== RiskController ====================


class RiskController:
    """Main risk management controller."""

    def __init__(
        self,
        config: RiskConfig,
        initial_equity: Decimal,
    ) -> None:
        """Initialize risk controller.

        Args:
            config: Risk configuration
            initial_equity: Initial account equity
        """
        self._config = config
        self._initial_equity = initial_equity
        self._current_equity = initial_equity

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

        # Position tracking
        self._positions: dict[str, Decimal] = {}

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


# ==================== Module Exports ====================


__all__ = [
    # Enums
    "RiskActionType",
    "RiskLevel",
    "RiskRuleType",
    # Models
    "ConcentrationAlert",
    "ConcentrationBreach",
    "DrawdownAlert",
    "LossAlert",
    "LossRecord",
    "RiskAction",
    "RiskConfig",
    "RiskRule",
    "RiskStatus",
    "RiskViolation",
    # Classes
    "ConcentrationChecker",
    "DrawdownMonitor",
    "LossLimiter",
    "RiskController",
]
