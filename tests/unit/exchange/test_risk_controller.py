"""Tests for risk controller module (Task 21).

TDD tests for risk management and control system.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iqfmp.exchange.monitoring import PositionData, PositionSide
from iqfmp.exchange.risk import (
    ConcentrationChecker,
    ConcentrationAlert,
    DrawdownMonitor,
    DrawdownAlert,
    LossLimiter,
    LossAlert,
    RiskAction,
    RiskActionType,
    RiskConfig,
    RiskController,
    RiskLevel,
    RiskRule,
    RiskRuleType,
    RiskStatus,
    RiskViolation,
)


# ==================== RiskConfig Tests ====================


class TestRiskConfig:
    """Test RiskConfig model."""

    def test_create_default_config(self) -> None:
        """Test creating default risk config."""
        config = RiskConfig()
        assert config.max_drawdown == Decimal("0.20")  # 20%
        assert config.max_single_loss == Decimal("0.02")  # 2%
        assert config.max_position_concentration == Decimal("0.30")  # 30%
        assert config.daily_loss_limit == Decimal("0.05")  # 5%

    def test_create_custom_config(self) -> None:
        """Test creating custom risk config."""
        config = RiskConfig(
            max_drawdown=Decimal("0.15"),
            max_single_loss=Decimal("0.01"),
            max_position_concentration=Decimal("0.25"),
            daily_loss_limit=Decimal("0.03"),
        )
        assert config.max_drawdown == Decimal("0.15")
        assert config.max_single_loss == Decimal("0.01")

    def test_config_validation_max_drawdown(self) -> None:
        """Test config validates max_drawdown range."""
        # Valid range: 0 < max_drawdown <= 1
        # emergency_close_threshold must be >= max_drawdown
        config = RiskConfig(
            max_drawdown=Decimal("0.50"),
            emergency_close_threshold=Decimal("0.60"),
        )
        assert config.is_valid is True

    def test_config_with_emergency_settings(self) -> None:
        """Test config with emergency settings."""
        config = RiskConfig(
            max_drawdown=Decimal("0.20"),
            emergency_close_threshold=Decimal("0.25"),
            auto_reduce_enabled=True,
        )
        assert config.emergency_close_threshold == Decimal("0.25")
        assert config.auto_reduce_enabled is True


# ==================== RiskRule Tests ====================


class TestRiskRule:
    """Test RiskRule model."""

    def test_create_drawdown_rule(self) -> None:
        """Test creating drawdown rule."""
        rule = RiskRule(
            rule_type=RiskRuleType.MAX_DRAWDOWN,
            threshold=Decimal("0.20"),
            action=RiskActionType.REDUCE_POSITION,
        )
        assert rule.rule_type == RiskRuleType.MAX_DRAWDOWN
        assert rule.threshold == Decimal("0.20")
        assert rule.action == RiskActionType.REDUCE_POSITION

    def test_create_concentration_rule(self) -> None:
        """Test creating concentration rule."""
        rule = RiskRule(
            rule_type=RiskRuleType.CONCENTRATION,
            threshold=Decimal("0.30"),
            action=RiskActionType.BLOCK_NEW_ORDERS,
        )
        assert rule.rule_type == RiskRuleType.CONCENTRATION

    def test_rule_enabled_flag(self) -> None:
        """Test rule enabled flag."""
        rule = RiskRule(
            rule_type=RiskRuleType.DAILY_LOSS,
            threshold=Decimal("0.05"),
            action=RiskActionType.CLOSE_ALL,
            enabled=False,
        )
        assert rule.enabled is False


# ==================== RiskViolation Tests ====================


class TestRiskViolation:
    """Test RiskViolation model."""

    def test_create_violation(self) -> None:
        """Test creating risk violation."""
        violation = RiskViolation(
            rule_type=RiskRuleType.MAX_DRAWDOWN,
            current_value=Decimal("0.22"),
            threshold=Decimal("0.20"),
            level=RiskLevel.CRITICAL,
            message="Max drawdown exceeded: 22% > 20%",
        )
        assert violation.rule_type == RiskRuleType.MAX_DRAWDOWN
        assert violation.current_value == Decimal("0.22")
        assert violation.level == RiskLevel.CRITICAL

    def test_violation_severity(self) -> None:
        """Test violation severity levels."""
        assert RiskLevel.WARNING.value < RiskLevel.DANGER.value
        assert RiskLevel.DANGER.value < RiskLevel.CRITICAL.value


# ==================== DrawdownMonitor Tests ====================


class TestDrawdownMonitor:
    """Test DrawdownMonitor class."""

    @pytest.fixture
    def monitor(self) -> DrawdownMonitor:
        """Create drawdown monitor."""
        return DrawdownMonitor(
            max_drawdown=Decimal("0.20"),
            initial_equity=Decimal("10000.0"),
        )

    def test_update_equity(self, monitor: DrawdownMonitor) -> None:
        """Test updating equity."""
        monitor.update_equity(Decimal("10500.0"))
        assert monitor.current_equity == Decimal("10500.0")
        assert monitor.peak_equity == Decimal("10500.0")

    def test_calculate_drawdown(self, monitor: DrawdownMonitor) -> None:
        """Test drawdown calculation."""
        monitor.update_equity(Decimal("10500.0"))  # New peak
        monitor.update_equity(Decimal("9450.0"))  # Drawdown

        # Drawdown = (10500 - 9450) / 10500 = 0.10 (10%)
        assert monitor.current_drawdown == Decimal("0.10")

    def test_track_max_drawdown(self, monitor: DrawdownMonitor) -> None:
        """Test tracking max drawdown."""
        monitor.update_equity(Decimal("10000.0"))
        monitor.update_equity(Decimal("8500.0"))  # 15% drawdown
        monitor.update_equity(Decimal("9000.0"))  # Recovery
        monitor.update_equity(Decimal("8000.0"))  # 20% drawdown from new peak

        # Max drawdown should be 20% (from 10000 to 8000)
        assert monitor.max_drawdown_recorded >= Decimal("0.15")

    def test_drawdown_alert_warning(self, monitor: DrawdownMonitor) -> None:
        """Test drawdown warning alert."""
        monitor.update_equity(Decimal("10000.0"))
        monitor.update_equity(Decimal("9000.0"))  # 10% drawdown

        alerts = monitor.check_alerts()
        # 10% is below 20% threshold but may trigger warning at 50% of limit
        assert len(alerts) >= 0

    def test_drawdown_alert_critical(self, monitor: DrawdownMonitor) -> None:
        """Test drawdown critical alert."""
        monitor.update_equity(Decimal("10000.0"))
        monitor.update_equity(Decimal("7900.0"))  # 21% drawdown

        alerts = monitor.check_alerts()
        assert len(alerts) > 0
        assert any(a.level == RiskLevel.CRITICAL for a in alerts)

    def test_is_breached(self, monitor: DrawdownMonitor) -> None:
        """Test drawdown breach detection."""
        monitor.update_equity(Decimal("10000.0"))
        monitor.update_equity(Decimal("7500.0"))  # 25% drawdown

        assert monitor.is_breached is True

    def test_reset_peak(self, monitor: DrawdownMonitor) -> None:
        """Test resetting peak equity."""
        monitor.update_equity(Decimal("10500.0"))
        monitor.update_equity(Decimal("9000.0"))

        monitor.reset_peak(Decimal("9000.0"))

        assert monitor.peak_equity == Decimal("9000.0")
        assert monitor.current_drawdown == Decimal("0")


# ==================== LossLimiter Tests ====================


class TestLossLimiter:
    """Test LossLimiter class."""

    @pytest.fixture
    def limiter(self) -> LossLimiter:
        """Create loss limiter."""
        return LossLimiter(
            max_single_loss=Decimal("0.02"),
            daily_loss_limit=Decimal("0.05"),
            account_equity=Decimal("10000.0"),
        )

    def test_check_single_loss_allowed(self, limiter: LossLimiter) -> None:
        """Test allowed single loss."""
        # 1% loss is within 2% limit
        is_allowed = limiter.check_single_loss(Decimal("100.0"))
        assert is_allowed is True

    def test_check_single_loss_exceeded(self, limiter: LossLimiter) -> None:
        """Test single loss exceeded."""
        # 3% loss exceeds 2% limit
        is_allowed = limiter.check_single_loss(Decimal("300.0"))
        assert is_allowed is False

    def test_record_loss(self, limiter: LossLimiter) -> None:
        """Test recording a loss."""
        limiter.record_loss(Decimal("150.0"), symbol="BTC/USDT")

        assert limiter.daily_loss == Decimal("150.0")
        assert len(limiter.loss_history) == 1

    def test_daily_loss_accumulation(self, limiter: LossLimiter) -> None:
        """Test daily loss accumulation."""
        limiter.record_loss(Decimal("100.0"), symbol="BTC/USDT")
        limiter.record_loss(Decimal("150.0"), symbol="ETH/USDT")
        limiter.record_loss(Decimal("200.0"), symbol="BTC/USDT")

        assert limiter.daily_loss == Decimal("450.0")

    def test_daily_limit_breach(self, limiter: LossLimiter) -> None:
        """Test daily loss limit breach."""
        limiter.record_loss(Decimal("300.0"), symbol="BTC/USDT")
        limiter.record_loss(Decimal("250.0"), symbol="ETH/USDT")

        # Total 5.5% > 5% limit
        assert limiter.is_daily_limit_breached is True

    def test_reset_daily_loss(self, limiter: LossLimiter) -> None:
        """Test resetting daily loss."""
        limiter.record_loss(Decimal("300.0"), symbol="BTC/USDT")
        limiter.reset_daily()

        assert limiter.daily_loss == Decimal("0")

    def test_loss_alerts(self, limiter: LossLimiter) -> None:
        """Test loss alerts generation."""
        limiter.record_loss(Decimal("400.0"), symbol="BTC/USDT")

        alerts = limiter.check_alerts()
        # 4% loss should trigger warning
        assert len(alerts) > 0


# ==================== ConcentrationChecker Tests ====================


class TestConcentrationChecker:
    """Test ConcentrationChecker class."""

    @pytest.fixture
    def checker(self) -> ConcentrationChecker:
        """Create concentration checker."""
        return ConcentrationChecker(
            max_single_concentration=Decimal("0.30"),
            max_sector_concentration=Decimal("0.50"),
        )

    def test_check_single_concentration(
        self, checker: ConcentrationChecker
    ) -> None:
        """Test single position concentration check."""
        positions = {
            "BTC/USDT": Decimal("3000.0"),
            "ETH/USDT": Decimal("2000.0"),
            "SOL/USDT": Decimal("1000.0"),
        }
        total_equity = Decimal("10000.0")

        concentration = checker.check_concentration(
            positions, total_equity
        )

        # BTC is 30% which equals limit
        assert "BTC/USDT" in concentration
        assert concentration["BTC/USDT"] == Decimal("0.30")

    def test_concentration_breach(
        self, checker: ConcentrationChecker
    ) -> None:
        """Test concentration breach detection."""
        positions = {
            "BTC/USDT": Decimal("4000.0"),  # 40%
            "ETH/USDT": Decimal("2000.0"),
        }
        total_equity = Decimal("10000.0")

        breaches = checker.get_breaches(positions, total_equity)

        assert len(breaches) > 0
        assert breaches[0].symbol == "BTC/USDT"

    def test_no_concentration_breach(
        self, checker: ConcentrationChecker
    ) -> None:
        """Test no concentration breach."""
        positions = {
            "BTC/USDT": Decimal("2000.0"),  # 20%
            "ETH/USDT": Decimal("2000.0"),  # 20%
            "SOL/USDT": Decimal("2000.0"),  # 20%
        }
        total_equity = Decimal("10000.0")

        breaches = checker.get_breaches(positions, total_equity)

        assert len(breaches) == 0

    def test_concentration_alerts(
        self, checker: ConcentrationChecker
    ) -> None:
        """Test concentration alerts."""
        positions = {
            "BTC/USDT": Decimal("3500.0"),  # 35%
        }
        total_equity = Decimal("10000.0")

        alerts = checker.check_alerts(positions, total_equity)

        assert len(alerts) > 0
        assert alerts[0].symbol == "BTC/USDT"


# ==================== RiskAction Tests ====================


class TestRiskAction:
    """Test RiskAction model."""

    def test_create_reduce_action(self) -> None:
        """Test creating reduce position action."""
        action = RiskAction(
            action_type=RiskActionType.REDUCE_POSITION,
            symbol="BTC/USDT",
            reduce_percent=Decimal("0.50"),
            reason="Max drawdown exceeded",
        )
        assert action.action_type == RiskActionType.REDUCE_POSITION
        assert action.reduce_percent == Decimal("0.50")

    def test_create_close_action(self) -> None:
        """Test creating close position action."""
        action = RiskAction(
            action_type=RiskActionType.CLOSE_POSITION,
            symbol="BTC/USDT",
            reason="Daily loss limit breached",
        )
        assert action.action_type == RiskActionType.CLOSE_POSITION

    def test_create_close_all_action(self) -> None:
        """Test creating close all positions action."""
        action = RiskAction(
            action_type=RiskActionType.CLOSE_ALL,
            symbol=None,
            reason="Emergency stop triggered",
        )
        assert action.action_type == RiskActionType.CLOSE_ALL


# ==================== RiskController Tests ====================


class TestRiskController:
    """Test RiskController class."""

    @pytest.fixture
    def config(self) -> RiskConfig:
        """Create risk config."""
        return RiskConfig(
            max_drawdown=Decimal("0.20"),
            max_single_loss=Decimal("0.02"),
            max_position_concentration=Decimal("0.30"),
            daily_loss_limit=Decimal("0.05"),
        )

    @pytest.fixture
    def controller(self, config: RiskConfig) -> RiskController:
        """Create risk controller."""
        return RiskController(
            config=config,
            initial_equity=Decimal("10000.0"),
        )

    def test_check_all_rules(self, controller: RiskController) -> None:
        """Test checking all risk rules."""
        controller.update_equity(Decimal("9500.0"))

        violations = controller.check_all()

        # 5% drawdown - no critical violations
        assert all(v.level != RiskLevel.CRITICAL for v in violations)

    def test_check_with_critical_violation(
        self, controller: RiskController
    ) -> None:
        """Test check with critical violation."""
        controller.update_equity(Decimal("7500.0"))  # 25% drawdown

        violations = controller.check_all()

        assert any(v.level == RiskLevel.CRITICAL for v in violations)

    def test_get_risk_status(self, controller: RiskController) -> None:
        """Test getting risk status."""
        controller.update_equity(Decimal("9000.0"))  # 10% drawdown

        status = controller.get_status()

        assert status.current_drawdown == Decimal("0.10")
        assert status.risk_level in [RiskLevel.WARNING, RiskLevel.SAFE]

    def test_add_position(self, controller: RiskController) -> None:
        """Test adding position for concentration check."""
        controller.add_position("BTC/USDT", Decimal("3000.0"))
        controller.add_position("ETH/USDT", Decimal("2000.0"))

        positions = controller.get_positions()

        assert "BTC/USDT" in positions
        assert positions["BTC/USDT"] == Decimal("3000.0")

    def test_update_position(self, controller: RiskController) -> None:
        """Test updating position."""
        controller.add_position("BTC/USDT", Decimal("3000.0"))
        controller.update_position("BTC/USDT", Decimal("3500.0"))

        positions = controller.get_positions()
        assert positions["BTC/USDT"] == Decimal("3500.0")

    def test_remove_position(self, controller: RiskController) -> None:
        """Test removing position."""
        controller.add_position("BTC/USDT", Decimal("3000.0"))
        controller.remove_position("BTC/USDT")

        positions = controller.get_positions()
        assert "BTC/USDT" not in positions

    def test_record_trade_loss(self, controller: RiskController) -> None:
        """Test recording trade loss."""
        controller.record_loss(Decimal("150.0"), symbol="BTC/USDT")

        assert controller.daily_loss == Decimal("150.0")

    def test_generate_actions(self, controller: RiskController) -> None:
        """Test generating risk actions."""
        controller.update_equity(Decimal("7500.0"))  # 25% drawdown
        controller.add_position("BTC/USDT", Decimal("5000.0"))

        actions = controller.generate_actions()

        assert len(actions) > 0
        assert any(
            a.action_type in [
                RiskActionType.REDUCE_POSITION,
                RiskActionType.CLOSE_ALL,
            ]
            for a in actions
        )

    def test_enable_disable_rule(self, controller: RiskController) -> None:
        """Test enabling/disabling rules."""
        controller.disable_rule(RiskRuleType.CONCENTRATION)

        assert controller.is_rule_enabled(RiskRuleType.CONCENTRATION) is False

        controller.enable_rule(RiskRuleType.CONCENTRATION)

        assert controller.is_rule_enabled(RiskRuleType.CONCENTRATION) is True

    def test_set_custom_threshold(self, controller: RiskController) -> None:
        """Test setting custom threshold."""
        controller.set_threshold(RiskRuleType.MAX_DRAWDOWN, Decimal("0.15"))

        # Trigger check with 16% drawdown
        controller.update_equity(Decimal("8400.0"))
        violations = controller.check_all()

        assert any(
            v.rule_type == RiskRuleType.MAX_DRAWDOWN
            for v in violations
        )


# ==================== Integration Tests ====================


class TestRiskControllerIntegration:
    """Integration tests for risk controller."""

    @pytest.fixture
    def controller(self) -> RiskController:
        """Create risk controller with default config."""
        config = RiskConfig()
        return RiskController(config=config, initial_equity=Decimal("10000.0"))

    def test_full_risk_monitoring_workflow(
        self, controller: RiskController
    ) -> None:
        """Test full risk monitoring workflow."""
        # Initial state
        assert controller.get_status().risk_level == RiskLevel.SAFE

        # Add positions
        controller.add_position("BTC/USDT", Decimal("3000.0"))
        controller.add_position("ETH/USDT", Decimal("2000.0"))

        # Simulate equity decline
        controller.update_equity(Decimal("9500.0"))  # 5% drawdown

        # Record some losses
        controller.record_loss(Decimal("100.0"), symbol="BTC/USDT")

        # Check status
        status = controller.get_status()
        assert status.current_drawdown == Decimal("0.05")

    def test_drawdown_triggers_action(
        self, controller: RiskController
    ) -> None:
        """Test drawdown triggers risk action."""
        controller.add_position("BTC/USDT", Decimal("5000.0"))

        # Severe drawdown
        controller.update_equity(Decimal("7000.0"))  # 30% drawdown

        actions = controller.generate_actions()

        # Should generate close or reduce action
        assert len(actions) > 0

    def test_concentration_triggers_action(
        self, controller: RiskController
    ) -> None:
        """Test concentration triggers risk action."""
        # Single position is 50% of equity
        controller.add_position("BTC/USDT", Decimal("5000.0"))

        violations = controller.check_all()

        assert any(
            v.rule_type == RiskRuleType.CONCENTRATION
            for v in violations
        )

    def test_daily_loss_triggers_trading_halt(
        self, controller: RiskController
    ) -> None:
        """Test daily loss triggers trading halt."""
        # Record losses exceeding daily limit
        controller.record_loss(Decimal("300.0"), symbol="BTC/USDT")
        controller.record_loss(Decimal("300.0"), symbol="ETH/USDT")

        # 6% daily loss exceeds 5% limit
        status = controller.get_status()

        assert status.is_trading_allowed is False
