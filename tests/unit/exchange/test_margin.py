"""Tests for Margin Calculation Engine.

Tests verify that MarginCalculator correctly:
1. Calculates initial margin
2. Calculates maintenance margin
3. Calculates liquidation prices
4. Determines margin status
5. Handles edge cases
"""

from decimal import Decimal

import pytest

from iqfmp.exchange.margin import (
    MarginCalculator,
    MarginConfig,
    MarginPosition,
    MarginResult,
    MarginStatus,
    MarginMode,
    PositionSide,
)


class TestMarginConfig:
    """Tests for MarginConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MarginConfig()
        assert config.initial_margin_rate == Decimal("0.10")
        assert config.maintenance_margin_rate == Decimal("0.04")
        assert config.taker_fee_rate == Decimal("0.0004")
        assert config.maker_fee_rate == Decimal("0.0002")

    def test_from_leverage_10x(self):
        """Test config creation from 10x leverage."""
        config = MarginConfig.from_leverage(10)
        assert config.initial_margin_rate == Decimal("0.1")
        assert config.maintenance_margin_rate == Decimal("0.025")

    def test_from_leverage_5x(self):
        """Test config creation from 5x leverage."""
        config = MarginConfig.from_leverage(5)
        assert config.initial_margin_rate == Decimal("0.2")
        assert config.maintenance_margin_rate == Decimal("0.01")

    def test_from_leverage_20x(self):
        """Test config creation from 20x leverage."""
        config = MarginConfig.from_leverage(20)
        assert config.initial_margin_rate == Decimal("0.05")
        assert config.maintenance_margin_rate == Decimal("0.04")

    def test_from_leverage_invalid_zero(self):
        """Test error on zero leverage."""
        with pytest.raises(ValueError, match="positive"):
            MarginConfig.from_leverage(0)

    def test_from_leverage_invalid_negative(self):
        """Test error on negative leverage."""
        with pytest.raises(ValueError, match="positive"):
            MarginConfig.from_leverage(-5)

    def test_from_leverage_invalid_too_high(self):
        """Test error on leverage over 125x."""
        with pytest.raises(ValueError, match="Maximum leverage"):
            MarginConfig.from_leverage(200)


class TestMarginPosition:
    """Tests for MarginPosition data class."""

    def test_position_value_long(self):
        """Test position value calculation for long."""
        position = MarginPosition(
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            quantity=Decimal("1.0"),
            entry_price=Decimal("2000"),
            mark_price=Decimal("2100"),
            leverage=10,
        )
        assert position.position_value == Decimal("2100")
        assert position.notional_value == Decimal("2000")

    def test_unrealized_pnl_long_profit(self):
        """Test unrealized PnL for profitable long."""
        position = MarginPosition(
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            quantity=Decimal("1.0"),
            entry_price=Decimal("2000"),
            mark_price=Decimal("2100"),
            leverage=10,
        )
        assert position.unrealized_pnl == Decimal("100")

    def test_unrealized_pnl_long_loss(self):
        """Test unrealized PnL for losing long."""
        position = MarginPosition(
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            quantity=Decimal("1.0"),
            entry_price=Decimal("2000"),
            mark_price=Decimal("1900"),
            leverage=10,
        )
        assert position.unrealized_pnl == Decimal("-100")

    def test_unrealized_pnl_short_profit(self):
        """Test unrealized PnL for profitable short."""
        position = MarginPosition(
            symbol="ETHUSDT",
            side=PositionSide.SHORT,
            quantity=Decimal("1.0"),
            entry_price=Decimal("2000"),
            mark_price=Decimal("1900"),
            leverage=10,
        )
        assert position.unrealized_pnl == Decimal("100")

    def test_unrealized_pnl_short_loss(self):
        """Test unrealized PnL for losing short."""
        position = MarginPosition(
            symbol="ETHUSDT",
            side=PositionSide.SHORT,
            quantity=Decimal("1.0"),
            entry_price=Decimal("2000"),
            mark_price=Decimal("2100"),
            leverage=10,
        )
        assert position.unrealized_pnl == Decimal("-100")


class TestMarginCalculator:
    """Tests for MarginCalculator."""

    @pytest.fixture
    def calculator(self) -> MarginCalculator:
        """Create default calculator."""
        return MarginCalculator()

    @pytest.fixture
    def calculator_10x(self) -> MarginCalculator:
        """Create 10x leverage calculator."""
        config = MarginConfig.from_leverage(10)
        return MarginCalculator(config)

    def test_calculate_initial_margin(self, calculator):
        """Test initial margin calculation."""
        margin = calculator.calculate_initial_margin(
            position_size=Decimal("1.0"),
            entry_price=Decimal("2000"),
            leverage=10,
        )
        # 1 * 2000 / 10 = 200
        assert margin == Decimal("200.00000000")

    def test_calculate_initial_margin_fractional(self, calculator):
        """Test initial margin with fractional position."""
        margin = calculator.calculate_initial_margin(
            position_size=Decimal("0.5"),
            entry_price=Decimal("3000"),
            leverage=5,
        )
        # 0.5 * 3000 / 5 = 300
        assert margin == Decimal("300.00000000")

    def test_calculate_initial_margin_high_leverage(self, calculator):
        """Test initial margin with high leverage."""
        margin = calculator.calculate_initial_margin(
            position_size=Decimal("1.0"),
            entry_price=Decimal("2000"),
            leverage=100,
        )
        # 1 * 2000 / 100 = 20
        assert margin == Decimal("20.00000000")

    def test_calculate_initial_margin_invalid_leverage(self, calculator):
        """Test error on invalid leverage."""
        with pytest.raises(ValueError):
            calculator.calculate_initial_margin(
                position_size=Decimal("1.0"),
                entry_price=Decimal("2000"),
                leverage=0,
            )

    def test_calculate_maintenance_margin(self, calculator):
        """Test maintenance margin calculation."""
        margin = calculator.calculate_maintenance_margin(
            position_size=Decimal("1.0"),
            mark_price=Decimal("2000"),
        )
        # 1 * 2000 * 0.04 = 80
        assert margin == Decimal("80.00000000")

    def test_calculate_margin_ratio_healthy(self, calculator):
        """Test margin ratio in healthy zone."""
        ratio = calculator.calculate_margin_ratio(
            margin_balance=Decimal("200"),
            maintenance_margin=Decimal("50"),
        )
        # 50 / 200 = 0.25
        assert ratio == Decimal("0.2500")

    def test_calculate_margin_ratio_warning(self, calculator):
        """Test margin ratio in warning zone."""
        ratio = calculator.calculate_margin_ratio(
            margin_balance=Decimal("100"),
            maintenance_margin=Decimal("55"),
        )
        # 55 / 100 = 0.55
        assert ratio == Decimal("0.5500")

    def test_calculate_margin_ratio_critical(self, calculator):
        """Test margin ratio in critical zone."""
        ratio = calculator.calculate_margin_ratio(
            margin_balance=Decimal("100"),
            maintenance_margin=Decimal("95"),
        )
        # 95 / 100 = 0.95
        assert ratio == Decimal("0.9500")

    def test_calculate_margin_ratio_zero_balance(self, calculator):
        """Test margin ratio with zero balance (liquidation)."""
        ratio = calculator.calculate_margin_ratio(
            margin_balance=Decimal("0"),
            maintenance_margin=Decimal("50"),
        )
        assert ratio == Decimal("1")

    def test_liquidation_price_long_10x(self, calculator_10x):
        """Test liquidation price for 10x long."""
        liq_price = calculator_10x.calculate_liquidation_price_long(
            entry_price=Decimal("2000"),
            leverage=10,
        )
        # LP = 2000 * (1 - 0.1 + 0.025) = 2000 * 0.925 = 1850
        assert liq_price == Decimal("1850.00")

    def test_liquidation_price_long_5x(self):
        """Test liquidation price for 5x long."""
        config = MarginConfig.from_leverage(5)
        calculator = MarginCalculator(config)

        liq_price = calculator.calculate_liquidation_price_long(
            entry_price=Decimal("2000"),
            leverage=5,
        )
        # LP = 2000 * (1 - 0.2 + 0.01) = 2000 * 0.81 = 1620
        assert liq_price == Decimal("1620.00")

    def test_liquidation_price_short_10x(self, calculator_10x):
        """Test liquidation price for 10x short."""
        liq_price = calculator_10x.calculate_liquidation_price_short(
            entry_price=Decimal("2000"),
            leverage=10,
        )
        # LP = 2000 * (1 + 0.1 - 0.025) = 2000 * 1.075 = 2150
        assert liq_price == Decimal("2150.00")

    def test_liquidation_price_short_5x(self):
        """Test liquidation price for 5x short."""
        config = MarginConfig.from_leverage(5)
        calculator = MarginCalculator(config)

        liq_price = calculator.calculate_liquidation_price_short(
            entry_price=Decimal("2000"),
            leverage=5,
        )
        # LP = 2000 * (1 + 0.2 - 0.01) = 2000 * 1.19 = 2380
        assert liq_price == Decimal("2380.00")

    def test_liquidation_price_invalid_leverage(self, calculator):
        """Test error on invalid leverage for liquidation."""
        with pytest.raises(ValueError):
            calculator.calculate_liquidation_price_long(
                entry_price=Decimal("2000"),
                leverage=-1,
            )

    def test_get_margin_status_healthy(self, calculator):
        """Test healthy margin status."""
        status = calculator.get_margin_status(Decimal("0.30"))
        assert status == MarginStatus.HEALTHY

    def test_get_margin_status_warning(self, calculator):
        """Test warning margin status."""
        status = calculator.get_margin_status(Decimal("0.55"))
        assert status == MarginStatus.WARNING

    def test_get_margin_status_danger(self, calculator):
        """Test danger margin status."""
        status = calculator.get_margin_status(Decimal("0.75"))
        assert status == MarginStatus.DANGER

    def test_get_margin_status_critical(self, calculator):
        """Test critical margin status."""
        status = calculator.get_margin_status(Decimal("0.92"))
        assert status == MarginStatus.CRITICAL

    def test_get_margin_status_liquidation(self, calculator):
        """Test liquidation margin status."""
        status = calculator.get_margin_status(Decimal("1.0"))
        assert status == MarginStatus.LIQUIDATION


class TestMarginCalculatorFullCalculation:
    """Tests for full margin calculation."""

    @pytest.fixture
    def calculator(self) -> MarginCalculator:
        """Create calculator with 10x config."""
        config = MarginConfig.from_leverage(10)
        return MarginCalculator(config)

    @pytest.fixture
    def long_position(self) -> MarginPosition:
        """Create sample long position."""
        return MarginPosition(
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            quantity=Decimal("1.0"),
            entry_price=Decimal("2000"),
            mark_price=Decimal("2050"),
            leverage=10,
        )

    @pytest.fixture
    def short_position(self) -> MarginPosition:
        """Create sample short position."""
        return MarginPosition(
            symbol="ETHUSDT",
            side=PositionSide.SHORT,
            quantity=Decimal("1.0"),
            entry_price=Decimal("2000"),
            mark_price=Decimal("1950"),
            leverage=10,
        )

    def test_calculate_margin_long_healthy(self, calculator, long_position):
        """Test full margin calculation for healthy long."""
        result = calculator.calculate_margin(
            position=long_position,
            margin_balance=Decimal("250"),  # Healthy margin
        )

        assert isinstance(result, MarginResult)
        assert result.initial_margin == Decimal("200.00000000")
        assert result.status == MarginStatus.HEALTHY
        assert result.is_safe is True

    def test_calculate_margin_long_warning(self, calculator, long_position):
        """Test full margin calculation for warning long."""
        result = calculator.calculate_margin(
            position=long_position,
            margin_balance=Decimal("100"),  # Warning zone
        )

        assert result.status in (MarginStatus.WARNING, MarginStatus.DANGER)

    def test_calculate_margin_short_healthy(self, calculator, short_position):
        """Test full margin calculation for healthy short."""
        result = calculator.calculate_margin(
            position=short_position,
            margin_balance=Decimal("250"),
        )

        assert result.status == MarginStatus.HEALTHY
        # Short liquidation price should be above entry
        assert result.liquidation_price > short_position.entry_price

    def test_calculate_margin_long_liquidation_price_direction(
        self, calculator, long_position
    ):
        """Test that long liquidation price is below entry."""
        result = calculator.calculate_margin(
            position=long_position,
            margin_balance=Decimal("200"),
        )

        # Long liquidation should be below entry price
        assert result.liquidation_price < long_position.entry_price

    def test_margin_result_distance_to_liquidation(self, calculator, long_position):
        """Test distance to liquidation calculation."""
        result = calculator.calculate_margin(
            position=long_position,
            margin_balance=Decimal("200"),
        )

        # Distance should be positive and less than 1
        assert result.distance_to_liquidation > Decimal("0")
        assert result.distance_to_liquidation < Decimal("1")


class TestMarginCall:
    """Tests for margin call detection."""

    @pytest.fixture
    def calculator(self) -> MarginCalculator:
        """Create calculator."""
        config = MarginConfig.from_leverage(10)
        return MarginCalculator(config)

    @pytest.fixture
    def position(self) -> MarginPosition:
        """Create test position."""
        return MarginPosition(
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            quantity=Decimal("1.0"),
            entry_price=Decimal("2000"),
            mark_price=Decimal("1900"),  # 5% loss
            leverage=10,
        )

    def test_check_margin_call_healthy(self, calculator, position):
        """Test no margin call for healthy position."""
        is_call, message = calculator.check_margin_call(
            position=position,
            margin_balance=Decimal("300"),  # Healthy margin
        )

        assert is_call is False
        assert "Healthy" in message

    def test_check_margin_call_warning(self, calculator, position):
        """Test warning message but no call."""
        is_call, message = calculator.check_margin_call(
            position=position,
            margin_balance=Decimal("100"),
        )

        # Warning may or may not trigger margin call
        assert "WARNING" in message or "Healthy" in message

    def test_check_margin_call_critical(self, calculator, position):
        """Test margin call for critical position."""
        is_call, message = calculator.check_margin_call(
            position=position,
            margin_balance=Decimal("55"),  # Very low margin
        )

        assert is_call is True
        assert "CRITICAL" in message or "DANGER" in message or "liquidation" in message.lower()


class TestMaxPositionSize:
    """Tests for max position size calculation."""

    @pytest.fixture
    def calculator(self) -> MarginCalculator:
        """Create calculator."""
        return MarginCalculator()

    def test_calculate_max_position_size_basic(self, calculator):
        """Test basic max position size calculation."""
        max_size = calculator.calculate_max_position_size(
            available_margin=Decimal("1000"),
            entry_price=Decimal("2000"),
            leverage=10,
        )
        # (1000 * 10) / 2000 = 5
        assert max_size == Decimal("5.000")

    def test_calculate_max_position_size_low_leverage(self, calculator):
        """Test max size with low leverage."""
        max_size = calculator.calculate_max_position_size(
            available_margin=Decimal("1000"),
            entry_price=Decimal("2000"),
            leverage=2,
        )
        # (1000 * 2) / 2000 = 1
        assert max_size == Decimal("1.000")

    def test_calculate_max_position_size_high_leverage(self, calculator):
        """Test max size with high leverage."""
        max_size = calculator.calculate_max_position_size(
            available_margin=Decimal("100"),
            entry_price=Decimal("2000"),
            leverage=50,
        )
        # (100 * 50) / 2000 = 2.5
        assert max_size == Decimal("2.500")

    def test_calculate_max_position_size_invalid_price(self, calculator):
        """Test error on zero price."""
        with pytest.raises(ValueError):
            calculator.calculate_max_position_size(
                available_margin=Decimal("1000"),
                entry_price=Decimal("0"),
                leverage=10,
            )

    def test_calculate_max_position_size_invalid_leverage(self, calculator):
        """Test error on invalid leverage."""
        with pytest.raises(ValueError):
            calculator.calculate_max_position_size(
                available_margin=Decimal("1000"),
                entry_price=Decimal("2000"),
                leverage=-1,
            )


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def calculator(self) -> MarginCalculator:
        """Create calculator."""
        return MarginCalculator()

    def test_very_small_position(self, calculator):
        """Test with very small position size."""
        margin = calculator.calculate_initial_margin(
            position_size=Decimal("0.001"),
            entry_price=Decimal("2000"),
            leverage=10,
        )
        assert margin > Decimal("0")

    def test_very_large_position(self, calculator):
        """Test with very large position size."""
        margin = calculator.calculate_initial_margin(
            position_size=Decimal("1000000"),
            entry_price=Decimal("2000"),
            leverage=10,
        )
        assert margin == Decimal("200000000.00000000")

    def test_high_price_asset(self, calculator):
        """Test with high price asset (like BTC)."""
        margin = calculator.calculate_initial_margin(
            position_size=Decimal("0.01"),
            entry_price=Decimal("100000"),  # $100k BTC
            leverage=10,
        )
        # 0.01 * 100000 / 10 = 100
        assert margin == Decimal("100.00000000")

    def test_low_price_asset(self, calculator):
        """Test with low price asset."""
        margin = calculator.calculate_initial_margin(
            position_size=Decimal("10000"),
            entry_price=Decimal("0.001"),  # Very low price
            leverage=10,
        )
        # 10000 * 0.001 / 10 = 1
        assert margin == Decimal("1.00000000")
