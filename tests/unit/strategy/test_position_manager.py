"""Tests for Position Manager (Task 16)."""

from datetime import datetime, timedelta
from typing import Any

import pytest
import pandas as pd
import numpy as np

from iqfmp.strategy.position import (
    Position,
    PositionSide,
    PositionStatus,
    PositionManager,
    PositionConfig,
    # Position Sizing
    PositionSizer,
    KellySizer,
    FixedSizer,
    RiskParitySizer,
    # Stop Loss
    StopLoss,
    PriceStopLoss,
    PercentStopLoss,
    TrailingStopLoss,
    TimeStopLoss,
    # Take Profit
    TakeProfit,
    FixedTakeProfit,
    TrailingTakeProfit,
    # Exceptions
    InvalidPositionError,
    InsufficientFundsError,
)


# ============ Position Tests ============

class TestPosition:
    """Tests for Position dataclass."""

    def test_create_long_position(self) -> None:
        """Test creating a long position."""
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
        )
        assert pos.symbol == "BTCUSDT"
        assert pos.side == PositionSide.LONG
        assert pos.entry_price == 50000.0
        assert pos.quantity == 0.1

    def test_create_short_position(self) -> None:
        """Test creating a short position."""
        pos = Position(
            symbol="ETHUSDT",
            side=PositionSide.SHORT,
            entry_price=3000.0,
            quantity=1.0,
        )
        assert pos.side == PositionSide.SHORT

    def test_position_pnl_long(self) -> None:
        """Test P&L calculation for long position."""
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
        )
        # Price goes up
        pnl = pos.calculate_pnl(55000.0)
        assert pnl == pytest.approx(500.0)  # (55000-50000) * 0.1

    def test_position_pnl_short(self) -> None:
        """Test P&L calculation for short position."""
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            entry_price=50000.0,
            quantity=0.1,
        )
        # Price goes down (profit for short)
        pnl = pos.calculate_pnl(45000.0)
        assert pnl == pytest.approx(500.0)  # (50000-45000) * 0.1

    def test_position_return_pct(self) -> None:
        """Test return percentage calculation."""
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
        )
        return_pct = pos.calculate_return_pct(55000.0)
        assert return_pct == pytest.approx(0.1)  # 10%


# ============ Position Sizing Tests ============

class TestKellySizer:
    """Tests for Kelly Criterion position sizing."""

    def test_basic_kelly(self) -> None:
        """Test basic Kelly calculation."""
        sizer = KellySizer()
        # win_rate=0.6, win_loss_ratio=2.0
        # Kelly = (0.6 * 2.0 - 0.4) / 2.0 = 0.4
        size = sizer.calculate(
            capital=100000.0,
            win_rate=0.6,
            win_loss_ratio=2.0,
        )
        assert size == pytest.approx(40000.0)

    def test_kelly_with_fraction(self) -> None:
        """Test Kelly with fraction (half-Kelly)."""
        sizer = KellySizer(fraction=0.5)
        size = sizer.calculate(
            capital=100000.0,
            win_rate=0.6,
            win_loss_ratio=2.0,
        )
        assert size == pytest.approx(20000.0)  # Half of full Kelly

    def test_kelly_max_position(self) -> None:
        """Test Kelly respects max position limit."""
        sizer = KellySizer(max_position_pct=0.2)
        size = sizer.calculate(
            capital=100000.0,
            win_rate=0.6,
            win_loss_ratio=2.0,
        )
        assert size == pytest.approx(20000.0)  # Capped at 20%

    def test_kelly_negative_edge(self) -> None:
        """Test Kelly returns 0 for negative edge."""
        sizer = KellySizer()
        # win_rate=0.3, win_loss_ratio=1.0 → negative Kelly
        size = sizer.calculate(
            capital=100000.0,
            win_rate=0.3,
            win_loss_ratio=1.0,
        )
        assert size == 0.0


class TestFixedSizer:
    """Tests for fixed position sizing."""

    def test_fixed_percent(self) -> None:
        """Test fixed percentage sizing."""
        sizer = FixedSizer(position_pct=0.1)
        size = sizer.calculate(capital=100000.0)
        assert size == pytest.approx(10000.0)

    def test_fixed_amount(self) -> None:
        """Test fixed amount sizing."""
        sizer = FixedSizer(fixed_amount=5000.0)
        size = sizer.calculate(capital=100000.0)
        assert size == pytest.approx(5000.0)

    def test_fixed_respects_capital(self) -> None:
        """Test fixed amount doesn't exceed capital."""
        sizer = FixedSizer(fixed_amount=150000.0)
        size = sizer.calculate(capital=100000.0)
        assert size == pytest.approx(100000.0)


class TestRiskParitySizer:
    """Tests for risk parity position sizing."""

    def test_risk_parity_basic(self) -> None:
        """Test basic risk parity sizing."""
        sizer = RiskParitySizer(target_risk=0.02)
        size = sizer.calculate(
            capital=100000.0,
            asset_volatility=0.5,  # 50% volatility
        )
        # target_risk / volatility * capital = 0.02/0.5 * 100000 = 4000
        assert size == pytest.approx(4000.0)

    def test_risk_parity_low_vol(self) -> None:
        """Test risk parity with low volatility."""
        sizer = RiskParitySizer(target_risk=0.02)
        size = sizer.calculate(
            capital=100000.0,
            asset_volatility=0.1,  # 10% volatility
        )
        assert size == pytest.approx(20000.0)


# ============ Stop Loss Tests ============

class TestPriceStopLoss:
    """Tests for price-based stop loss."""

    def test_long_stop_hit(self) -> None:
        """Test stop loss triggered for long position."""
        stop = PriceStopLoss(stop_price=48000.0)
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
        )
        assert stop.should_trigger(pos, current_price=47000.0)

    def test_long_stop_not_hit(self) -> None:
        """Test stop loss not triggered for long position."""
        stop = PriceStopLoss(stop_price=48000.0)
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
        )
        assert not stop.should_trigger(pos, current_price=52000.0)

    def test_short_stop_hit(self) -> None:
        """Test stop loss triggered for short position."""
        stop = PriceStopLoss(stop_price=52000.0)
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            entry_price=50000.0,
            quantity=0.1,
        )
        assert stop.should_trigger(pos, current_price=53000.0)


class TestPercentStopLoss:
    """Tests for percentage-based stop loss."""

    def test_percent_stop_long(self) -> None:
        """Test percentage stop for long position."""
        stop = PercentStopLoss(stop_pct=0.05)
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
        )
        # 5% below entry = 47500
        assert stop.should_trigger(pos, current_price=47000.0)
        assert not stop.should_trigger(pos, current_price=48000.0)

    def test_percent_stop_short(self) -> None:
        """Test percentage stop for short position."""
        stop = PercentStopLoss(stop_pct=0.05)
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            entry_price=50000.0,
            quantity=0.1,
        )
        # 5% above entry = 52500
        assert stop.should_trigger(pos, current_price=53000.0)
        assert not stop.should_trigger(pos, current_price=51000.0)


class TestTrailingStopLoss:
    """Tests for trailing stop loss."""

    def test_trailing_stop_updates(self) -> None:
        """Test trailing stop updates with price."""
        stop = TrailingStopLoss(trail_pct=0.05)
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
        )

        # Price moves up, stop should update
        stop.update(pos, high_price=55000.0)
        # Stop should be at 55000 * 0.95 = 52250
        assert not stop.should_trigger(pos, current_price=53000.0)
        assert stop.should_trigger(pos, current_price=52000.0)

    def test_trailing_stop_doesnt_lower(self) -> None:
        """Test trailing stop never lowers."""
        stop = TrailingStopLoss(trail_pct=0.05)
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
        )

        stop.update(pos, high_price=55000.0)
        stop.update(pos, high_price=53000.0)  # Lower high
        # Stop should still be at 52250, not 50350
        assert stop.should_trigger(pos, current_price=52000.0)


class TestTimeStopLoss:
    """Tests for time-based stop loss."""

    def test_time_stop_not_expired(self) -> None:
        """Test time stop before expiry."""
        stop = TimeStopLoss(max_hold_hours=24)
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
            entry_time=datetime.now(),
        )
        assert not stop.should_trigger(pos, current_time=datetime.now())

    def test_time_stop_expired(self) -> None:
        """Test time stop after expiry."""
        stop = TimeStopLoss(max_hold_hours=24)
        entry_time = datetime.now() - timedelta(hours=25)
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
            entry_time=entry_time,
        )
        assert stop.should_trigger(pos, current_time=datetime.now())


# ============ Take Profit Tests ============

class TestFixedTakeProfit:
    """Tests for fixed take profit."""

    def test_take_profit_long(self) -> None:
        """Test take profit for long position."""
        tp = FixedTakeProfit(profit_pct=0.1)
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
        )
        # 10% above entry = 55000
        assert tp.should_trigger(pos, current_price=56000.0)
        assert not tp.should_trigger(pos, current_price=54000.0)

    def test_take_profit_short(self) -> None:
        """Test take profit for short position."""
        tp = FixedTakeProfit(profit_pct=0.1)
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            entry_price=50000.0,
            quantity=0.1,
        )
        # 10% below entry = 45000
        assert tp.should_trigger(pos, current_price=44000.0)
        assert not tp.should_trigger(pos, current_price=46000.0)


class TestTrailingTakeProfit:
    """Tests for trailing take profit."""

    def test_trailing_tp_activates(self) -> None:
        """Test trailing TP activates at threshold."""
        tp = TrailingTakeProfit(
            activation_pct=0.05,
            trail_pct=0.02,
        )
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
        )

        # Not activated yet
        tp.update(pos, high_price=52000.0)
        assert not tp.is_activated

        # Activated at 5% profit
        tp.update(pos, high_price=53000.0)
        assert tp.is_activated

    def test_trailing_tp_triggers(self) -> None:
        """Test trailing TP triggers on pullback."""
        tp = TrailingTakeProfit(
            activation_pct=0.05,
            trail_pct=0.02,
        )
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
        )

        # Activate and update
        tp.update(pos, high_price=55000.0)
        # Trail level = 55000 * 0.98 = 53900
        assert not tp.should_trigger(pos, current_price=54000.0)
        assert tp.should_trigger(pos, current_price=53500.0)


# ============ Position Manager Tests ============

class TestPositionManager:
    """Tests for PositionManager."""

    @pytest.fixture
    def manager(self) -> PositionManager:
        """Create position manager."""
        config = PositionConfig(
            initial_capital=100000.0,
            max_positions=10,
            max_position_size=0.1,
        )
        return PositionManager(config)

    def test_open_long(self, manager: PositionManager) -> None:
        """Test opening a long position."""
        pos = manager.open_position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            price=50000.0,
            quantity=0.1,
        )
        assert pos is not None
        assert pos.status == PositionStatus.OPEN
        assert manager.get_position("BTCUSDT") is not None

    def test_open_short(self, manager: PositionManager) -> None:
        """Test opening a short position."""
        pos = manager.open_position(
            symbol="ETHUSDT",
            side=PositionSide.SHORT,
            price=3000.0,
            quantity=1.0,
        )
        assert pos.side == PositionSide.SHORT

    def test_close_position(self, manager: PositionManager) -> None:
        """Test closing a position."""
        manager.open_position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            price=50000.0,
            quantity=0.1,
        )

        result = manager.close_position("BTCUSDT", price=55000.0)
        assert result.pnl == pytest.approx(500.0)
        assert result.status == PositionStatus.CLOSED
        assert manager.get_position("BTCUSDT") is None

    def test_max_positions(self, manager: PositionManager) -> None:
        """Test max positions limit."""
        manager.config.max_positions = 2

        manager.open_position("BTC", PositionSide.LONG, 50000.0, 0.01)
        manager.open_position("ETH", PositionSide.LONG, 3000.0, 0.1)

        with pytest.raises(InvalidPositionError):
            manager.open_position("SOL", PositionSide.LONG, 100.0, 1.0)

    def test_insufficient_funds(self, manager: PositionManager) -> None:
        """Test insufficient funds handling."""
        with pytest.raises(InsufficientFundsError):
            manager.open_position(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                price=50000.0,
                quantity=100.0,  # Way too much
            )

    def test_process_signals(self, manager: PositionManager) -> None:
        """Test processing trading signals."""
        signals = pd.DataFrame({
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "signal": [1.0, -1.0],  # Long BTC, Short ETH
            "price": [50000.0, 3000.0],
        })

        positions = manager.process_signals(signals)
        assert len(positions) == 2
        assert positions[0].side == PositionSide.LONG
        assert positions[1].side == PositionSide.SHORT

    def test_update_stops(self, manager: PositionManager) -> None:
        """Test updating stop losses."""
        manager.add_stop_loss(PercentStopLoss(stop_pct=0.05))

        manager.open_position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            price=50000.0,
            quantity=0.1,
        )

        # Price drops 10%
        closed = manager.check_stops(
            prices={"BTCUSDT": 45000.0}
        )
        assert len(closed) == 1
        assert closed[0].symbol == "BTCUSDT"

    def test_portfolio_value(self, manager: PositionManager) -> None:
        """Test portfolio value calculation."""
        manager.open_position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            price=50000.0,
            quantity=0.1,
        )

        value = manager.get_portfolio_value(
            prices={"BTCUSDT": 55000.0}
        )
        # Initial capital - cost + current value
        # 100000 - 5000 + 5500 = 100500
        assert value == pytest.approx(100500.0)


class TestPositionManagerBoundary:
    """Boundary tests for PositionManager."""

    def test_zero_quantity(self) -> None:
        """Test zero quantity handling."""
        config = PositionConfig(initial_capital=100000.0)
        manager = PositionManager(config)

        with pytest.raises(InvalidPositionError):
            manager.open_position(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                price=50000.0,
                quantity=0.0,
            )

    def test_negative_price(self) -> None:
        """Test negative price handling."""
        config = PositionConfig(initial_capital=100000.0)
        manager = PositionManager(config)

        with pytest.raises(InvalidPositionError):
            manager.open_position(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                price=-50000.0,
                quantity=0.1,
            )

    def test_close_nonexistent(self) -> None:
        """Test closing non-existent position."""
        config = PositionConfig(initial_capital=100000.0)
        manager = PositionManager(config)

        with pytest.raises(InvalidPositionError):
            manager.close_position("BTCUSDT", price=50000.0)


class TestPositionManagerPerformance:
    """Performance tests for PositionManager."""

    def test_many_positions(self) -> None:
        """Test handling many positions."""
        import time

        config = PositionConfig(
            initial_capital=10000000.0,
            max_positions=1000,
        )
        manager = PositionManager(config)

        start = time.time()
        for i in range(100):
            manager.open_position(
                symbol=f"ASSET{i}",
                side=PositionSide.LONG,
                price=100.0,
                quantity=0.1,
            )
        elapsed = time.time() - start

        assert elapsed < 1.0  # Should be fast

    def test_many_signal_updates(self) -> None:
        """Test processing many signal updates."""
        import time

        config = PositionConfig(initial_capital=10000000.0)
        manager = PositionManager(config)

        signals = pd.DataFrame({
            "symbol": [f"ASSET{i}" for i in range(100)],
            "signal": [1.0] * 100,
            "price": [100.0] * 100,
        })

        start = time.time()
        manager.process_signals(signals)
        elapsed = time.time() - start

        assert elapsed < 1.0
