"""Tests for position monitoring module (Task 20).

TDD tests for position and PnL monitoring system.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iqfmp.exchange.adapter import (
    Balance,
    ExchangeAdapter,
    OrderSide,
)
from iqfmp.exchange.monitoring import (
    MarginAlert,
    MarginLevel,
    MarginMonitor,
    PnLCalculator,
    PnLRecord,
    PnLType,
    PositionData,
    PositionSide,
    PositionTracker,
    RealtimeUpdater,
    UpdateEvent,
    UpdateType,
)


# ==================== PositionData Tests ====================


class TestPositionData:
    """Test PositionData model."""

    def test_create_long_position(self) -> None:
        """Test creating a long position."""
        position = PositionData(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("51000.0"),
            leverage=10,
        )
        assert position.symbol == "BTC/USDT"
        assert position.side == PositionSide.LONG
        assert position.quantity == Decimal("0.5")
        assert position.entry_price == Decimal("50000.0")
        assert position.leverage == 10

    def test_create_short_position(self) -> None:
        """Test creating a short position."""
        position = PositionData(
            symbol="ETH/USDT",
            side=PositionSide.SHORT,
            quantity=Decimal("2.0"),
            entry_price=Decimal("3000.0"),
            current_price=Decimal("2900.0"),
            leverage=5,
        )
        assert position.side == PositionSide.SHORT

    def test_position_notional_value(self) -> None:
        """Test position notional value calculation."""
        position = PositionData(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("51000.0"),
            leverage=10,
        )
        # notional = quantity * current_price
        assert position.notional_value == Decimal("25500.0")

    def test_position_margin_used(self) -> None:
        """Test position margin used calculation."""
        position = PositionData(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("51000.0"),
            leverage=10,
        )
        # margin = notional / leverage = 25500 / 10 = 2550
        assert position.margin_used == Decimal("2550.0")

    def test_long_position_unrealized_pnl(self) -> None:
        """Test long position unrealized PnL."""
        position = PositionData(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("51000.0"),
            leverage=10,
        )
        # PnL = (current - entry) * quantity = (51000 - 50000) * 0.5 = 500
        assert position.unrealized_pnl == Decimal("500.0")

    def test_short_position_unrealized_pnl(self) -> None:
        """Test short position unrealized PnL."""
        position = PositionData(
            symbol="ETH/USDT",
            side=PositionSide.SHORT,
            quantity=Decimal("2.0"),
            entry_price=Decimal("3000.0"),
            current_price=Decimal("2900.0"),
            leverage=5,
        )
        # PnL = (entry - current) * quantity = (3000 - 2900) * 2 = 200
        assert position.unrealized_pnl == Decimal("200.0")

    def test_position_unrealized_pnl_percent(self) -> None:
        """Test unrealized PnL percentage."""
        position = PositionData(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("51000.0"),
            leverage=10,
        )
        # PnL% = unrealized_pnl / margin_used = 500 / 2550 ≈ 0.196
        assert float(position.unrealized_pnl_percent) == pytest.approx(
            0.196, rel=0.01
        )

    def test_position_liquidation_price_long(self) -> None:
        """Test liquidation price for long position."""
        position = PositionData(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("51000.0"),
            leverage=10,
            maintenance_margin_rate=Decimal("0.005"),
        )
        # For 10x leverage, liquidation ≈ entry * (1 - 1/leverage + mmr)
        assert position.liquidation_price < position.entry_price


# ==================== PositionTracker Tests ====================


class TestPositionTracker:
    """Test PositionTracker class."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create mock exchange adapter."""
        adapter = MagicMock(spec=ExchangeAdapter)
        return adapter

    @pytest.fixture
    def tracker(self, mock_adapter: MagicMock) -> PositionTracker:
        """Create position tracker."""
        return PositionTracker(adapter=mock_adapter)

    @pytest.mark.asyncio
    async def test_sync_positions(
        self, tracker: PositionTracker, mock_adapter: MagicMock
    ) -> None:
        """Test syncing positions from exchange."""
        mock_adapter.fetch_positions = AsyncMock(
            return_value=[
                {
                    "symbol": "BTC/USDT",
                    "side": "long",
                    "contracts": 0.5,
                    "entryPrice": 50000.0,
                    "markPrice": 51000.0,
                    "leverage": 10,
                }
            ]
        )

        await tracker.sync()

        positions = tracker.get_all_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_sync_multiple_positions(
        self, tracker: PositionTracker, mock_adapter: MagicMock
    ) -> None:
        """Test syncing multiple positions."""
        mock_adapter.fetch_positions = AsyncMock(
            return_value=[
                {
                    "symbol": "BTC/USDT",
                    "side": "long",
                    "contracts": 0.5,
                    "entryPrice": 50000.0,
                    "markPrice": 51000.0,
                    "leverage": 10,
                },
                {
                    "symbol": "ETH/USDT",
                    "side": "short",
                    "contracts": 2.0,
                    "entryPrice": 3000.0,
                    "markPrice": 2900.0,
                    "leverage": 5,
                },
            ]
        )

        await tracker.sync()

        assert len(tracker.get_all_positions()) == 2

    def test_get_position_by_symbol(self, tracker: PositionTracker) -> None:
        """Test getting position by symbol."""
        position = PositionData(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("51000.0"),
            leverage=10,
        )
        tracker.update_position(position)

        found = tracker.get_position("BTC/USDT")
        assert found is not None
        assert found.symbol == "BTC/USDT"

    def test_get_position_not_found(self, tracker: PositionTracker) -> None:
        """Test getting non-existent position."""
        found = tracker.get_position("XYZ/USDT")
        assert found is None

    def test_update_position_price(self, tracker: PositionTracker) -> None:
        """Test updating position current price."""
        position = PositionData(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("51000.0"),
            leverage=10,
        )
        tracker.update_position(position)

        tracker.update_price("BTC/USDT", Decimal("52000.0"))

        updated = tracker.get_position("BTC/USDT")
        assert updated is not None
        assert updated.current_price == Decimal("52000.0")

    def test_close_position(self, tracker: PositionTracker) -> None:
        """Test closing a position."""
        position = PositionData(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("51000.0"),
            leverage=10,
        )
        tracker.update_position(position)

        tracker.close_position("BTC/USDT")

        assert tracker.get_position("BTC/USDT") is None

    def test_total_unrealized_pnl(self, tracker: PositionTracker) -> None:
        """Test calculating total unrealized PnL."""
        pos1 = PositionData(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("51000.0"),
            leverage=10,
        )  # PnL = 500
        pos2 = PositionData(
            symbol="ETH/USDT",
            side=PositionSide.SHORT,
            quantity=Decimal("2.0"),
            entry_price=Decimal("3000.0"),
            current_price=Decimal("2900.0"),
            leverage=5,
        )  # PnL = 200

        tracker.update_position(pos1)
        tracker.update_position(pos2)

        assert tracker.total_unrealized_pnl == Decimal("700.0")

    def test_total_margin_used(self, tracker: PositionTracker) -> None:
        """Test calculating total margin used."""
        pos1 = PositionData(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("51000.0"),
            leverage=10,
        )  # margin = 2550
        pos2 = PositionData(
            symbol="ETH/USDT",
            side=PositionSide.SHORT,
            quantity=Decimal("2.0"),
            entry_price=Decimal("3000.0"),
            current_price=Decimal("2900.0"),
            leverage=5,
        )  # margin = 1160

        tracker.update_position(pos1)
        tracker.update_position(pos2)

        total_margin = tracker.total_margin_used
        assert total_margin == Decimal("2550.0") + Decimal("1160.0")


# ==================== PnLCalculator Tests ====================


class TestPnLCalculator:
    """Test PnLCalculator class."""

    @pytest.fixture
    def calculator(self) -> PnLCalculator:
        """Create PnL calculator."""
        return PnLCalculator()

    def test_calculate_unrealized_pnl_long(
        self, calculator: PnLCalculator
    ) -> None:
        """Test unrealized PnL calculation for long."""
        pnl = calculator.calculate_unrealized(
            side=PositionSide.LONG,
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("51000.0"),
        )
        assert pnl == Decimal("1000.0")

    def test_calculate_unrealized_pnl_short(
        self, calculator: PnLCalculator
    ) -> None:
        """Test unrealized PnL calculation for short."""
        pnl = calculator.calculate_unrealized(
            side=PositionSide.SHORT,
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("49000.0"),
        )
        assert pnl == Decimal("1000.0")

    def test_calculate_unrealized_pnl_loss(
        self, calculator: PnLCalculator
    ) -> None:
        """Test unrealized PnL calculation with loss."""
        pnl = calculator.calculate_unrealized(
            side=PositionSide.LONG,
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("48000.0"),
        )
        assert pnl == Decimal("-2000.0")

    def test_calculate_realized_pnl(self, calculator: PnLCalculator) -> None:
        """Test realized PnL calculation."""
        pnl = calculator.calculate_realized(
            side=PositionSide.LONG,
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.0"),
            exit_price=Decimal("52000.0"),
            commission=Decimal("10.0"),
        )
        # PnL = (52000 - 50000) * 1.0 - 10 = 1990
        assert pnl == Decimal("1990.0")

    def test_record_realized_pnl(self, calculator: PnLCalculator) -> None:
        """Test recording realized PnL."""
        record = PnLRecord(
            symbol="BTC/USDT",
            pnl_type=PnLType.REALIZED,
            amount=Decimal("1990.0"),
            timestamp=datetime.now(),
        )

        calculator.record(record)

        history = calculator.get_history("BTC/USDT")
        assert len(history) == 1
        assert history[0].amount == Decimal("1990.0")

    def test_total_realized_pnl(self, calculator: PnLCalculator) -> None:
        """Test total realized PnL."""
        calculator.record(
            PnLRecord(
                symbol="BTC/USDT",
                pnl_type=PnLType.REALIZED,
                amount=Decimal("1000.0"),
                timestamp=datetime.now(),
            )
        )
        calculator.record(
            PnLRecord(
                symbol="ETH/USDT",
                pnl_type=PnLType.REALIZED,
                amount=Decimal("500.0"),
                timestamp=datetime.now(),
            )
        )

        assert calculator.total_realized_pnl == Decimal("1500.0")

    def test_pnl_by_symbol(self, calculator: PnLCalculator) -> None:
        """Test PnL by symbol."""
        calculator.record(
            PnLRecord(
                symbol="BTC/USDT",
                pnl_type=PnLType.REALIZED,
                amount=Decimal("1000.0"),
                timestamp=datetime.now(),
            )
        )
        calculator.record(
            PnLRecord(
                symbol="BTC/USDT",
                pnl_type=PnLType.REALIZED,
                amount=Decimal("500.0"),
                timestamp=datetime.now(),
            )
        )

        assert calculator.get_symbol_pnl("BTC/USDT") == Decimal("1500.0")


# ==================== MarginMonitor Tests ====================


class TestMarginMonitor:
    """Test MarginMonitor class."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create mock exchange adapter."""
        adapter = MagicMock(spec=ExchangeAdapter)
        return adapter

    @pytest.fixture
    def monitor(self, mock_adapter: MagicMock) -> MarginMonitor:
        """Create margin monitor."""
        return MarginMonitor(adapter=mock_adapter)

    @pytest.mark.asyncio
    async def test_fetch_balance(
        self, monitor: MarginMonitor, mock_adapter: MagicMock
    ) -> None:
        """Test fetching balance."""
        mock_adapter.fetch_balance = AsyncMock(
            return_value=Balance(
                currency="USDT",
                total=10000.0,
                free=5000.0,
                used=5000.0,
            )
        )

        await monitor.update_balance()

        assert monitor.total_balance == Decimal("10000.0")
        assert monitor.available_balance == Decimal("5000.0")

    def test_margin_usage_calculation(self, monitor: MarginMonitor) -> None:
        """Test margin usage calculation."""
        monitor._total_balance = Decimal("10000.0")
        monitor._used_margin = Decimal("3000.0")

        # Usage = 3000 / 10000 = 0.3 (30%)
        assert monitor.margin_usage == Decimal("0.3")

    def test_margin_level_safe(self, monitor: MarginMonitor) -> None:
        """Test safe margin level."""
        monitor._total_balance = Decimal("10000.0")
        monitor._used_margin = Decimal("2000.0")

        assert monitor.margin_level == MarginLevel.SAFE

    def test_margin_level_warning(self, monitor: MarginMonitor) -> None:
        """Test warning margin level."""
        monitor._total_balance = Decimal("10000.0")
        monitor._used_margin = Decimal("5000.0")

        assert monitor.margin_level == MarginLevel.WARNING

    def test_margin_level_danger(self, monitor: MarginMonitor) -> None:
        """Test danger margin level."""
        monitor._total_balance = Decimal("10000.0")
        monitor._used_margin = Decimal("7500.0")

        assert monitor.margin_level == MarginLevel.DANGER

    def test_margin_level_critical(self, monitor: MarginMonitor) -> None:
        """Test critical margin level."""
        monitor._total_balance = Decimal("10000.0")
        monitor._used_margin = Decimal("9000.0")

        assert monitor.margin_level == MarginLevel.CRITICAL

    def test_check_margin_alerts(self, monitor: MarginMonitor) -> None:
        """Test margin alert generation."""
        monitor._total_balance = Decimal("10000.0")
        monitor._used_margin = Decimal("7500.0")

        alerts = monitor.check_alerts()

        assert len(alerts) > 0
        assert alerts[0].level == MarginLevel.DANGER

    def test_set_custom_thresholds(self, monitor: MarginMonitor) -> None:
        """Test setting custom margin thresholds."""
        monitor.set_thresholds(
            warning=Decimal("0.4"),
            danger=Decimal("0.6"),
            critical=Decimal("0.8"),
        )

        monitor._total_balance = Decimal("10000.0")
        monitor._used_margin = Decimal("4500.0")  # 45%

        assert monitor.margin_level == MarginLevel.WARNING


# ==================== RealtimeUpdater Tests ====================


class TestRealtimeUpdater:
    """Test RealtimeUpdater class."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create mock exchange adapter."""
        adapter = MagicMock(spec=ExchangeAdapter)
        return adapter

    @pytest.fixture
    def updater(self, mock_adapter: MagicMock) -> RealtimeUpdater:
        """Create realtime updater."""
        return RealtimeUpdater(adapter=mock_adapter)

    def test_subscribe_to_position_updates(
        self, updater: RealtimeUpdater
    ) -> None:
        """Test subscribing to position updates."""
        callback = MagicMock()

        updater.subscribe(UpdateType.POSITION, callback)

        assert updater.has_subscriber(UpdateType.POSITION)

    def test_subscribe_to_pnl_updates(self, updater: RealtimeUpdater) -> None:
        """Test subscribing to PnL updates."""
        callback = MagicMock()

        updater.subscribe(UpdateType.PNL, callback)

        assert updater.has_subscriber(UpdateType.PNL)

    def test_unsubscribe(self, updater: RealtimeUpdater) -> None:
        """Test unsubscribing from updates."""
        callback = MagicMock()

        sub_id = updater.subscribe(UpdateType.POSITION, callback)
        updater.unsubscribe(sub_id)

        assert not updater.has_subscriber(UpdateType.POSITION)

    def test_emit_position_update(self, updater: RealtimeUpdater) -> None:
        """Test emitting position update."""
        callback = MagicMock()
        updater.subscribe(UpdateType.POSITION, callback)

        event = UpdateEvent(
            type=UpdateType.POSITION,
            symbol="BTC/USDT",
            data={"quantity": 0.5, "unrealized_pnl": 500.0},
            timestamp=datetime.now(),
        )

        updater.emit(event)

        callback.assert_called_once()

    def test_emit_to_multiple_subscribers(
        self, updater: RealtimeUpdater
    ) -> None:
        """Test emitting to multiple subscribers."""
        callback1 = MagicMock()
        callback2 = MagicMock()

        updater.subscribe(UpdateType.POSITION, callback1)
        updater.subscribe(UpdateType.POSITION, callback2)

        event = UpdateEvent(
            type=UpdateType.POSITION,
            symbol="BTC/USDT",
            data={},
            timestamp=datetime.now(),
        )

        updater.emit(event)

        callback1.assert_called_once()
        callback2.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_websocket_stream(
        self, updater: RealtimeUpdater, mock_adapter: MagicMock
    ) -> None:
        """Test starting WebSocket stream."""
        mock_adapter.watch_positions = AsyncMock()

        await updater.start()

        assert updater.is_running is True

    @pytest.mark.asyncio
    async def test_stop_websocket_stream(
        self, updater: RealtimeUpdater, mock_adapter: MagicMock
    ) -> None:
        """Test stopping WebSocket stream."""
        updater._running = True

        await updater.stop()

        assert updater.is_running is False


# ==================== Integration Tests ====================


class TestPositionMonitoringIntegration:
    """Integration tests for position monitoring."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create mock exchange adapter."""
        adapter = MagicMock(spec=ExchangeAdapter)
        return adapter

    @pytest.mark.asyncio
    async def test_full_monitoring_workflow(
        self, mock_adapter: MagicMock
    ) -> None:
        """Test full position monitoring workflow."""
        tracker = PositionTracker(adapter=mock_adapter)
        calculator = PnLCalculator()
        monitor = MarginMonitor(adapter=mock_adapter)
        updater = RealtimeUpdater(adapter=mock_adapter)

        # Setup mocks
        mock_adapter.fetch_positions = AsyncMock(
            return_value=[
                {
                    "symbol": "BTC/USDT",
                    "side": "long",
                    "contracts": 0.5,
                    "entryPrice": 50000.0,
                    "markPrice": 51000.0,
                    "leverage": 10,
                }
            ]
        )
        mock_adapter.fetch_balance = AsyncMock(
            return_value=Balance(
                currency="USDT",
                total=10000.0,
                free=7000.0,
                used=3000.0,
            )
        )

        # Sync positions
        await tracker.sync()

        # Update balance
        await monitor.update_balance()

        # Get unrealized PnL
        position = tracker.get_position("BTC/USDT")
        assert position is not None
        assert position.unrealized_pnl == Decimal("500.0")

        # Check margin status
        assert monitor.margin_level == MarginLevel.SAFE

    @pytest.mark.asyncio
    async def test_position_price_update_flow(
        self, mock_adapter: MagicMock
    ) -> None:
        """Test position price update flow."""
        tracker = PositionTracker(adapter=mock_adapter)
        updater = RealtimeUpdater(adapter=mock_adapter)

        # Add position
        position = PositionData(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("51000.0"),
            leverage=10,
        )
        tracker.update_position(position)

        # Track PnL changes
        pnl_updates = []

        def on_update(event: UpdateEvent) -> None:
            pnl_updates.append(event)

        updater.subscribe(UpdateType.PNL, on_update)

        # Simulate price update
        tracker.update_price("BTC/USDT", Decimal("52000.0"))

        updated = tracker.get_position("BTC/USDT")
        assert updated is not None
        assert updated.unrealized_pnl == Decimal("1000.0")
