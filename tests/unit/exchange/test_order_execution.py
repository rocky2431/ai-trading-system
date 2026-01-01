"""Tests for order execution module (Task 19).

TDD tests for multi-directional order execution system.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iqfmp.exchange.adapter import (
    ExchangeAdapter,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from iqfmp.exchange.execution import (
    ExecutionResult,
    OrderAction,
    OrderDirection,
    OrderExecutionError,
    OrderExecutor,
    OrderManager,
    OrderRequest,
    OrderTimeout,
    PartialFill,
    PartialFillHandler,
    TimeoutHandler,
)


# ==================== OrderRequest Tests ====================


class TestOrderRequest:
    """Test OrderRequest model."""

    def test_create_market_order_request(self) -> None:
        """Test creating a market order request."""
        request = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            action=OrderAction.OPEN_LONG,
        )
        assert request.symbol == "BTC/USDT"
        assert request.side == OrderSide.BUY
        assert request.order_type == OrderType.MARKET
        assert request.quantity == Decimal("0.1")
        assert request.price is None
        assert request.action == OrderAction.OPEN_LONG

    def test_create_limit_order_request(self) -> None:
        """Test creating a limit order request."""
        request = OrderRequest(
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.5"),
            price=Decimal("2000.0"),
            action=OrderAction.CLOSE_LONG,
        )
        assert request.price == Decimal("2000.0")
        assert request.action == OrderAction.CLOSE_LONG

    def test_order_request_with_stop_loss(self) -> None:
        """Test order request with stop loss."""
        request = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            action=OrderAction.OPEN_LONG,
            stop_loss_price=Decimal("45000.0"),
        )
        assert request.stop_loss_price == Decimal("45000.0")

    def test_order_request_with_take_profit(self) -> None:
        """Test order request with take profit."""
        request = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            action=OrderAction.OPEN_LONG,
            take_profit_price=Decimal("55000.0"),
        )
        assert request.take_profit_price == Decimal("55000.0")

    def test_order_request_with_timeout(self) -> None:
        """Test order request with timeout."""
        request = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000.0"),
            action=OrderAction.OPEN_LONG,
            timeout_seconds=60,
        )
        assert request.timeout_seconds == 60


# ==================== OrderAction Tests ====================


class TestOrderAction:
    """Test OrderAction enum."""

    def test_open_long_action(self) -> None:
        """Test open long action."""
        assert OrderAction.OPEN_LONG.value == "open_long"
        assert OrderAction.OPEN_LONG.direction == OrderDirection.LONG
        assert OrderAction.OPEN_LONG.is_opening is True

    def test_close_long_action(self) -> None:
        """Test close long action."""
        assert OrderAction.CLOSE_LONG.value == "close_long"
        assert OrderAction.CLOSE_LONG.direction == OrderDirection.LONG
        assert OrderAction.CLOSE_LONG.is_opening is False

    def test_open_short_action(self) -> None:
        """Test open short action."""
        assert OrderAction.OPEN_SHORT.value == "open_short"
        assert OrderAction.OPEN_SHORT.direction == OrderDirection.SHORT
        assert OrderAction.OPEN_SHORT.is_opening is True

    def test_close_short_action(self) -> None:
        """Test close short action."""
        assert OrderAction.CLOSE_SHORT.value == "close_short"
        assert OrderAction.CLOSE_SHORT.direction == OrderDirection.SHORT
        assert OrderAction.CLOSE_SHORT.is_opening is False


# ==================== ExecutionResult Tests ====================


class TestExecutionResult:
    """Test ExecutionResult model."""

    def test_successful_execution_result(self) -> None:
        """Test successful execution result."""
        result = ExecutionResult(
            success=True,
            order_id="order_123",
            filled_quantity=Decimal("0.1"),
            average_price=Decimal("50000.0"),
            commission=Decimal("0.5"),
            status=OrderStatus.CLOSED,
        )
        assert result.success is True
        assert result.order_id == "order_123"
        assert result.filled_quantity == Decimal("0.1")
        assert result.average_price == Decimal("50000.0")
        assert result.commission == Decimal("0.5")
        assert result.error is None

    def test_failed_execution_result(self) -> None:
        """Test failed execution result."""
        result = ExecutionResult(
            success=False,
            error="Insufficient balance",
            status=OrderStatus.REJECTED,
        )
        assert result.success is False
        assert result.error == "Insufficient balance"
        assert result.order_id is None

    def test_partial_fill_execution_result(self) -> None:
        """Test partial fill execution result."""
        result = ExecutionResult(
            success=True,
            order_id="order_456",
            filled_quantity=Decimal("0.05"),
            remaining_quantity=Decimal("0.05"),
            average_price=Decimal("50000.0"),
            status=OrderStatus.OPEN,
        )
        assert result.remaining_quantity == Decimal("0.05")
        assert result.is_partial_fill is True


# ==================== OrderExecutor Tests ====================


class TestOrderExecutor:
    """Test OrderExecutor class."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create mock exchange adapter."""
        adapter = MagicMock(spec=ExchangeAdapter)
        return adapter

    @pytest.fixture
    def executor(self, mock_adapter: MagicMock) -> OrderExecutor:
        """Create order executor."""
        return OrderExecutor(adapter=mock_adapter)

    @pytest.mark.asyncio
    async def test_execute_market_buy_order(
        self, executor: OrderExecutor, mock_adapter: MagicMock
    ) -> None:
        """Test executing a market buy order."""
        mock_adapter.create_order = AsyncMock(
            return_value=Order(
                id="order_123",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                amount=0.1,
                filled=0.1,
                status=OrderStatus.CLOSED,
                timestamp=datetime.now(),
            )
        )

        request = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            action=OrderAction.OPEN_LONG,
        )

        result = await executor.execute(request)

        assert result.success is True
        assert result.order_id == "order_123"
        assert result.filled_quantity == Decimal("0.1")
        mock_adapter.create_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_market_sell_order(
        self, executor: OrderExecutor, mock_adapter: MagicMock
    ) -> None:
        """Test executing a market sell order."""
        mock_adapter.create_order = AsyncMock(
            return_value=Order(
                id="order_124",
                symbol="BTC/USDT",
                side=OrderSide.SELL,
                type=OrderType.MARKET,
                amount=0.1,
                filled=0.1,
                status=OrderStatus.CLOSED,
                timestamp=datetime.now(),
            )
        )

        request = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            action=OrderAction.CLOSE_LONG,
        )

        result = await executor.execute(request)

        assert result.success is True
        assert result.filled_quantity == Decimal("0.1")

    @pytest.mark.asyncio
    async def test_execute_limit_order(
        self, executor: OrderExecutor, mock_adapter: MagicMock
    ) -> None:
        """Test executing a limit order."""
        mock_adapter.create_order = AsyncMock(
            return_value=Order(
                id="order_125",
                symbol="ETH/USDT",
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                amount=1.0,
                price=2000.0,
                filled=0.0,
                status=OrderStatus.OPEN,
                timestamp=datetime.now(),
            )
        )

        request = OrderRequest(
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("2000.0"),
            action=OrderAction.OPEN_LONG,
        )

        result = await executor.execute(request)

        assert result.success is True
        assert result.order_id == "order_125"
        assert result.status == OrderStatus.OPEN

    @pytest.mark.asyncio
    async def test_execute_open_short(
        self, executor: OrderExecutor, mock_adapter: MagicMock
    ) -> None:
        """Test executing open short order."""
        mock_adapter.create_order = AsyncMock(
            return_value=Order(
                id="order_126",
                symbol="BTC/USDT",
                side=OrderSide.SELL,
                type=OrderType.MARKET,
                amount=0.1,
                filled=0.1,
                status=OrderStatus.CLOSED,
                timestamp=datetime.now(),
            )
        )

        request = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            action=OrderAction.OPEN_SHORT,
        )

        result = await executor.execute(request)

        assert result.success is True
        assert result.action == OrderAction.OPEN_SHORT

    @pytest.mark.asyncio
    async def test_execute_close_short(
        self, executor: OrderExecutor, mock_adapter: MagicMock
    ) -> None:
        """Test executing close short order."""
        mock_adapter.create_order = AsyncMock(
            return_value=Order(
                id="order_127",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                amount=0.1,
                filled=0.1,
                status=OrderStatus.CLOSED,
                timestamp=datetime.now(),
            )
        )

        request = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            action=OrderAction.CLOSE_SHORT,
        )

        result = await executor.execute(request)

        assert result.success is True
        assert result.action == OrderAction.CLOSE_SHORT

    @pytest.mark.asyncio
    async def test_execute_order_with_error(
        self, executor: OrderExecutor, mock_adapter: MagicMock
    ) -> None:
        """Test executing order with error."""
        mock_adapter.create_order = AsyncMock(
            side_effect=Exception("Insufficient balance")
        )

        request = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100.0"),
            action=OrderAction.OPEN_LONG,
        )

        result = await executor.execute(request)

        assert result.success is False
        assert "Insufficient balance" in str(result.error)

    def test_validate_order_request(self, executor: OrderExecutor) -> None:
        """Test order request validation."""
        # Valid request
        valid_request = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            action=OrderAction.OPEN_LONG,
        )
        assert executor.validate(valid_request) is True

    def test_validate_invalid_quantity(self, executor: OrderExecutor) -> None:
        """Test validation rejects zero or negative quantity."""
        request = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0"),
            action=OrderAction.OPEN_LONG,
        )
        assert executor.validate(request) is False

    def test_validate_limit_order_requires_price(
        self, executor: OrderExecutor
    ) -> None:
        """Test limit order requires price."""
        request = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=None,
            action=OrderAction.OPEN_LONG,
        )
        assert executor.validate(request) is False


# ==================== OrderManager Tests ====================


class TestOrderManager:
    """Test OrderManager class."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create mock exchange adapter."""
        adapter = MagicMock(spec=ExchangeAdapter)
        return adapter

    @pytest.fixture
    def manager(self, mock_adapter: MagicMock) -> OrderManager:
        """Create order manager."""
        return OrderManager(adapter=mock_adapter)

    @pytest.mark.asyncio
    async def test_track_order(
        self, manager: OrderManager, mock_adapter: MagicMock
    ) -> None:
        """Test tracking an order."""
        order = Order(
            id="order_001",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            amount=0.1,
            filled=0.1,
            status=OrderStatus.CLOSED,
            timestamp=datetime.now(),
        )

        manager.track(order)

        assert manager.get_order("order_001") == order
        assert len(manager.active_orders) == 0  # Closed order not active

    @pytest.mark.asyncio
    async def test_track_open_order(
        self, manager: OrderManager, mock_adapter: MagicMock
    ) -> None:
        """Test tracking an open order."""
        order = Order(
            id="order_002",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            amount=0.1,
            filled=0.0,
            status=OrderStatus.OPEN,
            timestamp=datetime.now(),
        )

        manager.track(order)

        assert "order_002" in manager.active_orders

    @pytest.mark.asyncio
    async def test_update_order_status(
        self, manager: OrderManager, mock_adapter: MagicMock
    ) -> None:
        """Test updating order status."""
        order = Order(
            id="order_003",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            amount=0.1,
            filled=0.0,
            status=OrderStatus.OPEN,
            timestamp=datetime.now(),
        )
        manager.track(order)

        mock_adapter.fetch_order = AsyncMock(
            return_value=Order(
                id="order_003",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                amount=0.1,
                filled=0.1,
                status=OrderStatus.CLOSED,
                timestamp=datetime.now(),
            )
        )

        updated = await manager.update_status("order_003")

        assert updated.status == OrderStatus.CLOSED
        assert "order_003" not in manager.active_orders

    @pytest.mark.asyncio
    async def test_cancel_order(
        self, manager: OrderManager, mock_adapter: MagicMock
    ) -> None:
        """Test cancelling an order."""
        order = Order(
            id="order_004",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            amount=0.1,
            filled=0.0,
            status=OrderStatus.OPEN,
            timestamp=datetime.now(),
        )
        manager.track(order)

        mock_adapter.cancel_order = AsyncMock(
            return_value=Order(
                id="order_004",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                amount=0.1,
                filled=0.0,
                status=OrderStatus.CANCELED,
                timestamp=datetime.now(),
            )
        )

        cancelled = await manager.cancel("order_004")

        assert cancelled.status == OrderStatus.CANCELED
        assert "order_004" not in manager.active_orders

    def test_get_orders_by_symbol(self, manager: OrderManager) -> None:
        """Test getting orders by symbol."""
        order1 = Order(
            id="order_005",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            amount=0.1,
            filled=0.1,
            status=OrderStatus.CLOSED,
            timestamp=datetime.now(),
        )
        order2 = Order(
            id="order_006",
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            amount=1.0,
            filled=1.0,
            status=OrderStatus.CLOSED,
            timestamp=datetime.now(),
        )
        manager.track(order1)
        manager.track(order2)

        btc_orders = manager.get_orders_by_symbol("BTC/USDT")

        assert len(btc_orders) == 1
        assert btc_orders[0].id == "order_005"

    def test_get_order_history(self, manager: OrderManager) -> None:
        """Test getting order history."""
        for i in range(5):
            order = Order(
                id=f"order_{i}",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                amount=0.1,
                filled=0.1,
                status=OrderStatus.CLOSED,
                timestamp=datetime.now(),
            )
            manager.track(order)

        history = manager.get_history(limit=3)

        assert len(history) == 3


# ==================== PartialFillHandler Tests ====================


class TestPartialFillHandler:
    """Test PartialFillHandler class."""

    @pytest.fixture
    def handler(self) -> PartialFillHandler:
        """Create partial fill handler."""
        return PartialFillHandler()

    def test_record_partial_fill(self, handler: PartialFillHandler) -> None:
        """Test recording a partial fill."""
        fill = PartialFill(
            order_id="order_100",
            filled_quantity=Decimal("0.05"),
            fill_price=Decimal("50000.0"),
            timestamp=datetime.now(),
        )

        handler.record(fill)

        fills = handler.get_fills("order_100")
        assert len(fills) == 1
        assert fills[0].filled_quantity == Decimal("0.05")

    def test_multiple_partial_fills(self, handler: PartialFillHandler) -> None:
        """Test multiple partial fills."""
        fill1 = PartialFill(
            order_id="order_101",
            filled_quantity=Decimal("0.03"),
            fill_price=Decimal("50000.0"),
            timestamp=datetime.now(),
        )
        fill2 = PartialFill(
            order_id="order_101",
            filled_quantity=Decimal("0.04"),
            fill_price=Decimal("50100.0"),
            timestamp=datetime.now(),
        )

        handler.record(fill1)
        handler.record(fill2)

        total = handler.get_total_filled("order_101")
        assert total == Decimal("0.07")

    def test_calculate_average_fill_price(
        self, handler: PartialFillHandler
    ) -> None:
        """Test calculating average fill price."""
        fill1 = PartialFill(
            order_id="order_102",
            filled_quantity=Decimal("0.5"),
            fill_price=Decimal("50000.0"),
            timestamp=datetime.now(),
        )
        fill2 = PartialFill(
            order_id="order_102",
            filled_quantity=Decimal("0.5"),
            fill_price=Decimal("51000.0"),
            timestamp=datetime.now(),
        )

        handler.record(fill1)
        handler.record(fill2)

        avg_price = handler.get_average_price("order_102")
        # (0.5 * 50000 + 0.5 * 51000) / 1.0 = 50500
        assert avg_price == Decimal("50500.0")

    def test_remaining_quantity(self, handler: PartialFillHandler) -> None:
        """Test calculating remaining quantity."""
        fill = PartialFill(
            order_id="order_103",
            filled_quantity=Decimal("0.3"),
            fill_price=Decimal("50000.0"),
            timestamp=datetime.now(),
        )

        handler.record(fill)

        remaining = handler.get_remaining(
            order_id="order_103", total_quantity=Decimal("1.0")
        )
        assert remaining == Decimal("0.7")

    def test_is_fully_filled(self, handler: PartialFillHandler) -> None:
        """Test checking if order is fully filled."""
        fill = PartialFill(
            order_id="order_104",
            filled_quantity=Decimal("1.0"),
            fill_price=Decimal("50000.0"),
            timestamp=datetime.now(),
        )

        handler.record(fill)

        assert (
            handler.is_fully_filled(
                order_id="order_104", total_quantity=Decimal("1.0")
            )
            is True
        )
        assert (
            handler.is_fully_filled(
                order_id="order_104", total_quantity=Decimal("2.0")
            )
            is False
        )


# ==================== TimeoutHandler Tests ====================


class TestTimeoutHandler:
    """Test TimeoutHandler class."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create mock exchange adapter."""
        adapter = MagicMock(spec=ExchangeAdapter)
        return adapter

    @pytest.fixture
    def handler(self, mock_adapter: MagicMock) -> TimeoutHandler:
        """Create timeout handler."""
        return TimeoutHandler(adapter=mock_adapter)

    def test_register_timeout(self, handler: TimeoutHandler) -> None:
        """Test registering order timeout."""
        timeout = OrderTimeout(
            order_id="order_200",
            timeout_seconds=60,
            created_at=datetime.now(),
        )

        handler.register(timeout)

        assert handler.has_timeout("order_200") is True

    def test_is_expired(self, handler: TimeoutHandler) -> None:
        """Test checking if order is expired."""
        expired_time = datetime.now() - timedelta(seconds=120)
        timeout = OrderTimeout(
            order_id="order_201",
            timeout_seconds=60,
            created_at=expired_time,
        )

        handler.register(timeout)

        assert handler.is_expired("order_201") is True

    def test_is_not_expired(self, handler: TimeoutHandler) -> None:
        """Test checking non-expired order."""
        timeout = OrderTimeout(
            order_id="order_202",
            timeout_seconds=60,
            created_at=datetime.now(),
        )

        handler.register(timeout)

        assert handler.is_expired("order_202") is False

    @pytest.mark.asyncio
    async def test_cancel_expired_order(
        self, handler: TimeoutHandler, mock_adapter: MagicMock
    ) -> None:
        """Test cancelling expired order."""
        expired_time = datetime.now() - timedelta(seconds=120)
        timeout = OrderTimeout(
            order_id="order_203",
            timeout_seconds=60,
            created_at=expired_time,
        )
        handler.register(timeout)

        mock_adapter.cancel_order = AsyncMock(
            return_value=Order(
                id="order_203",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                amount=0.1,
                filled=0.0,
                status=OrderStatus.CANCELED,
                timestamp=datetime.now(),
            )
        )

        cancelled = await handler.cancel_expired("order_203", "BTC/USDT")

        assert cancelled is True
        mock_adapter.cancel_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_all_timeouts(
        self, handler: TimeoutHandler, mock_adapter: MagicMock
    ) -> None:
        """Test checking all timeouts."""
        expired_time = datetime.now() - timedelta(seconds=120)

        # Register multiple timeouts
        handler.register(
            OrderTimeout(
                order_id="order_204",
                timeout_seconds=60,
                created_at=expired_time,
                symbol="BTC/USDT",
            )
        )
        handler.register(
            OrderTimeout(
                order_id="order_205",
                timeout_seconds=60,
                created_at=datetime.now(),
                symbol="ETH/USDT",
            )
        )

        mock_adapter.cancel_order = AsyncMock(
            return_value=Order(
                id="order_204",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                amount=0.1,
                filled=0.0,
                status=OrderStatus.CANCELED,
                timestamp=datetime.now(),
            )
        )

        expired_orders = await handler.check_and_cancel_expired()

        assert len(expired_orders) == 1
        assert "order_204" in expired_orders

    def test_unregister_timeout(self, handler: TimeoutHandler) -> None:
        """Test unregistering timeout."""
        timeout = OrderTimeout(
            order_id="order_206",
            timeout_seconds=60,
            created_at=datetime.now(),
        )

        handler.register(timeout)
        handler.unregister("order_206")

        assert handler.has_timeout("order_206") is False


# ==================== Integration Tests ====================


class TestOrderExecutionIntegration:
    """Integration tests for order execution."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create mock exchange adapter."""
        adapter = MagicMock(spec=ExchangeAdapter)
        return adapter

    @pytest.mark.asyncio
    async def test_full_long_position_lifecycle(
        self, mock_adapter: MagicMock
    ) -> None:
        """Test full long position lifecycle: open -> close."""
        executor = OrderExecutor(adapter=mock_adapter)
        manager = OrderManager(adapter=mock_adapter)

        # Open long
        mock_adapter.create_order = AsyncMock(
            return_value=Order(
                id="order_300",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                amount=0.1,
                filled=0.1,
                status=OrderStatus.CLOSED,
                timestamp=datetime.now(),
            )
        )

        open_request = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            action=OrderAction.OPEN_LONG,
        )

        open_result = await executor.execute(open_request)
        assert open_result.success is True

        # Close long
        mock_adapter.create_order = AsyncMock(
            return_value=Order(
                id="order_301",
                symbol="BTC/USDT",
                side=OrderSide.SELL,
                type=OrderType.MARKET,
                amount=0.1,
                filled=0.1,
                status=OrderStatus.CLOSED,
                timestamp=datetime.now(),
            )
        )

        close_request = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            action=OrderAction.CLOSE_LONG,
        )

        close_result = await executor.execute(close_request)
        assert close_result.success is True

    @pytest.mark.asyncio
    async def test_full_short_position_lifecycle(
        self, mock_adapter: MagicMock
    ) -> None:
        """Test full short position lifecycle: open -> close."""
        executor = OrderExecutor(adapter=mock_adapter)

        # Open short
        mock_adapter.create_order = AsyncMock(
            return_value=Order(
                id="order_302",
                symbol="BTC/USDT",
                side=OrderSide.SELL,
                type=OrderType.MARKET,
                amount=0.1,
                filled=0.1,
                status=OrderStatus.CLOSED,
                timestamp=datetime.now(),
            )
        )

        open_request = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            action=OrderAction.OPEN_SHORT,
        )

        open_result = await executor.execute(open_request)
        assert open_result.success is True

        # Close short (buy to cover)
        mock_adapter.create_order = AsyncMock(
            return_value=Order(
                id="order_303",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                amount=0.1,
                filled=0.1,
                status=OrderStatus.CLOSED,
                timestamp=datetime.now(),
            )
        )

        close_request = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            action=OrderAction.CLOSE_SHORT,
        )

        close_result = await executor.execute(close_request)
        assert close_result.success is True

    @pytest.mark.asyncio
    async def test_partial_fill_to_complete(
        self, mock_adapter: MagicMock
    ) -> None:
        """Test partial fill then complete fill."""
        executor = OrderExecutor(adapter=mock_adapter)
        manager = OrderManager(adapter=mock_adapter)
        fill_handler = PartialFillHandler()

        # Initial partial fill
        mock_adapter.create_order = AsyncMock(
            return_value=Order(
                id="order_304",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                amount=1.0,
                price=50000.0,
                filled=0.3,
                status=OrderStatus.OPEN,
                timestamp=datetime.now(),
            )
        )

        request = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000.0"),
            action=OrderAction.OPEN_LONG,
        )

        result = await executor.execute(request)
        assert result.is_partial_fill is True

        # Record partial fill
        fill_handler.record(
            PartialFill(
                order_id="order_304",
                filled_quantity=Decimal("0.3"),
                fill_price=Decimal("50000.0"),
                timestamp=datetime.now(),
            )
        )

        remaining = fill_handler.get_remaining(
            order_id="order_304", total_quantity=Decimal("1.0")
        )
        assert remaining == Decimal("0.7")
