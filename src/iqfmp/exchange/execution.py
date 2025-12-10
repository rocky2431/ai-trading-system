"""Order execution module for IQFMP.

Provides multi-directional order execution:
- OrderExecutor: Execute market/limit orders
- OrderManager: Order lifecycle management
- PartialFillHandler: Handle partial fills
- TimeoutHandler: Order timeout management
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Optional

from iqfmp.exchange.adapter import (
    ExchangeAdapter,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)


# ==================== Enums ====================


class OrderDirection(Enum):
    """Order direction for futures trading."""

    LONG = "long"
    SHORT = "short"


class OrderAction(Enum):
    """Order action types for futures trading."""

    OPEN_LONG = "open_long"
    CLOSE_LONG = "close_long"
    OPEN_SHORT = "open_short"
    CLOSE_SHORT = "close_short"

    @property
    def direction(self) -> OrderDirection:
        """Get the direction of this action."""
        if self in (OrderAction.OPEN_LONG, OrderAction.CLOSE_LONG):
            return OrderDirection.LONG
        return OrderDirection.SHORT

    @property
    def is_opening(self) -> bool:
        """Check if this action opens a position."""
        return self in (OrderAction.OPEN_LONG, OrderAction.OPEN_SHORT)


# ==================== Exceptions ====================


class OrderExecutionError(Exception):
    """Base exception for order execution errors."""

    pass


# ==================== Data Models ====================


@dataclass
class OrderRequest:
    """Order request model."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    action: OrderAction
    price: Optional[Decimal] = None
    stop_loss_price: Optional[Decimal] = None
    take_profit_price: Optional[Decimal] = None
    timeout_seconds: Optional[int] = None
    client_order_id: Optional[str] = None
    reduce_only: bool = False
    post_only: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Order execution result."""

    success: bool
    status: OrderStatus
    order_id: Optional[str] = None
    filled_quantity: Optional[Decimal] = None
    remaining_quantity: Optional[Decimal] = None
    average_price: Optional[Decimal] = None
    commission: Optional[Decimal] = None
    error: Optional[str] = None
    action: Optional[OrderAction] = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_partial_fill(self) -> bool:
        """Check if this is a partial fill."""
        return (
            self.success
            and self.remaining_quantity is not None
            and self.remaining_quantity > Decimal("0")
        )


@dataclass
class PartialFill:
    """Partial fill record."""

    order_id: str
    filled_quantity: Decimal
    fill_price: Decimal
    timestamp: datetime
    trade_id: Optional[str] = None
    commission: Optional[Decimal] = None


@dataclass
class OrderTimeout:
    """Order timeout configuration."""

    order_id: str
    timeout_seconds: int
    created_at: datetime
    symbol: Optional[str] = None
    callback: Optional[Callable[[str], None]] = None


# ==================== OrderExecutor ====================


class OrderExecutor:
    """Execute orders on exchange."""

    def __init__(
        self,
        adapter: ExchangeAdapter,
        default_timeout: int = 300,
    ) -> None:
        """Initialize order executor.

        Args:
            adapter: Exchange adapter
            default_timeout: Default order timeout in seconds
        """
        self._adapter = adapter
        self._default_timeout = default_timeout

    def validate(self, request: OrderRequest) -> bool:
        """Validate order request.

        Args:
            request: Order request to validate

        Returns:
            True if valid, False otherwise
        """
        # Check quantity
        if request.quantity <= Decimal("0"):
            return False

        # Limit orders require price
        if request.order_type == OrderType.LIMIT and request.price is None:
            return False

        # Stop limit orders require price
        if request.order_type == OrderType.STOP_LIMIT and request.price is None:
            return False

        return True

    async def execute(self, request: OrderRequest) -> ExecutionResult:
        """Execute an order.

        Args:
            request: Order request

        Returns:
            Execution result
        """
        try:
            # Create order on exchange
            order = await self._adapter.create_order(
                symbol=request.symbol,
                side=request.side,
                order_type=request.order_type,
                quantity=request.quantity,
                price=request.price,
            )

            # Calculate remaining quantity
            remaining = None
            if order.filled is not None and order.amount is not None:
                remaining = order.amount - order.filled
                if remaining <= 0:
                    remaining = None

            return ExecutionResult(
                success=True,
                order_id=order.id,
                filled_quantity=order.filled,
                remaining_quantity=remaining,
                average_price=getattr(order, 'price', None),
                status=order.status,
                action=request.action,
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                status=OrderStatus.REJECTED,
            )

    async def execute_with_stop_loss(
        self,
        request: OrderRequest,
    ) -> tuple[ExecutionResult, Optional[ExecutionResult]]:
        """Execute order with stop loss.

        Args:
            request: Order request with stop_loss_price

        Returns:
            Tuple of (main order result, stop loss order result)
        """
        # Execute main order
        main_result = await self.execute(request)

        if not main_result.success or request.stop_loss_price is None:
            return main_result, None

        # Create stop loss order
        stop_side = (
            OrderSide.SELL
            if request.side == OrderSide.BUY
            else OrderSide.BUY
        )

        stop_request = OrderRequest(
            symbol=request.symbol,
            side=stop_side,
            order_type=OrderType.STOP_MARKET,
            quantity=request.quantity,
            price=request.stop_loss_price,
            action=(
                OrderAction.CLOSE_LONG
                if request.action == OrderAction.OPEN_LONG
                else OrderAction.CLOSE_SHORT
            ),
            reduce_only=True,
        )

        stop_result = await self.execute(stop_request)

        return main_result, stop_result


# ==================== OrderManager ====================


class OrderManager:
    """Manage order lifecycle and tracking."""

    def __init__(self, adapter: ExchangeAdapter) -> None:
        """Initialize order manager.

        Args:
            adapter: Exchange adapter
        """
        self._adapter = adapter
        self._orders: dict[str, Order] = {}
        self._active_orders: set[str] = set()

    @property
    def active_orders(self) -> set[str]:
        """Get active order IDs."""
        return self._active_orders.copy()

    def track(self, order: Order) -> None:
        """Track an order.

        Args:
            order: Order to track
        """
        self._orders[order.id] = order

        if order.status in (OrderStatus.OPEN,):
            self._active_orders.add(order.id)
        else:
            self._active_orders.discard(order.id)

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order or None
        """
        return self._orders.get(order_id)

    async def update_status(self, order_id: str) -> Order:
        """Update order status from exchange.

        Args:
            order_id: Order ID

        Returns:
            Updated order
        """
        order = self._orders.get(order_id)
        if order is None:
            raise ValueError(f"Order {order_id} not found")

        updated = await self._adapter.fetch_order(order_id, order.symbol)
        self.track(updated)
        return updated

    async def cancel(self, order_id: str) -> Order:
        """Cancel an order.

        Args:
            order_id: Order ID

        Returns:
            Cancelled order
        """
        order = self._orders.get(order_id)
        if order is None:
            raise ValueError(f"Order {order_id} not found")

        cancelled = await self._adapter.cancel_order(order_id, order.symbol)
        self.track(cancelled)
        return cancelled

    def get_orders_by_symbol(self, symbol: str) -> list[Order]:
        """Get orders by symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of orders
        """
        return [o for o in self._orders.values() if o.symbol == symbol]

    def get_history(self, limit: int = 100) -> list[Order]:
        """Get order history.

        Args:
            limit: Maximum number of orders

        Returns:
            List of orders
        """
        orders = sorted(
            self._orders.values(),
            key=lambda o: o.timestamp,
            reverse=True,
        )
        return orders[:limit]


# ==================== PartialFillHandler ====================


class PartialFillHandler:
    """Handle partial fills for orders."""

    def __init__(self) -> None:
        """Initialize partial fill handler."""
        self._fills: dict[str, list[PartialFill]] = {}

    def record(self, fill: PartialFill) -> None:
        """Record a partial fill.

        Args:
            fill: Partial fill to record
        """
        if fill.order_id not in self._fills:
            self._fills[fill.order_id] = []
        self._fills[fill.order_id].append(fill)

    def get_fills(self, order_id: str) -> list[PartialFill]:
        """Get fills for an order.

        Args:
            order_id: Order ID

        Returns:
            List of partial fills
        """
        return self._fills.get(order_id, [])

    def get_total_filled(self, order_id: str) -> Decimal:
        """Get total filled quantity.

        Args:
            order_id: Order ID

        Returns:
            Total filled quantity
        """
        fills = self.get_fills(order_id)
        return sum((f.filled_quantity for f in fills), Decimal("0"))

    def get_average_price(self, order_id: str) -> Decimal:
        """Calculate average fill price.

        Args:
            order_id: Order ID

        Returns:
            Average fill price
        """
        fills = self.get_fills(order_id)
        if not fills:
            return Decimal("0")

        total_value = sum(
            f.filled_quantity * f.fill_price for f in fills
        )
        total_quantity = sum(f.filled_quantity for f in fills)

        if total_quantity == Decimal("0"):
            return Decimal("0")

        return total_value / total_quantity

    def get_remaining(
        self, order_id: str, total_quantity: Decimal
    ) -> Decimal:
        """Get remaining quantity to fill.

        Args:
            order_id: Order ID
            total_quantity: Total order quantity

        Returns:
            Remaining quantity
        """
        filled = self.get_total_filled(order_id)
        return total_quantity - filled

    def is_fully_filled(
        self, order_id: str, total_quantity: Decimal
    ) -> bool:
        """Check if order is fully filled.

        Args:
            order_id: Order ID
            total_quantity: Total order quantity

        Returns:
            True if fully filled
        """
        filled = self.get_total_filled(order_id)
        return filled >= total_quantity


# ==================== TimeoutHandler ====================


class TimeoutHandler:
    """Handle order timeouts."""

    def __init__(self, adapter: ExchangeAdapter) -> None:
        """Initialize timeout handler.

        Args:
            adapter: Exchange adapter
        """
        self._adapter = adapter
        self._timeouts: dict[str, OrderTimeout] = {}

    def register(self, timeout: OrderTimeout) -> None:
        """Register order timeout.

        Args:
            timeout: Timeout configuration
        """
        self._timeouts[timeout.order_id] = timeout

    def unregister(self, order_id: str) -> None:
        """Unregister order timeout.

        Args:
            order_id: Order ID
        """
        self._timeouts.pop(order_id, None)

    def has_timeout(self, order_id: str) -> bool:
        """Check if order has timeout registered.

        Args:
            order_id: Order ID

        Returns:
            True if timeout registered
        """
        return order_id in self._timeouts

    def is_expired(self, order_id: str) -> bool:
        """Check if order timeout has expired.

        Args:
            order_id: Order ID

        Returns:
            True if expired
        """
        timeout = self._timeouts.get(order_id)
        if timeout is None:
            return False

        expiry_time = timeout.created_at + timedelta(
            seconds=timeout.timeout_seconds
        )
        return datetime.now() > expiry_time

    async def cancel_expired(
        self, order_id: str, symbol: str
    ) -> bool:
        """Cancel an expired order.

        Args:
            order_id: Order ID
            symbol: Trading symbol

        Returns:
            True if cancelled successfully
        """
        if not self.is_expired(order_id):
            return False

        try:
            await self._adapter.cancel_order(order_id, symbol)
            self.unregister(order_id)
            return True
        except Exception:
            return False

    async def check_and_cancel_expired(self) -> list[str]:
        """Check all timeouts and cancel expired orders.

        Returns:
            List of cancelled order IDs
        """
        cancelled: list[str] = []

        for order_id, timeout in list(self._timeouts.items()):
            if self.is_expired(order_id):
                symbol = timeout.symbol or ""
                try:
                    await self._adapter.cancel_order(order_id, symbol)
                    cancelled.append(order_id)
                    self.unregister(order_id)

                    # Call callback if set
                    if timeout.callback:
                        timeout.callback(order_id)
                except Exception:
                    pass

        return cancelled


# ==================== Module Exports ====================


__all__ = [
    # Enums
    "OrderAction",
    "OrderDirection",
    # Exceptions
    "OrderExecutionError",
    # Models
    "ExecutionResult",
    "OrderRequest",
    "OrderTimeout",
    "PartialFill",
    # Classes
    "OrderExecutor",
    "OrderManager",
    "PartialFillHandler",
    "TimeoutHandler",
]
