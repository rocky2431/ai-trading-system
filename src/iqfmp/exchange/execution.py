"""Order execution module for IQFMP.

Provides multi-directional order execution:
- OrderExecutor: Execute market/limit orders
- OrderManager: Order lifecycle management
- PartialFillHandler: Handle partial fills
- TimeoutHandler: Order timeout management
"""

import hashlib
import json
import logging
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

logger = logging.getLogger(__name__)


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


# ==================== Idempotency Cache ====================


@dataclass
class IdempotencyCacheEntry:
    """Cache entry for idempotent order execution."""

    result: "ExecutionResult"
    timestamp: datetime
    request_hash: str


class IdempotencyCacheError(OrderExecutionError):
    """Raised when idempotency cache operations fail."""

    pass


def get_required_redis_client(error_cls: type[Exception], purpose: str):
    """Get Redis client or raise error. Critical state requires persistence.

    Args:
        error_cls: Exception class to raise on failure
        purpose: Description of why Redis is needed (for error messages)

    Returns:
        Redis client instance

    Raises:
        error_cls: If Redis is unavailable
    """
    try:
        from iqfmp.db import get_redis_client
        client = get_redis_client()
        if client is None:
            raise error_cls(
                f"Redis unavailable. {purpose} requires persistent storage "
                "per CLAUDE.md critical state rules."
            )
        return client
    except error_cls:
        raise
    except Exception as e:
        raise error_cls(f"Failed to connect to Redis for {purpose}: {e}") from e


class IdempotencyCache:
    """Redis-backed cache for preventing duplicate order execution.

    Uses client_order_id as the idempotency key.
    Entries expire after ttl_seconds via Redis TTL.

    Critical state per CLAUDE.md: Must be persistent to survive service restarts.
    Redis is REQUIRED - no in-memory fallback for production safety.

    For testing, use `IdempotencyCache(redis_client=mock_redis)` to inject a mock.
    """

    REDIS_KEY_PREFIX = "iqfmp:idempotency:"

    def __init__(self, ttl_seconds: int = 3600, redis_client: Any = None) -> None:
        """Initialize cache with Redis backend.

        Args:
            ttl_seconds: Time-to-live for cache entries (default: 1 hour)
            redis_client: Optional Redis client for dependency injection (testing)

        Raises:
            IdempotencyCacheError: If Redis is unavailable and no client injected
        """
        self._ttl_seconds = ttl_seconds
        self._ttl = timedelta(seconds=ttl_seconds)
        self._redis = redis_client if redis_client is not None else self._get_redis_client()

    def _get_redis_client(self):
        """Get Redis client. Raises if unavailable (critical state requires persistence)."""
        return get_required_redis_client(IdempotencyCacheError, "Idempotency cache")

    def _compute_hash(self, request: "OrderRequest") -> str:
        """Compute hash of request for validation."""
        key = f"{request.symbol}:{request.side.value}:{request.order_type.value}:{request.quantity}:{request.price}"
        return hashlib.md5(key.encode()).hexdigest()

    def _serialize_result(self, result: "ExecutionResult", request_hash: str) -> str:
        """Serialize ExecutionResult to JSON for Redis storage."""
        return json.dumps({
            "success": result.success,
            "status": result.status.value,
            "order_id": result.order_id,
            "filled_quantity": str(result.filled_quantity) if result.filled_quantity else None,
            "remaining_quantity": str(result.remaining_quantity) if result.remaining_quantity else None,
            "average_price": str(result.average_price) if result.average_price else None,
            "commission": str(result.commission) if result.commission else None,
            "error": result.error,
            "action": result.action.value if result.action else None,
            "timestamp": result.timestamp.isoformat(),
            "request_hash": request_hash,
        })

    def _deserialize_result(self, data: str) -> tuple["ExecutionResult", str]:
        """Deserialize JSON to ExecutionResult."""
        obj = json.loads(data)
        result = ExecutionResult(
            success=obj["success"],
            status=OrderStatus(obj["status"]),
            order_id=obj.get("order_id"),
            filled_quantity=Decimal(obj["filled_quantity"]) if obj.get("filled_quantity") else None,
            remaining_quantity=Decimal(obj["remaining_quantity"]) if obj.get("remaining_quantity") else None,
            average_price=Decimal(obj["average_price"]) if obj.get("average_price") else None,
            commission=Decimal(obj["commission"]) if obj.get("commission") else None,
            error=obj.get("error"),
            action=OrderAction(obj["action"]) if obj.get("action") else None,
            timestamp=datetime.fromisoformat(obj["timestamp"]),
        )
        return result, obj["request_hash"]

    def get(self, client_order_id: str, request: "OrderRequest") -> Optional["ExecutionResult"]:
        """Get cached result if exists and not expired.

        Args:
            client_order_id: Client-provided order ID
            request: Current request (for validation)

        Returns:
            Cached result or None if not found

        Raises:
            IdempotencyCacheError: If Redis operation fails
        """
        request_hash = self._compute_hash(request)

        try:
            key = f"{self.REDIS_KEY_PREFIX}{client_order_id}"
            data = self._redis.get(key)
            if data:
                result, stored_hash = self._deserialize_result(data)
                if stored_hash == request_hash:
                    logger.debug(f"Idempotency cache hit for {client_order_id}")
                    return result
                logger.warning(f"Idempotency hash mismatch for {client_order_id}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Corrupted idempotency cache entry for {client_order_id}: {e}")
            return None
        except Exception as e:
            raise IdempotencyCacheError(
                f"Redis get failed for idempotency key {client_order_id}: {e}"
            ) from e

    def set(self, client_order_id: str, request: "OrderRequest", result: "ExecutionResult") -> None:
        """Cache execution result.

        Args:
            client_order_id: Client-provided order ID
            request: Original request
            result: Execution result

        Raises:
            IdempotencyCacheError: If Redis operation fails
        """
        request_hash = self._compute_hash(request)

        try:
            key = f"{self.REDIS_KEY_PREFIX}{client_order_id}"
            data = self._serialize_result(result, request_hash)
            self._redis.setex(key, self._ttl_seconds, data)
            logger.debug(f"Cached idempotency result in Redis: {client_order_id}")
        except Exception as e:
            raise IdempotencyCacheError(
                f"Redis set failed for idempotency key {client_order_id}: {e}. "
                "Critical state persistence required."
            ) from e

    def delete(self, client_order_id: str) -> bool:
        """Delete a cached entry.

        Args:
            client_order_id: Client-provided order ID to delete

        Returns:
            True if entry was deleted, False if not found
        """
        try:
            key = f"{self.REDIS_KEY_PREFIX}{client_order_id}"
            return bool(self._redis.delete(key))
        except Exception as e:
            logger.warning(f"Redis delete failed for {client_order_id}: {e}")
            return False


# ==================== OrderExecutor ====================


class OrderExecutor:
    """Execute orders on exchange with idempotency support."""

    def __init__(
        self,
        adapter: ExchangeAdapter,
        default_timeout: int = 300,
        idempotency_ttl: int = 3600,
    ) -> None:
        """Initialize order executor.

        Args:
            adapter: Exchange adapter
            default_timeout: Default order timeout in seconds
            idempotency_ttl: TTL for idempotency cache entries in seconds
        """
        self._adapter = adapter
        self._default_timeout = default_timeout
        self._idempotency_cache = IdempotencyCache(ttl_seconds=idempotency_ttl)

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
        """Execute an order with idempotency support.

        Args:
            request: Order request

        Returns:
            Execution result

        Raises:
            IdempotencyCacheError: If Redis idempotency cache is unavailable or fails.

        Note:
            - If client_order_id is provided, the request is idempotent.
              Duplicate requests with the same client_order_id return cached results.
            - If stop_loss_price or take_profit_price is set, this method will
              create separate stop/TP orders after the main order succeeds.
        """
        # Check idempotency cache if client_order_id is provided
        if request.client_order_id:
            cached_result = self._idempotency_cache.get(
                request.client_order_id, request
            )
            if cached_result is not None:
                return cached_result

        try:
            # Build order params including reduce_only and post_only
            order_params: dict[str, Any] = {}
            if request.reduce_only:
                order_params["reduceOnly"] = True
            if request.post_only:
                order_params["postOnly"] = True

            # Create order on exchange
            order = await self._adapter.create_order(
                symbol=request.symbol,
                side=request.side,
                order_type=request.order_type,
                quantity=request.quantity,
                price=request.price,
                **order_params,
            )

            # Calculate remaining quantity
            remaining = None
            if order.filled is not None and order.amount is not None:
                remaining = order.amount - order.filled
                if remaining <= 0:
                    remaining = None

            result = ExecutionResult(
                success=True,
                order_id=order.id,
                filled_quantity=order.filled,
                remaining_quantity=remaining,
                average_price=getattr(order, 'price', None),
                status=order.status,
                action=request.action,
            )

            # Create stop loss order if specified and main order succeeded
            if request.stop_loss_price is not None and result.success:
                await self._create_stop_order(
                    request, request.stop_loss_price, is_stop_loss=True
                )

            # Create take profit order if specified and main order succeeded
            if request.take_profit_price is not None and result.success:
                await self._create_stop_order(
                    request, request.take_profit_price, is_stop_loss=False
                )

            # Cache result for idempotency
            if request.client_order_id:
                self._idempotency_cache.set(
                    request.client_order_id, request, result
                )

            return result

        except Exception as e:
            result = ExecutionResult(
                success=False,
                error=str(e),
                status=OrderStatus.REJECTED,
            )
            # Cache failed result too to prevent retry storms
            if request.client_order_id:
                self._idempotency_cache.set(
                    request.client_order_id, request, result
                )
            return result

    async def _create_stop_order(
        self,
        request: OrderRequest,
        trigger_price: Decimal,
        is_stop_loss: bool,
    ) -> Optional[ExecutionResult]:
        """Create a stop loss or take profit order.

        Args:
            request: Original order request
            trigger_price: Stop/TP trigger price
            is_stop_loss: True for stop loss, False for take profit

        Returns:
            Execution result for the stop order, or None if failed
        """
        try:
            # Opposite side for closing order
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
                price=trigger_price,
                action=(
                    OrderAction.CLOSE_LONG
                    if request.action == OrderAction.OPEN_LONG
                    else OrderAction.CLOSE_SHORT
                ),
                reduce_only=True,
                metadata={
                    "parent_order_id": request.client_order_id,
                    "order_type": "stop_loss" if is_stop_loss else "take_profit",
                },
            )

            order = await self._adapter.create_order(
                symbol=stop_request.symbol,
                side=stop_request.side,
                order_type=stop_request.order_type,
                quantity=stop_request.quantity,
                price=stop_request.price,
                reduceOnly=True,
            )

            return ExecutionResult(
                success=True,
                order_id=order.id,
                status=order.status,
                action=stop_request.action,
            )
        except Exception as e:
            order_type_name = "stop_loss" if is_stop_loss else "take_profit"
            logger.error(
                f"Failed to create {order_type_name} order for {request.symbol}: {e}. "
                f"Parent order may be unprotected."
            )
            # Return failure result instead of silently failing
            return ExecutionResult(
                success=False,
                error=f"Failed to create {order_type_name} order: {e}",
                status=OrderStatus.REJECTED,
                action=OrderAction.CLOSE_LONG if request.action == OrderAction.OPEN_LONG else OrderAction.CLOSE_SHORT,
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


class OrderManagerError(OrderExecutionError):
    """Raised when order manager operations fail."""

    pass


class OrderManager:
    """Manage order lifecycle and tracking with Redis persistence.

    Critical state per CLAUDE.md: Order tracking must be persistent to survive
    service restarts. Uses Redis for persistence.
    """

    REDIS_KEY_PREFIX = "iqfmp:orders:"
    REDIS_ACTIVE_KEY = "iqfmp:active_orders"

    def __init__(self, adapter: ExchangeAdapter, redis_client: Any = None) -> None:
        """Initialize order manager with Redis persistence.

        Args:
            adapter: Exchange adapter
            redis_client: Optional Redis client for dependency injection (testing)

        Raises:
            OrderManagerError: If Redis is unavailable and no client injected
        """
        self._adapter = adapter
        self._redis = redis_client if redis_client is not None else self._get_redis_client()

    def _get_redis_client(self):
        """Get Redis client. Raises if unavailable (critical state requires persistence)."""
        return get_required_redis_client(OrderManagerError, "Order tracking")

    def _serialize_order(self, order: Order) -> str:
        """Serialize Order to JSON for Redis storage."""
        return json.dumps({
            "id": order.id,
            "symbol": order.symbol,
            "side": order.side.value,
            "type": order.type.value,
            "amount": str(order.amount),
            "status": order.status.value,
            "price": str(order.price) if order.price else None,
            "filled": str(order.filled),
            "remaining": str(order.remaining),
            "cost": str(order.cost),
            "fee": str(order.fee),
            "timestamp": order.timestamp.isoformat() if order.timestamp else None,
            "metadata": order.metadata,
        })

    def _deserialize_order(self, data: str) -> Order:
        """Deserialize JSON to Order."""
        obj = json.loads(data)
        return Order(
            id=obj["id"],
            symbol=obj["symbol"],
            side=OrderSide(obj["side"]),
            type=OrderType(obj["type"]),
            amount=Decimal(obj["amount"]),
            status=OrderStatus(obj["status"]),
            price=Decimal(obj["price"]) if obj.get("price") else None,
            filled=Decimal(obj["filled"]),
            remaining=Decimal(obj["remaining"]),
            cost=Decimal(obj["cost"]),
            fee=Decimal(obj["fee"]),
            timestamp=datetime.fromisoformat(obj["timestamp"]) if obj.get("timestamp") else None,
            metadata=obj.get("metadata", {}),
        )

    @property
    def active_orders(self) -> set[str]:
        """Get active order IDs from Redis."""
        try:
            members = self._redis.smembers(self.REDIS_ACTIVE_KEY)
            return set(members) if members else set()
        except Exception as e:
            logger.error(f"Failed to get active orders from Redis: {e}")
            return set()

    def track(self, order: Order) -> None:
        """Track an order with Redis persistence.

        Args:
            order: Order to track
        """
        try:
            key = f"{self.REDIS_KEY_PREFIX}{order.id}"
            self._redis.set(key, self._serialize_order(order))

            if order.status == OrderStatus.OPEN:
                self._redis.sadd(self.REDIS_ACTIVE_KEY, order.id)
            else:
                self._redis.srem(self.REDIS_ACTIVE_KEY, order.id)

            logger.debug(f"Tracked order {order.id} with status {order.status.value}")
        except Exception as e:
            logger.error(f"Failed to track order {order.id} in Redis: {e}")
            raise OrderManagerError(f"Failed to track order: {e}") from e

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID from Redis.

        Args:
            order_id: Order ID

        Returns:
            Order or None
        """
        try:
            key = f"{self.REDIS_KEY_PREFIX}{order_id}"
            data = self._redis.get(key)
            if data:
                return self._deserialize_order(data)
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Corrupted order data for {order_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get order {order_id} from Redis: {e}")
            return None

    async def update_status(self, order_id: str) -> Order:
        """Update order status from exchange.

        Args:
            order_id: Order ID

        Returns:
            Updated order

        Raises:
            ValueError: If order not found
        """
        order = self.get_order(order_id)
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

        Raises:
            ValueError: If order not found
        """
        order = self.get_order(order_id)
        if order is None:
            raise ValueError(f"Order {order_id} not found")

        await self._adapter.cancel_order(order_id, order.symbol)
        # Update order status
        updated = await self._adapter.fetch_order(order_id, order.symbol)
        self.track(updated)
        return updated

    def get_orders_by_symbol(self, symbol: str) -> list[Order]:
        """Get orders by symbol from Redis.

        Args:
            symbol: Trading symbol

        Returns:
            List of orders
        """
        try:
            # Scan for all order keys
            cursor = 0
            orders = []
            while True:
                cursor, keys = self._redis.scan(cursor, match=f"{self.REDIS_KEY_PREFIX}*", count=100)
                for key in keys:
                    data = self._redis.get(key)
                    if data:
                        try:
                            order = self._deserialize_order(data)
                            if order.symbol == symbol:
                                orders.append(order)
                        except Exception as e:
                            logger.error(f"Failed to deserialize order from Redis key {key}: {e}")
                            continue
                if cursor == 0:
                    break
            return orders
        except Exception as e:
            logger.error(f"Failed to get orders by symbol {symbol}: {e}")
            return []

    def get_history(self, limit: int = 100) -> list[Order]:
        """Get order history from Redis.

        Args:
            limit: Maximum number of orders

        Returns:
            List of orders sorted by timestamp descending
        """
        try:
            # Scan for all order keys
            cursor = 0
            orders = []
            while True:
                cursor, keys = self._redis.scan(cursor, match=f"{self.REDIS_KEY_PREFIX}*", count=100)
                for key in keys:
                    data = self._redis.get(key)
                    if data:
                        try:
                            order = self._deserialize_order(data)
                            orders.append(order)
                        except Exception as e:
                            logger.error(f"Failed to deserialize order from Redis key {key}: {e}")
                            continue
                if cursor == 0:
                    break

            # Sort by timestamp descending (None timestamps go last)
            orders.sort(
                key=lambda o: o.timestamp if o.timestamp else datetime.min,
                reverse=True,
            )
            return orders[:limit]
        except Exception as e:
            logger.error(f"Failed to get order history: {e}")
            return []


# ==================== PartialFillHandler ====================


class PartialFillHandlerError(Exception):
    """Raised when partial fill handler operations fail."""

    pass


class PartialFillHandler:
    """Handle partial fills for orders.

    Uses Redis for persistence per CLAUDE.md critical state rules.
    Partial fills are critical for order reconciliation and audit.
    """

    REDIS_PREFIX = "partial_fills:"
    TTL_SECONDS = 86400 * 7  # 7 days

    def __init__(self, redis_client: Any = None) -> None:
        """Initialize partial fill handler with Redis persistence.

        Args:
            redis_client: Optional Redis client for dependency injection (testing)

        Raises:
            PartialFillHandlerError: If Redis is unavailable and no client injected
        """
        self._redis = redis_client if redis_client is not None else self._get_redis_client()

    def _get_redis_client(self) -> Any:
        """Get Redis client. Raises if unavailable (critical state requires persistence)."""
        return get_required_redis_client(PartialFillHandlerError, "Partial fills")

    def _serialize_fill(self, fill: PartialFill) -> str:
        """Serialize PartialFill to JSON."""
        return json.dumps({
            "order_id": fill.order_id,
            "filled_quantity": str(fill.filled_quantity),
            "fill_price": str(fill.fill_price),
            "timestamp": fill.timestamp.isoformat(),
            "trade_id": fill.trade_id,
            "commission": str(fill.commission) if fill.commission else None,
        })

    def _deserialize_fill(self, data: str) -> PartialFill:
        """Deserialize JSON to PartialFill."""
        obj = json.loads(data)
        return PartialFill(
            order_id=obj["order_id"],
            filled_quantity=Decimal(obj["filled_quantity"]),
            fill_price=Decimal(obj["fill_price"]),
            timestamp=datetime.fromisoformat(obj["timestamp"]),
            trade_id=obj.get("trade_id"),
            commission=Decimal(obj["commission"]) if obj.get("commission") else None,
        )

    def record(self, fill: PartialFill) -> None:
        """Record a partial fill.

        Args:
            fill: Partial fill to record

        Raises:
            PartialFillHandlerError: If persistence fails
        """
        try:
            key = f"{self.REDIS_PREFIX}{fill.order_id}"
            self._redis.rpush(key, self._serialize_fill(fill))
            self._redis.expire(key, self.TTL_SECONDS)
        except Exception as e:
            logger.error(f"Failed to record partial fill for order {fill.order_id}: {e}")
            raise PartialFillHandlerError(
                f"Failed to record partial fill for order {fill.order_id}: {e}. "
                "Order reconciliation will be incomplete."
            ) from e

    def get_fills(self, order_id: str) -> list[PartialFill]:
        """Get fills for an order.

        Args:
            order_id: Order ID

        Returns:
            List of partial fills

        Raises:
            PartialFillHandlerError: If Redis operation fails
        """
        try:
            key = f"{self.REDIS_PREFIX}{order_id}"
            fills_data = self._redis.lrange(key, 0, -1)
            fills = []
            for data in fills_data:
                try:
                    fills.append(self._deserialize_fill(data))
                except Exception as e:
                    logger.error(f"Failed to deserialize partial fill: {e}")
                    raise PartialFillHandlerError(
                        f"Failed to deserialize partial fill: {e}"
                    ) from e
            return fills
        except PartialFillHandlerError:
            raise
        except Exception as e:
            logger.error(f"Failed to get partial fills for order {order_id}: {e}")
            raise PartialFillHandlerError(
                f"Failed to get partial fills for order {order_id}: {e}. "
                "Order fill status unknown."
            ) from e

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


class TimeoutHandlerError(Exception):
    """Raised when timeout handler operations fail."""

    pass


class TimeoutHandler:
    """Handle order timeouts with Redis persistence.

    Critical state per CLAUDE.md: Timeouts must be persistent to survive
    service restarts. Uses Redis for persistence.
    """

    REDIS_KEY_PREFIX = "iqfmp:timeouts:"
    REDIS_INDEX_KEY = "iqfmp:timeouts:index"

    def __init__(
        self, adapter: ExchangeAdapter, redis_client: Any = None
    ) -> None:
        """Initialize timeout handler with Redis persistence.

        Args:
            adapter: Exchange adapter
            redis_client: Optional Redis client for dependency injection (testing)

        Raises:
            TimeoutHandlerError: If Redis is unavailable and no client injected
        """
        self._adapter = adapter
        self._redis = redis_client if redis_client is not None else self._get_redis_client()
        # Callbacks cannot be serialized to Redis, store in memory only
        # On restart, callbacks must be re-registered by callers
        self._callbacks: dict[str, Callable[[str], None]] = {}
        # Load existing timeouts from Redis
        self._load_timeouts()

    def _get_redis_client(self) -> Any:
        """Get Redis client. Raises if unavailable (critical state requires persistence)."""
        return get_required_redis_client(TimeoutHandlerError, "Timeout storage")

    def _serialize_timeout(self, timeout: OrderTimeout) -> str:
        """Serialize timeout to JSON (excluding callback)."""
        return json.dumps({
            "order_id": timeout.order_id,
            "timeout_seconds": timeout.timeout_seconds,
            "created_at": timeout.created_at.isoformat(),
            "symbol": timeout.symbol,
        })

    def _deserialize_timeout(self, data: str) -> OrderTimeout:
        """Deserialize JSON to OrderTimeout (without callback)."""
        obj = json.loads(data)
        return OrderTimeout(
            order_id=obj["order_id"],
            timeout_seconds=obj["timeout_seconds"],
            created_at=datetime.fromisoformat(obj["created_at"]),
            symbol=obj.get("symbol"),
            callback=None,  # Callbacks must be re-registered after restart
        )

    def _load_timeouts(self) -> None:
        """Load existing timeouts from Redis on startup."""
        try:
            order_ids = self._redis.smembers(self.REDIS_INDEX_KEY)
            if order_ids:
                for order_id in order_ids:
                    if isinstance(order_id, bytes):
                        order_id = order_id.decode("utf-8")
                    key = f"{self.REDIS_KEY_PREFIX}{order_id}"
                    data = self._redis.get(key)
                    if data:
                        # Just verify it exists and is valid
                        self._deserialize_timeout(data)
                logger.info(f"Loaded {len(order_ids)} timeouts from Redis")
        except Exception as e:
            logger.warning(f"Failed to load timeouts from Redis: {e}")

    def register(self, timeout: OrderTimeout) -> None:
        """Register order timeout.

        Args:
            timeout: Timeout configuration

        Raises:
            TimeoutHandlerError: If persistence fails
        """
        try:
            key = f"{self.REDIS_KEY_PREFIX}{timeout.order_id}"
            self._redis.set(key, self._serialize_timeout(timeout))
            self._redis.sadd(self.REDIS_INDEX_KEY, timeout.order_id)
            # Store callback in memory if provided
            if timeout.callback:
                self._callbacks[timeout.order_id] = timeout.callback
            logger.debug(f"Registered timeout for order {timeout.order_id}")
        except Exception as e:
            logger.error(f"Failed to register timeout for order {timeout.order_id}: {e}")
            raise TimeoutHandlerError(
                f"Failed to register timeout: {e}"
            ) from e

    def unregister(self, order_id: str) -> None:
        """Unregister order timeout.

        Args:
            order_id: Order ID

        Raises:
            TimeoutHandlerError: If persistence fails
        """
        try:
            key = f"{self.REDIS_KEY_PREFIX}{order_id}"
            self._redis.delete(key)
            self._redis.srem(self.REDIS_INDEX_KEY, order_id)
            self._callbacks.pop(order_id, None)
            logger.debug(f"Unregistered timeout for order {order_id}")
        except Exception as e:
            logger.error(f"Failed to unregister timeout for order {order_id}: {e}")
            raise TimeoutHandlerError(
                f"Failed to unregister timeout: {e}"
            ) from e

    def _get_timeout(self, order_id: str) -> Optional[OrderTimeout]:
        """Get timeout from Redis.

        Args:
            order_id: Order ID

        Returns:
            OrderTimeout if found, None otherwise
        """
        try:
            key = f"{self.REDIS_KEY_PREFIX}{order_id}"
            data = self._redis.get(key)
            if data:
                timeout = self._deserialize_timeout(data)
                # Restore callback from memory if available
                timeout.callback = self._callbacks.get(order_id)
                return timeout
        except Exception as e:
            logger.error(f"Failed to get timeout for order {order_id}: {e}")
            raise TimeoutHandlerError(f"Failed to get timeout: {e}") from e
        return None

    def has_timeout(self, order_id: str) -> bool:
        """Check if order has timeout registered.

        Args:
            order_id: Order ID

        Returns:
            True if timeout registered
        """
        try:
            return self._redis.sismember(self.REDIS_INDEX_KEY, order_id)
        except Exception as e:
            logger.error(f"Failed to check timeout for order {order_id}: {e}")
            raise TimeoutHandlerError(f"Failed to check timeout: {e}") from e

    def is_expired(self, order_id: str) -> bool:
        """Check if order timeout has expired.

        Args:
            order_id: Order ID

        Returns:
            True if expired
        """
        timeout = self._get_timeout(order_id)
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
            logger.info(f"Cancelled expired order {order_id} on {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel expired order {order_id} on {symbol}: {e}")
            raise OrderExecutionError(f"Failed to cancel expired order {order_id}: {e}") from e

    async def check_and_cancel_expired(self) -> list[str]:
        """Check all timeouts and cancel expired orders.

        Returns:
            List of cancelled order IDs
        """
        cancelled: list[str] = []

        try:
            order_ids = self._redis.smembers(self.REDIS_INDEX_KEY)
        except Exception as e:
            logger.error(f"Failed to get timeout list: {e}")
            raise TimeoutHandlerError(f"Failed to check timeouts: {e}") from e

        # Create a copy to avoid modifying set during iteration
        for order_id in list(order_ids):
            if isinstance(order_id, bytes):
                order_id = order_id.decode("utf-8")

            if self.is_expired(order_id):
                timeout = self._get_timeout(order_id)
                symbol = timeout.symbol if timeout else ""
                try:
                    await self._adapter.cancel_order(order_id, symbol or "")
                    cancelled.append(order_id)
                    self.unregister(order_id)
                    logger.info(f"Cancelled expired order {order_id}")

                    # Call callback if set
                    callback = self._callbacks.get(order_id)
                    if callback:
                        callback(order_id)
                except Exception as e:
                    logger.error(f"Failed to cancel expired order {order_id} on {symbol}: {e}")

        return cancelled


# ==================== Module Exports ====================


__all__ = [
    # Enums
    "OrderAction",
    "OrderDirection",
    # Exceptions
    "IdempotencyCacheError",
    "OrderExecutionError",
    "OrderManagerError",
    "PartialFillHandlerError",
    "TimeoutHandlerError",
    # Models
    "ExecutionResult",
    "IdempotencyCacheEntry",
    "OrderRequest",
    "OrderTimeout",
    "PartialFill",
    # Classes
    "IdempotencyCache",
    "OrderExecutor",
    "OrderManager",
    "PartialFillHandler",
    "TimeoutHandler",
]
