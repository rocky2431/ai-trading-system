"""Trading service for real-time trading operations.

Integrates with exchange adapters to provide:
- Position management
- Order execution
- Account monitoring
- Risk management
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import uuid4

from iqfmp.api.trading.schemas import (
    AccountInfo,
    CloseAllPositionsResponse,
    ClosePositionResponse,
    CreateOrderResponse,
    ExchangeStatus,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    PnLDataPoint,
    Position,
    PositionSide,
    RiskAlert,
    RiskLevel,
    RiskMetrics,
    TradingConfig,
    TradingState,
)
from iqfmp.exchange.adapter import (
    BinanceAdapter,
    ConnectionManager,
    ExchangeAdapter,
    ExchangeConfig,
    ExchangeType,
    OKXAdapter,
    Order as ExchangeOrder,
    OrderSide as ExchangeOrderSide,
    OrderType as ExchangeOrderType,
)

logger = logging.getLogger(__name__)


class TradingService:
    """Service for real-time trading operations."""

    def __init__(self) -> None:
        """Initialize trading service."""
        self._adapter: Optional[ExchangeAdapter] = None
        self._connection_manager: Optional[ConnectionManager] = None
        self._config = TradingConfig(
            enabled=False,
            symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            max_leverage=20,
            default_leverage=1,
        )
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._pnl_history: list[PnLDataPoint] = []
        self._connected = False
        self._last_heartbeat: Optional[datetime] = None
        self._error: Optional[str] = None

    async def initialize(self) -> bool:
        """Initialize connection to exchange.

        Returns:
            True if connected successfully.
        """
        try:
            # Get exchange credentials from environment
            exchange_id = os.environ.get("EXCHANGE_ID", "binance")
            api_key = os.environ.get("EXCHANGE_API_KEY", "")
            api_secret = os.environ.get("EXCHANGE_SECRET", "")
            passphrase = os.environ.get("EXCHANGE_PASSPHRASE")  # For OKX

            if not api_key or not api_secret:
                self._error = "Exchange API credentials not configured"
                logger.warning(self._error)
                return False

            # Determine exchange type
            exchange_type = ExchangeType.BINANCE
            if exchange_id.lower() == "okx":
                exchange_type = ExchangeType.OKX

            # Create exchange config
            config = ExchangeConfig(
                exchange_type=exchange_type,
                api_key=api_key,
                api_secret=api_secret,
                passphrase=passphrase,
                sandbox=os.environ.get("EXCHANGE_SANDBOX", "true").lower() == "true",
            )
            self._config.exchange_id = exchange_id

            # Create appropriate adapter
            if exchange_type == ExchangeType.BINANCE:
                self._adapter = BinanceAdapter(config)
            else:
                self._adapter = OKXAdapter(config)

            # Connect with retry logic
            self._connection_manager = ConnectionManager(max_retries=3)
            await self._connection_manager.connect(self._adapter)

            self._connected = True
            self._last_heartbeat = datetime.now(timezone.utc)
            self._error = None
            self._config.enabled = True

            logger.info(f"Connected to {exchange_id} exchange")
            return True

        except Exception as e:
            self._error = str(e)
            self._connected = False
            logger.error(f"Failed to connect to exchange: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from exchange."""
        if self._adapter:
            await self._adapter.disconnect()
        self._connected = False
        self._config.enabled = False

    # ==================== Position Management ====================

    async def get_positions(self) -> list[Position]:
        """Get all open positions.

        Returns:
            List of positions.
        """
        if not self._connected or not self._adapter:
            return list(self._positions.values())

        try:
            # Fetch positions from exchange
            # Note: ccxt doesn't have a unified positions method,
            # this would need exchange-specific implementation
            # For now, return cached positions
            return list(self._positions.values())
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return list(self._positions.values())

    async def close_position(
        self,
        position_id: str,
        price: Optional[float] = None,
    ) -> ClosePositionResponse:
        """Close a specific position.

        Args:
            position_id: Position ID to close.
            price: Optional limit price.

        Returns:
            Close position response.
        """
        position = self._positions.get(position_id)
        if not position:
            return ClosePositionResponse(
                success=False,
                message=f"Position {position_id} not found",
            )

        if not self._connected or not self._adapter:
            return ClosePositionResponse(
                success=False,
                message="Not connected to exchange",
            )

        try:
            # Determine close side
            close_side = (
                ExchangeOrderSide.SELL
                if position.side == PositionSide.LONG
                else ExchangeOrderSide.BUY
            )

            # Create close order
            order_type = (
                ExchangeOrderType.LIMIT if price else ExchangeOrderType.MARKET
            )

            order = await self._adapter.create_order(
                symbol=position.symbol,
                side=close_side,
                order_type=order_type,
                amount=position.size,
                price=price,
            )

            # Calculate realized PnL
            realized_pnl = position.unrealized_pnl

            # Remove position from cache
            del self._positions[position_id]

            return ClosePositionResponse(
                success=True,
                message=f"Position {position_id} closed",
                order_id=order.id,
                realized_pnl=realized_pnl,
            )

        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return ClosePositionResponse(
                success=False,
                message=str(e),
            )

    async def close_all_positions(self) -> CloseAllPositionsResponse:
        """Close all open positions.

        Returns:
            Close all positions response.
        """
        if not self._connected or not self._adapter:
            return CloseAllPositionsResponse(
                success=False,
                message="Not connected to exchange",
                closed_count=0,
                total_realized_pnl=0,
            )

        positions = list(self._positions.values())
        if not positions:
            return CloseAllPositionsResponse(
                success=True,
                message="No positions to close",
                closed_count=0,
                total_realized_pnl=0,
            )

        total_pnl = 0.0
        closed_count = 0
        errors = []

        for position in positions:
            result = await self.close_position(position.id)
            if result.success:
                closed_count += 1
                total_pnl += result.realized_pnl or 0
            else:
                errors.append(f"{position.symbol}: {result.message}")

        message = f"Closed {closed_count}/{len(positions)} positions"
        if errors:
            message += f". Errors: {'; '.join(errors)}"

        return CloseAllPositionsResponse(
            success=closed_count == len(positions),
            message=message,
            closed_count=closed_count,
            total_realized_pnl=total_pnl,
        )

    # ==================== Order Management ====================

    async def get_orders(self, symbol: Optional[str] = None) -> list[Order]:
        """Get open orders.

        Args:
            symbol: Optional symbol filter.

        Returns:
            List of orders.
        """
        orders = list(self._orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        size: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        leverage: int = 1,
        reduce_only: bool = False,
    ) -> CreateOrderResponse:
        """Create a new order.

        Args:
            symbol: Trading symbol.
            side: Buy or sell.
            order_type: Order type.
            size: Order size.
            price: Limit price (required for limit orders).
            stop_price: Stop trigger price.
            leverage: Position leverage.
            reduce_only: Only reduce position.

        Returns:
            Create order response.
        """
        if not self._connected or not self._adapter:
            return CreateOrderResponse(
                success=False,
                message="Not connected to exchange",
            )

        try:
            # Map order types
            exchange_side = (
                ExchangeOrderSide.BUY if side == OrderSide.BUY
                else ExchangeOrderSide.SELL
            )

            exchange_type = ExchangeOrderType.MARKET
            if order_type == OrderType.LIMIT:
                exchange_type = ExchangeOrderType.LIMIT
            elif order_type == OrderType.STOP:
                exchange_type = ExchangeOrderType.STOP_MARKET
            elif order_type == OrderType.STOP_LIMIT:
                exchange_type = ExchangeOrderType.STOP_LIMIT

            # Create order on exchange
            exchange_order = await self._adapter.create_order(
                symbol=symbol,
                side=exchange_side,
                order_type=exchange_type,
                amount=size,
                price=price,
            )

            # Convert to our order model
            order = self._convert_exchange_order(exchange_order)
            self._orders[order.id] = order

            return CreateOrderResponse(
                success=True,
                message="Order created successfully",
                order=order,
            )

        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            return CreateOrderResponse(
                success=False,
                message=str(e),
            )

    async def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        """Cancel an order.

        Args:
            order_id: Order ID.
            symbol: Trading symbol.

        Returns:
            Cancel result.
        """
        if not self._connected or not self._adapter:
            return {"success": False, "message": "Not connected to exchange"}

        try:
            await self._adapter.cancel_order(order_id, symbol)
            if order_id in self._orders:
                del self._orders[order_id]
            return {"success": True, "message": f"Order {order_id} canceled"}
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return {"success": False, "message": str(e)}

    async def cancel_all_orders(
        self, symbol: Optional[str] = None
    ) -> dict[str, Any]:
        """Cancel all open orders.

        Args:
            symbol: Optional symbol filter.

        Returns:
            Cancel result.
        """
        orders = await self.get_orders(symbol)
        canceled = 0
        errors = []

        for order in orders:
            result = await self.cancel_order(order.id, order.symbol)
            if result["success"]:
                canceled += 1
            else:
                errors.append(f"{order.id}: {result['message']}")

        return {
            "success": len(errors) == 0,
            "message": f"Canceled {canceled}/{len(orders)} orders",
            "canceled_count": canceled,
        }

    # ==================== Account & State ====================

    async def get_account_info(self) -> AccountInfo:
        """Get account information.

        Returns:
            Account info.
        """
        if not self._connected or not self._adapter:
            # Return default/cached values
            return AccountInfo(
                totalEquity=0,
                availableBalance=0,
                marginUsed=0,
                unrealizedPnl=0,
                realizedPnl=0,
                todayPnl=0,
                todayPnlPercent=0,
            )

        try:
            balances = await self._adapter.fetch_balance()
            usdt_balance = balances.get("USDT")

            total = usdt_balance.total if usdt_balance else 0
            free = usdt_balance.free if usdt_balance else 0
            used = usdt_balance.used if usdt_balance else 0

            # Calculate unrealized PnL from positions
            unrealized = sum(p.unrealized_pnl for p in self._positions.values())

            return AccountInfo(
                totalEquity=total,
                availableBalance=free,
                marginUsed=used,
                unrealizedPnl=unrealized,
                realizedPnl=0,  # Would need trade history
                todayPnl=0,
                todayPnlPercent=0,
            )

        except Exception as e:
            logger.error(f"Failed to fetch account info: {e}")
            return AccountInfo(
                totalEquity=0,
                availableBalance=0,
                marginUsed=0,
                unrealizedPnl=0,
                realizedPnl=0,
                todayPnl=0,
                todayPnlPercent=0,
            )

    async def get_risk_metrics(self) -> RiskMetrics:
        """Get risk metrics.

        Returns:
            Risk metrics.
        """
        account = await self.get_account_info()
        positions = await self.get_positions()

        # Calculate metrics
        margin_usage = 0.0
        if account.total_equity > 0:
            margin_usage = (account.margin_used / account.total_equity) * 100

        # Position concentration
        concentration = 0.0
        if positions and account.margin_used > 0:
            max_position = max(p.margin_used for p in positions)
            concentration = (max_position / account.margin_used) * 100

        # Determine risk level
        level = RiskLevel.NORMAL
        if margin_usage > 80 or concentration > 80:
            level = RiskLevel.CRITICAL
        elif margin_usage > 60 or concentration > 60:
            level = RiskLevel.DANGER
        elif margin_usage > 40 or concentration > 40:
            level = RiskLevel.WARNING

        # Generate alerts
        alerts = []
        if margin_usage > 60:
            alerts.append(RiskAlert(
                id=str(uuid4()),
                type="margin",
                message=f"Margin usage at {margin_usage:.1f}%",
                severity=RiskLevel.WARNING if margin_usage < 80 else RiskLevel.DANGER,
                timestamp=datetime.now(timezone.utc),
            ))

        return RiskMetrics(
            level=level,
            marginUsagePercent=margin_usage,
            maxDrawdownPercent=0,  # Would need historical data
            currentDrawdownPercent=0,
            dailyLossPercent=0,
            positionConcentration=concentration,
            alerts=alerts,
        )

    async def get_trading_state(self) -> TradingState:
        """Get complete trading state.

        Returns:
            Trading state.
        """
        account = await self.get_account_info()
        positions = await self.get_positions()
        orders = await self.get_orders()
        risk = await self.get_risk_metrics()

        return TradingState(
            account=account,
            positions=positions,
            openOrders=orders,
            pnlHistory=self._pnl_history[-24:],  # Last 24 data points
            risk=risk,
            isConnected=self._connected,
            lastUpdated=datetime.now(timezone.utc),
        )

    # ==================== Configuration ====================

    def get_config(self) -> TradingConfig:
        """Get trading configuration."""
        return self._config

    def update_config(
        self,
        enabled: Optional[bool] = None,
        symbols: Optional[list[str]] = None,
        max_leverage: Optional[int] = None,
        default_leverage: Optional[int] = None,
        risk_controls: Optional[dict[str, Any]] = None,
    ) -> TradingConfig:
        """Update trading configuration."""
        if enabled is not None:
            self._config.enabled = enabled
        if symbols is not None:
            self._config.symbols = symbols
        if max_leverage is not None:
            self._config.max_leverage = max_leverage
        if default_leverage is not None:
            self._config.default_leverage = default_leverage
        if risk_controls is not None:
            self._config.risk_controls = risk_controls
        return self._config

    def get_exchange_status(self) -> ExchangeStatus:
        """Get exchange connection status."""
        return ExchangeStatus(
            exchange_id=self._config.exchange_id or "none",
            connected=self._connected,
            last_heartbeat=self._last_heartbeat,
            error=self._error,
        )

    # ==================== Helpers ====================

    def _convert_exchange_order(self, exchange_order: ExchangeOrder) -> Order:
        """Convert exchange order to our order model."""
        return Order(
            id=exchange_order.id,
            symbol=exchange_order.symbol,
            side=OrderSide.BUY if exchange_order.side.value == "buy" else OrderSide.SELL,
            type=self._map_order_type(exchange_order.type),
            price=exchange_order.price,
            size=exchange_order.amount,
            filled=exchange_order.filled,
            remaining=exchange_order.remaining,
            status=self._map_order_status(exchange_order.status),
            createdAt=exchange_order.timestamp or datetime.now(timezone.utc),
        )

    def _map_order_type(self, exchange_type: ExchangeOrderType) -> OrderType:
        """Map exchange order type to our order type."""
        mapping = {
            ExchangeOrderType.LIMIT: OrderType.LIMIT,
            ExchangeOrderType.MARKET: OrderType.MARKET,
            ExchangeOrderType.STOP_LIMIT: OrderType.STOP_LIMIT,
            ExchangeOrderType.STOP_MARKET: OrderType.STOP,
        }
        return mapping.get(exchange_type, OrderType.MARKET)

    def _map_order_status(
        self, exchange_status: "ExchangeOrderStatus"
    ) -> OrderStatus:
        """Map exchange order status to our order status."""
        from iqfmp.exchange.adapter import OrderStatus as ExchangeOrderStatus
        mapping = {
            ExchangeOrderStatus.OPEN: OrderStatus.OPEN,
            ExchangeOrderStatus.CLOSED: OrderStatus.CLOSED,
            ExchangeOrderStatus.CANCELED: OrderStatus.CANCELED,
            ExchangeOrderStatus.EXPIRED: OrderStatus.EXPIRED,
            ExchangeOrderStatus.REJECTED: OrderStatus.REJECTED,
        }
        return mapping.get(exchange_status, OrderStatus.OPEN)


# Global service instance
_trading_service: Optional[TradingService] = None


def get_trading_service() -> TradingService:
    """Get or create trading service instance."""
    global _trading_service
    if _trading_service is None:
        _trading_service = TradingService()
    return _trading_service
