"""Trading API schemas.

Pydantic models for trading endpoints.

Note: Financial values use Decimal with custom JSON serializers per CLAUDE.md.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field, PlainSerializer, model_validator

# Custom Decimal type that serializes to string for JSON precision
DecimalStr = Annotated[
    Decimal,
    PlainSerializer(lambda x: str(x), return_type=str),
]


# ============== Enums ==============


class PositionSide(str, Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"


class OrderSide(str, Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""
    LIMIT = "limit"
    MARKET = "market"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order status."""
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"
    PARTIALLY_FILLED = "partially_filled"


class RiskLevel(str, Enum):
    """Risk level."""
    NORMAL = "normal"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"


# ============== Position Models ==============


class Position(BaseModel):
    """Trading position."""
    id: str
    symbol: str
    side: PositionSide
    size: DecimalStr
    entry_price: DecimalStr = Field(alias="entryPrice")
    mark_price: DecimalStr = Field(alias="markPrice")
    leverage: int = 1
    unrealized_pnl: DecimalStr = Field(alias="unrealizedPnl")
    unrealized_pnl_percent: DecimalStr = Field(alias="unrealizedPnlPercent")
    margin_used: DecimalStr = Field(alias="marginUsed")
    liquidation_price: Optional[DecimalStr] = Field(None, alias="liquidationPrice")
    created_at: datetime = Field(alias="createdAt")

    class Config:
        populate_by_name = True


class PositionResponse(BaseModel):
    """Response containing positions."""
    positions: list[Position]
    total: int


class ClosePositionRequest(BaseModel):
    """Request to close a position."""
    reduce_only: bool = True
    price: Optional[DecimalStr] = None  # For limit close


class ClosePositionResponse(BaseModel):
    """Response after closing a position."""
    success: bool
    message: str
    order_id: Optional[str] = None
    realized_pnl: Optional[DecimalStr] = None


class CloseAllPositionsResponse(BaseModel):
    """Response after closing all positions."""
    success: bool
    message: str
    closed_count: int
    total_realized_pnl: DecimalStr


# ============== Order Models ==============


class Order(BaseModel):
    """Trading order."""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    price: Optional[DecimalStr] = None
    size: DecimalStr
    filled: DecimalStr = Decimal("0")
    remaining: DecimalStr = Decimal("0")
    status: OrderStatus
    created_at: datetime = Field(alias="createdAt")
    updated_at: Optional[datetime] = Field(None, alias="updatedAt")

    class Config:
        populate_by_name = True


class OrderResponse(BaseModel):
    """Response containing orders."""
    orders: list[Order]
    total: int


class CreateOrderRequest(BaseModel):
    """Request to create an order."""
    symbol: str = Field(..., min_length=1, description="Trading pair symbol (e.g., BTCUSDT)")
    side: OrderSide
    type: OrderType
    size: DecimalStr = Field(..., gt=0, description="Order size, must be positive")
    price: Optional[DecimalStr] = Field(None, gt=0, description="Limit price, required for LIMIT orders")
    stop_price: Optional[DecimalStr] = Field(None, gt=0, description="Stop trigger price")
    leverage: int = Field(1, ge=1, le=125, description="Leverage multiplier (1-125)")
    reduce_only: bool = False
    post_only: bool = False
    client_order_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_order_type_requirements(self) -> "CreateOrderRequest":
        """Validate that LIMIT and STOP_LIMIT orders have price, STOP orders have stop_price."""
        if self.type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and self.price is None:
            raise ValueError(f"{self.type.value} orders require a price")
        if self.type in (OrderType.STOP, OrderType.STOP_LIMIT) and self.stop_price is None:
            raise ValueError(f"{self.type.value} orders require a stop_price")
        return self


class CreateOrderResponse(BaseModel):
    """Response after creating an order."""
    success: bool
    message: str
    order: Optional[Order] = None


class CancelOrderResponse(BaseModel):
    """Response after canceling an order."""
    success: bool
    message: str


class CancelAllOrdersResponse(BaseModel):
    """Response after canceling all orders."""
    success: bool
    message: str
    canceled_count: int


# ============== Account Models ==============


class AccountInfo(BaseModel):
    """Account information."""
    total_equity: DecimalStr = Field(alias="totalEquity")
    available_balance: DecimalStr = Field(alias="availableBalance")
    margin_used: DecimalStr = Field(alias="marginUsed")
    unrealized_pnl: DecimalStr = Field(alias="unrealizedPnl")
    realized_pnl: DecimalStr = Field(alias="realizedPnl")
    today_pnl: DecimalStr = Field(alias="todayPnl")
    today_pnl_percent: DecimalStr = Field(alias="todayPnlPercent")

    class Config:
        populate_by_name = True


class PnLDataPoint(BaseModel):
    """PnL history data point."""
    timestamp: datetime
    realized_pnl: DecimalStr = Field(alias="realizedPnl")
    unrealized_pnl: DecimalStr = Field(alias="unrealizedPnl")
    total_pnl: DecimalStr = Field(alias="totalPnl")
    equity: DecimalStr

    class Config:
        populate_by_name = True


# ============== Risk Models ==============


class RiskAlert(BaseModel):
    """Risk alert."""
    id: str
    type: str
    message: str
    severity: RiskLevel
    timestamp: datetime


class RiskMetrics(BaseModel):
    """Risk metrics."""
    level: RiskLevel
    margin_usage_percent: DecimalStr = Field(alias="marginUsagePercent")
    max_drawdown_percent: DecimalStr = Field(alias="maxDrawdownPercent")
    current_drawdown_percent: DecimalStr = Field(alias="currentDrawdownPercent")
    daily_loss_percent: DecimalStr = Field(alias="dailyLossPercent")
    position_concentration: DecimalStr = Field(alias="positionConcentration")
    alerts: list[RiskAlert]

    class Config:
        populate_by_name = True


# ============== Trading State ==============


class TradingState(BaseModel):
    """Complete trading state."""
    account: AccountInfo
    positions: list[Position]
    open_orders: list[Order] = Field(alias="openOrders")
    pnl_history: list[PnLDataPoint] = Field(alias="pnlHistory")
    risk: RiskMetrics
    is_connected: bool = Field(alias="isConnected")
    last_updated: datetime = Field(alias="lastUpdated")

    class Config:
        populate_by_name = True


# ============== Exchange Config ==============


class ExchangeStatus(BaseModel):
    """Exchange connection status."""
    exchange_id: str
    connected: bool
    last_heartbeat: Optional[datetime] = None
    error: Optional[str] = None


class TradingConfig(BaseModel):
    """Trading configuration."""
    enabled: bool
    exchange_id: Optional[str] = None
    symbols: list[str] = []
    max_leverage: int = 20
    default_leverage: int = 1
    risk_controls: dict[str, Any] = {}


class UpdateTradingConfigRequest(BaseModel):
    """Request to update trading config."""
    enabled: Optional[bool] = None
    symbols: Optional[list[str]] = None
    max_leverage: Optional[int] = None
    default_leverage: Optional[int] = None
    risk_controls: Optional[dict[str, Any]] = None


class TradingConfigResponse(BaseModel):
    """Response containing trading config."""
    config: TradingConfig
    exchange_status: Optional[ExchangeStatus] = None
