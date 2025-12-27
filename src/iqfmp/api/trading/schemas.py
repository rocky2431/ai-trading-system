"""Trading API schemas.

Pydantic models for trading endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


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
    size: float
    entry_price: float = Field(alias="entryPrice")
    mark_price: float = Field(alias="markPrice")
    leverage: int = 1
    unrealized_pnl: float = Field(alias="unrealizedPnl")
    unrealized_pnl_percent: float = Field(alias="unrealizedPnlPercent")
    margin_used: float = Field(alias="marginUsed")
    liquidation_price: Optional[float] = Field(None, alias="liquidationPrice")
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
    price: Optional[float] = None  # For limit close


class ClosePositionResponse(BaseModel):
    """Response after closing a position."""
    success: bool
    message: str
    order_id: Optional[str] = None
    realized_pnl: Optional[float] = None


class CloseAllPositionsResponse(BaseModel):
    """Response after closing all positions."""
    success: bool
    message: str
    closed_count: int
    total_realized_pnl: float


# ============== Order Models ==============


class Order(BaseModel):
    """Trading order."""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    price: Optional[float] = None
    size: float
    filled: float = 0
    remaining: float = 0
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
    symbol: str
    side: OrderSide
    type: OrderType
    size: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    leverage: int = 1
    reduce_only: bool = False
    post_only: bool = False
    client_order_id: Optional[str] = None


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
    total_equity: float = Field(alias="totalEquity")
    available_balance: float = Field(alias="availableBalance")
    margin_used: float = Field(alias="marginUsed")
    unrealized_pnl: float = Field(alias="unrealizedPnl")
    realized_pnl: float = Field(alias="realizedPnl")
    today_pnl: float = Field(alias="todayPnl")
    today_pnl_percent: float = Field(alias="todayPnlPercent")

    class Config:
        populate_by_name = True


class PnLDataPoint(BaseModel):
    """PnL history data point."""
    timestamp: datetime
    realized_pnl: float = Field(alias="realizedPnl")
    unrealized_pnl: float = Field(alias="unrealizedPnl")
    total_pnl: float = Field(alias="totalPnl")
    equity: float

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
    margin_usage_percent: float = Field(alias="marginUsagePercent")
    max_drawdown_percent: float = Field(alias="maxDrawdownPercent")
    current_drawdown_percent: float = Field(alias="currentDrawdownPercent")
    daily_loss_percent: float = Field(alias="dailyLossPercent")
    position_concentration: float = Field(alias="positionConcentration")
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
