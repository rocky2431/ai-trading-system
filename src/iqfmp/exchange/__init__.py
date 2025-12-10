"""Exchange module for IQFMP.

Provides tools for cryptocurrency exchange connections:
- ExchangeAdapter: Abstract exchange interface
- BinanceAdapter: Binance Futures connection
- OKXAdapter: OKX Swap connection
- ConnectionManager: Connection management
- OrderExecutor: Order execution engine
- OrderManager: Order lifecycle management
"""

from iqfmp.exchange.adapter import (
    # Adapters
    BinanceAdapter,
    ExchangeAdapter,
    OKXAdapter,
    # Models
    Balance,
    OHLCV,
    Order,
    OrderBook,
    Ticker,
    # Enums
    ConnectionStatus,
    ExchangeType,
    OrderSide,
    OrderStatus,
    OrderType,
    # Config
    ExchangeConfig,
    # Manager
    ConnectionManager,
    # Exceptions
    AuthenticationError,
    ConnectionError,
    ExchangeError,
    InsufficientFundsError,
    OrderNotFoundError,
)
from iqfmp.exchange.execution import (
    # Enums
    OrderAction,
    OrderDirection,
    # Exceptions
    OrderExecutionError,
    # Models
    ExecutionResult,
    OrderRequest,
    OrderTimeout,
    PartialFill,
    # Classes
    OrderExecutor,
    OrderManager,
    PartialFillHandler,
    TimeoutHandler,
)
from iqfmp.exchange.monitoring import (
    # Enums
    MarginLevel,
    PnLType,
    PositionSide,
    UpdateType,
    # Models
    MarginAlert,
    PnLRecord,
    PositionData,
    UpdateEvent,
    # Classes
    MarginMonitor,
    PnLCalculator,
    PositionTracker,
    RealtimeUpdater,
)
from iqfmp.exchange.risk import (
    # Enums
    RiskActionType,
    RiskLevel,
    RiskRuleType,
    # Models
    ConcentrationAlert,
    ConcentrationBreach,
    DrawdownAlert,
    LossAlert,
    LossRecord,
    RiskAction,
    RiskConfig,
    RiskRule,
    RiskStatus,
    RiskViolation,
    # Classes
    ConcentrationChecker,
    DrawdownMonitor,
    LossLimiter,
    RiskController,
)

__all__ = [
    # Adapters
    "BinanceAdapter",
    "ExchangeAdapter",
    "OKXAdapter",
    # Models
    "Balance",
    "OHLCV",
    "Order",
    "OrderBook",
    "Ticker",
    # Enums
    "ConnectionStatus",
    "ExchangeType",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    # Config
    "ExchangeConfig",
    # Manager
    "ConnectionManager",
    # Exceptions
    "AuthenticationError",
    "ConnectionError",
    "ExchangeError",
    "InsufficientFundsError",
    "OrderNotFoundError",
    # Execution - Enums
    "OrderAction",
    "OrderDirection",
    # Execution - Exceptions
    "OrderExecutionError",
    # Execution - Models
    "ExecutionResult",
    "OrderRequest",
    "OrderTimeout",
    "PartialFill",
    # Execution - Classes
    "OrderExecutor",
    "OrderManager",
    "PartialFillHandler",
    "TimeoutHandler",
    # Monitoring - Enums
    "MarginLevel",
    "PnLType",
    "PositionSide",
    "UpdateType",
    # Monitoring - Models
    "MarginAlert",
    "PnLRecord",
    "PositionData",
    "UpdateEvent",
    # Monitoring - Classes
    "MarginMonitor",
    "PnLCalculator",
    "PositionTracker",
    "RealtimeUpdater",
    # Risk - Enums
    "RiskActionType",
    "RiskLevel",
    "RiskRuleType",
    # Risk - Models
    "ConcentrationAlert",
    "ConcentrationBreach",
    "DrawdownAlert",
    "LossAlert",
    "LossRecord",
    "RiskAction",
    "RiskConfig",
    "RiskRule",
    "RiskStatus",
    "RiskViolation",
    # Risk - Classes
    "ConcentrationChecker",
    "DrawdownMonitor",
    "LossLimiter",
    "RiskController",
]
