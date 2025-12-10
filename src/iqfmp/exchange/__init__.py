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
]
