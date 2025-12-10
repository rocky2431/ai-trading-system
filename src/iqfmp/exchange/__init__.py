"""Exchange module for IQFMP.

Provides tools for cryptocurrency exchange connections:
- ExchangeAdapter: Abstract exchange interface
- BinanceAdapter: Binance Futures connection
- OKXAdapter: OKX Swap connection
- ConnectionManager: Connection management
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
]
