"""Exchange Adapter for cryptocurrency trading.

This module provides:
- ExchangeAdapter: Abstract base class for exchange connections
- BinanceAdapter: Binance Futures (USDT-M) implementation
- OKXAdapter: OKX Swap implementation
- ConnectionManager: Connection management with auto-reconnect
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

try:
    import ccxt
    import ccxt.async_support as ccxt_async
except ImportError:
    ccxt = None  # type: ignore
    ccxt_async = None  # type: ignore


# ============ Exceptions ============


class ExchangeError(Exception):
    """Base exception for exchange errors."""

    pass


class ConnectionError(ExchangeError):
    """Connection error."""

    pass


class AuthenticationError(ExchangeError):
    """Authentication error."""

    pass


class InsufficientFundsError(ExchangeError):
    """Insufficient funds error."""

    pass


class OrderNotFoundError(ExchangeError):
    """Order not found error."""

    pass


# ============ Enums ============


class ExchangeType(Enum):
    """Supported exchange types."""

    BINANCE = "binance"
    OKX = "okx"


class OrderSide(Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""

    LIMIT = "limit"
    MARKET = "market"
    STOP_LIMIT = "stop_limit"
    STOP_MARKET = "stop_market"


class OrderStatus(Enum):
    """Order status."""

    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"


class ConnectionStatus(Enum):
    """Connection status."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"


# ============ Data Models ============


@dataclass
class Ticker:
    """Market ticker data."""

    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume: Decimal = Decimal("0")
    timestamp: Optional[datetime] = None

    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> Decimal:
        """Calculate spread as percentage of mid price."""
        mid = (self.bid + self.ask) / 2
        if mid == Decimal("0"):
            return Decimal("0")
        return self.spread / mid


@dataclass
class OrderBook:
    """Order book data."""

    symbol: str
    bids: list[list[float]]  # [[price, amount], ...]
    asks: list[list[float]]
    timestamp: Optional[datetime] = None

    @property
    def best_bid(self) -> Optional[float]:
        """Get best bid price."""
        if not self.bids:
            return None
        return self.bids[0][0]

    @property
    def best_ask(self) -> Optional[float]:
        """Get best ask price."""
        if not self.asks:
            return None
        return self.asks[0][0]

    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price."""
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2


@dataclass
class OHLCV:
    """OHLCV candle data."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Balance:
    """Account balance."""

    currency: str
    free: Decimal
    used: Decimal
    total: Decimal


@dataclass
class Order:
    """Order data."""

    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    amount: Decimal
    status: OrderStatus
    price: Optional[Decimal] = None
    filled: Decimal = Decimal("0")
    remaining: Decimal = Decimal("0")
    cost: Decimal = Decimal("0")
    fee: Decimal = Decimal("0")
    timestamp: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate derived fields, normalize types, and validate invariants."""
        # Convert float inputs to Decimal for consistency
        if not isinstance(self.amount, Decimal):
            self.amount = Decimal(str(self.amount))
        if self.price is not None and not isinstance(self.price, Decimal):
            self.price = Decimal(str(self.price))
        if not isinstance(self.filled, Decimal):
            self.filled = Decimal(str(self.filled))
        if not isinstance(self.remaining, Decimal):
            self.remaining = Decimal(str(self.remaining))
        if not isinstance(self.cost, Decimal):
            self.cost = Decimal(str(self.cost))
        if not isinstance(self.fee, Decimal):
            self.fee = Decimal(str(self.fee))

        # Validate invariants
        if self.amount <= Decimal("0"):
            raise ValueError(f"amount must be positive, got {self.amount}")
        if self.type == OrderType.LIMIT and self.price is None:
            raise ValueError("LIMIT orders require price")
        if self.filled > self.amount:
            raise ValueError(f"filled ({self.filled}) cannot exceed amount ({self.amount})")

        if self.remaining == Decimal("0"):
            self.remaining = self.amount - self.filled
        if self.cost == Decimal("0") and self.price and self.filled:
            self.cost = self.price * self.filled

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == OrderStatus.CLOSED or self.filled >= self.amount


@dataclass
class Position:
    """Open position data."""

    symbol: str
    side: OrderSide
    size: Decimal
    entry_price: Decimal
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    leverage: int = 1
    liquidation_price: Optional[Decimal] = None
    margin: Decimal = Decimal("0")
    timestamp: Optional[datetime] = None

    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value of position."""
        return abs(self.size) * self.entry_price


# ============ Configuration ============


@dataclass
class ExchangeConfig:
    """Exchange configuration."""

    exchange_type: ExchangeType
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None  # For OKX
    sandbox: bool = False
    timeout: int = 30000
    rate_limit: bool = True
    options: dict[str, Any] = field(default_factory=dict)


# ============ Abstract Adapter ============


class ExchangeAdapter(ABC):
    """Abstract base class for exchange adapters."""

    def __init__(self, config: ExchangeConfig) -> None:
        """Initialize adapter.

        Args:
            config: Exchange configuration
        """
        self.config = config
        self._exchange: Any = None

    @abstractmethod
    async def connect(self) -> None:
        """Connect to exchange."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from exchange."""
        pass

    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> Ticker:
        """Fetch ticker for symbol."""
        pass

    @abstractmethod
    async def fetch_orderbook(
        self, symbol: str, limit: int = 20
    ) -> OrderBook:
        """Fetch order book for symbol."""
        pass

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        limit: int = 100,
    ) -> list[OHLCV]:
        """Fetch OHLCV data."""
        pass

    @abstractmethod
    async def fetch_balance(self) -> dict[str, Balance]:
        """Fetch account balance."""
        pass

    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        amount: Decimal,
        price: Optional[Decimal] = None,
        **kwargs: Any,
    ) -> Order:
        """Create an order.

        Args:
            symbol: Trading pair symbol
            side: Order side (buy/sell)
            order_type: Order type (limit/market/stop)
            amount: Order amount (Decimal for financial precision)
            price: Limit price (Decimal for financial precision)
            **kwargs: Additional order parameters (reduceOnly, postOnly, etc.)

        Returns:
            Created order
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order."""
        pass

    @abstractmethod
    async def fetch_order(self, order_id: str, symbol: str) -> Order:
        """Fetch order status."""
        pass

    @abstractmethod
    async def fetch_positions(self, symbol: Optional[str] = None) -> list[Position]:
        """Fetch open positions.

        Args:
            symbol: Optional symbol filter. If None, fetch all positions.

        Returns:
            List of open positions.
        """
        pass


# ============ CCXT Base Adapter ============


class CCXTBaseAdapter(ExchangeAdapter):
    """Base adapter with shared CCXT implementation.

    Subclasses only need to implement _setup_exchange() to configure
    the specific exchange instance (Binance, OKX, etc.).
    """

    # Override in subclass for exchange-specific error messages
    _exchange_name: str = "Exchange"

    def __init__(self, config: ExchangeConfig) -> None:
        """Initialize adapter.

        Args:
            config: Exchange configuration
        """
        super().__init__(config)
        self._setup_exchange()

    @abstractmethod
    def _setup_exchange(self) -> None:
        """Setup ccxt exchange instance. Must be implemented by subclass."""
        pass

    async def connect(self) -> None:
        """Connect to exchange."""
        try:
            await self._exchange.load_markets()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self._exchange_name}: {e}")

    async def disconnect(self) -> None:
        """Disconnect from exchange."""
        if hasattr(self._exchange, "close"):
            await self._exchange.close()

    async def fetch_ticker(self, symbol: str) -> Ticker:
        """Fetch ticker data."""
        try:
            data = await self._exchange.fetch_ticker(symbol)
            return Ticker(
                symbol=data["symbol"],
                bid=Decimal(str(data.get("bid", 0))),
                ask=Decimal(str(data.get("ask", 0))),
                last=Decimal(str(data.get("last", 0))),
                volume=Decimal(str(data.get("baseVolume", 0))),
                timestamp=datetime.fromtimestamp(
                    data["timestamp"] / 1000
                ) if data.get("timestamp") else None,
            )
        except Exception as e:
            raise ExchangeError(f"Failed to fetch ticker: {e}")

    async def fetch_orderbook(self, symbol: str, limit: int = 20) -> OrderBook:
        """Fetch order book data."""
        try:
            data = await self._exchange.fetch_order_book(symbol, limit)
            return OrderBook(
                symbol=data.get("symbol", symbol),
                bids=data.get("bids", []),
                asks=data.get("asks", []),
            )
        except Exception as e:
            raise ExchangeError(f"Failed to fetch orderbook: {e}")

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1m", limit: int = 100
    ) -> list[OHLCV]:
        """Fetch OHLCV candle data."""
        try:
            data = await self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return [
                OHLCV(
                    timestamp=datetime.fromtimestamp(candle[0] / 1000),
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[5]),
                )
                for candle in data
            ]
        except Exception as e:
            raise ExchangeError(f"Failed to fetch OHLCV: {e}")

    async def fetch_balance(self) -> dict[str, Balance]:
        """Fetch account balance."""
        try:
            data = await self._exchange.fetch_balance()
            result = {}
            for currency, balance in data.items():
                if isinstance(balance, dict) and "free" in balance:
                    result[currency] = Balance(
                        currency=currency,
                        free=Decimal(str(balance.get("free", 0))),
                        used=Decimal(str(balance.get("used", 0))),
                        total=Decimal(str(balance.get("total", 0))),
                    )
            return result
        except Exception as e:
            raise ExchangeError(f"Failed to fetch balance: {e}")

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        amount: Decimal,
        price: Optional[Decimal] = None,
        **kwargs: Any,
    ) -> Order:
        """Create an order."""
        try:
            params = {}
            if kwargs.get("reduceOnly"):
                params["reduceOnly"] = True
            if kwargs.get("postOnly"):
                params["postOnly"] = True

            data = await self._exchange.create_order(
                symbol=symbol,
                type=order_type.value,
                side=side.value,
                amount=float(amount),
                price=float(price) if price else None,
                params=params,
            )
            return self._parse_order(data)
        except Exception as e:
            if "insufficient" in str(e).lower():
                raise InsufficientFundsError(str(e))
            raise ExchangeError(f"Failed to create order: {e}")

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order."""
        try:
            await self._exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            if "not found" in str(e).lower():
                raise OrderNotFoundError(f"Order {order_id} not found")
            raise ExchangeError(f"Failed to cancel order: {e}")

    async def fetch_order(self, order_id: str, symbol: str) -> Order:
        """Fetch order status."""
        try:
            data = await self._exchange.fetch_order(order_id, symbol)
            return self._parse_order(data)
        except Exception as e:
            if "not found" in str(e).lower():
                raise OrderNotFoundError(f"Order {order_id} not found")
            raise ExchangeError(f"Failed to fetch order: {e}")

    def _parse_order(self, data: dict[str, Any]) -> Order:
        """Parse order data from ccxt format."""
        status_map = {
            "open": OrderStatus.OPEN,
            "closed": OrderStatus.CLOSED,
            "canceled": OrderStatus.CANCELED,
            "expired": OrderStatus.EXPIRED,
            "rejected": OrderStatus.REJECTED,
        }
        return Order(
            id=str(data["id"]),
            symbol=data["symbol"],
            side=OrderSide.BUY if data["side"] == "buy" else OrderSide.SELL,
            type=OrderType.LIMIT if data["type"] == "limit" else OrderType.MARKET,
            price=Decimal(str(data["price"])) if data.get("price") else None,
            amount=Decimal(str(data["amount"])),
            status=status_map.get(data["status"], OrderStatus.OPEN),
            filled=Decimal(str(data.get("filled", 0))),
        )

    async def fetch_positions(self, symbol: Optional[str] = None) -> list[Position]:
        """Fetch open positions."""
        try:
            positions = await self._exchange.fetch_positions(
                symbols=[symbol] if symbol else None
            )
            result = []
            for pos in positions:
                size = Decimal(str(pos.get("contracts", 0) or pos.get("contractSize", 0) or 0))
                if size == Decimal("0"):
                    continue

                side = OrderSide.BUY if pos.get("side") == "long" else OrderSide.SELL
                result.append(Position(
                    symbol=pos["symbol"],
                    side=side,
                    size=size,
                    entry_price=Decimal(str(pos.get("entryPrice", 0) or 0)),
                    unrealized_pnl=Decimal(str(pos.get("unrealizedPnl", 0) or 0)),
                    realized_pnl=Decimal(str(pos.get("realizedPnl", 0) or 0)),
                    leverage=int(pos.get("leverage", 1) or 1),
                    liquidation_price=Decimal(str(pos.get("liquidationPrice"))) if pos.get("liquidationPrice") else None,
                    margin=Decimal(str(pos.get("initialMargin", 0) or pos.get("margin", 0) or 0)),
                    timestamp=datetime.fromtimestamp(pos["timestamp"] / 1000) if pos.get("timestamp") else None,
                ))
            return result
        except Exception as e:
            raise ExchangeError(f"Failed to fetch positions: {e}")


# ============ Binance Adapter ============


class BinanceAdapter(CCXTBaseAdapter):
    """Binance Futures (USDT-M) adapter."""

    _exchange_name = "Binance"

    def _setup_exchange(self) -> None:
        """Setup Binance ccxt instance."""
        if ccxt is None:
            raise ImportError("ccxt is required for exchange connections")

        options = {
            "defaultType": "future",
            "adjustForTimeDifference": True,
            **self.config.options,
        }

        self._exchange = ccxt.binanceusdm({
            "apiKey": self.config.api_key,
            "secret": self.config.api_secret,
            "sandbox": self.config.sandbox,
            "timeout": self.config.timeout,
            "enableRateLimit": self.config.rate_limit,
            "options": options,
        })


# ============ OKX Adapter ============


class OKXAdapter(CCXTBaseAdapter):
    """OKX Swap adapter."""

    _exchange_name = "OKX"

    def _setup_exchange(self) -> None:
        """Setup OKX ccxt instance."""
        if ccxt is None:
            raise ImportError("ccxt is required for exchange connections")

        options = {
            "defaultType": "swap",
            **self.config.options,
        }

        self._exchange = ccxt.okx({
            "apiKey": self.config.api_key,
            "secret": self.config.api_secret,
            "password": self.config.passphrase,
            "sandbox": self.config.sandbox,
            "timeout": self.config.timeout,
            "enableRateLimit": self.config.rate_limit,
            "options": options,
        })


# ============ Connection Manager ============


class ConnectionManager:
    """Manages exchange connections with auto-reconnect."""

    def __init__(
        self,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        heartbeat_interval: float = 30.0,
    ) -> None:
        """Initialize connection manager.

        Args:
            max_retries: Maximum reconnection attempts
            retry_delay: Delay between retries (seconds)
            heartbeat_interval: Heartbeat check interval
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.heartbeat_interval = heartbeat_interval
        self.status = ConnectionStatus.DISCONNECTED
        self._adapter: Optional[ExchangeAdapter] = None

    async def connect(self, adapter: ExchangeAdapter) -> None:
        """Connect to exchange with retry logic.

        Args:
            adapter: Exchange adapter to connect

        Raises:
            ConnectionError: If connection fails after max retries
        """
        self._adapter = adapter
        self.status = ConnectionStatus.CONNECTING

        for attempt in range(self.max_retries):
            try:
                await adapter.connect()
                self.status = ConnectionStatus.CONNECTED
                return
            except ConnectionError:
                if attempt == self.max_retries - 1:
                    self.status = ConnectionStatus.DISCONNECTED
                    raise
                self.status = ConnectionStatus.RECONNECTING
                import asyncio
                await asyncio.sleep(self.retry_delay)

        self.status = ConnectionStatus.DISCONNECTED
        raise ConnectionError("Max retries exceeded")

    async def disconnect(self) -> None:
        """Disconnect from exchange."""
        if self._adapter:
            await self._adapter.disconnect()
        self.status = ConnectionStatus.DISCONNECTED
