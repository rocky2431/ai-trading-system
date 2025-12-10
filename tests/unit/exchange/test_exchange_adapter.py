"""Tests for Exchange Adapter (Task 18)."""

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iqfmp.exchange.adapter import (
    # Core
    ExchangeAdapter,
    ExchangeConfig,
    ExchangeType,
    # Implementations
    BinanceAdapter,
    OKXAdapter,
    # Models
    Ticker,
    OrderBook,
    OHLCV,
    Balance,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    # Connection
    ConnectionManager,
    ConnectionStatus,
    # Exceptions
    ExchangeError,
    ConnectionError,
    AuthenticationError,
    InsufficientFundsError,
    OrderNotFoundError,
)


# ============ Ticker Tests ============

class TestTicker:
    """Tests for Ticker model."""

    def test_create_ticker(self) -> None:
        """Test creating a ticker."""
        ticker = Ticker(
            symbol="BTC/USDT",
            bid=50000.0,
            ask=50010.0,
            last=50005.0,
            volume=1000.0,
            timestamp=datetime.now(),
        )
        assert ticker.symbol == "BTC/USDT"
        assert ticker.bid == 50000.0

    def test_ticker_spread(self) -> None:
        """Test spread calculation."""
        ticker = Ticker(
            symbol="BTC/USDT",
            bid=50000.0,
            ask=50010.0,
            last=50005.0,
            volume=1000.0,
        )
        assert ticker.spread == pytest.approx(10.0)

    def test_ticker_spread_pct(self) -> None:
        """Test spread percentage."""
        ticker = Ticker(
            symbol="BTC/USDT",
            bid=50000.0,
            ask=50010.0,
            last=50005.0,
            volume=1000.0,
        )
        assert ticker.spread_pct == pytest.approx(0.0002, rel=1e-3)  # ~0.02%


# ============ OrderBook Tests ============

class TestOrderBook:
    """Tests for OrderBook model."""

    def test_create_orderbook(self) -> None:
        """Test creating an order book."""
        book = OrderBook(
            symbol="BTC/USDT",
            bids=[[50000.0, 1.0], [49990.0, 2.0]],
            asks=[[50010.0, 1.5], [50020.0, 0.5]],
        )
        assert book.symbol == "BTC/USDT"
        assert len(book.bids) == 2

    def test_orderbook_best_bid(self) -> None:
        """Test best bid price."""
        book = OrderBook(
            symbol="BTC/USDT",
            bids=[[50000.0, 1.0], [49990.0, 2.0]],
            asks=[[50010.0, 1.5]],
        )
        assert book.best_bid == 50000.0

    def test_orderbook_best_ask(self) -> None:
        """Test best ask price."""
        book = OrderBook(
            symbol="BTC/USDT",
            bids=[[50000.0, 1.0]],
            asks=[[50010.0, 1.5], [50020.0, 0.5]],
        )
        assert book.best_ask == 50010.0

    def test_orderbook_mid_price(self) -> None:
        """Test mid price calculation."""
        book = OrderBook(
            symbol="BTC/USDT",
            bids=[[50000.0, 1.0]],
            asks=[[50010.0, 1.5]],
        )
        assert book.mid_price == pytest.approx(50005.0)


# ============ Order Tests ============

class TestOrder:
    """Tests for Order model."""

    def test_create_order(self) -> None:
        """Test creating an order."""
        order = Order(
            id="123456",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            price=50000.0,
            amount=0.1,
            status=OrderStatus.OPEN,
        )
        assert order.id == "123456"
        assert order.side == OrderSide.BUY

    def test_order_cost(self) -> None:
        """Test order cost calculation."""
        order = Order(
            id="123456",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            price=50000.0,
            amount=0.1,
            status=OrderStatus.CLOSED,
            filled=0.1,
        )
        assert order.cost == pytest.approx(5000.0)

    def test_order_is_filled(self) -> None:
        """Test filled check."""
        order = Order(
            id="123456",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            amount=0.1,
            status=OrderStatus.CLOSED,
            filled=0.1,
        )
        assert order.is_filled


# ============ ExchangeConfig Tests ============

class TestExchangeConfig:
    """Tests for ExchangeConfig."""

    def test_create_config(self) -> None:
        """Test creating config."""
        config = ExchangeConfig(
            exchange_type=ExchangeType.BINANCE,
            api_key="test_key",
            api_secret="test_secret",
        )
        assert config.exchange_type == ExchangeType.BINANCE

    def test_config_sandbox_mode(self) -> None:
        """Test sandbox mode config."""
        config = ExchangeConfig(
            exchange_type=ExchangeType.BINANCE,
            api_key="test_key",
            api_secret="test_secret",
            sandbox=True,
        )
        assert config.sandbox is True


# ============ ExchangeAdapter Interface Tests ============

class TestExchangeAdapterInterface:
    """Tests for ExchangeAdapter interface."""

    def test_adapter_is_abstract(self) -> None:
        """Test that adapter is abstract."""
        with pytest.raises(TypeError):
            ExchangeAdapter()  # type: ignore


# ============ BinanceAdapter Tests ============

class TestBinanceAdapter:
    """Tests for BinanceAdapter."""

    @pytest.fixture
    def config(self) -> ExchangeConfig:
        """Create test config."""
        return ExchangeConfig(
            exchange_type=ExchangeType.BINANCE,
            api_key="test_key",
            api_secret="test_secret",
            sandbox=True,
        )

    @pytest.fixture
    def adapter(self, config: ExchangeConfig) -> BinanceAdapter:
        """Create adapter with mocked ccxt."""
        with patch("iqfmp.exchange.adapter.ccxt") as mock_ccxt:
            mock_exchange = MagicMock()
            mock_ccxt.binanceusdm.return_value = mock_exchange
            adapter = BinanceAdapter(config)
            adapter._exchange = mock_exchange
            return adapter

    def test_adapter_creation(self, config: ExchangeConfig) -> None:
        """Test adapter creation."""
        with patch("iqfmp.exchange.adapter.ccxt"):
            adapter = BinanceAdapter(config)
            assert adapter is not None

    @pytest.mark.asyncio
    async def test_fetch_ticker(self, adapter: BinanceAdapter) -> None:
        """Test fetching ticker."""
        adapter._exchange.fetch_ticker = AsyncMock(return_value={
            "symbol": "BTC/USDT",
            "bid": 50000.0,
            "ask": 50010.0,
            "last": 50005.0,
            "baseVolume": 1000.0,
            "timestamp": 1704067200000,
        })

        ticker = await adapter.fetch_ticker("BTC/USDT")
        assert ticker.symbol == "BTC/USDT"
        assert ticker.bid == 50000.0

    @pytest.mark.asyncio
    async def test_fetch_orderbook(self, adapter: BinanceAdapter) -> None:
        """Test fetching order book."""
        adapter._exchange.fetch_order_book = AsyncMock(return_value={
            "symbol": "BTC/USDT",
            "bids": [[50000.0, 1.0], [49990.0, 2.0]],
            "asks": [[50010.0, 1.5], [50020.0, 0.5]],
        })

        book = await adapter.fetch_orderbook("BTC/USDT")
        assert book.symbol == "BTC/USDT"
        assert len(book.bids) == 2

    @pytest.mark.asyncio
    async def test_fetch_ohlcv(self, adapter: BinanceAdapter) -> None:
        """Test fetching OHLCV data."""
        adapter._exchange.fetch_ohlcv = AsyncMock(return_value=[
            [1704067200000, 50000.0, 50100.0, 49900.0, 50050.0, 100.0],
            [1704067260000, 50050.0, 50150.0, 50000.0, 50100.0, 150.0],
        ])

        ohlcv = await adapter.fetch_ohlcv("BTC/USDT", "1m", limit=2)
        assert len(ohlcv) == 2
        assert ohlcv[0].open == 50000.0

    @pytest.mark.asyncio
    async def test_fetch_balance(self, adapter: BinanceAdapter) -> None:
        """Test fetching balance."""
        adapter._exchange.fetch_balance = AsyncMock(return_value={
            "USDT": {"free": 10000.0, "used": 1000.0, "total": 11000.0},
            "BTC": {"free": 0.5, "used": 0.1, "total": 0.6},
        })

        balance = await adapter.fetch_balance()
        assert "USDT" in balance
        assert balance["USDT"].free == 10000.0

    @pytest.mark.asyncio
    async def test_create_order(self, adapter: BinanceAdapter) -> None:
        """Test creating an order."""
        adapter._exchange.create_order = AsyncMock(return_value={
            "id": "123456",
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "price": 50000.0,
            "amount": 0.1,
            "status": "open",
            "filled": 0.0,
        })

        order = await adapter.create_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=0.1,
            price=50000.0,
        )
        assert order.id == "123456"
        assert order.side == OrderSide.BUY

    @pytest.mark.asyncio
    async def test_cancel_order(self, adapter: BinanceAdapter) -> None:
        """Test canceling an order."""
        adapter._exchange.cancel_order = AsyncMock(return_value={
            "id": "123456",
            "symbol": "BTC/USDT",
            "status": "canceled",
        })

        result = await adapter.cancel_order("123456", "BTC/USDT")
        assert result is True

    @pytest.mark.asyncio
    async def test_fetch_order(self, adapter: BinanceAdapter) -> None:
        """Test fetching order status."""
        adapter._exchange.fetch_order = AsyncMock(return_value={
            "id": "123456",
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "price": 50000.0,
            "amount": 0.1,
            "status": "closed",
            "filled": 0.1,
        })

        order = await adapter.fetch_order("123456", "BTC/USDT")
        assert order.status == OrderStatus.CLOSED


# ============ OKXAdapter Tests ============

class TestOKXAdapter:
    """Tests for OKXAdapter."""

    @pytest.fixture
    def config(self) -> ExchangeConfig:
        """Create test config."""
        return ExchangeConfig(
            exchange_type=ExchangeType.OKX,
            api_key="test_key",
            api_secret="test_secret",
            passphrase="test_pass",
            sandbox=True,
        )

    @pytest.fixture
    def adapter(self, config: ExchangeConfig) -> OKXAdapter:
        """Create adapter with mocked ccxt."""
        with patch("iqfmp.exchange.adapter.ccxt") as mock_ccxt:
            mock_exchange = MagicMock()
            mock_ccxt.okx.return_value = mock_exchange
            adapter = OKXAdapter(config)
            adapter._exchange = mock_exchange
            return adapter

    def test_adapter_creation(self, config: ExchangeConfig) -> None:
        """Test adapter creation."""
        with patch("iqfmp.exchange.adapter.ccxt"):
            adapter = OKXAdapter(config)
            assert adapter is not None

    @pytest.mark.asyncio
    async def test_fetch_ticker(self, adapter: OKXAdapter) -> None:
        """Test fetching ticker."""
        adapter._exchange.fetch_ticker = AsyncMock(return_value={
            "symbol": "BTC/USDT:USDT",
            "bid": 50000.0,
            "ask": 50010.0,
            "last": 50005.0,
            "baseVolume": 1000.0,
            "timestamp": 1704067200000,
        })

        ticker = await adapter.fetch_ticker("BTC/USDT:USDT")
        assert ticker.bid == 50000.0


# ============ ConnectionManager Tests ============

class TestConnectionManager:
    """Tests for ConnectionManager."""

    @pytest.fixture
    def manager(self) -> ConnectionManager:
        """Create connection manager."""
        return ConnectionManager(
            max_retries=3,
            retry_delay=0.1,
            heartbeat_interval=1.0,
        )

    def test_manager_creation(self, manager: ConnectionManager) -> None:
        """Test manager creation."""
        assert manager.max_retries == 3
        assert manager.status == ConnectionStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_connect(self, manager: ConnectionManager) -> None:
        """Test connection."""
        mock_adapter = MagicMock()
        mock_adapter.connect = AsyncMock()

        await manager.connect(mock_adapter)
        assert manager.status == ConnectionStatus.CONNECTED

    @pytest.mark.asyncio
    async def test_disconnect(self, manager: ConnectionManager) -> None:
        """Test disconnection."""
        mock_adapter = MagicMock()
        mock_adapter.connect = AsyncMock()
        mock_adapter.disconnect = AsyncMock()

        await manager.connect(mock_adapter)
        await manager.disconnect()
        assert manager.status == ConnectionStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_reconnect_on_failure(self, manager: ConnectionManager) -> None:
        """Test automatic reconnection."""
        mock_adapter = MagicMock()
        # Fail first 2 times, succeed on 3rd
        mock_adapter.connect = AsyncMock(
            side_effect=[
                ConnectionError("Failed"),
                ConnectionError("Failed"),
                None,
            ]
        )

        await manager.connect(mock_adapter)
        assert manager.status == ConnectionStatus.CONNECTED
        assert mock_adapter.connect.call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, manager: ConnectionManager) -> None:
        """Test max retries exceeded."""
        mock_adapter = MagicMock()
        mock_adapter.connect = AsyncMock(
            side_effect=ConnectionError("Always fails")
        )

        with pytest.raises(ConnectionError):
            await manager.connect(mock_adapter)

        assert manager.status == ConnectionStatus.DISCONNECTED


# ============ Exception Tests ============

class TestExchangeExceptions:
    """Tests for exchange exceptions."""

    def test_exchange_error(self) -> None:
        """Test ExchangeError."""
        error = ExchangeError("Test error")
        assert str(error) == "Test error"

    def test_connection_error(self) -> None:
        """Test ConnectionError."""
        error = ConnectionError("Connection failed")
        assert "Connection" in str(error)

    def test_authentication_error(self) -> None:
        """Test AuthenticationError."""
        error = AuthenticationError("Invalid API key")
        assert "Invalid" in str(error)

    def test_insufficient_funds_error(self) -> None:
        """Test InsufficientFundsError."""
        error = InsufficientFundsError("Not enough USDT")
        assert "USDT" in str(error)

    def test_order_not_found_error(self) -> None:
        """Test OrderNotFoundError."""
        error = OrderNotFoundError("Order 123 not found")
        assert "123" in str(error)


# ============ Boundary Tests ============

class TestExchangeBoundary:
    """Boundary tests for exchange adapter."""

    @pytest.fixture
    def adapter(self) -> BinanceAdapter:
        """Create adapter with mocked ccxt."""
        config = ExchangeConfig(
            exchange_type=ExchangeType.BINANCE,
            api_key="test",
            api_secret="test",
        )
        with patch("iqfmp.exchange.adapter.ccxt"):
            return BinanceAdapter(config)

    @pytest.mark.asyncio
    async def test_invalid_symbol(self, adapter: BinanceAdapter) -> None:
        """Test invalid symbol handling."""
        adapter._exchange.fetch_ticker = AsyncMock(
            side_effect=Exception("Invalid symbol")
        )

        with pytest.raises(ExchangeError):
            await adapter.fetch_ticker("INVALID/PAIR")

    @pytest.mark.asyncio
    async def test_empty_orderbook(self, adapter: BinanceAdapter) -> None:
        """Test empty order book handling."""
        adapter._exchange.fetch_order_book = AsyncMock(return_value={
            "symbol": "BTC/USDT",
            "bids": [],
            "asks": [],
        })

        book = await adapter.fetch_orderbook("BTC/USDT")
        assert book.best_bid is None
        assert book.best_ask is None


# ============ Performance Tests ============

class TestExchangePerformance:
    """Performance tests for exchange adapter."""

    @pytest.mark.asyncio
    async def test_rapid_ticker_fetches(self) -> None:
        """Test rapid ticker fetches."""
        import time

        config = ExchangeConfig(
            exchange_type=ExchangeType.BINANCE,
            api_key="test",
            api_secret="test",
        )

        with patch("iqfmp.exchange.adapter.ccxt") as mock_ccxt:
            mock_exchange = MagicMock()
            mock_exchange.fetch_ticker = AsyncMock(return_value={
                "symbol": "BTC/USDT",
                "bid": 50000.0,
                "ask": 50010.0,
                "last": 50005.0,
                "baseVolume": 1000.0,
            })
            mock_ccxt.binanceusdm.return_value = mock_exchange

            adapter = BinanceAdapter(config)
            adapter._exchange = mock_exchange

            start = time.time()
            for _ in range(100):
                await adapter.fetch_ticker("BTC/USDT")
            elapsed = time.time() - start

            assert elapsed < 1.0  # Should be very fast with mocks
