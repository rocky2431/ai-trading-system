"""Unit tests for CryptoQlibBacktest unified engine.

Tests cover:
- CryptoBacktestConfig initialization
- CryptoExchange funding rate calculations
- CryptoQlibBacktest single-asset backtest
- CryptoQlibBacktest multi-asset backtest
- Liquidation detection
- Funding rate settlement

No mocks - uses real implementations with synthetic data.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from iqfmp.core.crypto_backtest import (
    BacktestTrade,
    CryptoBacktestConfig,
    CryptoBacktestResult,
    CryptoExchange,
    CryptoQlibBacktest,
    PositionType,
    SettlementEvent,
    SettlementRecord,
    run_crypto_backtest,
)
from iqfmp.exchange.margin import MarginCalculator, MarginConfig, MarginMode


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_config() -> CryptoBacktestConfig:
    """Default backtest configuration."""
    return CryptoBacktestConfig(
        initial_capital=100000.0,
        leverage=10,
        margin_mode=MarginMode.ISOLATED,
        commission_rate=0.0004,
        slippage_rate=0.0001,
        funding_enabled=True,
        funding_hours=[0, 8, 16],
        liquidation_enabled=True,
    )


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n_bars = 300  # Need >= 250 for Purged CV validation
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="h")

    close = 3000 + np.cumsum(np.random.randn(n_bars) * 10)
    high = close + np.random.rand(n_bars) * 20
    low = close - np.random.rand(n_bars) * 20
    open_ = close + np.random.randn(n_bars) * 5

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.uniform(1000, 5000, n_bars),
            "funding_rate": np.random.randn(n_bars) * 0.0001,
        },
        index=dates,
    )


@pytest.fixture
def momentum_signals(sample_ohlcv_data: pd.DataFrame) -> pd.Series:
    """Generate momentum-based trading signals."""
    returns = sample_ohlcv_data["close"].pct_change()
    signals = pd.Series(0, index=sample_ohlcv_data.index)
    signals[returns > 0.002] = 1
    signals[returns < -0.002] = -1
    return signals


# =============================================================================
# CryptoBacktestConfig Tests
# =============================================================================


class TestCryptoBacktestConfig:
    """Tests for CryptoBacktestConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CryptoBacktestConfig()
        assert config.initial_capital == 100000.0
        assert config.leverage == 10
        assert config.margin_mode == MarginMode.ISOLATED
        assert config.commission_rate == 0.0004
        assert config.funding_enabled is True
        assert config.funding_hours == [0, 8, 16]
        assert config.liquidation_enabled is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = CryptoBacktestConfig(
            initial_capital=50000.0,
            leverage=20,
            funding_enabled=False,
            liquidation_enabled=False,
        )
        assert config.initial_capital == 50000.0
        assert config.leverage == 20
        assert config.funding_enabled is False
        assert config.liquidation_enabled is False


# =============================================================================
# CryptoExchange Tests
# =============================================================================


class TestCryptoExchange:
    """Tests for CryptoExchange class."""

    def test_calculate_trading_costs(self, default_config: CryptoBacktestConfig) -> None:
        """Test trading cost calculation."""
        margin_config = MarginConfig.from_leverage(default_config.leverage)
        margin_calc = MarginCalculator(margin_config)
        exchange = CryptoExchange(default_config, margin_calc)

        # 1 ETH at $3000 = $3000 notional
        costs = exchange.calculate_trading_costs(1.0, 3000.0)

        # Commission: 3000 * 0.0004 = 1.2
        # Slippage: 3000 * 0.0001 = 0.3
        # Total: 1.5
        expected = 3000 * (0.0004 + 0.0001)
        assert abs(costs - expected) < 0.01

    def test_funding_payment_long_positive_rate(
        self, default_config: CryptoBacktestConfig
    ) -> None:
        """Test funding payment: long pays when rate is positive."""
        margin_config = MarginConfig.from_leverage(default_config.leverage)
        margin_calc = MarginCalculator(margin_config)
        exchange = CryptoExchange(default_config, margin_calc)

        # Long 1 ETH at $3000, funding rate 0.01%
        payment = exchange.calculate_funding_payment(
            position_size=1.0,
            position_type=PositionType.LONG,
            mark_price=3000.0,
            funding_rate=0.0001,  # 0.01%
        )

        # Long pays: -3000 * 0.0001 = -0.3
        assert payment < 0  # Long pays
        assert abs(payment - (-0.3)) < 0.01

    def test_funding_payment_short_positive_rate(
        self, default_config: CryptoBacktestConfig
    ) -> None:
        """Test funding payment: short receives when rate is positive."""
        margin_config = MarginConfig.from_leverage(default_config.leverage)
        margin_calc = MarginCalculator(margin_config)
        exchange = CryptoExchange(default_config, margin_calc)

        # Short 1 ETH at $3000, funding rate 0.01%
        payment = exchange.calculate_funding_payment(
            position_size=1.0,
            position_type=PositionType.SHORT,
            mark_price=3000.0,
            funding_rate=0.0001,
        )

        # Short receives: 3000 * 0.0001 = 0.3
        assert payment > 0  # Short receives
        assert abs(payment - 0.3) < 0.01

    def test_funding_payment_flat_position(
        self, default_config: CryptoBacktestConfig
    ) -> None:
        """Test funding payment: no payment for flat position."""
        margin_config = MarginConfig.from_leverage(default_config.leverage)
        margin_calc = MarginCalculator(margin_config)
        exchange = CryptoExchange(default_config, margin_calc)

        payment = exchange.calculate_funding_payment(
            position_size=0.0,
            position_type=PositionType.FLAT,
            mark_price=3000.0,
            funding_rate=0.0001,
        )

        assert payment == 0.0

    def test_should_settle_funding_correct_hour(
        self, default_config: CryptoBacktestConfig
    ) -> None:
        """Test funding settlement triggers at correct hours."""
        margin_config = MarginConfig.from_leverage(default_config.leverage)
        margin_calc = MarginCalculator(margin_config)
        exchange = CryptoExchange(default_config, margin_calc)

        # At 8:00 UTC (funding hour)
        ts_funding = datetime(2024, 1, 1, 8, 0, 0)
        assert exchange.should_settle_funding(ts_funding, "ETHUSDT") is True

        # At 10:00 UTC (not a funding hour)
        ts_no_funding = datetime(2024, 1, 1, 10, 0, 0)
        assert exchange.should_settle_funding(ts_no_funding, "ETHUSDT") is False


# =============================================================================
# CryptoQlibBacktest Tests
# =============================================================================


class TestCryptoQlibBacktest:
    """Tests for CryptoQlibBacktest engine."""

    def test_initialization(self, default_config: CryptoBacktestConfig) -> None:
        """Test engine initialization."""
        engine = CryptoQlibBacktest(default_config, ["ETHUSDT"])

        assert engine.config == default_config
        assert engine.symbols == ["ETHUSDT"]
        assert engine.margin_calc is not None
        assert engine.exchange is not None

    def test_run_backtest_returns_result(
        self,
        default_config: CryptoBacktestConfig,
        sample_ohlcv_data: pd.DataFrame,
        momentum_signals: pd.Series,
    ) -> None:
        """Test that backtest returns proper result object."""
        engine = CryptoQlibBacktest(default_config)
        result = engine.run(sample_ohlcv_data, momentum_signals, "ETHUSDT")

        assert isinstance(result, CryptoBacktestResult)
        assert hasattr(result, "total_return")
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "max_drawdown")
        assert hasattr(result, "n_trades")
        assert hasattr(result, "equity_curve")

    def test_run_backtest_equity_curve_length(
        self,
        default_config: CryptoBacktestConfig,
        sample_ohlcv_data: pd.DataFrame,
        momentum_signals: pd.Series,
    ) -> None:
        """Test that equity curve has correct length."""
        engine = CryptoQlibBacktest(default_config)
        result = engine.run(sample_ohlcv_data, momentum_signals, "ETHUSDT")

        assert len(result.equity_curve) == len(sample_ohlcv_data)

    def test_run_backtest_with_funding(
        self,
        sample_ohlcv_data: pd.DataFrame,
        momentum_signals: pd.Series,
    ) -> None:
        """Test backtest with funding enabled."""
        config = CryptoBacktestConfig(
            initial_capital=100000.0,
            leverage=10,
            funding_enabled=True,
        )
        engine = CryptoQlibBacktest(config)
        result = engine.run(sample_ohlcv_data, momentum_signals, "ETHUSDT")

        # Should have some funding settlements
        funding_events = [
            s for s in result.settlements if s.event_type == SettlementEvent.FUNDING
        ]
        assert len(funding_events) > 0

    def test_run_backtest_without_funding(
        self,
        sample_ohlcv_data: pd.DataFrame,
        momentum_signals: pd.Series,
    ) -> None:
        """Test backtest with funding disabled."""
        config = CryptoBacktestConfig(
            initial_capital=100000.0,
            leverage=10,
            funding_enabled=False,
        )
        engine = CryptoQlibBacktest(config)
        result = engine.run(sample_ohlcv_data, momentum_signals, "ETHUSDT")

        # Should have no funding settlements
        funding_events = [
            s for s in result.settlements if s.event_type == SettlementEvent.FUNDING
        ]
        assert len(funding_events) == 0
        assert result.total_funding_paid == 0.0
        assert result.total_funding_received == 0.0

    def test_run_backtest_validates_inputs(
        self, default_config: CryptoBacktestConfig
    ) -> None:
        """Test input validation."""
        engine = CryptoQlibBacktest(default_config)

        # Empty data
        empty_df = pd.DataFrame()
        empty_signals = pd.Series(dtype=float)
        with pytest.raises(ValueError, match="Data is empty"):
            engine.run(empty_df, empty_signals, "ETHUSDT")

        # Missing close column
        bad_df = pd.DataFrame(
            {"open": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3)
        )
        bad_signals = pd.Series([0, 0, 0], index=bad_df.index)
        with pytest.raises(ValueError, match="'close' column"):
            engine.run(bad_df, bad_signals, "ETHUSDT")


class TestCryptoQlibBacktestLiquidation:
    """Tests for liquidation detection."""

    def test_liquidation_triggered_on_large_loss(self) -> None:
        """Test that liquidation is triggered when price moves against position."""
        config = CryptoBacktestConfig(
            initial_capital=10000.0,
            leverage=10,  # 10x leverage
            funding_enabled=False,
            liquidation_enabled=True,
            strict_cv_mode=False,  # Skip CV validation for this edge case test
        )

        # Create price data that drops 15% to trigger liquidation
        n_bars = 50
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="h")

        close = np.concatenate(
            [
                np.ones(5) * 3000,  # Entry at 3000
                np.linspace(3000, 2550, 45),  # 15% drop
            ]
        )

        data = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.ones(n_bars) * 1000,
                "funding_rate": np.zeros(n_bars),
            },
            index=dates,
        )

        # Always long
        signals = pd.Series(1, index=dates)

        engine = CryptoQlibBacktest(config)
        result = engine.run(data, signals, "ETHUSDT")

        # Should have at least one liquidation
        assert result.n_liquidations >= 1

        # Check liquidation event exists
        liq_events = [
            s for s in result.settlements if s.event_type == SettlementEvent.LIQUIDATION
        ]
        assert len(liq_events) >= 1

    def test_no_liquidation_when_disabled(self) -> None:
        """Test that liquidation is not triggered when disabled."""
        config = CryptoBacktestConfig(
            initial_capital=10000.0,
            leverage=10,
            funding_enabled=False,
            liquidation_enabled=False,  # Disabled
            strict_cv_mode=False,  # Skip CV validation for this edge case test
        )

        # Same price drop data
        n_bars = 50
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="h")
        close = np.concatenate(
            [np.ones(5) * 3000, np.linspace(3000, 2550, 45)]
        )

        data = pd.DataFrame(
            {
                "close": close,
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "volume": np.ones(n_bars) * 1000,
                "funding_rate": np.zeros(n_bars),
            },
            index=dates,
        )
        signals = pd.Series(1, index=dates)

        engine = CryptoQlibBacktest(config)
        result = engine.run(data, signals, "ETHUSDT")

        # Should have no liquidations
        assert result.n_liquidations == 0


class TestCryptoQlibBacktestMultiAsset:
    """Tests for multi-asset backtesting."""

    def test_multi_asset_backtest_combines_results(self) -> None:
        """Test multi-asset backtest combines individual results."""
        np.random.seed(42)
        n_bars = 300  # Need >= 250 for Purged CV validation
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="h")

        # BTC data
        btc_close = 45000 + np.cumsum(np.random.randn(n_bars) * 100)
        btc_data = pd.DataFrame(
            {
                "close": btc_close,
                "open": btc_close * 0.999,
                "high": btc_close * 1.01,
                "low": btc_close * 0.99,
                "volume": np.ones(n_bars) * 1000,
                "funding_rate": np.zeros(n_bars),
            },
            index=dates,
        )

        # ETH data
        eth_close = 3000 + np.cumsum(np.random.randn(n_bars) * 20)
        eth_data = pd.DataFrame(
            {
                "close": eth_close,
                "open": eth_close * 0.999,
                "high": eth_close * 1.01,
                "low": eth_close * 0.99,
                "volume": np.ones(n_bars) * 1000,
                "funding_rate": np.zeros(n_bars),
            },
            index=dates,
        )

        # Simple signals
        btc_signals = pd.Series(1, index=dates)
        eth_signals = pd.Series(-1, index=dates)

        config = CryptoBacktestConfig(
            initial_capital=100000.0,
            leverage=5,
            funding_enabled=False,
        )

        engine = CryptoQlibBacktest(config, ["BTCUSDT", "ETHUSDT"])
        result = engine.run_multi_asset(
            data_dict={"BTCUSDT": btc_data, "ETHUSDT": eth_data},
            signals_dict={"BTCUSDT": btc_signals, "ETHUSDT": eth_signals},
            capital_allocation={"BTCUSDT": 0.6, "ETHUSDT": 0.4},
        )

        assert isinstance(result, CryptoBacktestResult)
        assert "BTCUSDT" in result.per_asset_pnl
        assert "ETHUSDT" in result.per_asset_pnl
        assert "BTCUSDT" in result.per_asset_trades
        assert "ETHUSDT" in result.per_asset_trades

    def test_multi_asset_validates_allocation(self) -> None:
        """Test that capital allocation must sum to 1.0."""
        config = CryptoBacktestConfig(initial_capital=100000.0)
        engine = CryptoQlibBacktest(config)

        # Create minimal data
        dates = pd.date_range("2024-01-01", periods=10, freq="h")
        data = pd.DataFrame(
            {"close": [100] * 10, "open": [100] * 10, "high": [101] * 10, "low": [99] * 10,
             "volume": [100] * 10, "funding_rate": [0] * 10},
            index=dates,
        )
        signals = pd.Series(0, index=dates)

        # Allocation sums to 0.9 (invalid)
        with pytest.raises(ValueError, match="sum to 1.0"):
            engine.run_multi_asset(
                data_dict={"BTC": data, "ETH": data},
                signals_dict={"BTC": signals, "ETH": signals},
                capital_allocation={"BTC": 0.5, "ETH": 0.4},
            )


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_run_crypto_backtest(
        self,
        sample_ohlcv_data: pd.DataFrame,
        momentum_signals: pd.Series,
    ) -> None:
        """Test run_crypto_backtest convenience function."""
        result = run_crypto_backtest(
            data=sample_ohlcv_data,
            signals=momentum_signals,
            initial_capital=50000.0,
            leverage=5,
            symbol="ETHUSDT",
        )

        assert isinstance(result, CryptoBacktestResult)
        assert result.equity_curve is not None


# =============================================================================
# Result Serialization Tests
# =============================================================================


class TestCryptoBacktestResult:
    """Tests for CryptoBacktestResult dataclass."""

    def test_to_dict(self) -> None:
        """Test result serialization to dictionary."""
        result = CryptoBacktestResult(
            total_return=0.15,
            annualized_return=0.45,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=0.10,
            win_rate=0.55,
            profit_factor=1.8,
            n_trades=100,
            n_liquidations=2,
            total_funding_paid=50.0,
            total_funding_received=30.0,
            net_funding=-20.0,
        )

        d = result.to_dict()

        assert d["metrics"]["total_return"] == 0.15
        assert d["metrics"]["sharpe_ratio"] == 1.5
        assert d["trading"]["n_trades"] == 100
        assert d["trading"]["n_liquidations"] == 2
        assert d["funding"]["total_paid"] == 50.0
        assert d["funding"]["net"] == -20.0
