"""Tests for Backtest Engine (Task 17)."""

from datetime import datetime, timedelta
from typing import Any

import pytest
import pandas as pd
import numpy as np

from iqfmp.strategy.backtest import (
    # Core
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    # Performance
    PerformanceMetrics,
    PerformanceCalculator,
    # Trade
    Trade,
    TradeType,
    TradeStatus,
    # Report
    BacktestReport,
    ReportConfig,
    # Exceptions
    BacktestError,
    InsufficientDataError,
)


# ============ Helper Functions ============

def create_sample_data(
    days: int = 100,
    start_price: float = 100.0,
    volatility: float = 0.02,
) -> pd.DataFrame:
    """Create sample OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=days, freq="D")

    returns = np.random.normal(0.0005, volatility, days)
    prices = start_price * np.cumprod(1 + returns)

    return pd.DataFrame({
        "open": prices * (1 - volatility / 2),
        "high": prices * (1 + volatility),
        "low": prices * (1 - volatility),
        "close": prices,
        "volume": np.random.randint(1000, 10000, days),
    }, index=dates)


def create_sample_signals(data: pd.DataFrame) -> pd.Series:
    """Create sample trading signals."""
    signals = pd.Series(0.0, index=data.index)
    # Simple momentum: buy when 5-day return > 0
    returns_5d = data["close"].pct_change(5)
    signals[returns_5d > 0.01] = 1.0  # Long
    signals[returns_5d < -0.01] = -1.0  # Short
    return signals


# ============ Trade Tests ============

class TestTrade:
    """Tests for Trade dataclass."""

    def test_create_long_trade(self) -> None:
        """Test creating a long trade."""
        trade = Trade(
            symbol="BTCUSDT",
            trade_type=TradeType.LONG,
            entry_price=50000.0,
            quantity=0.1,
            entry_time=datetime.now(),
        )
        assert trade.symbol == "BTCUSDT"
        assert trade.trade_type == TradeType.LONG

    def test_create_short_trade(self) -> None:
        """Test creating a short trade."""
        trade = Trade(
            symbol="BTCUSDT",
            trade_type=TradeType.SHORT,
            entry_price=50000.0,
            quantity=0.1,
            entry_time=datetime.now(),
        )
        assert trade.trade_type == TradeType.SHORT

    def test_trade_pnl_long(self) -> None:
        """Test P&L for long trade."""
        trade = Trade(
            symbol="BTCUSDT",
            trade_type=TradeType.LONG,
            entry_price=50000.0,
            exit_price=55000.0,
            quantity=0.1,
            entry_time=datetime.now(),
        )
        assert trade.pnl == pytest.approx(500.0)

    def test_trade_pnl_short(self) -> None:
        """Test P&L for short trade."""
        trade = Trade(
            symbol="BTCUSDT",
            trade_type=TradeType.SHORT,
            entry_price=50000.0,
            exit_price=45000.0,
            quantity=0.1,
            entry_time=datetime.now(),
        )
        assert trade.pnl == pytest.approx(500.0)

    def test_trade_return_pct(self) -> None:
        """Test return percentage calculation."""
        trade = Trade(
            symbol="BTCUSDT",
            trade_type=TradeType.LONG,
            entry_price=50000.0,
            exit_price=55000.0,
            quantity=0.1,
            entry_time=datetime.now(),
        )
        assert trade.return_pct == pytest.approx(0.1)  # 10%


# ============ Performance Metrics Tests ============

class TestPerformanceMetrics:
    """Tests for PerformanceMetrics."""

    def test_metrics_creation(self) -> None:
        """Test creating metrics object."""
        metrics = PerformanceMetrics(
            total_return=0.25,
            sharpe_ratio=1.5,
            max_drawdown=0.15,
            win_rate=0.6,
        )
        assert metrics.total_return == 0.25
        assert metrics.sharpe_ratio == 1.5

    def test_metrics_to_dict(self) -> None:
        """Test metrics serialization."""
        metrics = PerformanceMetrics(
            total_return=0.25,
            sharpe_ratio=1.5,
            max_drawdown=0.15,
            win_rate=0.6,
        )
        d = metrics.to_dict()
        assert "total_return" in d
        assert "sharpe_ratio" in d


class TestPerformanceCalculator:
    """Tests for PerformanceCalculator."""

    @pytest.fixture
    def equity_curve(self) -> pd.Series:
        """Create sample equity curve."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        equity = 100000 * np.cumprod(1 + returns)
        return pd.Series(equity, index=dates)

    def test_calculate_sharpe(self, equity_curve: pd.Series) -> None:
        """Test Sharpe ratio calculation."""
        calc = PerformanceCalculator()
        sharpe = calc.calculate_sharpe(equity_curve)
        assert isinstance(sharpe, float)
        assert sharpe > -5.0 and sharpe < 5.0  # Reasonable bounds

    def test_calculate_max_drawdown(self, equity_curve: pd.Series) -> None:
        """Test max drawdown calculation."""
        calc = PerformanceCalculator()
        mdd = calc.calculate_max_drawdown(equity_curve)
        assert 0.0 <= mdd <= 1.0

    def test_calculate_win_rate(self) -> None:
        """Test win rate calculation."""
        calc = PerformanceCalculator()
        trades = [
            Trade("A", TradeType.LONG, 100, 0.1, datetime.now(), exit_price=110),
            Trade("B", TradeType.LONG, 100, 0.1, datetime.now(), exit_price=90),
            Trade("C", TradeType.LONG, 100, 0.1, datetime.now(), exit_price=120),
        ]
        win_rate = calc.calculate_win_rate(trades)
        assert win_rate == pytest.approx(2/3)

    def test_calculate_profit_factor(self) -> None:
        """Test profit factor calculation."""
        calc = PerformanceCalculator()
        trades = [
            Trade("A", TradeType.LONG, 100, 1.0, datetime.now(), exit_price=110),  # +10
            Trade("B", TradeType.LONG, 100, 1.0, datetime.now(), exit_price=95),   # -5
            Trade("C", TradeType.LONG, 100, 1.0, datetime.now(), exit_price=120),  # +20
        ]
        pf = calc.calculate_profit_factor(trades)
        assert pf == pytest.approx(30/5)  # 6.0

    def test_calculate_sortino(self, equity_curve: pd.Series) -> None:
        """Test Sortino ratio calculation."""
        calc = PerformanceCalculator()
        sortino = calc.calculate_sortino(equity_curve)
        assert isinstance(sortino, float)

    def test_calculate_calmar(self, equity_curve: pd.Series) -> None:
        """Test Calmar ratio calculation."""
        calc = PerformanceCalculator()
        calmar = calc.calculate_calmar(equity_curve)
        assert isinstance(calmar, float)

    def test_calculate_all_metrics(self, equity_curve: pd.Series) -> None:
        """Test calculating all metrics."""
        calc = PerformanceCalculator()
        trades = [
            Trade("A", TradeType.LONG, 100, 1.0, datetime.now(), exit_price=110),
            Trade("B", TradeType.LONG, 100, 1.0, datetime.now(), exit_price=95),
        ]
        metrics = calc.calculate_all(equity_curve, trades)
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.sharpe_ratio is not None
        assert metrics.max_drawdown is not None


# ============ Backtest Engine Tests ============

class TestBacktestEngine:
    """Tests for BacktestEngine."""

    @pytest.fixture
    def config(self) -> BacktestConfig:
        """Create backtest config."""
        return BacktestConfig(
            initial_capital=100000.0,
            commission=0.001,
            slippage=0.0005,
        )

    @pytest.fixture
    def engine(self, config: BacktestConfig) -> BacktestEngine:
        """Create backtest engine."""
        return BacktestEngine(config)

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample data."""
        return create_sample_data(days=100)

    def test_engine_creation(self, engine: BacktestEngine) -> None:
        """Test engine creation."""
        assert engine is not None
        assert engine.config.initial_capital == 100000.0

    def test_run_backtest(
        self, engine: BacktestEngine, sample_data: pd.DataFrame
    ) -> None:
        """Test running a basic backtest."""
        signals = create_sample_signals(sample_data)
        result = engine.run(sample_data, signals)
        assert isinstance(result, BacktestResult)
        assert result.equity_curve is not None

    def test_backtest_with_long_only(
        self, engine: BacktestEngine, sample_data: pd.DataFrame
    ) -> None:
        """Test backtest with long-only signals."""
        signals = pd.Series(1.0, index=sample_data.index)  # Always long
        result = engine.run(sample_data, signals)
        assert len(result.trades) > 0

    def test_backtest_with_short_only(
        self, engine: BacktestEngine, sample_data: pd.DataFrame
    ) -> None:
        """Test backtest with short-only signals."""
        signals = pd.Series(-1.0, index=sample_data.index)  # Always short
        result = engine.run(sample_data, signals)
        assert any(t.trade_type == TradeType.SHORT for t in result.trades)

    def test_backtest_applies_commission(
        self, sample_data: pd.DataFrame
    ) -> None:
        """Test that commission is applied."""
        config_no_comm = BacktestConfig(initial_capital=100000.0, commission=0.0)
        config_with_comm = BacktestConfig(initial_capital=100000.0, commission=0.01)

        engine_no_comm = BacktestEngine(config_no_comm)
        engine_with_comm = BacktestEngine(config_with_comm)

        signals = create_sample_signals(sample_data)

        result_no_comm = engine_no_comm.run(sample_data, signals)
        result_with_comm = engine_with_comm.run(sample_data, signals)

        # Commission should reduce final equity
        assert result_with_comm.final_equity < result_no_comm.final_equity

    def test_backtest_applies_slippage(
        self, sample_data: pd.DataFrame
    ) -> None:
        """Test that slippage is applied."""
        config_no_slip = BacktestConfig(initial_capital=100000.0, slippage=0.0)
        config_with_slip = BacktestConfig(initial_capital=100000.0, slippage=0.01)

        engine_no_slip = BacktestEngine(config_no_slip)
        engine_with_slip = BacktestEngine(config_with_slip)

        signals = create_sample_signals(sample_data)

        result_no_slip = engine_no_slip.run(sample_data, signals)
        result_with_slip = engine_with_slip.run(sample_data, signals)

        # Slippage should affect the results (different final equity)
        # Note: The exact direction depends on trade patterns
        assert result_with_slip.final_equity != result_no_slip.final_equity

    def test_backtest_tracks_equity(
        self, engine: BacktestEngine, sample_data: pd.DataFrame
    ) -> None:
        """Test equity curve tracking."""
        signals = create_sample_signals(sample_data)
        result = engine.run(sample_data, signals)

        assert len(result.equity_curve) == len(sample_data)
        assert result.equity_curve.iloc[0] == pytest.approx(100000.0)

    def test_backtest_insufficient_data(
        self, engine: BacktestEngine
    ) -> None:
        """Test with insufficient data."""
        data = create_sample_data(days=2)
        signals = pd.Series(1.0, index=data.index)

        with pytest.raises(InsufficientDataError):
            engine.run(data, signals)


# ============ Backtest Result Tests ============

class TestBacktestResult:
    """Tests for BacktestResult."""

    @pytest.fixture
    def sample_result(self) -> BacktestResult:
        """Create sample result."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        equity = pd.Series(
            100000 * np.cumprod(1 + np.random.normal(0.001, 0.02, 100)),
            index=dates
        )
        trades = [
            Trade("BTC", TradeType.LONG, 100, 1.0, dates[0], exit_price=110,
                  exit_time=dates[10]),
            Trade("BTC", TradeType.SHORT, 110, 1.0, dates[20], exit_price=100,
                  exit_time=dates[30]),
        ]
        return BacktestResult(
            equity_curve=equity,
            trades=trades,
            initial_capital=100000.0,
        )

    def test_result_final_equity(self, sample_result: BacktestResult) -> None:
        """Test final equity calculation."""
        assert sample_result.final_equity == sample_result.equity_curve.iloc[-1]

    def test_result_total_return(self, sample_result: BacktestResult) -> None:
        """Test total return calculation."""
        expected = (sample_result.final_equity / 100000.0) - 1
        assert sample_result.total_return == pytest.approx(expected)

    def test_result_trade_count(self, sample_result: BacktestResult) -> None:
        """Test trade count."""
        assert sample_result.trade_count == 2

    def test_result_to_dict(self, sample_result: BacktestResult) -> None:
        """Test result serialization."""
        d = sample_result.to_dict()
        assert "final_equity" in d
        assert "total_return" in d
        assert "trade_count" in d

    def test_result_get_metrics(self, sample_result: BacktestResult) -> None:
        """Test getting performance metrics."""
        metrics = sample_result.get_metrics()
        assert isinstance(metrics, PerformanceMetrics)


# ============ Backtest Report Tests ============

class TestBacktestReport:
    """Tests for BacktestReport."""

    @pytest.fixture
    def sample_result(self) -> BacktestResult:
        """Create sample result for report."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        np.random.seed(42)
        equity = pd.Series(
            100000 * np.cumprod(1 + np.random.normal(0.001, 0.02, 100)),
            index=dates
        )
        trades = [
            Trade("BTC", TradeType.LONG, 100, 1.0, dates[0], exit_price=110,
                  exit_time=dates[10]),
        ]
        return BacktestResult(
            equity_curve=equity,
            trades=trades,
            initial_capital=100000.0,
        )

    def test_create_report(self, sample_result: BacktestResult) -> None:
        """Test creating a report."""
        report = BacktestReport(sample_result)
        assert report is not None

    def test_report_summary(self, sample_result: BacktestResult) -> None:
        """Test report summary generation."""
        report = BacktestReport(sample_result)
        summary = report.get_summary()
        assert "initial_capital" in summary
        assert "final_equity" in summary
        assert "sharpe_ratio" in summary

    def test_report_to_markdown(self, sample_result: BacktestResult) -> None:
        """Test markdown export."""
        report = BacktestReport(sample_result)
        md = report.to_markdown()
        assert "# Backtest Report" in md
        assert "Performance" in md

    def test_report_monthly_breakdown(self, sample_result: BacktestResult) -> None:
        """Test monthly performance breakdown."""
        report = BacktestReport(sample_result)
        monthly = report.get_monthly_returns()
        assert isinstance(monthly, pd.Series)

    def test_report_trade_statistics(self, sample_result: BacktestResult) -> None:
        """Test trade statistics."""
        report = BacktestReport(sample_result)
        stats = report.get_trade_statistics()
        assert "total_trades" in stats
        assert "winning_trades" in stats
        assert "losing_trades" in stats


# ============ Boundary Tests ============

class TestBacktestBoundary:
    """Boundary tests for backtest engine."""

    def test_zero_signals(self) -> None:
        """Test with all zero signals."""
        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)
        data = create_sample_data(days=100)
        signals = pd.Series(0.0, index=data.index)

        result = engine.run(data, signals)
        # No trades should be made
        assert len(result.trades) == 0

    def test_minimum_data(self) -> None:
        """Test with minimum required data."""
        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)
        data = create_sample_data(days=10)
        signals = create_sample_signals(data)

        result = engine.run(data, signals)
        assert result is not None

    def test_large_dataset(self) -> None:
        """Test with large dataset."""
        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)
        data = create_sample_data(days=1000)
        signals = create_sample_signals(data)

        import time
        start = time.time()
        result = engine.run(data, signals)
        elapsed = time.time() - start

        assert elapsed < 5.0  # Should complete in reasonable time
        assert result is not None


# ============ Exception Tests ============

class TestBacktestExceptions:
    """Exception handling tests."""

    def test_mismatched_index(self) -> None:
        """Test with mismatched data and signal index."""
        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)

        data = create_sample_data(days=100)
        # Create signals with different index
        wrong_dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        signals = pd.Series(1.0, index=wrong_dates)

        with pytest.raises(BacktestError):
            engine.run(data, signals)

    def test_invalid_capital(self) -> None:
        """Test with invalid initial capital."""
        with pytest.raises(BacktestError):
            BacktestConfig(initial_capital=-1000.0)

    def test_invalid_commission(self) -> None:
        """Test with invalid commission rate."""
        with pytest.raises(BacktestError):
            BacktestConfig(initial_capital=100000.0, commission=-0.01)


# ============ Performance Tests ============

class TestBacktestPerformance:
    """Performance tests for backtest engine."""

    def test_multiple_symbols(self) -> None:
        """Test backtesting multiple symbols."""
        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)

        # Create multi-symbol data
        data_btc = create_sample_data(days=100, start_price=50000.0)
        data_eth = create_sample_data(days=100, start_price=3000.0)

        signals_btc = create_sample_signals(data_btc)
        signals_eth = create_sample_signals(data_eth)

        result_btc = engine.run(data_btc, signals_btc)
        result_eth = engine.run(data_eth, signals_eth)

        assert result_btc is not None
        assert result_eth is not None

    def test_rapid_backtests(self) -> None:
        """Test running many backtests quickly."""
        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)
        data = create_sample_data(days=50)

        import time
        start = time.time()
        for _ in range(100):
            signals = pd.Series(np.random.choice([-1, 0, 1], 50), index=data.index)
            engine.run(data, signals)
        elapsed = time.time() - start

        assert elapsed < 10.0  # 100 backtests in under 10 seconds
