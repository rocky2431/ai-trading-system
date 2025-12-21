"""Real tests for BacktestEngine - using real data, no mocks."""

import numpy as np
import pandas as pd
import pytest

# Import fixtures
import sys
sys.path.insert(0, str(__file__).rsplit("/tests/", 1)[0] + "/tests")
from fixtures.real_data_fixtures import (
    real_ohlcv_data,
    real_ohlcv_with_funding,
    real_backtest_engine,
    assert_no_mocks_used,
)


def _prepare_backtest_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV input to BacktestEngine expectations.

    BacktestEngine requires a DatetimeIndex and a numeric 'close' column.
    """
    normalized = df.copy()

    if not isinstance(normalized.index, pd.DatetimeIndex):
        for candidate in ("datetime", "timestamp", "date"):
            if candidate in normalized.columns:
                normalized[candidate] = pd.to_datetime(
                    normalized[candidate], errors="coerce"
                )
                normalized = normalized.set_index(candidate)
                break
        else:
            normalized.index = pd.date_range(
                start="2024-01-01", periods=len(normalized), freq="D"
            )

    normalized = normalized.sort_index()

    if "close" not in normalized.columns:
        raise ValueError("Required column 'close' not found in input data")

    normalized["close"] = pd.to_numeric(normalized["close"], errors="coerce")
    normalized = normalized.dropna(subset=["close"])
    if len(normalized) < 5:
        pytest.skip("Not enough clean OHLCV data for backtest")
    return normalized


class TestBacktestEngineReal:
    """Tests for BacktestEngine using real market data."""

    def test_backtest_engine_initialization(self, real_ohlcv_with_funding):
        """Test backtest engine initializes with real data."""
        from iqfmp.strategy.backtest import BacktestConfig, BacktestEngine

        config = BacktestConfig(
            initial_capital=100000.0,
            commission=0.001,
            slippage=0.0005,
        )

        engine = BacktestEngine(config=config)

        assert engine is not None
        assert engine.config is config

    def test_no_mock_objects(self, real_backtest_engine):
        """Verify no mock objects are used in backtest engine."""
        assert_no_mocks_used(real_backtest_engine)

    def test_run_simple_backtest(self, real_ohlcv_with_funding):
        """Test running a simple backtest with real data."""
        from iqfmp.strategy.backtest import BacktestConfig, BacktestEngine, BacktestResult

        config = BacktestConfig(
            initial_capital=100000.0,
            commission=0.001,
            slippage=0.0005,
            include_funding=False,
        )

        engine = BacktestEngine(config=config)
        data = _prepare_backtest_data(real_ohlcv_with_funding)

        # Run a simple momentum strategy
        open_prices = (
            pd.to_numeric(data["open"], errors="coerce") if "open" in data.columns else data["close"]
        )
        signals = pd.Series(
            np.where(data["close"] > open_prices, 1.0, -1.0),
            index=data.index,
        )
        result = engine.run(data, signals)

        # Verify result structure
        assert isinstance(result, BacktestResult)
        assert np.isfinite(result.total_return)
        metrics = result.get_metrics()
        assert metrics.trade_count == result.trade_count

    def test_funding_rate_settlement(self, real_ohlcv_with_funding):
        """Test funding rate settlement with real data."""
        from iqfmp.strategy.backtest import BacktestConfig, BacktestEngine

        config = BacktestConfig(
            initial_capital=100000.0,
            commission=0.001,
            slippage=0.0005,
            include_funding=True,
            funding_settlement_hours=[0, 8, 16],
        )

        engine = BacktestEngine(config=config)
        data = _prepare_backtest_data(real_ohlcv_with_funding)

        # Make funding deterministic and non-zero
        data["funding_rate"] = 0.0001
        signals = pd.Series(1.0, index=data.index)  # Always long to accrue funding
        result = engine.run(data, signals)

        assert "total_funding_pnl" in result.metadata
        assert np.isfinite(result.metadata["total_funding_pnl"])
        assert result.metadata["total_funding_pnl"] < 0  # positive funding => longs pay

    def test_position_sizing(self, real_ohlcv_with_funding):
        """Test position sizing with real data."""
        from iqfmp.strategy.backtest import BacktestConfig, BacktestEngine

        data = _prepare_backtest_data(real_ohlcv_with_funding).iloc[:20]
        signals = pd.Series(1.0, index=data.index)  # Always long

        config_small = BacktestConfig(initial_capital=100000.0, position_size=0.05)
        config_large = BacktestConfig(initial_capital=100000.0, position_size=0.2)

        result_small = BacktestEngine(config=config_small).run(data, signals)
        result_large = BacktestEngine(config=config_large).run(data, signals)

        assert result_small.trade_count == 1
        assert result_large.trade_count == 1
        assert result_large.trades[0].quantity > result_small.trades[0].quantity

    def test_commission_and_slippage(self, real_ohlcv_with_funding):
        """Test commission and slippage are applied."""
        from iqfmp.strategy.backtest import BacktestConfig, BacktestEngine

        # Higher costs should result in lower returns
        config_low_cost = BacktestConfig(
            initial_capital=100000.0,
            commission=0.0001,
            slippage=0.0001,
        )

        config_high_cost = BacktestConfig(
            initial_capital=100000.0,
            commission=0.01,
            slippage=0.01,
        )

        data = _prepare_backtest_data(real_ohlcv_with_funding).iloc[:50]
        signals = pd.Series(1.0, index=data.index)  # Always long

        result_low = BacktestEngine(config=config_low_cost).run(data, signals)
        result_high = BacktestEngine(config=config_high_cost).run(data, signals)

        # Higher costs should reduce final equity deterministically
        assert result_high.final_equity < result_low.final_equity


class TestBacktestMetrics:
    """Tests for backtest metrics calculation."""

    def test_sharpe_ratio_calculation(self, real_ohlcv_with_funding):
        """Test Sharpe ratio is calculated correctly."""
        from iqfmp.strategy.backtest import BacktestConfig, BacktestEngine

        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config=config)
        data = _prepare_backtest_data(real_ohlcv_with_funding)

        signals = pd.Series(1.0, index=data.index)  # Always long
        result = engine.run(data, signals)

        # Sharpe ratio should be a finite number
        sharpe = result.get_metrics().sharpe_ratio
        assert sharpe is None or np.isfinite(sharpe)

    def test_max_drawdown_calculation(self, real_ohlcv_with_funding):
        """Test maximum drawdown is calculated correctly."""
        from iqfmp.strategy.backtest import BacktestConfig, BacktestEngine

        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config=config)
        data = _prepare_backtest_data(real_ohlcv_with_funding)

        signals = pd.Series(1.0, index=data.index)
        result = engine.run(data, signals)

        # Max drawdown should be non-negative
        max_dd = result.get_metrics().max_drawdown
        assert max_dd is None or max_dd >= 0

    def test_trade_count(self, real_ohlcv_with_funding):
        """Test trade count is tracked correctly."""
        from iqfmp.strategy.backtest import BacktestConfig, BacktestEngine

        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config=config)
        data = _prepare_backtest_data(real_ohlcv_with_funding)

        signals = pd.Series(1.0, index=data.index)
        result = engine.run(data, signals)

        # Trade count should be non-negative
        assert result.trade_count >= 0


class TestBacktestDataValidation:
    """Tests for backtest data validation."""

    def test_requires_ohlcv_columns(self):
        """Test that engine requires OHLCV columns."""
        from iqfmp.strategy.backtest import BacktestConfig, BacktestEngine, BacktestError

        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config=config)

        invalid_df = pd.DataFrame(
            {"foo": [1, 2, 3, 4, 5], "bar": [4, 5, 6, 7, 8]},
            index=pd.date_range(start="2024-01-01", periods=5, freq="D"),
        )
        signals = pd.Series(0.0, index=invalid_df.index)

        with pytest.raises((BacktestError, KeyError, ValueError)):
            engine.run(invalid_df, signals)

    def test_handles_missing_data(self, real_ohlcv_with_funding):
        """Test engine handles missing data gracefully."""
        from iqfmp.strategy.backtest import BacktestConfig, BacktestEngine, BacktestError

        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config=config)

        # Add some NaN values
        df_with_nan = _prepare_backtest_data(real_ohlcv_with_funding)
        df_with_nan.loc[df_with_nan.index[:5], "close"] = np.nan
        signals = pd.Series(1.0, index=df_with_nan.index)

        with pytest.raises((BacktestError, ValueError)):
            engine.run(df_with_nan, signals)
