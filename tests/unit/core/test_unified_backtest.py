"""Unit tests for Unified Backtest Framework.

Tests cover:
- UnifiedBacktestParams configuration
- NestedExecutionConfig validation
- NestedExecutionLevel configuration
- UnifiedBacktestRunner mode selection
- CryptoNestedBacktest integration
- Factory functions and prebuilt configs

No mocks - uses real implementations with synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iqfmp.core.crypto_backtest import CryptoBacktestConfig, CryptoBacktestResult
from iqfmp.core.qlib_backtest_adapter import QlibBacktestConfig
from iqfmp.core.unified_backtest import (
    # Configuration
    BacktestMode,
    CryptoNestedBacktest,
    ExecutionFrequency,
    InnerStrategyType,
    NestedExecutionConfig,
    NestedExecutionLevel,
    UnifiedBacktestParams,
    # Core classes
    UnifiedBacktestRunner,
    create_backtest_runner,
    create_crypto_nested_config,
    create_hft_nested_config,
    create_standard_nested_config,
    # Factory functions
    run_backtest,
)
from iqfmp.exchange.margin import MarginMode

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Create sample OHLCV data for testing.

    Uses daily frequency to satisfy BacktestEngine's 30-day minimum requirement.
    """
    np.random.seed(42)
    n_bars = 365  # One year of daily data
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="D")

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


@pytest.fixture
def signals_dataframe(momentum_signals: pd.Series) -> pd.DataFrame:
    """Convert signals to DataFrame format."""
    return momentum_signals.to_frame(name="ETHUSDT")


# =============================================================================
# ExecutionFrequency Tests
# =============================================================================


class TestExecutionFrequency:
    """Tests for ExecutionFrequency enum."""

    def test_all_frequencies_defined(self) -> None:
        """Verify all expected frequencies are defined."""
        expected = ["DAY", "HOUR_4", "HOUR_1", "MIN_30", "MIN_15", "MIN_5", "MIN_1"]
        actual = [f.name for f in ExecutionFrequency]
        assert set(actual) == set(expected)

    def test_to_qlib_freq_day(self) -> None:
        """Test day frequency conversion."""
        assert ExecutionFrequency.DAY.to_qlib_freq() == "day"

    def test_to_qlib_freq_30min(self) -> None:
        """Test 30-minute frequency conversion."""
        assert ExecutionFrequency.MIN_30.to_qlib_freq() == "30min"

    def test_to_qlib_freq_5min(self) -> None:
        """Test 5-minute frequency conversion."""
        assert ExecutionFrequency.MIN_5.to_qlib_freq() == "5min"

    def test_to_qlib_freq_1h(self) -> None:
        """Test 1-hour frequency conversion."""
        assert ExecutionFrequency.HOUR_1.to_qlib_freq() == "60min"


# =============================================================================
# InnerStrategyType Tests
# =============================================================================


class TestInnerStrategyType:
    """Tests for InnerStrategyType enum."""

    def test_all_strategies_defined(self) -> None:
        """Verify all expected strategies are defined."""
        expected = ["TWAP", "VWAP", "SBB_EMA", "PASSIVE"]
        actual = [s.name for s in InnerStrategyType]
        assert set(actual) == set(expected)


# =============================================================================
# NestedExecutionLevel Tests
# =============================================================================


class TestNestedExecutionLevel:
    """Tests for NestedExecutionLevel dataclass."""

    def test_default_creation(self) -> None:
        """Test creating level with defaults."""
        level = NestedExecutionLevel(frequency=ExecutionFrequency.DAY)
        assert level.frequency == ExecutionFrequency.DAY
        assert level.strategy_type == InnerStrategyType.TWAP
        assert level.strategy_kwargs == {}

    def test_custom_strategy(self) -> None:
        """Test creating level with custom strategy."""
        level = NestedExecutionLevel(
            frequency=ExecutionFrequency.MIN_30,
            strategy_type=InnerStrategyType.SBB_EMA,
            strategy_kwargs={"hold_thresh": 1.0},
        )
        assert level.strategy_type == InnerStrategyType.SBB_EMA
        assert level.strategy_kwargs["hold_thresh"] == 1.0

    def test_to_qlib_config(self) -> None:
        """Test conversion to Qlib config dict."""
        level = NestedExecutionLevel(
            frequency=ExecutionFrequency.MIN_5,
            strategy_type=InnerStrategyType.TWAP,
        )
        config = level.to_qlib_config()

        assert config["time_per_step"] == "5min"
        assert config["strategy_class"] == "TWAPStrategy"
        assert "strategy_kwargs" in config


# =============================================================================
# NestedExecutionConfig Tests
# =============================================================================


class TestNestedExecutionConfig:
    """Tests for NestedExecutionConfig dataclass."""

    def test_default_creation(self) -> None:
        """Test creating config with defaults."""
        config = NestedExecutionConfig()
        assert len(config.levels) == 3
        assert config.levels[0].frequency == ExecutionFrequency.DAY
        assert config.levels[1].frequency == ExecutionFrequency.MIN_30
        assert config.levels[2].frequency == ExecutionFrequency.MIN_5

    def test_custom_levels(self) -> None:
        """Test creating config with custom levels."""
        levels = [
            NestedExecutionLevel(ExecutionFrequency.HOUR_4),
            NestedExecutionLevel(ExecutionFrequency.MIN_15),
        ]
        config = NestedExecutionConfig(levels=levels)
        assert len(config.levels) == 2

    def test_validate_success(self) -> None:
        """Test validation passes for valid config."""
        config = NestedExecutionConfig()
        # Config is valid if construction succeeds (validates in __post_init__)
        config.validate()  # Should not raise

    def test_validate_fails_single_level(self) -> None:
        """Test validation fails for single level at construction time."""
        with pytest.raises(ValueError, match="at least 2 levels"):
            # Validation now happens in __post_init__
            NestedExecutionConfig(
                levels=[NestedExecutionLevel(ExecutionFrequency.DAY)]
            )

    def test_validate_fails_wrong_order(self) -> None:
        """Test validation fails for wrong frequency order at construction time."""
        with pytest.raises(ValueError, match="decreasing frequency order"):
            # Validation now happens in __post_init__
            NestedExecutionConfig(
                levels=[
                    NestedExecutionLevel(ExecutionFrequency.MIN_5),  # Smaller first
                    NestedExecutionLevel(ExecutionFrequency.DAY),   # Larger second
                ]
            )

    def test_build_qlib_executor_config(self) -> None:
        """Test building Qlib executor config."""
        config = NestedExecutionConfig(
            levels=[
                NestedExecutionLevel(ExecutionFrequency.DAY, InnerStrategyType.SBB_EMA),
                NestedExecutionLevel(ExecutionFrequency.MIN_5, InnerStrategyType.TWAP),
            ]
        )
        executor_config = config.build_qlib_executor_config()

        # Outer level should be NestedExecutor
        assert executor_config["class"] == "NestedExecutor"
        assert executor_config["kwargs"]["time_per_step"] == "day"

        # Inner level should be SimulatorExecutor
        inner = executor_config["kwargs"]["inner_executor"]
        assert inner["class"] == "SimulatorExecutor"
        assert inner["kwargs"]["time_per_step"] == "5min"


# =============================================================================
# UnifiedBacktestParams Tests
# =============================================================================


class TestUnifiedBacktestParams:
    """Tests for UnifiedBacktestParams dataclass."""

    def test_default_values(self) -> None:
        """Test default parameter values."""
        params = UnifiedBacktestParams()
        assert params.initial_capital == 100_000.0
        assert params.commission_rate == 0.0004
        assert params.leverage == 1
        assert params.mode == BacktestMode.STANDARD

    def test_to_qlib_config(self) -> None:
        """Test conversion to Qlib config."""
        params = UnifiedBacktestParams(
            start_time="2023-01-01",
            end_time="2024-01-01",
            initial_capital=50000.0,
            commission_rate=0.001,
        )
        qlib_config = params.to_qlib_config()

        assert isinstance(qlib_config, QlibBacktestConfig)
        assert qlib_config.account == 50000.0
        assert qlib_config.start_time == "2023-01-01"
        assert qlib_config.exchange_kwargs["open_cost"] == 0.001

    def test_to_crypto_config(self) -> None:
        """Test conversion to crypto config."""
        params = UnifiedBacktestParams(
            initial_capital=100000.0,
            leverage=10,
            funding_enabled=True,
            margin_mode=MarginMode.ISOLATED,  # Use enum directly
        )
        crypto_config = params.to_crypto_config()

        assert isinstance(crypto_config, CryptoBacktestConfig)
        assert crypto_config.initial_capital == 100000.0
        assert crypto_config.leverage == 10
        assert crypto_config.funding_enabled is True
        assert crypto_config.margin_mode == MarginMode.ISOLATED

    def test_to_nested_executor_config_default(self) -> None:
        """Test nested executor config with default settings."""
        params = UnifiedBacktestParams(mode=BacktestMode.NESTED)
        executor_config = params.to_nested_executor_config()

        assert "class" in executor_config
        assert executor_config["class"] == "NestedExecutor"

    def test_to_nested_executor_config_custom(self) -> None:
        """Test nested executor config with custom settings."""
        nested = NestedExecutionConfig(
            levels=[
                NestedExecutionLevel(ExecutionFrequency.HOUR_1),
                NestedExecutionLevel(ExecutionFrequency.MIN_1),
            ]
        )
        params = UnifiedBacktestParams(
            mode=BacktestMode.NESTED,
            nested_config=nested,
        )
        executor_config = params.to_nested_executor_config()

        assert executor_config["kwargs"]["time_per_step"] == "60min"

    def test_validation_initial_capital_positive(self) -> None:
        """Test that initial_capital must be positive."""
        with pytest.raises(ValueError, match="initial_capital must be positive"):
            UnifiedBacktestParams(initial_capital=-1000.0)

    def test_validation_initial_capital_zero(self) -> None:
        """Test that initial_capital cannot be zero."""
        with pytest.raises(ValueError, match="initial_capital must be positive"):
            UnifiedBacktestParams(initial_capital=0)

    def test_validation_leverage_minimum(self) -> None:
        """Test that leverage must be >= 1."""
        with pytest.raises(ValueError, match="leverage must be >= 1"):
            UnifiedBacktestParams(leverage=0)

    def test_validation_max_position_pct_range(self) -> None:
        """Test that max_position_pct must be in (0, 1]."""
        with pytest.raises(ValueError, match="max_position_pct must be in"):
            UnifiedBacktestParams(max_position_pct=1.5)
        with pytest.raises(ValueError, match="max_position_pct must be in"):
            UnifiedBacktestParams(max_position_pct=0)

    def test_validation_commission_rate_non_negative(self) -> None:
        """Test that commission_rate must be non-negative."""
        with pytest.raises(ValueError, match="commission_rate must be non-negative"):
            UnifiedBacktestParams(commission_rate=-0.001)

    def test_validation_slippage_rate_non_negative(self) -> None:
        """Test that slippage_rate must be non-negative."""
        with pytest.raises(ValueError, match="slippage_rate must be non-negative"):
            UnifiedBacktestParams(slippage_rate=-0.001)

    def test_validation_topk_positive(self) -> None:
        """Test that topk must be positive."""
        with pytest.raises(ValueError, match="topk must be positive"):
            UnifiedBacktestParams(topk=0)

    def test_validation_n_drop_non_negative(self) -> None:
        """Test that n_drop must be non-negative."""
        with pytest.raises(ValueError, match="n_drop must be non-negative"):
            UnifiedBacktestParams(n_drop=-1)

    def test_cross_margin_mode(self) -> None:
        """Test cross margin mode configuration."""
        params = UnifiedBacktestParams(
            margin_mode=MarginMode.CROSS,
        )
        crypto_config = params.to_crypto_config()
        assert crypto_config.margin_mode == MarginMode.CROSS


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_backtest_runner(self) -> None:
        """Test creating backtest runner."""
        runner = create_backtest_runner()
        assert isinstance(runner, UnifiedBacktestRunner)

    def test_create_standard_nested_config(self) -> None:
        """Test creating standard nested config."""
        config = create_standard_nested_config()
        assert len(config.levels) == 3
        assert config.levels[0].frequency == ExecutionFrequency.DAY
        assert config.levels[2].frequency == ExecutionFrequency.MIN_5

    def test_create_crypto_nested_config(self) -> None:
        """Test creating crypto nested config."""
        config = create_crypto_nested_config()
        assert len(config.levels) == 3
        assert config.levels[0].frequency == ExecutionFrequency.HOUR_4

    def test_create_hft_nested_config(self) -> None:
        """Test creating HFT nested config."""
        config = create_hft_nested_config()
        assert len(config.levels) == 3
        assert config.levels[2].frequency == ExecutionFrequency.MIN_1


# =============================================================================
# UnifiedBacktestRunner Tests
# =============================================================================


class TestUnifiedBacktestRunner:
    """Tests for UnifiedBacktestRunner."""

    def test_init(self) -> None:
        """Test runner initialization."""
        runner = UnifiedBacktestRunner()
        # Runner is now stateless - no internal state to check
        assert isinstance(runner, UnifiedBacktestRunner)

    def test_run_standard_mode(
        self,
        sample_ohlcv_data: pd.DataFrame,
        momentum_signals: pd.Series,
    ) -> None:
        """Test running in standard mode."""
        runner = UnifiedBacktestRunner()
        params = UnifiedBacktestParams(
            mode=BacktestMode.STANDARD,
            initial_capital=100000.0,
            strict_cv_mode=False,  # Disable for test
            # Match date range to sample data (starts 2024-01-01)
            start_time="2024-01-01",
            end_time="2024-12-31",
        )

        result = runner.run(
            signals=momentum_signals,
            data=sample_ohlcv_data,
            params=params,
        )

        assert isinstance(result, dict)
        assert "total_return" in result

    def test_run_crypto_mode(
        self,
        sample_ohlcv_data: pd.DataFrame,
        momentum_signals: pd.Series,
    ) -> None:
        """Test running in crypto mode."""
        runner = UnifiedBacktestRunner()
        params = UnifiedBacktestParams(
            mode=BacktestMode.CRYPTO,
            initial_capital=100000.0,
            leverage=10,
            funding_enabled=True,
            strict_cv_mode=False,  # Disable for test
        )

        result = runner.run(
            signals=momentum_signals,
            data=sample_ohlcv_data,
            params=params,
        )

        assert isinstance(result, CryptoBacktestResult)
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "total_funding_paid")

    def test_run_qlib_mode(
        self,
        sample_ohlcv_data: pd.DataFrame,
        signals_dataframe: pd.DataFrame,
    ) -> None:
        """Test running in Qlib mode (uses pandas fallback if Qlib unavailable)."""
        runner = UnifiedBacktestRunner()
        params = UnifiedBacktestParams(
            mode=BacktestMode.QLIB,
            initial_capital=100000.0,
            topk=10,
            n_drop=2,
        )

        result = runner.run(
            signals=signals_dataframe,
            data=sample_ohlcv_data,
            params=params,
        )

        # Result could be dict (from Qlib/fallback)
        assert isinstance(result, dict)

    def test_run_invalid_mode_fails(
        self,
        sample_ohlcv_data: pd.DataFrame,
        momentum_signals: pd.Series,
    ) -> None:
        """Test that invalid mode string raises ValueError."""
        with pytest.raises(ValueError, match="is not a valid BacktestMode"):
            run_backtest(
                signals=momentum_signals,
                data=sample_ohlcv_data,
                mode="invalid_mode",
            )


# =============================================================================
# CryptoNestedBacktest Tests
# =============================================================================


class TestCryptoNestedBacktest:
    """Tests for CryptoNestedBacktest."""

    def test_init_with_default_params(self) -> None:
        """Test initialization with default parameters."""
        params = UnifiedBacktestParams(
            mode=BacktestMode.NESTED,
            leverage=5,
        )
        engine = CryptoNestedBacktest(params)

        assert engine.params == params
        assert engine.crypto_config.leverage == 5

    def test_init_with_nested_config(self) -> None:
        """Test initialization with custom nested config."""
        nested = create_crypto_nested_config()
        params = UnifiedBacktestParams(
            mode=BacktestMode.NESTED,
            nested_config=nested,
        )
        engine = CryptoNestedBacktest(params)

        assert engine.params.nested_config == nested

    def test_run_fallback(
        self,
        sample_ohlcv_data: pd.DataFrame,
        signals_dataframe: pd.DataFrame,
    ) -> None:
        """Test fallback execution when Qlib nested not available."""
        params = UnifiedBacktestParams(
            mode=BacktestMode.NESTED,
            initial_capital=100000.0,
            strict_cv_mode=False,
        )
        engine = CryptoNestedBacktest(params)

        # This should fall back to single-level crypto backtest
        result = engine._run_fallback(
            signals=signals_dataframe,
            price_data=sample_ohlcv_data,
            funding_rates=None,
        )

        assert isinstance(result, CryptoBacktestResult)

    def test_run_with_allow_fallback(
        self,
        sample_ohlcv_data: pd.DataFrame,
        signals_dataframe: pd.DataFrame,
    ) -> None:
        """Test run with allow_fallback parameter."""
        params = UnifiedBacktestParams(
            mode=BacktestMode.NESTED,
            initial_capital=100000.0,
            strict_cv_mode=False,
        )
        engine = CryptoNestedBacktest(params)

        # With allow_fallback=True, should succeed even if nested fails
        result = engine.run(
            signals=signals_dataframe,
            price_data=sample_ohlcv_data,
            allow_fallback=True,
        )

        assert isinstance(result, CryptoBacktestResult)


# =============================================================================
# run_backtest Convenience Function Tests
# =============================================================================


class TestRunBacktestFunction:
    """Tests for run_backtest convenience function."""

    def test_run_backtest_standard(
        self,
        sample_ohlcv_data: pd.DataFrame,
        momentum_signals: pd.Series,
    ) -> None:
        """Test run_backtest with standard mode."""
        result = run_backtest(
            signals=momentum_signals,
            data=sample_ohlcv_data,
            mode="standard",
            initial_capital=50000.0,
            strict_cv_mode=False,
            # Match date range to sample data (starts 2024-01-01)
            start_time="2024-01-01",
            end_time="2024-12-31",
        )

        assert isinstance(result, dict)

    def test_run_backtest_crypto(
        self,
        sample_ohlcv_data: pd.DataFrame,
        momentum_signals: pd.Series,
    ) -> None:
        """Test run_backtest with crypto mode."""
        result = run_backtest(
            signals=momentum_signals,
            data=sample_ohlcv_data,
            mode="crypto",
            leverage=5,
            funding_enabled=True,
            strict_cv_mode=False,
        )

        assert isinstance(result, CryptoBacktestResult)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the unified backtest framework."""

    def test_end_to_end_standard_backtest(
        self,
        sample_ohlcv_data: pd.DataFrame,
        momentum_signals: pd.Series,
    ) -> None:
        """Test complete standard backtest workflow."""
        # Create params - use full data range
        params = UnifiedBacktestParams(
            start_time="2024-01-01",
            end_time="2024-12-31",
            initial_capital=100000.0,
            mode=BacktestMode.STANDARD,
            strict_cv_mode=False,
        )

        # Create runner
        runner = create_backtest_runner()

        # Run backtest
        result = runner.run(
            signals=momentum_signals,
            data=sample_ohlcv_data,
            params=params,
        )

        # Verify result structure
        assert "total_return" in result
        assert "sharpe_ratio" in result

    def test_end_to_end_crypto_backtest_with_funding(
        self,
        sample_ohlcv_data: pd.DataFrame,
        momentum_signals: pd.Series,
    ) -> None:
        """Test complete crypto backtest with funding rates."""
        # Create params with crypto-specific settings
        params = UnifiedBacktestParams(
            initial_capital=100000.0,
            mode=BacktestMode.CRYPTO,
            leverage=10,
            funding_enabled=True,
            liquidation_enabled=True,
            strict_cv_mode=False,
        )

        # Create runner
        runner = create_backtest_runner()

        # Run backtest
        result = runner.run(
            signals=momentum_signals,
            data=sample_ohlcv_data,
            params=params,
        )

        # Verify crypto-specific fields
        assert isinstance(result, CryptoBacktestResult)
        assert hasattr(result, "net_funding")
        assert hasattr(result, "total_funding_paid")
        assert hasattr(result, "total_funding_received")

    def test_param_conversion_preserves_values(self) -> None:
        """Test that parameter conversion preserves all values."""
        params = UnifiedBacktestParams(
            initial_capital=75000.0,
            commission_rate=0.0005,
            slippage_rate=0.0002,
            leverage=20,
            funding_enabled=False,
        )

        # Convert to crypto config
        crypto_config = params.to_crypto_config()
        assert crypto_config.initial_capital == 75000.0
        assert crypto_config.commission_rate == 0.0005
        assert crypto_config.slippage_rate == 0.0002
        assert crypto_config.leverage == 20
        assert crypto_config.funding_enabled is False

        # Convert to Qlib config
        qlib_config = params.to_qlib_config()
        assert qlib_config.account == 75000.0
        assert qlib_config.exchange_kwargs["open_cost"] == 0.0005
