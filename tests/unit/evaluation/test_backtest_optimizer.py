"""Tests for BacktestOptimizer - Optuna-based backtest parameter optimization."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from iqfmp.evaluation.backtest_optimizer import (
    BacktestOptimizationConfig,
    BacktestOptimizationMetric,
    BacktestOptimizationResult,
    BacktestOptimizer,
    BacktestPrunerType,
    BacktestSamplerType,
    BacktestSearchSpace,
    BacktestTrialResult,
    create_crypto_optimization_config,
    optimize_backtest,
)
from iqfmp.core.unified_backtest import BacktestMode, UnifiedBacktestParams


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_signals() -> pd.DataFrame:
    """Create sample signals DataFrame for testing."""
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    np.random.seed(42)
    signals = pd.DataFrame(
        {
            "signal": np.random.randn(252).cumsum(),
        },
        index=dates,
    )
    return signals


@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    """Create sample OHLCV price data for testing."""
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    np.random.seed(42)

    # Generate realistic price data
    returns = np.random.randn(252) * 0.02
    close = 100 * np.exp(returns.cumsum())

    df = pd.DataFrame(
        {
            "open": close * (1 + np.random.randn(252) * 0.005),
            "high": close * (1 + np.abs(np.random.randn(252)) * 0.01),
            "low": close * (1 - np.abs(np.random.randn(252)) * 0.01),
            "close": close,
            "volume": np.random.randint(1000, 10000, 252),
        },
        index=dates,
    )
    return df


@pytest.fixture
def base_params() -> UnifiedBacktestParams:
    """Create base backtest parameters."""
    return UnifiedBacktestParams(
        start_time="2023-01-01",
        end_time="2023-12-31",
        initial_capital=100_000.0,
        commission_rate=0.0004,
        mode=BacktestMode.CRYPTO,
    )


@pytest.fixture
def basic_config() -> BacktestOptimizationConfig:
    """Create basic optimization config."""
    return BacktestOptimizationConfig(
        n_trials=5,
        n_jobs=1,
        metric=BacktestOptimizationMetric.SHARPE,
        sampler=BacktestSamplerType.RANDOM,
        pruner=BacktestPrunerType.NONE,
    )


# =============================================================================
# BacktestSearchSpace Tests
# =============================================================================


class TestBacktestSearchSpace:
    """Tests for BacktestSearchSpace dataclass."""

    def test_float_search_space(self) -> None:
        """Test float search space creation."""
        space = BacktestSearchSpace(
            name="leverage",
            param_type="float",
            bounds=(1.0, 5.0),
        )
        assert space.name == "leverage"
        assert space.param_type == "float"
        assert space.bounds == (1.0, 5.0)

    def test_int_search_space(self) -> None:
        """Test int search space creation."""
        space = BacktestSearchSpace(
            name="topk",
            param_type="int",
            bounds=(10, 100),
            step=10,
        )
        assert space.name == "topk"
        assert space.param_type == "int"
        assert space.step == 10

    def test_categorical_search_space(self) -> None:
        """Test categorical search space creation."""
        space = BacktestSearchSpace(
            name="mode",
            param_type="categorical",
            bounds=["standard", "nested", "crypto"],
        )
        assert space.name == "mode"
        assert space.bounds == ["standard", "nested", "crypto"]

    def test_log_scale_search_space(self) -> None:
        """Test log-scale search space creation."""
        space = BacktestSearchSpace(
            name="commission",
            param_type="float",
            bounds=(0.0001, 0.01),
            log_scale=True,
        )
        assert space.log_scale is True

    def test_search_space_validation_float(self) -> None:
        """Test validation for float search space."""
        space = BacktestSearchSpace(
            name="test",
            param_type="float",
            bounds=(1.0, 5.0),
        )
        space.validate()  # Should not raise

    def test_search_space_validation_invalid_bounds(self) -> None:
        """Test validation fails for invalid bounds at construction."""
        with pytest.raises(ValueError, match="low must be less than high"):
            BacktestSearchSpace(
                name="test",
                param_type="float",
                bounds=(5.0, 1.0),  # low > high
            )

    def test_search_space_disabled(self) -> None:
        """Test disabled search space skips validation."""
        space = BacktestSearchSpace(
            name="test",
            param_type="float",
            bounds=(5.0, 1.0),  # Invalid but disabled
            enabled=False,
        )
        space.validate()  # Should not raise because disabled


# =============================================================================
# BacktestOptimizationConfig Tests
# =============================================================================


class TestBacktestOptimizationConfig:
    """Tests for BacktestOptimizationConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = BacktestOptimizationConfig()
        assert config.n_trials == 100
        assert config.n_jobs == 1
        assert config.metric == BacktestOptimizationMetric.SHARPE
        assert config.sampler == BacktestSamplerType.TPE
        assert config.pruner == BacktestPrunerType.MEDIAN

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = BacktestOptimizationConfig(
            n_trials=50,
            n_jobs=4,
            metric=BacktestOptimizationMetric.CALMAR,
            sampler=BacktestSamplerType.CMAES,
            pruner=BacktestPrunerType.HYPERBAND,
            timeout=3600,
        )
        assert config.n_trials == 50
        assert config.n_jobs == 4
        assert config.metric == BacktestOptimizationMetric.CALMAR
        assert config.timeout == 3600

    def test_custom_search_spaces(self) -> None:
        """Test config with custom search spaces."""
        spaces = [
            BacktestSearchSpace(
                name="leverage",
                param_type="int",
                bounds=(1, 10),
            ),
        ]
        config = BacktestOptimizationConfig(
            n_trials=10,
            custom_search_spaces=spaces,
        )
        assert len(config.custom_search_spaces) == 1
        assert config.custom_search_spaces[0].name == "leverage"

    def test_config_validation_invalid_n_trials(self) -> None:
        """Test that invalid n_trials raises error."""
        with pytest.raises(ValueError, match="n_trials must be at least 1"):
            BacktestOptimizationConfig(n_trials=0)

    def test_config_validation_invalid_direction(self) -> None:
        """Test that invalid direction raises error."""
        with pytest.raises(ValueError, match="direction must be"):
            BacktestOptimizationConfig(direction="invalid")


# =============================================================================
# BacktestOptimizer Tests
# =============================================================================


class TestBacktestOptimizer:
    """Tests for BacktestOptimizer class."""

    def test_optimizer_creation(
        self,
        sample_signals: pd.DataFrame,
        sample_price_data: pd.DataFrame,
        base_params: UnifiedBacktestParams,
        basic_config: BacktestOptimizationConfig,
    ) -> None:
        """Test optimizer instance creation."""
        optimizer = BacktestOptimizer(
            signals=sample_signals,
            price_data=sample_price_data,
            config=basic_config,
            base_params=base_params,
        )
        assert optimizer is not None
        assert optimizer.config == basic_config
        assert optimizer.base_params == base_params

    def test_optimizer_with_funding_rates(
        self,
        sample_signals: pd.DataFrame,
        sample_price_data: pd.DataFrame,
        base_params: UnifiedBacktestParams,
        basic_config: BacktestOptimizationConfig,
    ) -> None:
        """Test optimizer with funding rates data."""
        funding = pd.DataFrame(
            {"rate": np.random.randn(252) * 0.0001},
            index=sample_signals.index,
        )
        optimizer = BacktestOptimizer(
            signals=sample_signals,
            price_data=sample_price_data,
            config=basic_config,
            base_params=base_params,
            funding_rates=funding,
        )
        assert optimizer.funding_rates is not None

    def test_get_search_space_info(
        self,
        sample_signals: pd.DataFrame,
        sample_price_data: pd.DataFrame,
        base_params: UnifiedBacktestParams,
        basic_config: BacktestOptimizationConfig,
    ) -> None:
        """Test getting search space information."""
        optimizer = BacktestOptimizer(
            signals=sample_signals,
            price_data=sample_price_data,
            config=basic_config,
            base_params=base_params,
        )
        info = optimizer.get_search_space_info()
        assert isinstance(info, list)
        assert all("name" in s for s in info)
        assert all("type" in s for s in info)


# =============================================================================
# BacktestTrialResult Tests
# =============================================================================


class TestBacktestTrialResult:
    """Tests for BacktestTrialResult dataclass."""

    def test_trial_result_creation(self) -> None:
        """Test creating a trial result."""
        result = BacktestTrialResult(
            trial_number=0,
            params={"leverage": 2},
            metric_value=1.5,
            metric_name="sharpe",
            duration_seconds=0.5,
            status="completed",
        )
        assert result.trial_number == 0
        assert result.params == {"leverage": 2}
        assert result.metric_value == 1.5
        assert result.metric_name == "sharpe"
        assert result.status == "completed"

    def test_trial_result_with_backtest_result(self) -> None:
        """Test trial result with backtest data."""
        result = BacktestTrialResult(
            trial_number=1,
            params={"leverage": 3},
            metric_value=2.0,
            metric_name="sharpe",
            duration_seconds=0.6,
            status="completed",
            backtest_result={"sharpe_ratio": 2.0, "total_return": 0.15},
        )
        assert result.backtest_result["sharpe_ratio"] == 2.0


# =============================================================================
# BacktestOptimizationResult Tests
# =============================================================================


class TestBacktestOptimizationResult:
    """Tests for BacktestOptimizationResult dataclass."""

    def test_result_creation(self) -> None:
        """Test result creation with trial data."""
        trials = [
            BacktestTrialResult(
                trial_number=0,
                params={"leverage": 2},
                metric_value=1.5,
                metric_name="sharpe",
                duration_seconds=0.5,
                status="completed",
            ),
            BacktestTrialResult(
                trial_number=1,
                params={"leverage": 3},
                metric_value=2.0,
                metric_name="sharpe",
                duration_seconds=0.6,
                status="completed",
            ),
        ]
        now = datetime.now()
        result = BacktestOptimizationResult(
            optimization_id="test-123",
            best_params={"leverage": 3},
            best_value=2.0,
            best_trial_number=1,
            n_trials_completed=2,
            n_trials_total=2,
            metric_name="sharpe",
            direction="maximize",
            optimization_history=[1.5, 2.0],
            param_importance={"leverage": 1.0},
            all_trials=trials,
            started_at=now,
            completed_at=now,
            duration_seconds=1.1,
        )

        assert result.best_trial_number == 1
        assert result.best_value == 2.0
        assert result.best_params["leverage"] == 3
        assert result.n_trials_completed == 2


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_crypto_optimization_config(self) -> None:
        """Test create_crypto_optimization_config factory."""
        config = create_crypto_optimization_config(
            n_trials=50,
            optimize_leverage=True,
            max_leverage=5,
        )

        assert config.n_trials == 50
        # Should have leverage search space
        has_leverage = any(
            s.name == "leverage" for s in config.custom_search_spaces
        )
        assert has_leverage

    def test_create_crypto_config_search_spaces(self) -> None:
        """Test that crypto config creates proper search spaces."""
        config = create_crypto_optimization_config(
            n_trials=10,
            optimize_leverage=True,
            max_leverage=10,
        )

        leverage_space = next(
            (s for s in config.custom_search_spaces if s.name == "leverage"),
            None,
        )
        assert leverage_space is not None
        assert leverage_space.bounds == (1, 10)

    def test_create_crypto_config_no_leverage(self) -> None:
        """Test crypto config without leverage optimization."""
        config = create_crypto_optimization_config(
            n_trials=20,
            optimize_leverage=False,
        )
        has_leverage = any(
            s.name == "leverage" for s in config.custom_search_spaces
        )
        assert not has_leverage


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enum values."""

    def test_optimization_metrics(self) -> None:
        """Test all optimization metrics are defined."""
        assert BacktestOptimizationMetric.SHARPE.value == "sharpe"
        assert BacktestOptimizationMetric.CALMAR.value == "calmar"
        assert BacktestOptimizationMetric.TOTAL_RETURN.value == "total_return"
        assert BacktestOptimizationMetric.ANNUAL_RETURN.value == "annual_return"
        assert BacktestOptimizationMetric.SORTINO.value == "sortino"
        assert BacktestOptimizationMetric.MAX_DRAWDOWN.value == "max_drawdown"
        assert BacktestOptimizationMetric.WIN_RATE.value == "win_rate"
        assert BacktestOptimizationMetric.PROFIT_FACTOR.value == "profit_factor"

    def test_sampler_types(self) -> None:
        """Test all sampler types are defined."""
        assert BacktestSamplerType.TPE.value == "tpe"
        assert BacktestSamplerType.CMAES.value == "cmaes"
        assert BacktestSamplerType.RANDOM.value == "random"
        assert BacktestSamplerType.GRID.value == "grid"

    def test_pruner_types(self) -> None:
        """Test all pruner types are defined."""
        assert BacktestPrunerType.MEDIAN.value == "median"
        assert BacktestPrunerType.HYPERBAND.value == "hyperband"
        assert BacktestPrunerType.PERCENTILE.value == "percentile"
        assert BacktestPrunerType.NONE.value == "none"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_search_space_categorical_validation(self) -> None:
        """Test categorical search space needs at least 2 choices at construction."""
        with pytest.raises(ValueError, match="at least 2 choices"):
            BacktestSearchSpace(
                name="mode",
                param_type="categorical",
                bounds=["only_one"],  # Only one choice - invalid
            )

    def test_search_space_unknown_type(self) -> None:
        """Test unknown param type raises error at construction."""
        with pytest.raises(ValueError, match="Unknown param_type"):
            BacktestSearchSpace(
                name="test",
                param_type="unknown",
                bounds=(1.0, 5.0),
            )
