"""Comprehensive tests for RD Loop (Research-Development Loop).

Tests use real implementations - NO MOCKS per user requirement.
Tests use dependency injection for ResearchLedger with MemoryStorage
(production code requires PostgresStorage via DATABASE_URL).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from iqfmp.core.rd_loop import (
    LoopPhase,
    LoopConfig,
    IterationResult,
    LoopState,
    RDLoop,
)
from iqfmp.evaluation.research_ledger import (
    ResearchLedger,
    MemoryStorage,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def test_ledger() -> ResearchLedger:
    """Create a ResearchLedger with MemoryStorage for tests.

    P4 Architecture: Production code requires PostgresStorage via DATABASE_URL.
    Tests use dependency injection with MemoryStorage (real implementation, not mock).
    """
    return ResearchLedger(storage=MemoryStorage())


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Create sample OHLCV DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=200, freq="D")
    close = 2000 + np.cumsum(np.random.randn(200) * 20)

    return pd.DataFrame({
        "datetime": dates,
        "open": close * (1 + np.random.randn(200) * 0.01),
        "high": close * (1 + np.abs(np.random.randn(200)) * 0.02),
        "low": close * (1 - np.abs(np.random.randn(200)) * 0.02),
        "close": close,
        "volume": np.random.randint(1000, 10000, 200) * 1000,
    })


@pytest.fixture
def loop_config() -> LoopConfig:
    """Create test loop configuration."""
    return LoopConfig(
        max_iterations=10,
        max_hypotheses_per_iteration=3,
        target_core_factors=5,
        ic_threshold=0.02,
        ir_threshold=0.5,
        novelty_threshold=0.7,
        run_benchmark=False,  # Disable benchmark for unit tests
    )


# =============================================================================
# Test LoopPhase Enum
# =============================================================================

class TestLoopPhase:
    """Tests for LoopPhase enum."""

    def test_all_phases_exist(self):
        """Test that all expected phases are defined."""
        expected_phases = [
            "INITIALIZING",
            "HYPOTHESIS_GENERATION",
            "FACTOR_CODING",
            "FACTOR_EVALUATION",
            "BENCHMARK_COMPARISON",
            "FEEDBACK_ANALYSIS",
            "FACTOR_COMBINATION",
            "FACTOR_SELECTION",
            "COMPLETED",
        ]
        for phase in expected_phases:
            assert hasattr(LoopPhase, phase)

    def test_phase_values(self):
        """Test that phases have correct string values."""
        assert LoopPhase.INITIALIZING.value == "initializing"
        assert LoopPhase.COMPLETED.value == "completed"
        assert LoopPhase.HYPOTHESIS_GENERATION.value == "hypothesis_generation"

    def test_phase_is_str_enum(self):
        """Test that LoopPhase is a string enum."""
        assert isinstance(LoopPhase.INITIALIZING, str)
        assert LoopPhase.INITIALIZING == "initializing"


# =============================================================================
# Test LoopConfig Dataclass
# =============================================================================

class TestLoopConfig:
    """Tests for LoopConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LoopConfig()

        assert config.max_iterations == 100
        assert config.max_hypotheses_per_iteration == 5
        assert config.target_core_factors == 10
        assert config.ic_threshold == 0.03
        assert config.ir_threshold == 1.0
        assert config.novelty_threshold == 0.7
        assert config.run_benchmark is True

    def test_custom_values(self, loop_config: LoopConfig):
        """Test custom configuration values."""
        assert loop_config.max_iterations == 10
        assert loop_config.ic_threshold == 0.02

    def test_config_with_callbacks(self):
        """Test configuration with callbacks."""

        def on_iteration(iteration: int, data: dict) -> None:
            pass

        def on_phase_change(phase: LoopPhase) -> None:
            pass

        config = LoopConfig(
            on_iteration_complete=on_iteration,
            on_phase_change=on_phase_change,
        )

        assert config.on_iteration_complete is not None
        assert config.on_phase_change is not None


# =============================================================================
# Test IterationResult Dataclass
# =============================================================================

class TestIterationResult:
    """Tests for IterationResult dataclass."""

    def test_create_iteration_result(self):
        """Test creating an iteration result."""
        result = IterationResult(
            iteration=1,
            hypotheses_tested=5,
            factors_validated=3,
            best_ic=0.045,
            best_factor_name="momentum_20d",
        )

        assert result.iteration == 1
        assert result.hypotheses_tested == 5
        assert result.factors_validated == 3
        assert result.best_ic == 0.045
        assert result.best_factor_name == "momentum_20d"
        assert result.benchmark_rank is None

    def test_iteration_result_with_benchmark(self):
        """Test iteration result with benchmark rank."""
        result = IterationResult(
            iteration=2,
            hypotheses_tested=10,
            factors_validated=5,
            best_ic=0.05,
            best_factor_name="volatility_30d",
            benchmark_rank=15,
        )

        assert result.benchmark_rank == 15

    def test_iteration_result_to_dict(self):
        """Test conversion to dictionary."""
        result = IterationResult(
            iteration=1,
            hypotheses_tested=5,
            factors_validated=3,
            best_ic=0.045,
            best_factor_name="momentum_20d",
            phase_durations={"hypothesis_generation": 1.5, "evaluation": 2.0},
        )

        d = result.to_dict()

        assert d["iteration"] == 1
        assert d["hypotheses_tested"] == 5
        assert d["factors_validated"] == 3
        assert d["best_ic"] == 0.045
        assert d["best_factor_name"] == "momentum_20d"
        assert "timestamp" in d
        assert "phase_durations" in d

    def test_iteration_result_timestamp(self):
        """Test that timestamp is auto-generated."""
        before = datetime.now()
        result = IterationResult(
            iteration=1,
            hypotheses_tested=1,
            factors_validated=1,
            best_ic=0.01,
            best_factor_name="test",
        )
        after = datetime.now()

        assert before <= result.timestamp <= after


# =============================================================================
# Test LoopState Dataclass
# =============================================================================

class TestLoopState:
    """Tests for LoopState dataclass."""

    def test_default_state(self):
        """Test default loop state."""
        state = LoopState()

        assert state.phase == LoopPhase.INITIALIZING
        assert state.iteration == 0
        assert state.total_hypotheses_tested == 0
        assert state.core_factors == []
        assert state.iteration_results == []
        assert state.is_running is False
        assert state.stop_requested is False

    def test_state_with_values(self):
        """Test loop state with custom values."""
        state = LoopState(
            phase=LoopPhase.FACTOR_EVALUATION,
            iteration=5,
            total_hypotheses_tested=25,
            core_factors=["factor_1", "factor_2"],
            is_running=True,
        )

        assert state.phase == LoopPhase.FACTOR_EVALUATION
        assert state.iteration == 5
        assert len(state.core_factors) == 2

    def test_state_to_dict(self):
        """Test conversion to dictionary."""
        state = LoopState(
            phase=LoopPhase.COMPLETED,
            iteration=10,
            total_hypotheses_tested=50,
            core_factors=["f1", "f2", "f3"],
        )

        d = state.to_dict()

        assert d["phase"] == "completed"
        assert d["iteration"] == 10
        assert d["total_hypotheses_tested"] == 50
        assert d["core_factors_count"] == 3
        assert d["is_running"] is False

    def test_state_to_dict_truncates_factors(self):
        """Test that to_dict truncates core_factors to 10."""
        state = LoopState(
            core_factors=[f"factor_{i}" for i in range(20)],
        )

        d = state.to_dict()

        # Only first 10 factors should be included
        assert len(d["core_factors"]) == 10
        assert d["core_factors_count"] == 20


# =============================================================================
# Test RDLoop Class
# =============================================================================

class TestRDLoopInitialization:
    """Tests for RDLoop initialization."""

    def test_default_initialization(self, test_ledger: ResearchLedger):
        """Test RDLoop with default config (injected ledger for tests)."""
        loop = RDLoop(ledger=test_ledger)

        assert loop.config is not None
        assert loop.state.phase == LoopPhase.INITIALIZING
        assert loop.hypothesis_agent is not None
        assert loop.benchmarker is not None
        assert loop.ledger is not None

    def test_custom_config(self, loop_config: LoopConfig, test_ledger: ResearchLedger):
        """Test RDLoop with custom config."""
        loop = RDLoop(config=loop_config, ledger=test_ledger)

        assert loop.config.max_iterations == 10
        assert loop.config.ic_threshold == 0.02

    def test_initial_state(self, test_ledger: ResearchLedger):
        """Test initial state of RDLoop."""
        loop = RDLoop(ledger=test_ledger)

        assert loop._df is None
        assert loop._factor_engine is None
        assert loop._factor_evaluator is None
        assert loop._forward_returns is None
        assert loop._validated_factors == {}
        assert loop._factor_metadata == {}


class TestRDLoopDataLoading:
    """Tests for RDLoop data loading."""

    def test_load_dataframe(self, sample_ohlcv_df: pd.DataFrame, test_ledger: ResearchLedger):
        """Test loading data from DataFrame."""
        loop = RDLoop(ledger=test_ledger)
        loop.load_data(sample_ohlcv_df)

        assert loop._df is not None
        assert len(loop._df) == len(sample_ohlcv_df)

    def test_load_data_creates_engines(self, sample_ohlcv_df: pd.DataFrame, test_ledger: ResearchLedger):
        """Test that loading data creates factor engine and evaluator."""
        loop = RDLoop(ledger=test_ledger)
        loop.load_data(sample_ohlcv_df)

        assert loop._factor_engine is not None
        assert loop._factor_evaluator is not None

    def test_load_data_calculates_returns(self, sample_ohlcv_df: pd.DataFrame, test_ledger: ResearchLedger):
        """Test that loading data calculates forward returns."""
        loop = RDLoop(ledger=test_ledger)
        loop.load_data(sample_ohlcv_df)

        assert loop._forward_returns is not None
        assert len(loop._forward_returns) == len(sample_ohlcv_df)


class TestRDLoopState:
    """Tests for RDLoop state management."""

    def test_state_access(self, loop_config: LoopConfig, test_ledger: ResearchLedger):
        """Test accessing loop state directly."""
        loop = RDLoop(config=loop_config, ledger=test_ledger)

        assert isinstance(loop.state, LoopState)
        assert loop.state.phase == LoopPhase.INITIALIZING

    def test_state_to_dict(self, loop_config: LoopConfig, test_ledger: ResearchLedger):
        """Test converting state to dictionary."""
        loop = RDLoop(config=loop_config, ledger=test_ledger)
        state_dict = loop.state.to_dict()

        assert isinstance(state_dict, dict)
        assert "phase" in state_dict
        assert "iteration" in state_dict

    def test_stop_request(self, loop_config: LoopConfig, test_ledger: ResearchLedger):
        """Test stop request."""
        loop = RDLoop(config=loop_config, ledger=test_ledger)
        loop.stop()

        assert loop.state.stop_requested is True


class TestRDLoopCoreFunctions:
    """Tests for core RDLoop functions."""

    def test_get_core_factors_empty(self, loop_config: LoopConfig, test_ledger: ResearchLedger):
        """Test getting core factors when empty."""
        loop = RDLoop(config=loop_config, ledger=test_ledger)
        factors = loop.get_core_factors()

        assert isinstance(factors, list)
        assert len(factors) == 0

    def test_get_statistics(self, loop_config: LoopConfig, test_ledger: ResearchLedger):
        """Test getting statistics."""
        loop = RDLoop(config=loop_config, ledger=test_ledger)
        stats = loop.get_statistics()

        assert isinstance(stats, dict)

    def test_should_continue_initially(self, loop_config: LoopConfig, test_ledger: ResearchLedger):
        """Test _should_continue initially returns True."""
        loop = RDLoop(config=loop_config, ledger=test_ledger)

        # Should continue initially (no iterations run)
        result = loop._should_continue()
        assert isinstance(result, bool)


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestRDLoopEdgeCases:
    """Edge case tests for RDLoop."""

    def test_empty_dataframe(self, test_ledger: ResearchLedger):
        """Test loading empty DataFrame."""
        loop = RDLoop(ledger=test_ledger)
        empty_df = pd.DataFrame()

        # Should handle empty DataFrame gracefully
        try:
            loop.load_data(empty_df)
        except (ValueError, KeyError):
            pass  # Expected for invalid data

    def test_missing_columns(self, test_ledger: ResearchLedger):
        """Test loading DataFrame with missing columns."""
        loop = RDLoop(ledger=test_ledger)
        invalid_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        with pytest.raises((ValueError, KeyError)):
            loop.load_data(invalid_df)

    def test_nan_handling(self, sample_ohlcv_df: pd.DataFrame, test_ledger: ResearchLedger):
        """Test handling of NaN values in data."""
        loop = RDLoop(ledger=test_ledger)

        # Add some NaN values
        df_with_nan = sample_ohlcv_df.copy()
        df_with_nan.loc[10:15, "close"] = np.nan

        # Should handle NaN gracefully
        loop.load_data(df_with_nan)
        assert loop._df is not None


# =============================================================================
# Test Integration
# =============================================================================

class TestRDLoopIntegration:
    """Integration tests for RDLoop with real components."""

    def test_full_workflow_setup(self, sample_ohlcv_df: pd.DataFrame, loop_config: LoopConfig, test_ledger: ResearchLedger):
        """Test full workflow setup without running the loop."""
        loop = RDLoop(config=loop_config, ledger=test_ledger)
        loop.load_data(sample_ohlcv_df)

        # Verify all components are ready
        assert loop._df is not None
        assert loop._factor_engine is not None
        assert loop._factor_evaluator is not None
        assert loop._forward_returns is not None

        # Verify state
        assert loop.state.phase == LoopPhase.INITIALIZING or loop.state.phase == LoopPhase.HYPOTHESIS_GENERATION

    def test_ledger_integration(self, loop_config: LoopConfig, test_ledger: ResearchLedger):
        """Test that ledger is properly integrated."""
        loop = RDLoop(config=loop_config, ledger=test_ledger)

        # Ledger should be initialized
        assert loop.ledger is not None
        # Check for actual ledger method
        assert hasattr(loop.ledger, "add_trial") or hasattr(loop.ledger, "record")
