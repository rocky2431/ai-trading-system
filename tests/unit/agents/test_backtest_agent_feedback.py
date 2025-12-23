"""Tests for BacktestAgent Feedback Loop, Research Ledger, and Purged CV Gatekeeper.

This test file validates the three critical architecture fixes:
1. ACTION 1: Feedback Loop - Agent retries based on backtest results
2. ACTION 2: Research Ledger - Records experiments, calculates dynamic thresholds
3. ACTION 3: Purged CV Gatekeeper - Rejects factors with prob_overfit > threshold

These tests use REAL implementations, no mocks.
"""

import pytest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

from iqfmp.agents.backtest_agent import (
    BacktestOptimizationAgent,
    BacktestConfig,
    BacktestMetrics,
    OverfitRejectionError,
    DynamicThresholdNotMetError,
)
from iqfmp.agents.orchestrator import AgentState


# Skip if Qlib not available
pytest.importorskip("qlib")


class TestFeedbackLoop:
    """Test ACTION 1: Feedback Loop activation."""

    def test_feedback_loop_triggers_on_low_sharpe(self):
        """Test that feedback loop is triggered when Sharpe is below threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BacktestConfig(
                min_sharpe=2.0,  # High threshold - will fail
                enable_feedback_loop=True,
                max_retry_attempts=3,
                ledger_path=f"{tmpdir}/ledger.json",
            )
            agent = BacktestOptimizationAgent(config)

            # Create state with low Sharpe result
            state = AgentState(
                context={
                    "strategy_result": {"name": "test_strategy"},
                    "factor_name": "test_factor",
                    "factor_family": "momentum",
                    "price_data": self._create_test_price_data(),
                    "strategy_signals": self._create_test_signals(),
                    "retry_count": 0,
                }
            )

            import asyncio
            result_state = asyncio.run(agent.optimize(state))

            # Should trigger retry
            assert result_state.context.get("should_retry") is True
            assert result_state.context.get("feedback_for_factor_gen") is not None
            assert "suggestions" in result_state.context["feedback_for_factor_gen"]
            assert result_state.context["retry_count"] == 1

    def test_feedback_loop_stops_after_max_retries(self):
        """Test that feedback loop stops after max retry attempts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BacktestConfig(
                min_sharpe=2.0,
                enable_feedback_loop=True,
                max_retry_attempts=3,
                ledger_path=f"{tmpdir}/ledger.json",
            )
            agent = BacktestOptimizationAgent(config)

            # Already at max retries
            state = AgentState(
                context={
                    "strategy_result": {"name": "test_strategy"},
                    "factor_name": "test_factor",
                    "factor_family": "momentum",
                    "price_data": self._create_test_price_data(),
                    "strategy_signals": self._create_test_signals(),
                    "retry_count": 3,  # Already at max
                }
            )

            import asyncio
            result_state = asyncio.run(agent.optimize(state))

            # Should NOT trigger retry (at max)
            assert result_state.context.get("should_retry") is False

    def test_feedback_suggestions_are_generated(self):
        """Test that actionable suggestions are generated on failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BacktestConfig(
                min_sharpe=2.0,
                enable_feedback_loop=True,
                ledger_path=f"{tmpdir}/ledger.json",
            )
            agent = BacktestOptimizationAgent(config)

            metrics = BacktestMetrics(
                sharpe_ratio=0.3,  # Very low
                max_drawdown=0.35,  # High
                win_rate=0.40,  # Low
            )

            suggestions = agent._generate_feedback_suggestions(
                metrics=metrics,
                prob_overfit=0.6,  # High overfit
                threshold_result=None,
            )

            assert len(suggestions) > 0
            assert any("SIMPLIFY" in s for s in suggestions)  # High overfit
            assert any("ALPHA" in s or "TIMING" in s for s in suggestions)  # Low sharpe

    def _create_test_price_data(self, n=500):
        """Create test price data."""
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        np.random.seed(42)
        returns = np.random.randn(n) * 0.02
        close = 100 * (1 + returns).cumprod()
        return pd.DataFrame({
            "close": close,
            "returns": returns,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "volume": np.random.randint(1000, 10000, n),
        }, index=dates)

    def _create_test_signals(self, n=500):
        """Create test signals."""
        np.random.seed(42)
        signals = np.random.choice([-1, 0, 1], n, p=[0.2, 0.6, 0.2])
        return [{"combined_signal": s} for s in signals]


class TestResearchLedger:
    """Test ACTION 2: Research Ledger integration."""

    def test_ledger_records_trials(self):
        """Test that trials are recorded to the ledger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = f"{tmpdir}/ledger.json"
            config = BacktestConfig(
                enable_dynamic_threshold=True,
                ledger_path=ledger_path,
            )
            agent = BacktestOptimizationAgent(config)

            # Verify ledger is initialized
            assert agent.research_ledger is not None
            initial_count = agent.research_ledger.get_trial_count()

            # Run optimization
            state = AgentState(
                context={
                    "strategy_result": {"name": "test_strategy"},
                    "factor_name": "test_factor",
                    "factor_family": "momentum",
                    "price_data": self._create_test_price_data(),
                    "strategy_signals": self._create_test_signals(),
                }
            )

            import asyncio
            asyncio.run(agent.optimize(state))

            # Verify trial was recorded
            assert agent.research_ledger.get_trial_count() == initial_count + 1

    def test_dynamic_threshold_increases_with_trials(self):
        """Test that threshold increases as more trials are conducted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = f"{tmpdir}/ledger.json"
            config = BacktestConfig(
                enable_dynamic_threshold=True,
                ledger_path=ledger_path,
            )
            agent = BacktestOptimizationAgent(config)

            threshold_1 = agent.research_ledger.get_current_threshold()

            # Add many trials
            from iqfmp.evaluation.research_ledger import TrialRecord
            for i in range(20):
                trial = TrialRecord(
                    factor_name=f"factor_{i}",
                    factor_family="test",
                    sharpe_ratio=1.5,
                )
                agent.research_ledger.record(trial)

            threshold_21 = agent.research_ledger.get_current_threshold()

            # Threshold should increase with more trials
            assert threshold_21 > threshold_1

    def test_ledger_file_persists(self):
        """Test that ledger file is persisted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = f"{tmpdir}/ledger.json"

            # Create agent and run optimization
            config = BacktestConfig(
                enable_dynamic_threshold=True,
                ledger_path=ledger_path,
            )
            agent1 = BacktestOptimizationAgent(config)

            state = AgentState(
                context={
                    "strategy_result": {"name": "test"},
                    "factor_name": "persistent_factor",
                    "factor_family": "test",
                    "price_data": self._create_test_price_data(),
                    "strategy_signals": self._create_test_signals(),
                }
            )

            import asyncio
            asyncio.run(agent1.optimize(state))

            # Create new agent with same ledger path
            agent2 = BacktestOptimizationAgent(config)

            # Should have the trial from agent1
            assert agent2.research_ledger.get_trial_count() >= 1

    def _create_test_price_data(self, n=500):
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        np.random.seed(42)
        returns = np.random.randn(n) * 0.02
        close = 100 * (1 + returns).cumprod()
        return pd.DataFrame({
            "close": close,
            "returns": returns,
        }, index=dates)

    def _create_test_signals(self, n=500):
        np.random.seed(42)
        signals = np.random.choice([-1, 0, 1], n)
        return [{"combined_signal": s} for s in signals]


class TestPurgedCVGatekeeper:
    """Test ACTION 3: Purged CV Gatekeeper."""

    def test_gatekeeper_rejects_high_overfit(self):
        """Test that factors with high prob_overfit are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BacktestConfig(
                enable_overfit_gatekeeper=True,
                prob_overfit_threshold=0.05,  # 5% threshold
                ledger_path=f"{tmpdir}/ledger.json",
            )
            agent = BacktestOptimizationAgent(config)

            # State with high prob_overfit
            state = AgentState(
                context={
                    "strategy_result": {"name": "test"},
                    "factor_name": "overfit_factor",
                    "factor_family": "test",
                    "price_data": self._create_test_price_data(),
                    "strategy_signals": self._create_test_signals(),
                    "prob_overfit": 0.3,  # 30% - much higher than 5% threshold
                }
            )

            import asyncio
            result_state = asyncio.run(agent.optimize(state))

            # Factor should be rejected
            assert result_state.context.get("passes_backtest") is False
            assert "REJECTED by Purged CV Gatekeeper" in result_state.context.get("rejection_reason", "")

    def test_gatekeeper_passes_low_overfit(self):
        """Test that factors with low prob_overfit pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BacktestConfig(
                enable_overfit_gatekeeper=True,
                prob_overfit_threshold=0.5,  # High threshold for test
                min_sharpe=0.0,  # Low threshold to pass
                ledger_path=f"{tmpdir}/ledger.json",
            )
            agent = BacktestOptimizationAgent(config)

            state = AgentState(
                context={
                    "strategy_result": {"name": "test"},
                    "factor_name": "good_factor",
                    "factor_family": "test",
                    "price_data": self._create_test_price_data(),
                    "strategy_signals": self._create_test_signals(),
                    "prob_overfit": 0.01,  # 1% - well below threshold
                }
            )

            import asyncio
            result_state = asyncio.run(agent.optimize(state))

            # Factor should pass (unless other constraints fail)
            assert result_state.context.get("rejection_reason") is None or \
                   "Purged CV Gatekeeper" not in result_state.context.get("rejection_reason", "")

    def test_gatekeeper_disabled_when_configured(self):
        """Test that gatekeeper can be disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BacktestConfig(
                enable_overfit_gatekeeper=False,  # Disabled
                min_sharpe=0.0,
                ledger_path=f"{tmpdir}/ledger.json",
            )
            agent = BacktestOptimizationAgent(config)

            state = AgentState(
                context={
                    "strategy_result": {"name": "test"},
                    "factor_name": "any_factor",
                    "factor_family": "test",
                    "price_data": self._create_test_price_data(),
                    "strategy_signals": self._create_test_signals(),
                    "prob_overfit": 0.99,  # Very high - but gatekeeper disabled
                }
            )

            import asyncio
            result_state = asyncio.run(agent.optimize(state))

            # Should not reject based on overfit (gatekeeper disabled)
            rejection = result_state.context.get("rejection_reason", "")
            if rejection:
                assert "Purged CV Gatekeeper" not in rejection

    def _create_test_price_data(self, n=500):
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        np.random.seed(42)
        returns = np.random.randn(n) * 0.02
        close = 100 * (1 + returns).cumprod()
        return pd.DataFrame({
            "close": close,
            "returns": returns,
        }, index=dates)

    def _create_test_signals(self, n=500):
        np.random.seed(42)
        signals = np.random.choice([-1, 0, 1], n)
        return [{"combined_signal": s} for s in signals]


class TestIntegration:
    """Integration tests for all three features working together."""

    def test_full_workflow_with_feedback_and_ledger(self):
        """Test complete workflow with feedback loop and ledger recording."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BacktestConfig(
                min_sharpe=0.5,
                enable_feedback_loop=True,
                enable_dynamic_threshold=True,
                enable_overfit_gatekeeper=True,
                prob_overfit_threshold=0.5,
                ledger_path=f"{tmpdir}/ledger.json",
            )
            agent = BacktestOptimizationAgent(config)

            state = AgentState(
                context={
                    "strategy_result": {"name": "integration_test"},
                    "factor_name": "momentum_cross",
                    "factor_family": "momentum",
                    "price_data": self._create_test_price_data(),
                    "strategy_signals": self._create_test_signals(),
                    "prob_overfit": 0.1,  # Reasonable
                }
            )

            import asyncio
            result_state = asyncio.run(agent.optimize(state))

            # Verify all systems engaged
            assert "backtest_metrics" in result_state.context
            assert result_state.context.get("total_trials", 0) >= 1

            # If passed, should have these fields
            if result_state.context.get("passes_backtest"):
                assert result_state.context.get("should_retry") is False
            else:
                # If failed, feedback should be provided
                if result_state.context.get("should_retry"):
                    assert result_state.context.get("feedback_for_factor_gen") is not None

    def _create_test_price_data(self, n=500):
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        np.random.seed(42)
        returns = np.random.randn(n) * 0.02
        close = 100 * (1 + returns).cumprod()
        return pd.DataFrame({
            "close": close,
            "returns": returns,
        }, index=dates)

    def _create_test_signals(self, n=500):
        np.random.seed(42)
        signals = np.random.choice([-1, 0, 1], n)
        return [{"combined_signal": s} for s in signals]
