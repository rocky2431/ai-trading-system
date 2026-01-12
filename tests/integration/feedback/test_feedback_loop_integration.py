"""Integration tests for the closed-loop factor mining system.

Tests the complete feedback loop integrating:
- StructuredFeedback
- PatternMemory
- FeedbackPromptBuilder
- FeedbackLoop
- FactorEvaluator (mocked evaluate method)
- FactorGenerationAgent (mocked LLM calls)
- HypothesisAgent (mocked LLM calls)
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iqfmp.feedback.feedback_loop import (
    FeedbackLoop,
    LoopConfig,
)
from iqfmp.feedback.pattern_memory import PatternMemory, PatternRecord
from iqfmp.feedback.prompt_builder import FeedbackPromptBuilder
from iqfmp.feedback.structured_feedback import FailureReason, StructuredFeedback


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for PatternMemory."""
    store = MagicMock()
    store.add_pattern = MagicMock(return_value="pattern-123")
    store.search_patterns = MagicMock(return_value=[])
    store.get_pattern_stats = MagicMock(
        return_value={"total": 0, "success": 0, "failure": 0}
    )
    return store


@pytest.fixture
def pattern_memory(mock_vector_store):
    """Create a PatternMemory with mocked vector store."""
    return PatternMemory(
        vector_store=mock_vector_store,
        session_factory=None,  # No DB for unit test
    )


@pytest.fixture
def prompt_builder():
    """Create a FeedbackPromptBuilder."""
    return FeedbackPromptBuilder()


@pytest.fixture
def mock_generated_factor():
    """Create a mock factor with proper structure."""
    factor = MagicMock()
    factor.name = "momentum_volume_factor"
    factor.code = "Ref($close, -1) * $volume / Mean($volume, 20)"
    factor.family = MagicMock()
    factor.family.value = "momentum"
    return factor


@pytest.fixture
def mock_factor_generator(mock_generated_factor):
    """Create a mock factor generator that produces consistent factors."""
    generator = MagicMock()
    generator.generate_with_feedback = AsyncMock(return_value=mock_generated_factor)
    return generator


@pytest.fixture
def mock_evaluator():
    """Create a mock evaluator with configurable results."""
    evaluator = MagicMock()
    return evaluator


@pytest.fixture
def mock_hypothesis_agent():
    """Create a mock hypothesis agent."""
    agent = MagicMock()
    agent.refine_hypothesis_with_feedback = AsyncMock(
        return_value="Refined: Momentum with volume confirmation works better"
    )
    return agent


def create_mock_evaluation(ic: float, passes: bool = False) -> MagicMock:
    """Helper to create mock evaluation results."""
    metrics = MagicMock()
    metrics.ic = ic
    metrics.ir = ic * 30  # IR roughly 30x IC
    metrics.sharpe_ratio = ic * 50
    metrics.max_drawdown = 0.15 - ic  # Lower IC = higher drawdown

    evaluation = MagicMock()
    evaluation.metrics = metrics
    evaluation.passes_threshold = passes
    evaluation.trial_id = f"trial-ic-{ic}"
    evaluation.to_structured_feedback = MagicMock(
        return_value=StructuredFeedback(
            factor_name="test_factor",
            hypothesis="test hypothesis",
            factor_code="test code",
            ic=ic,
            ir=metrics.ir,
            sharpe=metrics.sharpe_ratio,
            max_drawdown=metrics.max_drawdown,
            passes_threshold=passes,
            failure_reasons=[] if passes else [FailureReason.LOW_IC],
            failure_details=(
                {} if passes else {FailureReason.LOW_IC: f"IC={ic:.4f}, need >= 0.03"}
            ),
            suggestions=[] if passes else ["Try different lookback periods"],
            trial_id=evaluation.trial_id,
        )
    )
    return evaluation


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
class TestFeedbackLoopIntegration:
    """Integration tests for the complete feedback loop."""

    @pytest.mark.asyncio
    async def test_complete_loop_with_success(
        self,
        mock_factor_generator,
        mock_evaluator,
        mock_hypothesis_agent,
        pattern_memory,
        prompt_builder,
    ):
        """Test complete loop from start to successful factor."""
        config = LoopConfig(
            max_iterations=5,
            ic_threshold=0.03,
            enable_hypothesis_refinement=True,
            enable_pattern_memory=True,
            hypothesis_refinement_interval=2,
        )

        # ICs that improve and eventually pass threshold
        ics = [0.02, 0.025, 0.028, 0.035]
        evaluations = [
            create_mock_evaluation(ic, passes=(ic >= 0.03)) for ic in ics
        ]
        mock_evaluator.evaluate = MagicMock(side_effect=evaluations)

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            hypothesis_agent=mock_hypothesis_agent,
            pattern_memory=pattern_memory,
            prompt_builder=prompt_builder,
            config=config,
        )

        result = await loop.run_loop(
            initial_hypothesis="Price momentum predicts short-term returns",
            factor_family="momentum",
        )

        # Verify success
        assert result.success is True
        assert result.total_iterations == 4
        assert result.best_ic == 0.035
        assert len(result.improvement_history) == 4

        # Verify hypothesis refinement was called (at iteration 2)
        assert mock_hypothesis_agent.refine_hypothesis_with_feedback.call_count == 1

        # Verify factor generator received feedback after first iteration
        calls = mock_factor_generator.generate_with_feedback.call_args_list
        assert len(calls) == 4

        # First call should have no feedback
        assert calls[0][1]["feedback"] is None

        # Subsequent calls should have feedback
        for i in range(1, len(calls)):
            assert calls[i][1]["feedback"] is not None

    @pytest.mark.asyncio
    async def test_loop_with_early_stop(
        self,
        mock_factor_generator,
        mock_evaluator,
        pattern_memory,
    ):
        """Test loop stops early when no improvement."""
        config = LoopConfig(
            max_iterations=10,
            early_stop_no_improvement=2,
            enable_hypothesis_refinement=False,
            enable_pattern_memory=True,
        )

        # ICs that don't improve after first
        ics = [0.025, 0.023, 0.024]  # Best is first, then no improvement
        evaluations = [create_mock_evaluation(ic) for ic in ics]
        mock_evaluator.evaluate = MagicMock(side_effect=evaluations)

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            pattern_memory=pattern_memory,
            config=config,
        )

        result = await loop.run_loop(
            initial_hypothesis="Volume spike indicates price movement",
        )

        # Should stop after 3 iterations (1 best + 2 no improvement)
        assert result.success is False
        assert result.total_iterations == 3
        assert result.best_ic == 0.025

    @pytest.mark.asyncio
    async def test_pattern_memory_records_patterns(
        self,
        mock_factor_generator,
        mock_evaluator,
        mock_vector_store,
    ):
        """Test that patterns are recorded to memory."""
        pattern_memory = PatternMemory(
            vector_store=mock_vector_store,
            session_factory=None,
        )

        config = LoopConfig(
            max_iterations=3,
            enable_pattern_memory=True,
        )

        # First two fail, third succeeds
        evaluations = [
            create_mock_evaluation(0.02, False),
            create_mock_evaluation(0.025, False),
            create_mock_evaluation(0.04, True),
        ]
        mock_evaluator.evaluate = MagicMock(side_effect=evaluations)

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            pattern_memory=pattern_memory,
            config=config,
        )

        await loop.run_loop(initial_hypothesis="Test hypothesis")

        # Verify vector store was called to add patterns
        # 2 failures + 1 success = 3 patterns
        assert mock_vector_store.add_pattern.call_count == 3

        # Check pattern types
        pattern_calls = mock_vector_store.add_pattern.call_args_list
        pattern_types = [call[1]["pattern_type"] for call in pattern_calls]
        assert pattern_types.count("failure") == 2
        assert pattern_types.count("success") == 1

    @pytest.mark.asyncio
    async def test_similar_patterns_passed_to_generator(
        self,
        mock_factor_generator,
        mock_evaluator,
    ):
        """Test that similar patterns are passed to factor generator."""
        mock_vector_store = MagicMock()

        # Mock similar patterns retrieval
        success_pattern = PatternRecord(
            pattern_id="success-1",
            pattern_type="success",
            hypothesis="Previous successful momentum",
            factor_code="Ref($close, -1) * $volume",
            factor_family="momentum",
            metrics={"ic": 0.05, "ir": 1.5},
        )

        failure_pattern = PatternRecord(
            pattern_id="failure-1",
            pattern_type="failure",
            hypothesis="Previous failed attempt",
            factor_code="$close / $open",
            factor_family="momentum",
            metrics={"ic": 0.01},
            feedback="Too simple",
        )

        mock_vector_store.search_patterns = MagicMock(
            side_effect=[
                # First call: successes for iteration 1 (returns dict list, not MagicMock)
                [
                    {
                        "pattern_id": success_pattern.pattern_id,
                        "pattern_type": "success",
                        "hypothesis": success_pattern.hypothesis,
                        "factor_code": success_pattern.factor_code,
                        "factor_family": success_pattern.factor_family,
                        "metrics": success_pattern.metrics,
                        "score": 0.95,
                    }
                ],
                # Second call: failures for iteration 1
                [
                    {
                        "pattern_id": failure_pattern.pattern_id,
                        "pattern_type": "failure",
                        "hypothesis": failure_pattern.hypothesis,
                        "factor_code": failure_pattern.factor_code,
                        "factor_family": failure_pattern.factor_family,
                        "metrics": failure_pattern.metrics,
                        "feedback": failure_pattern.feedback,
                        "score": 0.85,
                    }
                ],
                # Additional calls return empty
                [], [],
            ]
        )
        mock_vector_store.add_pattern = MagicMock(return_value="new-pattern")
        mock_vector_store.get_pattern_stats = MagicMock(return_value={})

        pattern_memory = PatternMemory(
            vector_store=mock_vector_store,
            session_factory=None,
        )

        config = LoopConfig(
            max_iterations=2,
            enable_pattern_memory=True,
        )

        evaluations = [
            create_mock_evaluation(0.035, True),
            create_mock_evaluation(0.04, True),
        ]
        mock_evaluator.evaluate = MagicMock(side_effect=evaluations)

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            pattern_memory=pattern_memory,
            config=config,
        )

        await loop.run_loop(initial_hypothesis="Test hypothesis")

        # Verify generator received similar patterns
        first_call = mock_factor_generator.generate_with_feedback.call_args_list[0]
        assert first_call[1]["similar_successes"] is not None
        assert first_call[1]["similar_failures"] is not None

    @pytest.mark.asyncio
    async def test_structured_feedback_flow(
        self,
        mock_factor_generator,
        mock_evaluator,
        pattern_memory,
    ):
        """Test that structured feedback flows correctly through loop."""
        config = LoopConfig(
            max_iterations=3,
            enable_pattern_memory=True,
        )

        evaluations = [
            create_mock_evaluation(0.02, False),
            create_mock_evaluation(0.025, False),
            create_mock_evaluation(0.035, True),
        ]
        mock_evaluator.evaluate = MagicMock(side_effect=evaluations)

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            pattern_memory=pattern_memory,
            config=config,
        )

        result = await loop.run_loop(initial_hypothesis="Test hypothesis")

        # Verify iteration results contain proper feedback
        for i, iteration in enumerate(result.iterations):
            assert iteration.feedback is not None
            assert isinstance(iteration.feedback, StructuredFeedback)
            assert iteration.feedback.ic == evaluations[i].metrics.ic

            if i < 2:  # First two failed
                assert not iteration.feedback.passes_threshold
                assert FailureReason.LOW_IC in iteration.feedback.failure_reasons
            else:  # Third succeeded
                assert iteration.feedback.passes_threshold

    @pytest.mark.asyncio
    async def test_prompt_builder_integration(
        self,
        mock_factor_generator,
        mock_evaluator,
        pattern_memory,
        prompt_builder,
    ):
        """Test prompt builder is used correctly in loop."""
        config = LoopConfig(
            max_iterations=2,
            enable_pattern_memory=True,
        )

        evaluations = [
            create_mock_evaluation(0.02, False),
            create_mock_evaluation(0.035, True),
        ]
        mock_evaluator.evaluate = MagicMock(side_effect=evaluations)

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            pattern_memory=pattern_memory,
            prompt_builder=prompt_builder,
            config=config,
        )

        await loop.run_loop(initial_hypothesis="Test hypothesis")

        # Generator should be called twice
        assert mock_factor_generator.generate_with_feedback.call_count == 2

        # Second call should have feedback from first iteration
        second_call = mock_factor_generator.generate_with_feedback.call_args_list[1]
        feedback = second_call[1]["feedback"]

        assert feedback is not None
        assert feedback.ic == 0.02
        assert not feedback.passes_threshold

    @pytest.mark.asyncio
    async def test_hypothesis_refinement_with_family_stats(
        self,
        mock_factor_generator,
        mock_evaluator,
        mock_hypothesis_agent,
        mock_vector_store,
    ):
        """Test hypothesis refinement receives family statistics."""
        # Setup mock to return family stats
        mock_vector_store.get_pattern_stats = MagicMock(
            return_value={
                "success": 5,
                "failure": 15,
            }
        )
        mock_vector_store.search_patterns = MagicMock(return_value=[])
        mock_vector_store.add_pattern = MagicMock(return_value="pattern-id")

        pattern_memory = PatternMemory(
            vector_store=mock_vector_store,
            session_factory=None,
        )

        config = LoopConfig(
            max_iterations=4,
            enable_hypothesis_refinement=True,
            hypothesis_refinement_interval=2,
            early_stop_no_improvement=10,
        )

        # All iterations fail to trigger refinement
        evaluations = [create_mock_evaluation(0.02 + i * 0.002) for i in range(4)]
        mock_evaluator.evaluate = MagicMock(side_effect=evaluations)

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            hypothesis_agent=mock_hypothesis_agent,
            pattern_memory=pattern_memory,
            config=config,
        )

        await loop.run_loop(
            initial_hypothesis="Test hypothesis",
            factor_family="momentum",
        )

        # Verify hypothesis agent was called with family stats
        refinement_calls = mock_hypothesis_agent.refine_hypothesis_with_feedback.call_args_list
        assert len(refinement_calls) == 2  # At iterations 2 and 4

        # Check that family_statistics was passed (may be dict or None)
        for call in refinement_calls:
            assert "family_statistics" in call[1]

    @pytest.mark.asyncio
    async def test_loop_result_serialization(
        self,
        mock_factor_generator,
        mock_evaluator,
        pattern_memory,
    ):
        """Test that loop result can be serialized to dict."""
        config = LoopConfig(max_iterations=2)

        evaluations = [
            create_mock_evaluation(0.025, False),
            create_mock_evaluation(0.035, True),
        ]
        mock_evaluator.evaluate = MagicMock(side_effect=evaluations)

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            pattern_memory=pattern_memory,
            config=config,
        )

        result = await loop.run_loop(initial_hypothesis="Test hypothesis")

        # Serialize to dict
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["success"] is True
        assert result_dict["total_iterations"] == 2
        assert result_dict["best_ic"] == 0.035
        assert "final_factor" in result_dict
        assert "iterations_summary" in result_dict
        assert len(result_dict["iterations_summary"]) == 2

    @pytest.mark.asyncio
    async def test_error_resilience(
        self,
        mock_factor_generator,
        mock_evaluator,
        pattern_memory,
    ):
        """Test loop continues despite transient errors."""
        config = LoopConfig(
            max_iterations=4,
            early_stop_no_improvement=10,
        )

        # Simulate: error, success, error, success
        mock_factor_generator.generate_with_feedback = AsyncMock(
            side_effect=[
                Exception("Transient error 1"),
                MagicMock(
                    name="factor_1",
                    code="code_1",
                    family=MagicMock(value="momentum"),
                ),
                Exception("Transient error 2"),
                MagicMock(
                    name="factor_2",
                    code="code_2",
                    family=MagicMock(value="momentum"),
                ),
            ]
        )

        evaluations = [
            create_mock_evaluation(0.025, False),
            create_mock_evaluation(0.035, True),
        ]
        mock_evaluator.evaluate = MagicMock(side_effect=evaluations)

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            pattern_memory=pattern_memory,
            config=config,
        )

        result = await loop.run_loop(initial_hypothesis="Test hypothesis")

        # Should succeed despite errors
        assert result.success is True
        # All iterations recorded (including failed ones for observability)
        # 4 iterations: error, success, error, success
        assert result.total_iterations == 4
        # Only 2 successful evaluations in improvement_history
        assert len(result.improvement_history) == 2
        # Errors are tracked
        assert len(result.errors) == 2

    @pytest.mark.asyncio
    async def test_config_updates_during_loop(
        self,
        mock_factor_generator,
        mock_evaluator,
    ):
        """Test config can be updated via set_config."""
        initial_config = LoopConfig(max_iterations=5)

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            config=initial_config,
        )

        assert loop.get_config().max_iterations == 5

        new_config = LoopConfig(max_iterations=10, ic_threshold=0.05)
        loop.set_config(new_config)

        assert loop.get_config().max_iterations == 10
        assert loop.get_config().ic_threshold == 0.05


@pytest.mark.integration
class TestFeedbackComponentIntegration:
    """Tests for integration between feedback components."""

    def test_structured_feedback_to_prompt_context(self, prompt_builder):
        """Test StructuredFeedback generates valid prompt context."""
        feedback = StructuredFeedback(
            factor_name="test_momentum",
            hypothesis="Price momentum predicts returns",
            factor_code="Ref($close, -1) / Ref($close, -5)",
            ic=0.025,
            ir=0.9,
            sharpe=1.2,
            max_drawdown=0.15,
            passes_threshold=False,
            failure_reasons=[FailureReason.LOW_IC, FailureReason.LOW_IR],
            failure_details={
                FailureReason.LOW_IC: "IC=0.025, need >= 0.03",
                FailureReason.LOW_IR: "IR=0.9, need >= 1.0",
            },
            suggestions=["Try different lookback", "Add volume confirmation"],
        )

        # Generate prompt context
        context = feedback.to_prompt_context()

        # Verify context structure
        assert "FAILED" in context
        assert "test_momentum" in context
        assert "IC: 0.025" in context or "IC=0.025" in context
        assert "Improvement Suggestions" in context or "Try different lookback" in context

    def test_prompt_builder_with_pattern_records(self, prompt_builder):
        """Test prompt builder correctly formats pattern records."""
        feedback = StructuredFeedback(
            factor_name="test",
            hypothesis="test",
            factor_code="test",
            ic=0.02,
            ir=0.8,
            sharpe=1.0,
            max_drawdown=0.18,
            passes_threshold=False,
            failure_reasons=[FailureReason.LOW_IC],
        )

        success_patterns = [
            PatternRecord(
                pattern_id="s1",
                pattern_type="success",
                hypothesis="Good momentum",
                factor_code="Ref($close, -1) * $volume",
                factor_family="momentum",
                metrics={"ic": 0.05, "ir": 1.5},
            )
        ]

        failure_patterns = [
            PatternRecord(
                pattern_id="f1",
                pattern_type="failure",
                hypothesis="Bad approach",
                factor_code="$close / $open",
                factor_family="momentum",
                metrics={"ic": 0.01},
                feedback="Too simple",
            )
        ]

        system_prompt, user_prompt = prompt_builder.build_factor_refinement_prompt(
            feedback=feedback,
            similar_successes=success_patterns,
            similar_failures=failure_patterns,
        )

        # Verify prompts contain pattern information
        assert "Successful Similar Factors" in user_prompt
        assert "Avoid These Patterns" in user_prompt
        assert success_patterns[0].factor_code in user_prompt
        assert failure_patterns[0].factor_code in user_prompt

    def test_pattern_record_roundtrip(self, mock_vector_store):
        """Test PatternRecord can be stored and retrieved."""
        pattern_memory = PatternMemory(
            vector_store=mock_vector_store,
            session_factory=None,
        )

        # Record a success
        pattern_id = pattern_memory.record_success(
            hypothesis="Test momentum hypothesis",
            factor_code="Ref($close, -1) / Ref($close, -5)",
            factor_family="momentum",
            metrics={"ic": 0.05, "ir": 1.5, "sharpe": 2.0},
        )

        # Verify vector store was called
        mock_vector_store.add_pattern.assert_called_once()
        call_kwargs = mock_vector_store.add_pattern.call_args[1]

        assert call_kwargs["pattern_type"] == "success"
        assert call_kwargs["hypothesis"] == "Test momentum hypothesis"
        assert call_kwargs["family"] == "momentum"
        assert call_kwargs["metrics"]["ic"] == 0.05

    def test_feedback_generates_suggestions(self):
        """Test StructuredFeedback generates appropriate suggestions."""
        # Low IC feedback
        low_ic_feedback = StructuredFeedback(
            factor_name="weak_factor",
            hypothesis="test",
            factor_code="test",
            ic=0.01,
            ir=0.5,
            sharpe=0.8,
            max_drawdown=0.2,
            passes_threshold=False,
            failure_reasons=[FailureReason.LOW_IC, FailureReason.LOW_IR],
        )

        # Should have suggestions
        context = low_ic_feedback.to_prompt_context()
        assert "FAILED" in context
        assert "IC" in context

        # Success feedback
        success_feedback = StructuredFeedback(
            factor_name="good_factor",
            hypothesis="test",
            factor_code="test",
            ic=0.05,
            ir=1.5,
            sharpe=2.0,
            max_drawdown=0.1,
            passes_threshold=True,
            failure_reasons=[],
        )

        context = success_feedback.to_prompt_context()
        assert "SUCCESS" in context or "passed" in context


@pytest.mark.integration
class TestEndToEndScenarios:
    """End-to-end scenario tests."""

    @pytest.mark.asyncio
    async def test_momentum_factor_discovery_scenario(
        self,
        mock_factor_generator,
        mock_evaluator,
        mock_hypothesis_agent,
        mock_vector_store,
    ):
        """Simulate complete momentum factor discovery."""
        mock_vector_store.search_patterns = MagicMock(return_value=[])
        mock_vector_store.add_pattern = MagicMock(return_value="pattern-id")
        mock_vector_store.get_pattern_stats = MagicMock(return_value={})

        pattern_memory = PatternMemory(
            vector_store=mock_vector_store,
            session_factory=None,
        )

        config = LoopConfig(
            max_iterations=5,
            ic_threshold=0.03,
            enable_hypothesis_refinement=True,
            enable_pattern_memory=True,
            hypothesis_refinement_interval=2,
            early_stop_no_improvement=3,
        )

        # Simulate realistic IC progression
        ics = [0.018, 0.022, 0.027, 0.032]
        evaluations = [create_mock_evaluation(ic, ic >= 0.03) for ic in ics]
        mock_evaluator.evaluate = MagicMock(side_effect=evaluations)

        # Simulate factors getting progressively better
        factors = []
        for i, ic in enumerate(ics):
            factor = MagicMock()
            factor.name = f"momentum_v{i+1}"
            factor.code = f"Ref($close, -{5+i*5}) / Ref($close, -{10+i*5})"
            factor.family = MagicMock()
            factor.family.value = "momentum"
            factors.append(factor)

        mock_factor_generator.generate_with_feedback = AsyncMock(
            side_effect=factors
        )

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            hypothesis_agent=mock_hypothesis_agent,
            pattern_memory=pattern_memory,
            config=config,
        )

        result = await loop.run_loop(
            initial_hypothesis="Short-term price momentum predicts next-day returns",
            factor_family="momentum",
        )

        # Verify scenario outcome
        assert result.success is True
        assert result.total_iterations == 4
        assert result.best_ic == 0.032
        assert result.final_factor.name == "momentum_v4"

        # Verify learning occurred
        assert len(result.improvement_history) == 4
        assert result.improvement_history[-1] > result.improvement_history[0]

        # Verify patterns were recorded (3 failures + 1 success)
        assert mock_vector_store.add_pattern.call_count == 4

    @pytest.mark.asyncio
    async def test_failed_exploration_scenario(
        self,
        mock_factor_generator,
        mock_evaluator,
        mock_vector_store,
    ):
        """Simulate scenario where no good factor is found."""
        mock_vector_store.search_patterns = MagicMock(return_value=[])
        mock_vector_store.add_pattern = MagicMock(return_value="pattern-id")
        mock_vector_store.get_pattern_stats = MagicMock(return_value={})

        pattern_memory = PatternMemory(
            vector_store=mock_vector_store,
            session_factory=None,
        )

        config = LoopConfig(
            max_iterations=5,
            early_stop_no_improvement=2,
            enable_pattern_memory=True,
        )

        # ICs plateau and don't improve
        ics = [0.015, 0.018, 0.016, 0.017]
        evaluations = [create_mock_evaluation(ic) for ic in ics]
        mock_evaluator.evaluate = MagicMock(side_effect=evaluations)

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            pattern_memory=pattern_memory,
            config=config,
        )

        result = await loop.run_loop(
            initial_hypothesis="Random hypothesis that won't work",
        )

        # Should fail with early stop
        assert result.success is False
        # With default improvement_threshold=0.005:
        # - Iteration 0: IC=0.015, first → significant_improvement=True
        # - Iteration 1: IC=0.018, delta=0.003 < 0.005 → no significant improvement
        # - Iteration 2: IC=0.016, delta<0 → no significant improvement (count=2)
        # Early stop after 3 iterations (no_improvement_count reaches 2)
        assert result.total_iterations == 3
        # Best IC is still tracked correctly
        assert result.best_ic == 0.018

        # All patterns should be recorded as failures
        pattern_types = [
            call[1]["pattern_type"]
            for call in mock_vector_store.add_pattern.call_args_list
        ]
        assert all(pt == "failure" for pt in pattern_types)
