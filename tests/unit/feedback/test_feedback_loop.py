"""Unit tests for FeedbackLoop closed-loop coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iqfmp.feedback.feedback_loop import (
    FeedbackLoop,
    IterationResult,
    LoopConfig,
    LoopResult,
)
from iqfmp.feedback.pattern_memory import PatternRecord
from iqfmp.feedback.structured_feedback import FailureReason, StructuredFeedback


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def loop_config():
    """Create a test loop configuration."""
    return LoopConfig(
        max_iterations=3,
        ic_threshold=0.03,
        improvement_threshold=0.005,
        enable_hypothesis_refinement=True,
        enable_pattern_memory=True,
        early_stop_no_improvement=2,
        hypothesis_refinement_interval=2,
    )


@pytest.fixture
def mock_factor():
    """Create a mock generated factor."""
    factor = MagicMock()
    factor.name = "test_momentum_factor"
    factor.code = "Ref($close, -1) / Ref($close, -5)"
    factor.family = MagicMock()
    factor.family.value = "momentum"
    return factor


@pytest.fixture
def mock_metrics():
    """Create mock evaluation metrics."""
    metrics = MagicMock()
    metrics.ic = 0.025
    metrics.ir = 0.9
    metrics.sharpe_ratio = 1.2
    metrics.max_drawdown = 0.15
    return metrics


@pytest.fixture
def mock_evaluation(mock_metrics):
    """Create a mock evaluation result."""
    evaluation = MagicMock()
    evaluation.metrics = mock_metrics
    evaluation.passes_threshold = False
    evaluation.trial_id = "trial-001"
    return evaluation


@pytest.fixture
def sample_feedback():
    """Create a sample structured feedback."""
    return StructuredFeedback(
        factor_name="test_momentum",
        hypothesis="Momentum factors based on recent price changes",
        factor_code="Ref($close, -1) / Ref($close, -5)",
        ic=0.025,
        ir=0.9,
        sharpe=1.2,
        max_drawdown=0.15,
        passes_threshold=False,
        failure_reasons=[FailureReason.LOW_IC],
        failure_details={FailureReason.LOW_IC: "IC=0.025, need >= 0.03"},
        suggestions=["Try different lookback periods"],
        trial_id="trial-001",
    )


@pytest.fixture
def success_feedback():
    """Create a successful evaluation feedback."""
    return StructuredFeedback(
        factor_name="good_momentum",
        hypothesis="Strong momentum with volume",
        factor_code="Ref($close, -1) * $volume / Mean($volume, 20)",
        ic=0.045,
        ir=1.5,
        sharpe=2.0,
        max_drawdown=0.12,
        passes_threshold=True,
        failure_reasons=[],
        failure_details={},
        suggestions=[],
        trial_id="trial-002",
    )


@pytest.fixture
def mock_factor_generator():
    """Create a mock factor generation agent."""
    generator = AsyncMock()
    return generator


@pytest.fixture
def mock_evaluator():
    """Create a mock factor evaluator."""
    evaluator = MagicMock()
    return evaluator


@pytest.fixture
def mock_hypothesis_agent():
    """Create a mock hypothesis agent."""
    agent = AsyncMock()
    return agent


@pytest.fixture
def mock_pattern_memory():
    """Create a mock pattern memory."""
    memory = MagicMock()
    memory.retrieve_similar_successes = MagicMock(return_value=[])
    memory.retrieve_similar_failures = MagicMock(return_value=[])
    memory.record_success = MagicMock()
    memory.record_failure = MagicMock()
    memory.get_family_statistics = MagicMock(return_value={})
    return memory


# ============================================================================
# LoopConfig Tests
# ============================================================================


class TestLoopConfig:
    """Tests for LoopConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LoopConfig()

        assert config.max_iterations == 5
        assert config.ic_threshold == 0.03
        assert config.improvement_threshold == 0.005
        assert config.enable_hypothesis_refinement is True
        assert config.enable_pattern_memory is True
        assert config.early_stop_no_improvement == 2
        assert config.hypothesis_refinement_interval == 2

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LoopConfig(
            max_iterations=10,
            ic_threshold=0.05,
            improvement_threshold=0.01,
            enable_hypothesis_refinement=False,
            enable_pattern_memory=False,
            early_stop_no_improvement=3,
            hypothesis_refinement_interval=3,
        )

        assert config.max_iterations == 10
        assert config.ic_threshold == 0.05
        assert config.improvement_threshold == 0.01
        assert config.enable_hypothesis_refinement is False
        assert config.enable_pattern_memory is False
        assert config.early_stop_no_improvement == 3
        assert config.hypothesis_refinement_interval == 3


# ============================================================================
# IterationResult Tests
# ============================================================================


class TestIterationResult:
    """Tests for IterationResult dataclass."""

    def test_iteration_result_creation(
        self, mock_factor, mock_evaluation, sample_feedback
    ):
        """Test creating an iteration result."""
        result = IterationResult(
            iteration=0,
            hypothesis="Test hypothesis",
            factor=mock_factor,
            evaluation=mock_evaluation,
            feedback=sample_feedback,
            improved=True,
            improvement_delta=0.01,
        )

        assert result.iteration == 0
        assert result.hypothesis == "Test hypothesis"
        assert result.factor == mock_factor
        assert result.evaluation == mock_evaluation
        assert result.feedback == sample_feedback
        assert result.improved is True
        assert result.improvement_delta == 0.01


# ============================================================================
# LoopResult Tests
# ============================================================================


class TestLoopResult:
    """Tests for LoopResult dataclass."""

    def test_loop_result_success(self, mock_factor, mock_evaluation):
        """Test successful loop result."""
        result = LoopResult(
            success=True,
            final_factor=mock_factor,
            final_evaluation=mock_evaluation,
            iterations=[],
            improvement_history=[0.02, 0.03, 0.045],
        )

        assert result.success is True
        assert result.final_factor == mock_factor
        # total_iterations and best_ic are now computed properties
        assert result.total_iterations == 0  # No iterations in list
        assert result.best_ic == 0.045  # Max of improvement_history
        assert len(result.improvement_history) == 3

    def test_loop_result_failure(self):
        """Test failed loop result."""
        result = LoopResult(
            success=False,
            final_factor=None,
            final_evaluation=None,
            iterations=[],
            improvement_history=[0.015, 0.018, 0.02, 0.019, 0.02],
        )

        assert result.success is False
        assert result.final_factor is None
        assert result.final_evaluation is None
        # total_iterations is computed from iterations list
        assert result.total_iterations == 0
        # best_ic is computed from improvement_history
        assert result.best_ic == 0.02

    def test_to_dict_with_factor(self, mock_factor, mock_evaluation, sample_feedback):
        """Test to_dict serialization with factor."""
        iteration = IterationResult(
            iteration=0,
            hypothesis="Test",
            factor=mock_factor,
            evaluation=mock_evaluation,
            feedback=sample_feedback,
            improved=True,
            improvement_delta=0.01,
        )

        result = LoopResult(
            success=True,
            final_factor=mock_factor,
            final_evaluation=mock_evaluation,
            iterations=[iteration],
            improvement_history=[0.045],
        )

        data = result.to_dict()

        assert data["success"] is True
        # Properties are computed and included in to_dict
        assert data["total_iterations"] == 1
        assert data["best_ic"] == 0.045
        assert data["improvement_history"] == [0.045]
        assert data["final_factor"]["name"] == "test_momentum_factor"
        assert len(data["iterations_summary"]) == 1

    def test_to_dict_without_factor(self):
        """Test to_dict serialization without factor."""
        result = LoopResult(
            success=False,
            final_factor=None,
            final_evaluation=None,
            iterations=[],
            improvement_history=[],
        )

        data = result.to_dict()

        assert data["success"] is False
        assert data["final_factor"] is None
        assert data["iterations_summary"] == []
        # Computed properties
        assert data["total_iterations"] == 0
        assert data["best_ic"] == 0.0


# ============================================================================
# FeedbackLoop Tests
# ============================================================================


class TestFeedbackLoopInit:
    """Tests for FeedbackLoop initialization."""

    def test_basic_init(self, mock_factor_generator, mock_evaluator):
        """Test basic initialization."""
        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
        )

        assert loop.factor_generator == mock_factor_generator
        assert loop.evaluator == mock_evaluator
        assert loop.hypothesis_agent is None
        assert loop.pattern_memory is None
        assert loop.config is not None
        assert loop.prompt_builder is not None

    def test_full_init(
        self,
        mock_factor_generator,
        mock_evaluator,
        mock_hypothesis_agent,
        mock_pattern_memory,
        loop_config,
    ):
        """Test initialization with all components."""
        prompt_builder = MagicMock()

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            hypothesis_agent=mock_hypothesis_agent,
            pattern_memory=mock_pattern_memory,
            prompt_builder=prompt_builder,
            config=loop_config,
        )

        assert loop.factor_generator == mock_factor_generator
        assert loop.evaluator == mock_evaluator
        assert loop.hypothesis_agent == mock_hypothesis_agent
        assert loop.pattern_memory == mock_pattern_memory
        assert loop.prompt_builder == prompt_builder
        assert loop.config == loop_config

    def test_default_config(self, mock_factor_generator, mock_evaluator):
        """Test that default config is created when not provided."""
        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
        )

        assert loop.config.max_iterations == 5
        assert loop.config.ic_threshold == 0.03

    def test_get_config(self, mock_factor_generator, mock_evaluator, loop_config):
        """Test get_config method."""
        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            config=loop_config,
        )

        assert loop.get_config() == loop_config

    def test_set_config(self, mock_factor_generator, mock_evaluator):
        """Test set_config method."""
        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
        )

        new_config = LoopConfig(max_iterations=10)
        loop.set_config(new_config)

        assert loop.config == new_config
        assert loop.config.max_iterations == 10


class TestFeedbackLoopRunLoop:
    """Tests for FeedbackLoop.run_loop method."""

    @pytest.mark.asyncio
    async def test_success_on_first_iteration(
        self,
        mock_factor_generator,
        mock_evaluator,
        mock_factor,
        loop_config,
    ):
        """Test loop terminates on first success."""
        # Setup: factor passes threshold on first try
        mock_factor_generator.generate_with_feedback = AsyncMock(return_value=mock_factor)

        mock_metrics = MagicMock()
        mock_metrics.ic = 0.045
        mock_metrics.ir = 1.5
        mock_metrics.sharpe_ratio = 2.0
        mock_metrics.max_drawdown = 0.12

        mock_evaluation = MagicMock()
        mock_evaluation.metrics = mock_metrics
        mock_evaluation.passes_threshold = True
        mock_evaluation.trial_id = "trial-001"
        mock_evaluation.to_structured_feedback = MagicMock(
            return_value=StructuredFeedback(
                factor_name="test",
                hypothesis="test",
                factor_code="test",
                ic=0.045,
                ir=1.5,
                sharpe=2.0,
                max_drawdown=0.12,
                passes_threshold=True,
                failure_reasons=[],
            )
        )

        mock_evaluator.evaluate = MagicMock(return_value=mock_evaluation)

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            config=loop_config,
        )

        result = await loop.run_loop(
            initial_hypothesis="Test momentum hypothesis",
            factor_family=None,
        )

        assert result.success is True
        assert result.total_iterations == 1
        assert result.best_ic == 0.045
        assert mock_factor_generator.generate_with_feedback.call_count == 1

    @pytest.mark.asyncio
    async def test_max_iterations_reached(
        self,
        mock_factor_generator,
        mock_evaluator,
        mock_factor,
    ):
        """Test loop stops at max iterations."""
        config = LoopConfig(
            max_iterations=3,
            early_stop_no_improvement=10,  # Disable early stop
        )

        mock_factor_generator.generate_with_feedback = AsyncMock(return_value=mock_factor)

        # Create metrics that improve each iteration but never pass threshold
        ics = [0.02, 0.025, 0.028]

        def make_evaluation(ic):
            metrics = MagicMock()
            metrics.ic = ic
            metrics.ir = 0.9
            metrics.sharpe_ratio = 1.0
            metrics.max_drawdown = 0.15

            evaluation = MagicMock()
            evaluation.metrics = metrics
            evaluation.passes_threshold = False
            evaluation.trial_id = f"trial-{ic}"
            evaluation.to_structured_feedback = MagicMock(
                return_value=StructuredFeedback(
                    factor_name="test",
                    hypothesis="test",
                    factor_code="test",
                    ic=ic,
                    ir=0.9,
                    sharpe=1.0,
                    max_drawdown=0.15,
                    passes_threshold=False,
                    failure_reasons=[FailureReason.LOW_IC],
                )
            )
            return evaluation

        mock_evaluator.evaluate = MagicMock(
            side_effect=[make_evaluation(ic) for ic in ics]
        )

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            config=config,
        )

        result = await loop.run_loop(initial_hypothesis="Test hypothesis")

        assert result.success is False
        assert result.total_iterations == 3
        assert result.best_ic == 0.028
        assert len(result.improvement_history) == 3

    @pytest.mark.asyncio
    async def test_early_stop_no_improvement(
        self,
        mock_factor_generator,
        mock_evaluator,
        mock_factor,
    ):
        """Test early stop when no improvement."""
        config = LoopConfig(
            max_iterations=10,
            early_stop_no_improvement=2,
        )

        mock_factor_generator.generate_with_feedback = AsyncMock(return_value=mock_factor)

        # ICs that don't improve after first iteration
        ics = [0.025, 0.023, 0.022]  # Best is first, then no improvement

        def make_evaluation(ic):
            metrics = MagicMock()
            metrics.ic = ic
            metrics.ir = 0.9
            metrics.sharpe_ratio = 1.0
            metrics.max_drawdown = 0.15

            evaluation = MagicMock()
            evaluation.metrics = metrics
            evaluation.passes_threshold = False
            evaluation.trial_id = f"trial-{ic}"
            evaluation.to_structured_feedback = MagicMock(
                return_value=StructuredFeedback(
                    factor_name="test",
                    hypothesis="test",
                    factor_code="test",
                    ic=ic,
                    ir=0.9,
                    sharpe=1.0,
                    max_drawdown=0.15,
                    passes_threshold=False,
                    failure_reasons=[FailureReason.LOW_IC],
                )
            )
            return evaluation

        mock_evaluator.evaluate = MagicMock(
            side_effect=[make_evaluation(ic) for ic in ics]
        )

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            config=config,
        )

        result = await loop.run_loop(initial_hypothesis="Test hypothesis")

        assert result.success is False
        # Should stop after 3 iterations (1 improvement + 2 no improvement)
        assert result.total_iterations == 3
        assert result.best_ic == 0.025

    @pytest.mark.asyncio
    async def test_pattern_memory_integration(
        self,
        mock_factor_generator,
        mock_evaluator,
        mock_pattern_memory,
        mock_factor,
    ):
        """Test pattern memory is used during loop."""
        config = LoopConfig(
            max_iterations=2,
            enable_pattern_memory=True,
        )

        mock_factor_generator.generate_with_feedback = AsyncMock(return_value=mock_factor)

        mock_pattern_memory.retrieve_similar_successes = MagicMock(
            return_value=[
                PatternRecord(
                    pattern_id="success-1",
                    pattern_type="success",
                    hypothesis="Similar success",
                    factor_code="success_code",
                    factor_family="momentum",
                    metrics={"ic": 0.05},
                )
            ]
        )

        mock_pattern_memory.retrieve_similar_failures = MagicMock(
            return_value=[
                PatternRecord(
                    pattern_id="failure-1",
                    pattern_type="failure",
                    hypothesis="Similar failure",
                    factor_code="failure_code",
                    factor_family="momentum",
                    metrics={"ic": 0.01},
                )
            ]
        )

        # Create evaluations
        def make_evaluation(passes):
            metrics = MagicMock()
            metrics.ic = 0.04 if passes else 0.02
            metrics.ir = 1.2 if passes else 0.8
            metrics.sharpe_ratio = 1.5
            metrics.max_drawdown = 0.12

            evaluation = MagicMock()
            evaluation.metrics = metrics
            evaluation.passes_threshold = passes
            evaluation.trial_id = "trial-001"
            evaluation.to_structured_feedback = MagicMock(
                return_value=StructuredFeedback(
                    factor_name="test",
                    hypothesis="test",
                    factor_code="test",
                    ic=metrics.ic,
                    ir=metrics.ir,
                    sharpe=1.5,
                    max_drawdown=0.12,
                    passes_threshold=passes,
                    failure_reasons=[] if passes else [FailureReason.LOW_IC],
                )
            )
            return evaluation

        mock_evaluator.evaluate = MagicMock(
            side_effect=[make_evaluation(False), make_evaluation(True)]
        )

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            pattern_memory=mock_pattern_memory,
            config=config,
        )

        result = await loop.run_loop(initial_hypothesis="Test hypothesis")

        # Verify pattern memory was queried
        assert mock_pattern_memory.retrieve_similar_successes.call_count >= 1
        assert mock_pattern_memory.retrieve_similar_failures.call_count >= 1

        # Verify patterns were recorded
        assert mock_pattern_memory.record_failure.call_count >= 1
        assert mock_pattern_memory.record_success.call_count >= 1

    @pytest.mark.asyncio
    async def test_hypothesis_refinement(
        self,
        mock_factor_generator,
        mock_evaluator,
        mock_hypothesis_agent,
        mock_factor,
    ):
        """Test hypothesis refinement at intervals."""
        config = LoopConfig(
            max_iterations=4,
            enable_hypothesis_refinement=True,
            hypothesis_refinement_interval=2,
            early_stop_no_improvement=10,
        )

        mock_factor_generator.generate_with_feedback = AsyncMock(return_value=mock_factor)

        mock_hypothesis_agent.refine_hypothesis_with_feedback = AsyncMock(
            return_value="Refined hypothesis with volume confirmation"
        )

        # Create 4 evaluations that don't pass
        def make_evaluation(i):
            metrics = MagicMock()
            metrics.ic = 0.02 + i * 0.002  # Slowly improving
            metrics.ir = 0.9
            metrics.sharpe_ratio = 1.0
            metrics.max_drawdown = 0.15

            evaluation = MagicMock()
            evaluation.metrics = metrics
            evaluation.passes_threshold = False
            evaluation.trial_id = f"trial-{i}"
            evaluation.to_structured_feedback = MagicMock(
                return_value=StructuredFeedback(
                    factor_name="test",
                    hypothesis="test",
                    factor_code="test",
                    ic=metrics.ic,
                    ir=0.9,
                    sharpe=1.0,
                    max_drawdown=0.15,
                    passes_threshold=False,
                    failure_reasons=[FailureReason.LOW_IC],
                )
            )
            return evaluation

        mock_evaluator.evaluate = MagicMock(
            side_effect=[make_evaluation(i) for i in range(4)]
        )

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            hypothesis_agent=mock_hypothesis_agent,
            config=config,
        )

        await loop.run_loop(initial_hypothesis="Initial hypothesis")

        # Hypothesis should be refined after iteration 2 (index 1)
        # And after iteration 4 (index 3)
        # So it should be called twice
        assert mock_hypothesis_agent.refine_hypothesis_with_feedback.call_count == 2

    @pytest.mark.asyncio
    async def test_generation_error_continues(
        self,
        mock_factor_generator,
        mock_evaluator,
        mock_factor,
    ):
        """Test loop continues after generation error."""
        config = LoopConfig(max_iterations=3)

        # First call raises error, second succeeds
        mock_factor_generator.generate_with_feedback = AsyncMock(
            side_effect=[
                Exception("Generation failed"),
                mock_factor,
                mock_factor,
            ]
        )

        def make_evaluation():
            metrics = MagicMock()
            metrics.ic = 0.04
            metrics.ir = 1.2
            metrics.sharpe_ratio = 1.5
            metrics.max_drawdown = 0.12

            evaluation = MagicMock()
            evaluation.metrics = metrics
            evaluation.passes_threshold = True
            evaluation.trial_id = "trial-001"
            evaluation.to_structured_feedback = MagicMock(
                return_value=StructuredFeedback(
                    factor_name="test",
                    hypothesis="test",
                    factor_code="test",
                    ic=0.04,
                    ir=1.2,
                    sharpe=1.5,
                    max_drawdown=0.12,
                    passes_threshold=True,
                    failure_reasons=[],
                )
            )
            return evaluation

        mock_evaluator.evaluate = MagicMock(return_value=make_evaluation())

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            config=config,
        )

        result = await loop.run_loop(initial_hypothesis="Test hypothesis")

        # Loop should succeed despite first iteration failing
        assert result.success is True
        assert mock_factor_generator.generate_with_feedback.call_count == 2

    @pytest.mark.asyncio
    async def test_evaluation_error_continues(
        self,
        mock_factor_generator,
        mock_evaluator,
        mock_factor,
    ):
        """Test loop continues after evaluation error."""
        config = LoopConfig(max_iterations=3)

        mock_factor_generator.generate_with_feedback = AsyncMock(return_value=mock_factor)

        def make_evaluation():
            metrics = MagicMock()
            metrics.ic = 0.04
            metrics.ir = 1.2
            metrics.sharpe_ratio = 1.5
            metrics.max_drawdown = 0.12

            evaluation = MagicMock()
            evaluation.metrics = metrics
            evaluation.passes_threshold = True
            evaluation.trial_id = "trial-001"
            evaluation.to_structured_feedback = MagicMock(
                return_value=StructuredFeedback(
                    factor_name="test",
                    hypothesis="test",
                    factor_code="test",
                    ic=0.04,
                    ir=1.2,
                    sharpe=1.5,
                    max_drawdown=0.12,
                    passes_threshold=True,
                    failure_reasons=[],
                )
            )
            return evaluation

        # First call raises error, second succeeds
        mock_evaluator.evaluate = MagicMock(
            side_effect=[
                Exception("Evaluation failed"),
                make_evaluation(),
            ]
        )

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            config=config,
        )

        result = await loop.run_loop(initial_hypothesis="Test hypothesis")

        # Loop should succeed despite first evaluation failing
        assert result.success is True

    @pytest.mark.asyncio
    async def test_improvement_tracking(
        self,
        mock_factor_generator,
        mock_evaluator,
        mock_factor,
    ):
        """Test improvement delta tracking."""
        config = LoopConfig(
            max_iterations=4,
            early_stop_no_improvement=10,
        )

        mock_factor_generator.generate_with_feedback = AsyncMock(return_value=mock_factor)

        # ICs: improving then declining
        ics = [0.02, 0.025, 0.03, 0.028]

        def make_evaluation(ic):
            metrics = MagicMock()
            metrics.ic = ic
            metrics.ir = 0.9
            metrics.sharpe_ratio = 1.0
            metrics.max_drawdown = 0.15

            evaluation = MagicMock()
            evaluation.metrics = metrics
            evaluation.passes_threshold = ic >= 0.03
            evaluation.trial_id = f"trial-{ic}"
            evaluation.to_structured_feedback = MagicMock(
                return_value=StructuredFeedback(
                    factor_name="test",
                    hypothesis="test",
                    factor_code="test",
                    ic=ic,
                    ir=0.9,
                    sharpe=1.0,
                    max_drawdown=0.15,
                    passes_threshold=ic >= 0.03,
                    failure_reasons=[] if ic >= 0.03 else [FailureReason.LOW_IC],
                )
            )
            return evaluation

        mock_evaluator.evaluate = MagicMock(
            side_effect=[make_evaluation(ic) for ic in ics]
        )

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            config=config,
        )

        result = await loop.run_loop(initial_hypothesis="Test hypothesis")

        # Should stop at iteration 3 (index 2) because it passed threshold
        assert result.total_iterations == 3
        assert result.best_ic == 0.03
        assert result.improvement_history == [0.02, 0.025, 0.03]

    @pytest.mark.asyncio
    async def test_disabled_pattern_memory(
        self,
        mock_factor_generator,
        mock_evaluator,
        mock_pattern_memory,
        mock_factor,
    ):
        """Test pattern memory can be disabled."""
        config = LoopConfig(
            max_iterations=1,
            enable_pattern_memory=False,
        )

        mock_factor_generator.generate_with_feedback = AsyncMock(return_value=mock_factor)

        metrics = MagicMock()
        metrics.ic = 0.04
        metrics.ir = 1.2
        metrics.sharpe_ratio = 1.5
        metrics.max_drawdown = 0.12

        evaluation = MagicMock()
        evaluation.metrics = metrics
        evaluation.passes_threshold = True
        evaluation.trial_id = "trial-001"
        evaluation.to_structured_feedback = MagicMock(
            return_value=StructuredFeedback(
                factor_name="test",
                hypothesis="test",
                factor_code="test",
                ic=0.04,
                ir=1.2,
                sharpe=1.5,
                max_drawdown=0.12,
                passes_threshold=True,
                failure_reasons=[],
            )
        )

        mock_evaluator.evaluate = MagicMock(return_value=evaluation)

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            pattern_memory=mock_pattern_memory,
            config=config,
        )

        await loop.run_loop(initial_hypothesis="Test hypothesis")

        # Pattern memory should not be used when disabled
        mock_pattern_memory.retrieve_similar_successes.assert_not_called()
        mock_pattern_memory.retrieve_similar_failures.assert_not_called()

    @pytest.mark.asyncio
    async def test_disabled_hypothesis_refinement(
        self,
        mock_factor_generator,
        mock_evaluator,
        mock_hypothesis_agent,
        mock_factor,
    ):
        """Test hypothesis refinement can be disabled."""
        config = LoopConfig(
            max_iterations=4,
            enable_hypothesis_refinement=False,
            early_stop_no_improvement=10,
        )

        mock_factor_generator.generate_with_feedback = AsyncMock(return_value=mock_factor)

        def make_evaluation():
            metrics = MagicMock()
            metrics.ic = 0.02
            metrics.ir = 0.9
            metrics.sharpe_ratio = 1.0
            metrics.max_drawdown = 0.15

            evaluation = MagicMock()
            evaluation.metrics = metrics
            evaluation.passes_threshold = False
            evaluation.trial_id = "trial-001"
            evaluation.to_structured_feedback = MagicMock(
                return_value=StructuredFeedback(
                    factor_name="test",
                    hypothesis="test",
                    factor_code="test",
                    ic=0.02,
                    ir=0.9,
                    sharpe=1.0,
                    max_drawdown=0.15,
                    passes_threshold=False,
                    failure_reasons=[FailureReason.LOW_IC],
                )
            )
            return evaluation

        mock_evaluator.evaluate = MagicMock(
            side_effect=[make_evaluation() for _ in range(4)]
        )

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            hypothesis_agent=mock_hypothesis_agent,
            config=config,
        )

        await loop.run_loop(initial_hypothesis="Test hypothesis")

        # Hypothesis refinement should not be called when disabled
        mock_hypothesis_agent.refine_hypothesis_with_feedback.assert_not_called()

    @pytest.mark.asyncio
    async def test_hypothesis_refinement_error_continues(
        self,
        mock_factor_generator,
        mock_evaluator,
        mock_hypothesis_agent,
        mock_factor,
    ):
        """Test loop continues after hypothesis refinement error."""
        config = LoopConfig(
            max_iterations=3,
            enable_hypothesis_refinement=True,
            hypothesis_refinement_interval=2,
            early_stop_no_improvement=10,
        )

        mock_factor_generator.generate_with_feedback = AsyncMock(return_value=mock_factor)

        mock_hypothesis_agent.refine_hypothesis_with_feedback = AsyncMock(
            side_effect=Exception("Refinement failed")
        )

        def make_evaluation(i):
            metrics = MagicMock()
            metrics.ic = 0.02 + i * 0.005
            metrics.ir = 0.9
            metrics.sharpe_ratio = 1.0
            metrics.max_drawdown = 0.15

            evaluation = MagicMock()
            evaluation.metrics = metrics
            evaluation.passes_threshold = False
            evaluation.trial_id = f"trial-{i}"
            evaluation.to_structured_feedback = MagicMock(
                return_value=StructuredFeedback(
                    factor_name="test",
                    hypothesis="test",
                    factor_code="test",
                    ic=metrics.ic,
                    ir=0.9,
                    sharpe=1.0,
                    max_drawdown=0.15,
                    passes_threshold=False,
                    failure_reasons=[FailureReason.LOW_IC],
                )
            )
            return evaluation

        mock_evaluator.evaluate = MagicMock(
            side_effect=[make_evaluation(i) for i in range(3)]
        )

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            hypothesis_agent=mock_hypothesis_agent,
            config=config,
        )

        result = await loop.run_loop(initial_hypothesis="Test hypothesis")

        # Loop should complete despite refinement errors
        assert result.total_iterations == 3

    @pytest.mark.asyncio
    async def test_factor_family_passed_to_generator(
        self,
        mock_factor_generator,
        mock_evaluator,
        mock_factor,
    ):
        """Test factor family is passed to generator."""
        config = LoopConfig(max_iterations=1)

        mock_factor_generator.generate_with_feedback = AsyncMock(return_value=mock_factor)

        metrics = MagicMock()
        metrics.ic = 0.04
        metrics.ir = 1.2
        metrics.sharpe_ratio = 1.5
        metrics.max_drawdown = 0.12

        evaluation = MagicMock()
        evaluation.metrics = metrics
        evaluation.passes_threshold = True
        evaluation.trial_id = "trial-001"
        evaluation.to_structured_feedback = MagicMock(
            return_value=StructuredFeedback(
                factor_name="test",
                hypothesis="test",
                factor_code="test",
                ic=0.04,
                ir=1.2,
                sharpe=1.5,
                max_drawdown=0.12,
                passes_threshold=True,
                failure_reasons=[],
            )
        )

        mock_evaluator.evaluate = MagicMock(return_value=evaluation)

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            config=config,
        )

        test_family = "momentum"
        await loop.run_loop(
            initial_hypothesis="Test hypothesis",
            factor_family=test_family,
        )

        # Verify factor family was passed to generator
        call_kwargs = mock_factor_generator.generate_with_feedback.call_args[1]
        assert call_kwargs["factor_family"] == test_family

    @pytest.mark.asyncio
    async def test_empty_iterations_result(
        self,
        mock_factor_generator,
        mock_evaluator,
    ):
        """Test result when all iterations fail with errors."""
        config = LoopConfig(max_iterations=2)

        # All generations fail
        mock_factor_generator.generate_with_feedback = AsyncMock(
            side_effect=Exception("Generation failed")
        )

        loop = FeedbackLoop(
            factor_generator=mock_factor_generator,
            evaluator=mock_evaluator,
            config=config,
        )

        result = await loop.run_loop(initial_hypothesis="Test hypothesis")

        assert result.success is False
        # Failed iterations are now recorded (P0-3/4 fix)
        assert result.total_iterations == 2
        # All iterations have errors
        assert len(result.errors) == 2
        assert all("Generation failed" in e for e in result.errors)
        # No successful evaluations
        assert result.best_ic == 0.0
        assert result.final_factor is None
        assert result.improvement_history == []
