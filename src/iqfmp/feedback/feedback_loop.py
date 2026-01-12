"""Feedback loop coordinator for closed-loop factor mining.

This module implements the main closed-loop coordinator that orchestrates
the iterative factor discovery process:

    State(hypothesis) -> Action(generate) -> Reward(evaluate) -> State'(refined)

The loop continues until either:
- A factor passes the evaluation threshold
- Maximum iterations reached
- No improvement detected for N consecutive iterations (early stop)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import pandas as pd

    from iqfmp.agents.factor_generation import FactorGenerationAgent, GeneratedFactor
    from iqfmp.agents.hypothesis_agent import HypothesisAgent
    from iqfmp.evaluation.factor_evaluator import EvaluationResult, FactorEvaluator
    from iqfmp.feedback.pattern_memory import PatternMemory
    from iqfmp.feedback.prompt_builder import FeedbackPromptBuilder
    from iqfmp.feedback.structured_feedback import StructuredFeedback

logger = logging.getLogger(__name__)


def _get_family_value(family: Any) -> str:
    """Extract string value from factor family.

    Handles both enum-like objects with .value attribute and plain strings.
    """
    return family.value if hasattr(family, "value") else str(family)


@dataclass
class LoopConfig:
    """Configuration for the feedback loop.

    Attributes:
        max_iterations: Maximum number of mining iterations
        ic_threshold: Minimum IC threshold for success (used for early success detection)
        improvement_threshold: Minimum IC improvement to count as progress
        enable_hypothesis_refinement: Whether to refine hypotheses
        enable_pattern_memory: Whether to use pattern memory
        early_stop_no_improvement: Stop after N iterations with no improvement
        hypothesis_refinement_interval: Refine hypothesis every N iterations
    """

    max_iterations: int = 5
    ic_threshold: float = 0.03
    improvement_threshold: float = 0.005
    enable_hypothesis_refinement: bool = True
    enable_pattern_memory: bool = True
    early_stop_no_improvement: int = 2
    hypothesis_refinement_interval: int = 2

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
        if not 0 < self.ic_threshold < 1:
            raise ValueError(f"ic_threshold must be in (0, 1), got {self.ic_threshold}")
        if self.improvement_threshold < 0:
            raise ValueError(f"improvement_threshold cannot be negative, got {self.improvement_threshold}")
        if self.early_stop_no_improvement < 1:
            raise ValueError(f"early_stop_no_improvement must be >= 1, got {self.early_stop_no_improvement}")
        if self.hypothesis_refinement_interval < 1:
            raise ValueError(f"hypothesis_refinement_interval must be >= 1, got {self.hypothesis_refinement_interval}")


@dataclass
class IterationResult:
    """Result of a single mining iteration.

    Attributes:
        iteration: Iteration number (0-indexed)
        hypothesis: Hypothesis used for this iteration
        factor: Generated factor (None if generation failed)
        evaluation: Evaluation result (None if evaluation failed)
        feedback: Structured feedback (None if no feedback generated)
        improved: Whether this iteration improved on previous best
        improvement_delta: Change in IC from previous best
        error: Error message if iteration failed
    """

    iteration: int
    hypothesis: str
    factor: Optional["GeneratedFactor"]
    evaluation: Optional["EvaluationResult"]
    feedback: Optional["StructuredFeedback"]
    improved: bool
    improvement_delta: float
    error: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate iteration result."""
        if self.iteration < 0:
            raise ValueError(f"iteration must be >= 0, got {self.iteration}")


@dataclass
class LoopResult:
    """Result of the complete feedback loop.

    Attributes:
        success: Whether a factor passed the threshold
        final_factor: Best factor found (if any)
        final_evaluation: Evaluation of best factor
        iterations: List of all iteration results
        improvement_history: List of ICs per iteration (only successful evaluations)
        errors: List of errors that occurred during the loop
    """

    success: bool
    final_factor: Optional["GeneratedFactor"]
    final_evaluation: Optional["EvaluationResult"]
    iterations: list[IterationResult]
    improvement_history: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate loop result invariants."""
        if self.success and (self.final_factor is None or self.final_evaluation is None):
            raise ValueError("success=True requires final_factor and final_evaluation to be set")

    @property
    def total_iterations(self) -> int:
        """Number of iterations run (computed from iterations list)."""
        return len(self.iterations)

    @property
    def best_ic(self) -> float:
        """Best IC achieved (computed from improvement_history)."""
        if not self.improvement_history:
            return 0.0
        return max(self.improvement_history)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        iterations_summary = []
        for it in self.iterations:
            summary: dict[str, Any] = {
                "iteration": it.iteration,
                "improved": it.improved,
            }
            if it.evaluation is not None:
                summary["ic"] = it.evaluation.metrics.ic
            if it.error is not None:
                summary["error"] = it.error
            iterations_summary.append(summary)

        return {
            "success": self.success,
            "final_factor": (
                {
                    "name": self.final_factor.name,
                    "code": self.final_factor.code,
                    "family": str(self.final_factor.family),
                }
                if self.final_factor
                else None
            ),
            "total_iterations": self.total_iterations,
            "best_ic": self.best_ic,
            "improvement_history": self.improvement_history,
            "iterations_summary": iterations_summary,
            "errors": self.errors,
        }


class FeedbackLoop:
    """Closed-loop coordinator for iterative factor mining.

    Orchestrates the feedback loop between:
    - Factor generation (with feedback context)
    - Factor evaluation
    - Pattern memory (success/failure recording)
    - Hypothesis refinement (optional)

    Usage:
        loop = FeedbackLoop(
            factor_generator=generator,
            evaluator=evaluator,
            hypothesis_agent=hypothesis_agent,  # optional
            pattern_memory=memory,              # optional
        )

        result = await loop.run_loop(
            initial_hypothesis="Momentum factors predict short-term returns",
            factor_family=FactorFamily.MOMENTUM,
            market_data=data,
        )

        if result.success:
            print(f"Found factor with IC={result.best_ic:.4f}")
    """

    def __init__(
        self,
        factor_generator: "FactorGenerationAgent",
        evaluator: "FactorEvaluator",
        hypothesis_agent: Optional["HypothesisAgent"] = None,
        pattern_memory: Optional["PatternMemory"] = None,
        prompt_builder: Optional["FeedbackPromptBuilder"] = None,
        config: Optional[LoopConfig] = None,
    ):
        """Initialize the feedback loop.

        Args:
            factor_generator: Agent for generating factors
            evaluator: Evaluator for testing factors
            hypothesis_agent: Optional agent for hypothesis refinement
            pattern_memory: Optional pattern storage for learning
            prompt_builder: Optional custom prompt builder
            config: Loop configuration
        """
        self.factor_generator = factor_generator
        self.evaluator = evaluator
        self.hypothesis_agent = hypothesis_agent
        self.pattern_memory = pattern_memory
        self.config = config or LoopConfig()

        # Initialize prompt builder
        if prompt_builder:
            self.prompt_builder = prompt_builder
        else:
            from iqfmp.feedback.prompt_builder import FeedbackPromptBuilder

            self.prompt_builder = FeedbackPromptBuilder()

    async def run_loop(
        self,
        initial_hypothesis: str,
        factor_family: Optional[Any] = None,
        market_data: Optional["pd.DataFrame"] = None,
    ) -> LoopResult:
        """Run the closed-loop factor mining process.

        Args:
            initial_hypothesis: Starting research hypothesis
            factor_family: Optional factor family constraint
            market_data: Optional market data for evaluation

        Returns:
            LoopResult with success status and best factor found
        """
        iterations: list[IterationResult] = []
        errors: list[str] = []
        current_hypothesis = initial_hypothesis
        previous_feedback: Optional["StructuredFeedback"] = None
        best_ic = float("-inf")
        best_factor: Optional["GeneratedFactor"] = None
        best_evaluation: Optional["EvaluationResult"] = None
        no_improvement_count = 0

        logger.info(
            f"Starting feedback loop with max_iterations={self.config.max_iterations}"
        )

        for i in range(self.config.max_iterations):
            logger.info(f"Mining iteration {i + 1}/{self.config.max_iterations}")

            # Retrieve similar patterns if pattern memory is enabled
            similar_successes = None
            similar_failures = None
            if self.pattern_memory and self.config.enable_pattern_memory:
                similar_successes = self.pattern_memory.retrieve_similar_successes(
                    current_hypothesis, limit=3
                )
                similar_failures = self.pattern_memory.retrieve_similar_failures(
                    current_hypothesis, limit=3
                )
                logger.debug(
                    f"Retrieved {len(similar_successes)} successes, "
                    f"{len(similar_failures)} failures from memory"
                )

            # Generate factor with feedback context
            try:
                factor = await self.factor_generator.generate_with_feedback(
                    user_request=current_hypothesis,
                    factor_family=factor_family,
                    feedback=previous_feedback,
                    similar_successes=similar_successes,
                    similar_failures=similar_failures,
                )
            except Exception as e:
                error_msg = f"Factor generation failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                # Record failed iteration
                iterations.append(
                    IterationResult(
                        iteration=i,
                        hypothesis=current_hypothesis,
                        factor=None,
                        evaluation=None,
                        feedback=None,
                        improved=False,
                        improvement_delta=0.0,
                        error=error_msg,
                    )
                )
                continue

            # Evaluate factor
            try:
                evaluation = self.evaluator.evaluate(
                    factor_name=factor.name,
                    factor_family=_get_family_value(factor.family),
                    data=market_data,
                )
            except Exception as e:
                error_msg = f"Factor evaluation failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                # Record failed iteration with factor but no evaluation
                iterations.append(
                    IterationResult(
                        iteration=i,
                        hypothesis=current_hypothesis,
                        factor=factor,
                        evaluation=None,
                        feedback=None,
                        improved=False,
                        improvement_delta=0.0,
                        error=error_msg,
                    )
                )
                continue

            # Generate structured feedback
            feedback = evaluation.to_structured_feedback(
                hypothesis=current_hypothesis,
                factor_code=factor.code,
            )

            # Calculate improvement
            current_ic = evaluation.metrics.ic
            is_first_evaluation = best_ic == float("-inf")
            improvement_delta = current_ic - best_ic if not is_first_evaluation else 0

            # Track actual best (regardless of threshold)
            is_actually_better = is_first_evaluation or current_ic > best_ic

            # Significant improvement for early stopping (requires threshold)
            if is_first_evaluation:
                significant_improvement = True
            else:
                significant_improvement = improvement_delta >= self.config.improvement_threshold

            # Update best if actually better
            if is_actually_better:
                best_ic = current_ic
                best_factor = factor
                best_evaluation = evaluation
                logger.info(f"New best IC={current_ic:.4f} (delta={improvement_delta:.4f})")

            # Early stop counter based on significant improvement
            if significant_improvement:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                logger.info(f"No significant improvement. IC={current_ic:.4f}, best={best_ic:.4f}")

            # For iteration result, "improved" means actual improvement
            improved = is_actually_better

            # Record iteration result
            iteration_result = IterationResult(
                iteration=i,
                hypothesis=current_hypothesis,
                factor=factor,
                evaluation=evaluation,
                feedback=feedback,
                improved=improved,
                improvement_delta=improvement_delta,
            )
            iterations.append(iteration_result)

            # Record pattern in memory
            if self.pattern_memory and self.config.enable_pattern_memory:
                family_value = _get_family_value(factor.family)
                if evaluation.passes_threshold:
                    self.pattern_memory.record_success(
                        hypothesis=current_hypothesis,
                        factor_code=factor.code,
                        factor_family=family_value,
                        metrics={
                            "ic": current_ic,
                            "ir": evaluation.metrics.ir,
                            "sharpe": evaluation.metrics.sharpe_ratio,
                            "max_drawdown": evaluation.metrics.max_drawdown,
                        },
                        trial_id=evaluation.trial_id,
                    )
                else:
                    self.pattern_memory.record_failure(
                        hypothesis=current_hypothesis,
                        factor_code=factor.code,
                        factor_family=family_value,
                        feedback=feedback,
                    )

            # Check termination conditions - use configured IC threshold
            passes_config_threshold = current_ic >= self.config.ic_threshold
            if passes_config_threshold:
                logger.info(
                    f"Success! Factor passed threshold with IC={current_ic:.4f} "
                    f"(threshold={self.config.ic_threshold})"
                )
                break

            if no_improvement_count >= self.config.early_stop_no_improvement:
                logger.info(
                    f"Early stop: no improvement for {no_improvement_count} iterations"
                )
                break

            # Prepare for next iteration
            previous_feedback = feedback

            # Refine hypothesis if enabled and at interval
            if (
                self.config.enable_hypothesis_refinement
                and self.hypothesis_agent
                and (i + 1) % self.config.hypothesis_refinement_interval == 0
            ):
                family_stats = None
                if self.pattern_memory:
                    family_stats = self.pattern_memory.get_family_statistics(
                        _get_family_value(factor.family)
                    )

                try:
                    current_hypothesis = (
                        await self.hypothesis_agent.refine_hypothesis_with_feedback(
                            original_hypothesis=current_hypothesis,
                            feedback=feedback,
                            family_statistics=family_stats,
                        )
                    )
                    logger.info(f"Hypothesis refined: {current_hypothesis[:100]}...")
                except Exception as e:
                    logger.warning(f"Hypothesis refinement failed: {e}")

        # Build final result
        # improvement_history only includes successful evaluations
        improvement_history = [
            it.evaluation.metrics.ic
            for it in iterations
            if it.evaluation is not None
        ]

        # Determine success using configured IC threshold
        final_success = (
            best_evaluation is not None
            and best_evaluation.metrics.ic >= self.config.ic_threshold
        )

        return LoopResult(
            success=final_success,
            final_factor=best_factor,
            final_evaluation=best_evaluation,
            iterations=iterations,
            improvement_history=improvement_history,
            errors=errors,
        )

    def get_config(self) -> LoopConfig:
        """Get current loop configuration."""
        return self.config

    def set_config(self, config: LoopConfig) -> None:
        """Update loop configuration."""
        self.config = config
