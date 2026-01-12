"""Feedback prompt builder for closed-loop factor mining.

This module constructs LLM prompts that incorporate feedback context
from previous iterations, enabling the model to learn from past
successes and failures.

Key features:
- Injects structured feedback from evaluation results
- Includes success examples for inspiration
- Warns against failure patterns to avoid
- Supports hypothesis refinement prompts
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from iqfmp.feedback.pattern_memory import PatternRecord
    from iqfmp.feedback.structured_feedback import FailureReason, StructuredFeedback


# Data-driven instruction mapping for failure reasons
def _get_failure_instruction_mapping() -> dict["FailureReason", str]:
    """Get mapping of failure reasons to generation instructions.

    Lazy initialization to avoid circular imports.
    """
    from iqfmp.feedback.structured_feedback import FailureReason

    return {
        FailureReason.LOW_IC: (
            "Focus on improving predictive power. Consider:\n"
            "- Different data fields or combinations\n"
            "- Longer lookback periods\n"
            "- Cross-sectional comparisons"
        ),
        FailureReason.LOW_IR: (
            "Focus on signal consistency. Consider:\n"
            "- Adding smoothing or averaging\n"
            "- Rank-based transformations\n"
            "- Z-score normalization"
        ),
        FailureReason.HIGH_DRAWDOWN: (
            "Focus on risk management. Consider:\n"
            "- Volatility scaling\n"
            "- Conditional signals\n"
            "- Market regime awareness"
        ),
        FailureReason.OVERFITTING: (
            "Focus on robustness. Consider:\n"
            "- Simpler expressions with fewer parameters\n"
            "- Robust statistics (median vs mean)\n"
            "- Longer time horizons"
        ),
        FailureReason.LOOKAHEAD_BIAS: (
            "CRITICAL: Fix lookahead bias. Ensure:\n"
            "- All data uses proper lag: Ref($field, -1)\n"
            "- No future information in calculations\n"
            "- Check all time references"
        ),
        FailureReason.INSTABILITY: (
            "Focus on stability. Consider:\n"
            "- Market regime awareness\n"
            "- Adaptive parameters\n"
            "- Multiple time horizons"
        ),
        FailureReason.LOW_SHARPE: (
            "Focus on risk-adjusted returns. Consider:\n"
            "- Volatility targeting\n"
            "- Signal filtering\n"
            "- Position sizing awareness"
        ),
        FailureReason.LOW_WIN_RATE: (
            "Focus on trade frequency. Consider:\n"
            "- Mean-reversion components\n"
            "- Confirmation signals\n"
            "- Filtering false positives"
        ),
    }

logger = logging.getLogger(__name__)


class FeedbackPromptBuilder:
    """Builds LLM prompts with feedback context for iterative factor improvement.

    This builder creates prompts that help the LLM:
    1. Understand what went wrong in previous attempts
    2. Learn from successful similar factors
    3. Avoid patterns that led to failures
    4. Generate improved factors based on feedback

    Usage:
        builder = FeedbackPromptBuilder()

        system_prompt, user_prompt = builder.build_factor_refinement_prompt(
            feedback=feedback,
            similar_successes=successes,
            similar_failures=failures,
        )
    """

    # Default system prompts
    FACTOR_GENERATION_SYSTEM_PROMPT = """You are a quantitative factor development expert.
Your task is to generate improved Qlib-compatible factor expressions based on feedback.

RULES:
1. DO NOT repeat the same approach that failed
2. Learn from successful similar factors provided as examples
3. Avoid patterns that led to failures
4. Generate a single, valid Qlib expression
5. Focus on the specific issues identified in the feedback

OUTPUT FORMAT:
Return ONLY a valid Qlib expression. Do not include explanations or markdown formatting.
Examples of valid expressions:
- Ref($close, -1) / Ref($close, -5)
- Mean($volume, 20) / Mean($volume, 5)
- ($high - $low) / $close
"""

    HYPOTHESIS_REFINEMENT_SYSTEM_PROMPT = """You are a quantitative research hypothesis expert.
Your task is to refine research hypotheses based on factor evaluation feedback.

The original hypothesis led to a factor that failed evaluation.
Generate an IMPROVED hypothesis that addresses the identified issues.

RULES:
1. Address the specific weaknesses identified in the feedback
2. Take a different angle on the same market phenomenon
3. Be specific and testable
4. Consider the family's historical success patterns

OUTPUT FORMAT:
Return ONLY the refined hypothesis text. No additional formatting or explanation.
"""

    def build_factor_refinement_prompt(
        self,
        feedback: "StructuredFeedback",
        similar_successes: Optional[list["PatternRecord"]] = None,
        similar_failures: Optional[list["PatternRecord"]] = None,
        base_prompt: Optional[str] = None,
    ) -> tuple[str, str]:
        """Build prompts for factor refinement based on feedback.

        Args:
            feedback: Structured feedback from previous evaluation
            similar_successes: Success patterns for inspiration
            similar_failures: Failure patterns to avoid
            base_prompt: Optional base user prompt to enhance

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = self.FACTOR_GENERATION_SYSTEM_PROMPT

        # Build user prompt sections
        sections = []

        # Add feedback context
        sections.append(feedback.to_prompt_context())

        # Add success examples if available
        if similar_successes:
            success_section = self._format_success_examples(similar_successes)
            if success_section:
                sections.append(success_section)

        # Add failure warnings if available
        if similar_failures:
            failure_section = self._format_failure_warnings(similar_failures)
            if failure_section:
                sections.append(failure_section)

        # Add base prompt if provided
        if base_prompt:
            sections.append(f"\n## Original Request\n{base_prompt}")

        # Add generation instruction
        sections.append(self._get_generation_instruction(feedback))

        user_prompt = "\n".join(sections)

        return system_prompt, user_prompt

    def build_hypothesis_refinement_prompt(
        self,
        original_hypothesis: str,
        feedback: "StructuredFeedback",
        family_statistics: Optional[dict[str, Any]] = None,
    ) -> tuple[str, str]:
        """Build prompts for hypothesis refinement.

        Args:
            original_hypothesis: The original research hypothesis
            feedback: Structured feedback from evaluation
            family_statistics: Optional statistics for the factor family

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = self.HYPOTHESIS_REFINEMENT_SYSTEM_PROMPT

        # Build user prompt
        sections = []

        # Add original hypothesis
        sections.append(f"## Original Hypothesis\n{original_hypothesis}")

        # Add evaluation feedback
        sections.append(f"\n## Evaluation Feedback\n{feedback.to_prompt_context()}")

        # Add family statistics if available
        if family_statistics:
            stats_section = self._format_family_statistics(family_statistics)
            if stats_section:
                sections.append(stats_section)

        # Add instruction
        sections.append("""
## Task
Generate a REFINED hypothesis that:
1. Addresses the identified weaknesses
2. Takes a different angle on the same market phenomenon
3. Is specific and testable

Return ONLY the refined hypothesis text.
""")

        user_prompt = "\n".join(sections)

        return system_prompt, user_prompt

    def _format_success_examples(
        self,
        patterns: list["PatternRecord"],
    ) -> str:
        """Format success patterns as examples.

        Args:
            patterns: List of success patterns

        Returns:
            Formatted string for prompt inclusion
        """
        if not patterns:
            return ""

        lines = ["## Successful Similar Factors (for inspiration)"]

        for i, pattern in enumerate(patterns[:3], 1):
            ic = pattern.metrics.get("ic", 0)
            ir = pattern.metrics.get("ir", 0)

            lines.append(f"\n### Example {i} (IC={ic:.4f}, IR={ir:.2f})")
            lines.append(f"**Hypothesis:** {pattern.hypothesis[:200]}...")
            lines.append(f"**Code:** `{pattern.factor_code}`")

        lines.append("\nLearn from these patterns but DO NOT copy them directly.")

        return "\n".join(lines)

    def _format_failure_warnings(
        self,
        patterns: list["PatternRecord"],
    ) -> str:
        """Format failure patterns as warnings.

        Args:
            patterns: List of failure patterns

        Returns:
            Formatted string for prompt inclusion
        """
        if not patterns:
            return ""

        lines = ["## Avoid These Patterns (led to failures)"]

        for i, pattern in enumerate(patterns[:3], 1):
            reasons = ", ".join(pattern.failure_reasons[:3])
            lines.append(f"\n### Warning {i}: {reasons}")
            lines.append(f"**Failed approach:** `{pattern.factor_code}`")
            lines.append(f"**Why it failed:** {pattern.feedback[:200] if pattern.feedback else 'Unknown'}...")

        lines.append("\nDO NOT use similar approaches to those shown above.")

        return "\n".join(lines)

    def _format_family_statistics(
        self,
        stats: dict[str, Any],
    ) -> str:
        """Format family statistics for prompt.

        Args:
            stats: Family statistics dictionary

        Returns:
            Formatted string for prompt inclusion
        """
        lines = ["\n## Factor Family Statistics"]

        success_rate = stats.get("success_rate", 0)
        lines.append(f"- Success rate: {success_rate:.1%}")

        common_failures = stats.get("common_failures", [])
        if common_failures:
            lines.append(f"- Common failures: {', '.join(common_failures)}")

        best_ic = stats.get("best_ic", 0)
        if best_ic > 0:
            lines.append(f"- Best IC achieved: {best_ic:.4f}")

        success_count = stats.get("success_count", 0)
        failure_count = stats.get("failure_count", 0)
        lines.append(f"- Total attempts: {success_count + failure_count} ({success_count} successes, {failure_count} failures)")

        return "\n".join(lines)

    def _get_generation_instruction(
        self,
        feedback: "StructuredFeedback",
    ) -> str:
        """Get generation instruction based on failure type.

        Args:
            feedback: Structured feedback

        Returns:
            Instruction string tailored to the failure type
        """
        instructions = ["\n## Generation Instruction"]

        # Tailor instruction based on primary failure using data-driven mapping
        if feedback.failure_reasons:
            primary = feedback.failure_reasons[0]
            instruction_mapping = _get_failure_instruction_mapping()

            instruction = instruction_mapping.get(
                primary,
                "Generate an improved factor that addresses the identified issues.\n"
                "Try a fundamentally different approach.",
            )
            instructions.append(instruction)
        else:
            instructions.append(
                "Generate a Qlib-compatible factor expression.\n"
                "Focus on creating a factor with strong predictive power and consistency."
            )

        instructions.append("\nReturn ONLY the Qlib expression. No explanations.")

        return "\n".join(instructions)

    def build_combined_prompt(
        self,
        user_request: str,
        feedback: Optional["StructuredFeedback"] = None,
        similar_successes: Optional[list["PatternRecord"]] = None,
        similar_failures: Optional[list["PatternRecord"]] = None,
        family_statistics: Optional[dict[str, Any]] = None,
    ) -> tuple[str, str]:
        """Build a combined prompt with all available context.

        This is a convenience method that combines all feedback elements
        into a comprehensive prompt.

        Args:
            user_request: Original user request/hypothesis
            feedback: Structured feedback from previous attempt
            similar_successes: Success patterns
            similar_failures: Failure patterns
            family_statistics: Family statistics

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = self.FACTOR_GENERATION_SYSTEM_PROMPT

        sections = [f"## User Request\n{user_request}"]

        # Add feedback if available
        if feedback:
            sections.append(f"\n{feedback.to_prompt_context()}")

        # Add success examples
        if similar_successes:
            success_section = self._format_success_examples(similar_successes)
            if success_section:
                sections.append(f"\n{success_section}")

        # Add failure warnings
        if similar_failures:
            failure_section = self._format_failure_warnings(similar_failures)
            if failure_section:
                sections.append(f"\n{failure_section}")

        # Add family statistics
        if family_statistics:
            stats_section = self._format_family_statistics(family_statistics)
            if stats_section:
                sections.append(stats_section)

        # Add generation instruction
        if feedback:
            sections.append(self._get_generation_instruction(feedback))
        else:
            sections.append(
                "\n## Instruction\n"
                "Generate a Qlib-compatible factor expression based on the request above.\n"
                "Return ONLY the expression, no explanations."
            )

        user_prompt = "\n".join(sections)

        return system_prompt, user_prompt
