"""Unit tests for FeedbackPromptBuilder."""

import pytest

from iqfmp.feedback.pattern_memory import PatternRecord
from iqfmp.feedback.prompt_builder import FeedbackPromptBuilder
from iqfmp.feedback.structured_feedback import FailureReason, StructuredFeedback


@pytest.fixture
def prompt_builder():
    """Create a FeedbackPromptBuilder instance."""
    return FeedbackPromptBuilder()


@pytest.fixture
def sample_feedback():
    """Create a sample StructuredFeedback."""
    return StructuredFeedback(
        factor_name="test_momentum",
        hypothesis="Momentum factors based on recent price changes",
        factor_code="Ref($close, -1) / Ref($close, -5)",
        ic=0.02,
        ir=0.8,
        sharpe=0.5,
        max_drawdown=0.18,
        passes_threshold=False,
        failure_reasons=[FailureReason.LOW_IC, FailureReason.LOW_IR],
        failure_details={
            FailureReason.LOW_IC: "IC=0.02, need >= 0.03",
            FailureReason.LOW_IR: "IR=0.8, need >= 1.0",
        },
        suggestions=["Try different data fields"],
        trial_id="trial-001",
    )


@pytest.fixture
def success_patterns():
    """Create sample success patterns."""
    return [
        PatternRecord(
            pattern_id="success-1",
            pattern_type="success",
            hypothesis="Price momentum with volume confirmation",
            factor_code="Ref($close, -1) * Mean($volume, 20) / Mean($volume, 5)",
            factor_family="momentum",
            metrics={"ic": 0.05, "ir": 1.5, "sharpe": 2.0},
        ),
        PatternRecord(
            pattern_id="success-2",
            pattern_type="success",
            hypothesis="Volatility-adjusted returns",
            factor_code="($close - Ref($close, -5)) / Std($close, 20)",
            factor_family="momentum",
            metrics={"ic": 0.04, "ir": 1.2, "sharpe": 1.8},
        ),
    ]


@pytest.fixture
def failure_patterns():
    """Create sample failure patterns."""
    return [
        PatternRecord(
            pattern_id="failure-1",
            pattern_type="failure",
            hypothesis="Simple price ratio",
            factor_code="$close / $open",
            factor_family="momentum",
            metrics={"ic": 0.01, "ir": 0.4},
            feedback="Low IC and IR, too simple",
            failure_reasons=["low_ic", "low_ir"],
        ),
        PatternRecord(
            pattern_id="failure-2",
            pattern_type="failure",
            hypothesis="Volume momentum",
            factor_code="$volume / Mean($volume, 5)",
            factor_family="volume",
            metrics={"ic": 0.015, "ir": 0.6},
            feedback="Not predictive enough",
            failure_reasons=["low_ic"],
        ),
    ]


class TestBuildFactorRefinementPrompt:
    """Tests for build_factor_refinement_prompt method."""

    def test_basic_prompt_generation(self, prompt_builder, sample_feedback):
        """Test basic prompt generation with feedback."""
        system_prompt, user_prompt = prompt_builder.build_factor_refinement_prompt(
            feedback=sample_feedback,
        )

        # System prompt should contain rules
        assert "DO NOT repeat" in system_prompt
        assert "Qlib" in system_prompt

        # User prompt should contain feedback context
        assert "FAILED" in user_prompt
        assert sample_feedback.factor_name in user_prompt
        assert "IC: 0.02" in user_prompt

    def test_prompt_with_success_examples(
        self, prompt_builder, sample_feedback, success_patterns
    ):
        """Test prompt includes success examples."""
        system_prompt, user_prompt = prompt_builder.build_factor_refinement_prompt(
            feedback=sample_feedback,
            similar_successes=success_patterns,
        )

        assert "Successful Similar Factors" in user_prompt
        assert "Example 1" in user_prompt
        assert success_patterns[0].factor_code in user_prompt
        assert "DO NOT copy them directly" in user_prompt

    def test_prompt_with_failure_warnings(
        self, prompt_builder, sample_feedback, failure_patterns
    ):
        """Test prompt includes failure warnings."""
        system_prompt, user_prompt = prompt_builder.build_factor_refinement_prompt(
            feedback=sample_feedback,
            similar_failures=failure_patterns,
        )

        assert "Avoid These Patterns" in user_prompt
        assert "Warning 1" in user_prompt
        assert failure_patterns[0].factor_code in user_prompt
        assert "DO NOT use similar approaches" in user_prompt

    def test_prompt_with_all_context(
        self, prompt_builder, sample_feedback, success_patterns, failure_patterns
    ):
        """Test prompt with all context elements."""
        system_prompt, user_prompt = prompt_builder.build_factor_refinement_prompt(
            feedback=sample_feedback,
            similar_successes=success_patterns,
            similar_failures=failure_patterns,
            base_prompt="Generate a momentum factor",
        )

        # Should have all sections
        assert "FAILED" in user_prompt
        assert "Successful Similar Factors" in user_prompt
        assert "Avoid These Patterns" in user_prompt
        assert "Original Request" in user_prompt
        assert "Generation Instruction" in user_prompt

    def test_empty_patterns_lists(self, prompt_builder, sample_feedback):
        """Test with empty pattern lists."""
        system_prompt, user_prompt = prompt_builder.build_factor_refinement_prompt(
            feedback=sample_feedback,
            similar_successes=[],
            similar_failures=[],
        )

        # Should not include empty sections
        assert "Successful Similar Factors" not in user_prompt
        assert "Avoid These Patterns" not in user_prompt


class TestBuildHypothesisRefinementPrompt:
    """Tests for build_hypothesis_refinement_prompt method."""

    def test_basic_hypothesis_prompt(self, prompt_builder, sample_feedback):
        """Test basic hypothesis refinement prompt."""
        original = "Price momentum predicts future returns"

        system_prompt, user_prompt = prompt_builder.build_hypothesis_refinement_prompt(
            original_hypothesis=original,
            feedback=sample_feedback,
        )

        # System prompt should be for hypothesis refinement
        assert "hypothesis" in system_prompt.lower()
        assert "IMPROVED" in system_prompt

        # User prompt should contain original and feedback
        assert "Original Hypothesis" in user_prompt
        assert original in user_prompt
        assert "Evaluation Feedback" in user_prompt

    def test_hypothesis_prompt_with_family_stats(self, prompt_builder, sample_feedback):
        """Test hypothesis prompt with family statistics."""
        stats = {
            "success_rate": 0.25,
            "common_failures": ["low_ic", "overfitting"],
            "best_ic": 0.06,
            "success_count": 5,
            "failure_count": 15,
        }

        system_prompt, user_prompt = prompt_builder.build_hypothesis_refinement_prompt(
            original_hypothesis="Test hypothesis",
            feedback=sample_feedback,
            family_statistics=stats,
        )

        assert "Factor Family Statistics" in user_prompt
        assert "25.0%" in user_prompt
        assert "low_ic" in user_prompt
        assert "0.0600" in user_prompt


class TestGenerationInstructions:
    """Tests for generation instruction tailoring."""

    def test_low_ic_instruction(self, prompt_builder):
        """Test instruction for low IC failure."""
        feedback = StructuredFeedback(
            factor_name="test",
            hypothesis="test",
            factor_code="test",
            ic=0.01,
            ir=1.2,
            sharpe=1.0,
            max_drawdown=0.1,
            passes_threshold=False,
            failure_reasons=[FailureReason.LOW_IC],
        )

        _, user_prompt = prompt_builder.build_factor_refinement_prompt(feedback=feedback)

        assert "predictive power" in user_prompt.lower()
        assert "data fields" in user_prompt.lower()

    def test_low_ir_instruction(self, prompt_builder):
        """Test instruction for low IR failure."""
        feedback = StructuredFeedback(
            factor_name="test",
            hypothesis="test",
            factor_code="test",
            ic=0.05,
            ir=0.5,
            sharpe=1.0,
            max_drawdown=0.1,
            passes_threshold=False,
            failure_reasons=[FailureReason.LOW_IR],
        )

        _, user_prompt = prompt_builder.build_factor_refinement_prompt(feedback=feedback)

        assert "consistency" in user_prompt.lower()
        assert "smoothing" in user_prompt.lower()

    def test_high_drawdown_instruction(self, prompt_builder):
        """Test instruction for high drawdown failure."""
        feedback = StructuredFeedback(
            factor_name="test",
            hypothesis="test",
            factor_code="test",
            ic=0.05,
            ir=1.5,
            sharpe=1.0,
            max_drawdown=0.3,
            passes_threshold=False,
            failure_reasons=[FailureReason.HIGH_DRAWDOWN],
        )

        _, user_prompt = prompt_builder.build_factor_refinement_prompt(feedback=feedback)

        assert "risk" in user_prompt.lower()
        assert "volatility" in user_prompt.lower()

    def test_overfitting_instruction(self, prompt_builder):
        """Test instruction for overfitting failure."""
        feedback = StructuredFeedback(
            factor_name="test",
            hypothesis="test",
            factor_code="test",
            ic=0.05,
            ir=1.5,
            sharpe=2.0,
            max_drawdown=0.1,
            passes_threshold=False,
            failure_reasons=[FailureReason.OVERFITTING],
        )

        _, user_prompt = prompt_builder.build_factor_refinement_prompt(feedback=feedback)

        assert "robustness" in user_prompt.lower()
        assert "simpler" in user_prompt.lower()

    def test_lookahead_bias_instruction(self, prompt_builder):
        """Test instruction for lookahead bias failure."""
        feedback = StructuredFeedback(
            factor_name="test",
            hypothesis="test",
            factor_code="test",
            ic=0.05,
            ir=1.5,
            sharpe=2.0,
            max_drawdown=0.1,
            passes_threshold=False,
            failure_reasons=[FailureReason.LOOKAHEAD_BIAS],
        )

        _, user_prompt = prompt_builder.build_factor_refinement_prompt(feedback=feedback)

        assert "CRITICAL" in user_prompt
        assert "lookahead" in user_prompt.lower()
        assert "Ref" in user_prompt


class TestBuildCombinedPrompt:
    """Tests for build_combined_prompt method."""

    def test_combined_prompt_basic(self, prompt_builder):
        """Test combined prompt with just user request."""
        system_prompt, user_prompt = prompt_builder.build_combined_prompt(
            user_request="Generate a momentum factor",
        )

        assert "User Request" in user_prompt
        assert "momentum factor" in user_prompt
        assert "Instruction" in user_prompt

    def test_combined_prompt_with_all_context(
        self, prompt_builder, sample_feedback, success_patterns, failure_patterns
    ):
        """Test combined prompt with all context."""
        stats = {
            "success_rate": 0.3,
            "common_failures": ["low_ic"],
            "best_ic": 0.05,
            "success_count": 3,
            "failure_count": 7,
        }

        system_prompt, user_prompt = prompt_builder.build_combined_prompt(
            user_request="Generate improved momentum factor",
            feedback=sample_feedback,
            similar_successes=success_patterns,
            similar_failures=failure_patterns,
            family_statistics=stats,
        )

        # Should include all sections
        assert "User Request" in user_prompt
        assert "FAILED" in user_prompt
        assert "Successful Similar Factors" in user_prompt
        assert "Avoid These Patterns" in user_prompt
        assert "Factor Family Statistics" in user_prompt
        assert "Generation Instruction" in user_prompt


class TestFormatMethods:
    """Tests for internal formatting methods."""

    def test_format_success_examples_empty(self, prompt_builder):
        """Test formatting empty success list."""
        result = prompt_builder._format_success_examples([])
        assert result == ""

    def test_format_failure_warnings_empty(self, prompt_builder):
        """Test formatting empty failure list."""
        result = prompt_builder._format_failure_warnings([])
        assert result == ""

    def test_format_success_examples_limits_to_three(
        self, prompt_builder, success_patterns
    ):
        """Test that only first 3 examples are included."""
        # Add more patterns
        many_patterns = success_patterns + [
            PatternRecord(
                pattern_id=f"success-{i}",
                pattern_type="success",
                hypothesis=f"Hypothesis {i}",
                factor_code=f"code_{i}",
                factor_family="momentum",
                metrics={"ic": 0.04},
            )
            for i in range(3, 10)
        ]

        result = prompt_builder._format_success_examples(many_patterns)

        # Should have exactly 3 examples
        assert result.count("### Example") == 3
        assert "Example 1" in result
        assert "Example 2" in result
        assert "Example 3" in result
        assert "Example 4" not in result

    def test_format_family_statistics(self, prompt_builder):
        """Test formatting family statistics."""
        stats = {
            "success_rate": 0.25,
            "common_failures": ["low_ic", "high_drawdown"],
            "best_ic": 0.055,
            "success_count": 10,
            "failure_count": 30,
        }

        result = prompt_builder._format_family_statistics(stats)

        assert "Factor Family Statistics" in result
        assert "25.0%" in result
        assert "low_ic" in result
        assert "high_drawdown" in result
        assert "0.0550" in result
        assert "40" in result  # Total attempts
