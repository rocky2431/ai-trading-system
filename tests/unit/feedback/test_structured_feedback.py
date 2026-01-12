"""Unit tests for StructuredFeedback."""

import pytest
from unittest.mock import MagicMock

from iqfmp.feedback.structured_feedback import (
    FailureReason,
    StructuredFeedback,
    _calculate_std,
    _generate_suggestions,
)


class TestFailureReason:
    """Tests for FailureReason enum."""

    def test_failure_reason_values(self):
        """Test all failure reason values."""
        assert FailureReason.LOW_IC.value == "low_ic"
        assert FailureReason.LOW_IR.value == "low_ir"
        assert FailureReason.HIGH_DRAWDOWN.value == "high_drawdown"
        assert FailureReason.OVERFITTING.value == "overfitting"
        assert FailureReason.LOOKAHEAD_BIAS.value == "lookahead_bias"
        assert FailureReason.INSTABILITY.value == "instability"
        assert FailureReason.DUPLICATE.value == "duplicate"
        assert FailureReason.LOW_SHARPE.value == "low_sharpe"
        assert FailureReason.LOW_WIN_RATE.value == "low_win_rate"

    def test_failure_reason_is_string_enum(self):
        """Test that FailureReason is a string enum."""
        assert isinstance(FailureReason.LOW_IC, str)
        assert FailureReason.LOW_IC == "low_ic"


class TestStructuredFeedback:
    """Tests for StructuredFeedback dataclass."""

    def test_basic_creation(self):
        """Test basic StructuredFeedback creation."""
        feedback = StructuredFeedback(
            factor_name="test_factor",
            hypothesis="Price momentum predicts returns",
            factor_code="Ref($close, -1) / Ref($close, -5) - 1",
            ic=0.02,
            ir=0.8,
            sharpe=1.2,
            max_drawdown=0.15,
            passes_threshold=False,
        )

        assert feedback.factor_name == "test_factor"
        assert feedback.hypothesis == "Price momentum predicts returns"
        assert feedback.ic == 0.02
        assert feedback.ir == 0.8
        assert feedback.sharpe == 1.2
        assert feedback.max_drawdown == 0.15
        assert feedback.passes_threshold is False
        assert feedback.confidence == 0.5  # default

    def test_creation_with_failure_reasons(self):
        """Test creation with failure reasons."""
        feedback = StructuredFeedback(
            factor_name="test_factor",
            hypothesis="Test hypothesis",
            factor_code="$close",
            ic=0.01,
            ir=0.5,
            sharpe=0.8,
            max_drawdown=0.25,
            passes_threshold=False,
            failure_reasons=[
                FailureReason.LOW_IC,
                FailureReason.LOW_IR,
                FailureReason.HIGH_DRAWDOWN,
            ],
            failure_details={
                FailureReason.LOW_IC: "IC=0.01, need >= 0.03",
                FailureReason.LOW_IR: "IR=0.5, need >= 1.0",
            },
        )

        assert len(feedback.failure_reasons) == 3
        assert FailureReason.LOW_IC in feedback.failure_reasons
        assert FailureReason.LOW_IR in feedback.failure_reasons
        assert FailureReason.HIGH_DRAWDOWN in feedback.failure_reasons
        assert FailureReason.LOW_IC in feedback.failure_details

    def test_validation_empty_factor_name(self):
        """Test validation rejects empty factor_name."""
        with pytest.raises(ValueError, match="factor_name cannot be empty"):
            StructuredFeedback(
                factor_name="",
                hypothesis="Test",
                factor_code="$close",
                ic=0.02,
                ir=0.8,
                sharpe=1.2,
                max_drawdown=0.15,
                passes_threshold=False,
            )

    def test_validation_empty_hypothesis(self):
        """Test validation rejects empty hypothesis."""
        with pytest.raises(ValueError, match="hypothesis cannot be empty"):
            StructuredFeedback(
                factor_name="test",
                hypothesis="",
                factor_code="$close",
                ic=0.02,
                ir=0.8,
                sharpe=1.2,
                max_drawdown=0.15,
                passes_threshold=False,
            )

    def test_validation_invalid_confidence(self):
        """Test validation rejects invalid confidence."""
        with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
            StructuredFeedback(
                factor_name="test",
                hypothesis="Test",
                factor_code="$close",
                ic=0.02,
                ir=0.8,
                sharpe=1.2,
                max_drawdown=0.15,
                passes_threshold=False,
                confidence=1.5,
            )

    def test_to_prompt_context_success(self):
        """Test to_prompt_context for successful factors."""
        feedback = StructuredFeedback(
            factor_name="momentum_factor",
            hypothesis="Price momentum predicts returns",
            factor_code="Ref($close, -1) / Ref($close, -5) - 1",
            ic=0.05,
            ir=1.5,
            sharpe=2.0,
            max_drawdown=0.10,
            passes_threshold=True,
        )

        context = feedback.to_prompt_context()

        assert "SUCCESS" in context
        assert "momentum_factor" in context
        assert "0.0500" in context  # IC
        assert "1.50" in context  # IR
        assert "2.00" in context  # Sharpe

    def test_to_prompt_context_failure(self):
        """Test to_prompt_context for failed factors."""
        feedback = StructuredFeedback(
            factor_name="bad_factor",
            hypothesis="Failed hypothesis",
            factor_code="$close",
            ic=0.01,
            ir=0.5,
            sharpe=0.8,
            max_drawdown=0.25,
            passes_threshold=False,
            failure_reasons=[FailureReason.LOW_IC, FailureReason.LOW_IR],
            failure_details={
                FailureReason.LOW_IC: "IC too low",
                FailureReason.LOW_IR: "IR too low",
            },
            suggestions=["Try different approach", "Use more data"],
        )

        context = feedback.to_prompt_context()

        assert "FAILED" in context
        assert "bad_factor" in context
        assert "low_ic" in context
        assert "low_ir" in context
        assert "IC too low" in context
        assert "Try different approach" in context
        assert "DIFFERENT approach" in context  # Instruction to try different

    def test_to_dict(self):
        """Test to_dict serialization."""
        feedback = StructuredFeedback(
            factor_name="test_factor",
            hypothesis="Test hypothesis",
            factor_code="$close",
            ic=0.02,
            ir=0.8,
            sharpe=1.2,
            max_drawdown=0.15,
            passes_threshold=False,
            failure_reasons=[FailureReason.LOW_IC],
            failure_details={FailureReason.LOW_IC: "Too low"},
            suggestions=["Try harder"],
            confidence=0.7,
            trial_id="trial_123",
        )

        data = feedback.to_dict()

        assert data["factor_name"] == "test_factor"
        assert data["ic"] == 0.02
        assert data["failure_reasons"] == ["low_ic"]
        assert data["failure_details"] == {"low_ic": "Too low"}
        assert data["confidence"] == 0.7
        assert data["trial_id"] == "trial_123"

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "factor_name": "test_factor",
            "hypothesis": "Test hypothesis",
            "factor_code": "$close",
            "ic": 0.02,
            "ir": 0.8,
            "sharpe": 1.2,
            "max_drawdown": 0.15,
            "passes_threshold": False,
            "failure_reasons": ["low_ic", "low_ir"],
            "failure_details": {"low_ic": "Too low"},
            "suggestions": ["Try harder"],
            "confidence": 0.7,
            "trial_id": "trial_123",
        }

        feedback = StructuredFeedback.from_dict(data)

        assert feedback.factor_name == "test_factor"
        assert feedback.ic == 0.02
        assert FailureReason.LOW_IC in feedback.failure_reasons
        assert FailureReason.LOW_IR in feedback.failure_reasons
        assert feedback.confidence == 0.7

    def test_roundtrip_serialization(self):
        """Test that to_dict and from_dict are inverses."""
        original = StructuredFeedback(
            factor_name="test_factor",
            hypothesis="Test hypothesis",
            factor_code="$close",
            ic=0.02,
            ir=0.8,
            sharpe=1.2,
            max_drawdown=0.15,
            passes_threshold=False,
            failure_reasons=[FailureReason.LOW_IC],
            failure_details={FailureReason.LOW_IC: "Too low"},
            suggestions=["Try harder"],
            confidence=0.7,
            trial_id="trial_123",
            stability_score=0.8,
            cv_consistency=0.9,
            win_rate=0.55,
        )

        data = original.to_dict()
        restored = StructuredFeedback.from_dict(data)

        assert restored.factor_name == original.factor_name
        assert restored.ic == original.ic
        assert restored.failure_reasons == original.failure_reasons
        assert restored.confidence == original.confidence
        assert restored.stability_score == original.stability_score


class TestFromEvaluationResult:
    """Tests for from_evaluation_result class method."""

    def _create_mock_evaluation_result(
        self,
        ic: float = 0.02,
        ir: float = 0.8,
        sharpe: float = 1.2,
        max_drawdown: float = 0.15,
        win_rate: float = 0.5,
        passes_threshold: bool = False,
        has_cv_results: bool = False,
        has_stability_report: bool = False,
        has_lookahead_result: bool = False,
    ):
        """Create a mock EvaluationResult."""
        result = MagicMock()
        result.factor_name = "test_factor"
        result.passes_threshold = passes_threshold
        result.trial_id = "trial_123"

        # Metrics
        result.metrics.ic = ic
        result.metrics.ir = ir
        result.metrics.sharpe_ratio = sharpe
        result.metrics.max_drawdown = max_drawdown
        result.metrics.win_rate = win_rate

        # Optional CV results
        if has_cv_results:
            cv1 = MagicMock()
            cv1.metrics.sharpe_ratio = 1.0
            cv2 = MagicMock()
            cv2.metrics.sharpe_ratio = 1.2
            cv3 = MagicMock()
            cv3.metrics.sharpe_ratio = 1.1
            result.cv_results = [cv1, cv2, cv3]
        else:
            result.cv_results = None

        # Optional stability report
        if has_stability_report:
            result.stability_report.overall_score.value = 0.7
        else:
            result.stability_report = None

        # Optional lookahead result
        if has_lookahead_result:
            result.lookahead_result.has_bias = True
            inst = MagicMock()
            inst.location = "Ref($close, 1)"
            inst.severity.value = "CRITICAL"
            result.lookahead_result.instances = [inst]
        else:
            result.lookahead_result = None

        return result

    def test_from_evaluation_result_low_ic(self):
        """Test classification of low IC failure."""
        result = self._create_mock_evaluation_result(ic=0.01)

        feedback = StructuredFeedback.from_evaluation_result(
            result=result,
            hypothesis="Test hypothesis",
            factor_code="$close",
        )

        assert FailureReason.LOW_IC in feedback.failure_reasons
        assert "IC=0.0100" in feedback.failure_details[FailureReason.LOW_IC]

    def test_from_evaluation_result_low_ir(self):
        """Test classification of low IR failure."""
        result = self._create_mock_evaluation_result(ir=0.5)

        feedback = StructuredFeedback.from_evaluation_result(
            result=result,
            hypothesis="Test hypothesis",
            factor_code="$close",
        )

        assert FailureReason.LOW_IR in feedback.failure_reasons

    def test_from_evaluation_result_high_drawdown(self):
        """Test classification of high drawdown failure."""
        result = self._create_mock_evaluation_result(max_drawdown=0.25)

        feedback = StructuredFeedback.from_evaluation_result(
            result=result,
            hypothesis="Test hypothesis",
            factor_code="$close",
        )

        assert FailureReason.HIGH_DRAWDOWN in feedback.failure_reasons

    def test_from_evaluation_result_low_sharpe(self):
        """Test classification of low Sharpe failure."""
        result = self._create_mock_evaluation_result(sharpe=1.0)

        feedback = StructuredFeedback.from_evaluation_result(
            result=result,
            hypothesis="Test hypothesis",
            factor_code="$close",
        )

        assert FailureReason.LOW_SHARPE in feedback.failure_reasons

    def test_from_evaluation_result_low_win_rate(self):
        """Test classification of low win rate failure."""
        result = self._create_mock_evaluation_result(win_rate=0.40)

        feedback = StructuredFeedback.from_evaluation_result(
            result=result,
            hypothesis="Test hypothesis",
            factor_code="$close",
        )

        assert FailureReason.LOW_WIN_RATE in feedback.failure_reasons

    def test_from_evaluation_result_multiple_failures(self):
        """Test classification of multiple failures."""
        result = self._create_mock_evaluation_result(
            ic=0.01, ir=0.5, max_drawdown=0.25, sharpe=1.0, win_rate=0.40
        )

        feedback = StructuredFeedback.from_evaluation_result(
            result=result,
            hypothesis="Test hypothesis",
            factor_code="$close",
        )

        assert len(feedback.failure_reasons) == 5
        assert FailureReason.LOW_IC in feedback.failure_reasons
        assert FailureReason.LOW_IR in feedback.failure_reasons
        assert FailureReason.HIGH_DRAWDOWN in feedback.failure_reasons
        assert FailureReason.LOW_SHARPE in feedback.failure_reasons
        assert FailureReason.LOW_WIN_RATE in feedback.failure_reasons

    def test_from_evaluation_result_success(self):
        """Test that passing factor has no failure reasons."""
        result = self._create_mock_evaluation_result(
            ic=0.05, ir=1.5, sharpe=2.0, max_drawdown=0.10, win_rate=0.55,
            passes_threshold=True
        )

        feedback = StructuredFeedback.from_evaluation_result(
            result=result,
            hypothesis="Test hypothesis",
            factor_code="$close",
        )

        assert feedback.passes_threshold is True
        assert len(feedback.failure_reasons) == 0

    def test_from_evaluation_result_with_lookahead(self):
        """Test lookahead bias detection."""
        result = self._create_mock_evaluation_result(has_lookahead_result=True)

        feedback = StructuredFeedback.from_evaluation_result(
            result=result,
            hypothesis="Test hypothesis",
            factor_code="$close",
        )

        assert FailureReason.LOOKAHEAD_BIAS in feedback.failure_reasons

    def test_suggestions_generated(self):
        """Test that suggestions are generated for failures."""
        result = self._create_mock_evaluation_result(ic=0.01, ir=0.5)

        feedback = StructuredFeedback.from_evaluation_result(
            result=result,
            hypothesis="Test hypothesis",
            factor_code="$close",
        )

        assert len(feedback.suggestions) > 0
        # Should have suggestions for both LOW_IC and LOW_IR
        assert any("signal" in s.lower() for s in feedback.suggestions)


class TestGenerateSuggestions:
    """Tests for _generate_suggestions helper."""

    def test_low_ic_suggestions(self):
        """Test suggestions for low IC."""
        suggestions = _generate_suggestions([FailureReason.LOW_IC])

        assert len(suggestions) >= 2
        assert any("data fields" in s for s in suggestions)

    def test_low_ir_suggestions(self):
        """Test suggestions for low IR."""
        suggestions = _generate_suggestions([FailureReason.LOW_IR])

        assert len(suggestions) >= 2
        assert any("smoothing" in s or "averaging" in s for s in suggestions)

    def test_high_drawdown_suggestions(self):
        """Test suggestions for high drawdown."""
        suggestions = _generate_suggestions([FailureReason.HIGH_DRAWDOWN])

        assert len(suggestions) >= 2
        assert any("risk" in s.lower() for s in suggestions)

    def test_overfitting_suggestions(self):
        """Test suggestions for overfitting."""
        suggestions = _generate_suggestions([FailureReason.OVERFITTING])

        assert len(suggestions) >= 2
        assert any("simplify" in s.lower() for s in suggestions)

    def test_lookahead_suggestions(self):
        """Test suggestions for lookahead bias."""
        suggestions = _generate_suggestions([FailureReason.LOOKAHEAD_BIAS])

        assert len(suggestions) >= 2
        assert any("lag" in s.lower() or "ref" in s.lower() for s in suggestions)

    def test_no_failures_gets_generic_suggestions(self):
        """Test that no failures still gets generic suggestions."""
        suggestions = _generate_suggestions([])

        assert len(suggestions) >= 2
        assert any("different" in s.lower() for s in suggestions)


class TestCalculateStd:
    """Tests for _calculate_std helper."""

    def test_empty_list(self):
        """Test std of empty list."""
        assert _calculate_std([]) == 0.0

    def test_single_value(self):
        """Test std of single value."""
        assert _calculate_std([5.0]) == 0.0

    def test_two_values(self):
        """Test std of two values."""
        std = _calculate_std([1.0, 3.0])
        # Sample std of [1, 3] is sqrt((1+1)/1) = sqrt(2) ≈ 1.414
        assert abs(std - 1.414) < 0.01

    def test_multiple_values(self):
        """Test std of multiple values."""
        std = _calculate_std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        # Known sample std is approximately 2.138
        assert abs(std - 2.138) < 0.01

    def test_identical_values(self):
        """Test std of identical values."""
        assert _calculate_std([5.0, 5.0, 5.0]) == 0.0
