"""Tests for LookaheadBiasDetector - Detection of future data leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iqfmp.evaluation.lookahead_detector import (
    AuditReport,
    BiasInstance,
    BiasType,
    DetectionMode,
    DetectionResult,
    ICDecayAnalysis,
    ICDecayAnalyzer,
    LookaheadBiasDetector,
    LookaheadBiasError,
    PythonCodeAnalyzer,
    QlibExpressionAnalyzer,
    SeverityLevel,
    TemporalAlignmentAuditor,
    audit_alignment,
    check_code,
    check_expression,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_factor_df() -> pd.DataFrame:
    """Create sample factor DataFrame."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    return pd.DataFrame(
        {
            "factor": np.random.randn(100),
            "_date": dates,
        },
        index=dates,
    )


@pytest.fixture
def sample_target_df() -> pd.DataFrame:
    """Create sample target DataFrame with forward returns."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    return pd.DataFrame(
        {
            "return": np.random.randn(100) * 0.02,
            "_date": dates,
        },
        index=dates,
    )


@pytest.fixture
def sample_returns() -> pd.Series:
    """Create sample returns Series."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    return pd.Series(np.random.randn(100) * 0.02, index=dates, name="returns")


# =============================================================================
# BiasInstance and DetectionResult Tests
# =============================================================================


class TestBiasInstance:
    """Tests for BiasInstance dataclass."""

    def test_bias_instance_creation(self) -> None:
        """Test creating a bias instance."""
        instance = BiasInstance(
            bias_type=BiasType.FUTURE_REFERENCE,
            severity=SeverityLevel.CRITICAL,
            location="line 10",
            description="Negative shift accesses future data",
            evidence=".shift(-1)",
            suggestion="Use positive shift value",
        )
        assert instance.bias_type == BiasType.FUTURE_REFERENCE
        assert instance.severity == SeverityLevel.CRITICAL
        assert "shift" in instance.evidence


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_empty_result(self) -> None:
        """Test empty detection result."""
        result = DetectionResult(has_bias=False)
        assert not result.has_bias
        assert result.critical_count == 0
        assert result.warning_count == 0

    def test_result_with_instances(self) -> None:
        """Test detection result with bias instances."""
        instances = [
            BiasInstance(
                bias_type=BiasType.FUTURE_REFERENCE,
                severity=SeverityLevel.CRITICAL,
                location="line 1",
                description="Test",
                evidence="code",
                suggestion="fix",
            ),
            BiasInstance(
                bias_type=BiasType.NEGATIVE_SHIFT,
                severity=SeverityLevel.WARNING,
                location="line 2",
                description="Test2",
                evidence="code2",
                suggestion="fix2",
            ),
        ]
        result = DetectionResult(has_bias=True, instances=instances)
        assert result.has_bias
        assert result.critical_count == 1
        assert result.warning_count == 1

    def test_result_to_dict(self) -> None:
        """Test converting detection result to dict."""
        instances = [
            BiasInstance(
                bias_type=BiasType.FUTURE_REFERENCE,
                severity=SeverityLevel.CRITICAL,
                location="line 1",
                description="Test",
                evidence="code",
                suggestion="fix",
            ),
        ]
        result = DetectionResult(has_bias=True, instances=instances)
        result_dict = result.to_dict()

        assert "has_bias" in result_dict
        assert "instances" in result_dict
        assert len(result_dict["instances"]) == 1


# =============================================================================
# QlibExpressionAnalyzer Tests
# =============================================================================


class TestQlibExpressionAnalyzer:
    """Tests for Qlib expression analysis."""

    def test_clean_expression(self) -> None:
        """Test that clean expressions pass."""
        analyzer = QlibExpressionAnalyzer()
        result = analyzer.analyze("Ref($close, 1)")  # Yesterday's close - OK
        assert not result.has_bias

    def test_detect_negative_ref(self) -> None:
        """Test detection of negative Ref (future data)."""
        analyzer = QlibExpressionAnalyzer()
        result = analyzer.analyze("Ref($close, -1)")  # Tomorrow's close - BAD
        assert result.has_bias
        assert result.critical_count == 1
        assert any(
            i.bias_type == BiasType.FUTURE_REFERENCE for i in result.instances
        )

    def test_detect_negative_shift(self) -> None:
        """Test detection of negative Shift."""
        analyzer = QlibExpressionAnalyzer()
        result = analyzer.analyze("Shift($volume, -5)")
        assert result.has_bias

    def test_detect_negative_delay(self) -> None:
        """Test detection of negative Delay."""
        analyzer = QlibExpressionAnalyzer()
        result = analyzer.analyze("Delay($open, -3)")
        assert result.has_bias

    def test_detect_negative_rolling_period(self) -> None:
        """Test detection of negative rolling period."""
        analyzer = QlibExpressionAnalyzer()
        result = analyzer.analyze("Mean($close, -5)")
        assert result.has_bias
        assert any(
            i.bias_type == BiasType.FORWARD_LOOKING for i in result.instances
        )

    def test_complex_expression_with_bias(self) -> None:
        """Test complex expression containing bias."""
        analyzer = QlibExpressionAnalyzer()
        result = analyzer.analyze(
            "Ref($close, 1) + Ref($close, -1) / Mean($volume, 20)"
        )
        assert result.has_bias
        assert result.critical_count >= 1

    def test_future_return_reference(self) -> None:
        """Test detection of future return reference."""
        analyzer = QlibExpressionAnalyzer()
        result = analyzer.analyze("Ref($return, -1)")
        assert result.has_bias


# =============================================================================
# PythonCodeAnalyzer Tests
# =============================================================================


class TestPythonCodeAnalyzer:
    """Tests for Python code analysis."""

    def test_clean_code(self) -> None:
        """Test that clean code passes."""
        analyzer = PythonCodeAnalyzer()
        result = analyzer.analyze("df['close'].shift(1)")  # Past data - OK
        assert not result.has_bias

    def test_detect_negative_shift(self) -> None:
        """Test detection of negative shift."""
        analyzer = PythonCodeAnalyzer()
        result = analyzer.analyze("df['close'].shift(-1)")
        assert result.has_bias
        assert any(
            i.bias_type == BiasType.NEGATIVE_SHIFT for i in result.instances
        )

    def test_detect_negative_pct_change(self) -> None:
        """Test detection of negative pct_change."""
        analyzer = PythonCodeAnalyzer()
        result = analyzer.analyze("df['close'].pct_change(-1)")
        assert result.has_bias

    def test_detect_negative_diff(self) -> None:
        """Test detection of negative diff."""
        analyzer = PythonCodeAnalyzer()
        result = analyzer.analyze("df['price'].diff(-2)")
        assert result.has_bias

    def test_multiline_code(self) -> None:
        """Test analysis of multiline code."""
        code = """
def calculate_factor(df):
    momentum = df['close'].shift(1) / df['close'].shift(5)
    future = df['close'].shift(-1)  # Lookahead!
    return momentum * future
"""
        analyzer = PythonCodeAnalyzer()
        result = analyzer.analyze(code)
        assert result.has_bias
        assert result.critical_count >= 1

    def test_syntax_error_handling(self) -> None:
        """Test that syntax errors are handled gracefully."""
        analyzer = PythonCodeAnalyzer()
        result = analyzer.analyze("def broken(:")
        # Should not raise, but may have info-level instance
        assert not result.has_bias or result.critical_count == 0

    def test_suspicious_method_names(self) -> None:
        """Test detection of suspicious method names."""
        analyzer = PythonCodeAnalyzer()
        # The analyzer checks for exact method names like "future", "forward", "next"
        result = analyzer.analyze("df.future()")
        # Should flag as warning
        assert any(i.severity == SeverityLevel.WARNING for i in result.instances)


# =============================================================================
# TemporalAlignmentAuditor Tests
# =============================================================================


class TestTemporalAlignmentAuditor:
    """Tests for temporal alignment auditing."""

    def test_aligned_data(
        self, sample_factor_df: pd.DataFrame, sample_target_df: pd.DataFrame
    ) -> None:
        """Test properly aligned data."""
        auditor = TemporalAlignmentAuditor()
        report = auditor.audit(sample_factor_df, sample_target_df)

        assert isinstance(report, AuditReport)
        assert report.overlap_days > 0

    def test_misaligned_dates(self) -> None:
        """Test detection of misaligned dates."""
        factor_dates = pd.date_range("2023-02-01", periods=50, freq="D")
        target_dates = pd.date_range("2023-01-01", periods=50, freq="D")

        factor_df = pd.DataFrame({"factor": np.random.randn(50)}, index=factor_dates)
        target_df = pd.DataFrame({"return": np.random.randn(50)}, index=target_dates)

        auditor = TemporalAlignmentAuditor()
        report = auditor.audit(factor_df, target_df)

        # Target starts before factor - suspicious
        assert len(report.detection_result.instances) > 0

    def test_missing_timestamps(self) -> None:
        """Test handling of missing timestamps."""
        # Use string indices that cannot be converted to datetime
        factor_df = pd.DataFrame(
            {"factor": [1, 2, 3]}, index=["asset_a", "asset_b", "asset_c"]
        )
        target_df = pd.DataFrame(
            {"return": [0.1, 0.2, 0.3]}, index=["asset_a", "asset_b", "asset_c"]
        )

        auditor = TemporalAlignmentAuditor()
        report = auditor.audit(factor_df, target_df)

        # Should indicate alignment issues due to inability to extract timestamps
        assert not report.is_aligned
        assert any("timestamp" in r.lower() for r in report.recommendations)

    def test_gap_detection(self) -> None:
        """Test detection of gaps between factor and target."""
        factor_dates = pd.date_range("2023-01-01", periods=30, freq="D")
        target_dates = pd.date_range("2023-03-01", periods=30, freq="D")  # Gap

        factor_df = pd.DataFrame({"factor": np.random.randn(30)}, index=factor_dates)
        target_df = pd.DataFrame({"return": np.random.randn(30)}, index=target_dates)

        auditor = TemporalAlignmentAuditor()
        report = auditor.audit(factor_df, target_df)

        assert report.gap_days > 0


# =============================================================================
# ICDecayAnalyzer Tests
# =============================================================================


class TestICDecayAnalyzer:
    """Tests for IC decay analysis."""

    def test_normal_decay(self) -> None:
        """Test normal IC decay pattern (no suspicion)."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=200, freq="D")

        # Create factor with moderate correlation to returns
        returns = pd.Series(np.random.randn(200) * 0.02, index=dates)
        factor = returns.shift(1) * 0.3 + np.random.randn(200) * 0.7
        factor = pd.Series(factor.values, index=dates)

        analyzer = ICDecayAnalyzer(max_lag=5, suspicion_threshold=3.0)
        analysis = analyzer.analyze(factor, returns)

        assert isinstance(analysis, ICDecayAnalysis)
        assert len(analysis.ic_values) > 0

    def test_suspicious_decay_pattern(self) -> None:
        """Test suspicious IC decay (possible lookahead)."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=200, freq="D")

        # Create factor that "knows" future returns (lookahead bias)
        returns = pd.Series(np.random.randn(200) * 0.02, index=dates)
        # Factor is essentially future returns (extreme lookahead)
        factor = returns * 0.9 + np.random.randn(200) * 0.1
        factor = pd.Series(factor.values, index=dates)

        analyzer = ICDecayAnalyzer(max_lag=5, suspicion_threshold=2.0)
        analysis = analyzer.analyze(factor, returns)

        # Should be suspicious because IC_0 will be very high
        assert analysis.is_suspicious or abs(analysis.ic_values[0]) > 0.3

    def test_insufficient_data(self) -> None:
        """Test handling of insufficient data."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        factor = pd.Series(np.random.randn(10), index=dates)
        returns = pd.Series(np.random.randn(10) * 0.02, index=dates)

        analyzer = ICDecayAnalyzer(max_lag=5)
        analysis = analyzer.analyze(factor, returns)

        # Should handle gracefully
        assert isinstance(analysis, ICDecayAnalysis)

    def test_half_life_calculation(self) -> None:
        """Test IC half-life calculation."""
        analyzer = ICDecayAnalyzer()

        # Test with known IC values
        ic_values = [0.2, 0.15, 0.1, 0.05, 0.02]
        half_life = analyzer._calculate_half_life(ic_values)

        # Half of 0.2 is 0.1, which is at index 2
        assert half_life == 2.0


# =============================================================================
# LookaheadBiasDetector Tests
# =============================================================================


class TestLookaheadBiasDetector:
    """Tests for main detector class."""

    def test_detector_creation(self) -> None:
        """Test detector instance creation."""
        detector = LookaheadBiasDetector()
        assert detector.mode == DetectionMode.LENIENT

        strict_detector = LookaheadBiasDetector(mode=DetectionMode.STRICT)
        assert strict_detector.mode == DetectionMode.STRICT

    def test_check_qlib_expression(self) -> None:
        """Test Qlib expression checking."""
        detector = LookaheadBiasDetector()

        # Clean expression
        result = detector.check_qlib_expression("Ref($close, 1)")
        assert not result.has_bias

        # Biased expression
        result = detector.check_qlib_expression("Ref($close, -1)")
        assert result.has_bias

    def test_check_python_code(self) -> None:
        """Test Python code checking."""
        detector = LookaheadBiasDetector()

        # Clean code
        result = detector.check_python_code("df.shift(1)")
        assert not result.has_bias

        # Biased code
        result = detector.check_python_code("df.shift(-1)")
        assert result.has_bias

    def test_audit_temporal_alignment(
        self, sample_factor_df: pd.DataFrame, sample_target_df: pd.DataFrame
    ) -> None:
        """Test temporal alignment auditing."""
        detector = LookaheadBiasDetector()
        report = detector.audit_temporal_alignment(sample_factor_df, sample_target_df)

        assert isinstance(report, AuditReport)

    def test_ic_decay_analysis(self, sample_returns: pd.Series) -> None:
        """Test IC decay analysis."""
        np.random.seed(42)
        factor = pd.Series(np.random.randn(100), index=sample_returns.index)

        detector = LookaheadBiasDetector()
        analysis = detector.ic_decay_analysis(factor, sample_returns)

        assert isinstance(analysis, ICDecayAnalysis)

    def test_strict_mode_raises(self) -> None:
        """Test that strict mode raises exception on bias."""
        detector = LookaheadBiasDetector(mode=DetectionMode.STRICT)

        with pytest.raises(LookaheadBiasError):
            detector.check_qlib_expression("Ref($close, -1)")

    def test_lenient_mode_warns(self) -> None:
        """Test that lenient mode logs warning but doesn't raise."""
        detector = LookaheadBiasDetector(mode=DetectionMode.LENIENT)

        # Should not raise
        result = detector.check_qlib_expression("Ref($close, -1)")
        assert result.has_bias

    def test_audit_mode(self) -> None:
        """Test audit mode (report only)."""
        detector = LookaheadBiasDetector(mode=DetectionMode.AUDIT)

        # Should not raise or warn
        result = detector.check_qlib_expression("Ref($close, -1)")
        assert result.has_bias


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_check_expression(self) -> None:
        """Test check_expression convenience function."""
        result = check_expression("Ref($close, 1)")
        assert not result.has_bias

        result = check_expression("Ref($close, -1)")
        assert result.has_bias

    def test_check_code(self) -> None:
        """Test check_code convenience function."""
        result = check_code("df.shift(1)")
        assert not result.has_bias

        result = check_code("df.shift(-1)")
        assert result.has_bias

    def test_audit_alignment(
        self, sample_factor_df: pd.DataFrame, sample_target_df: pd.DataFrame
    ) -> None:
        """Test audit_alignment convenience function."""
        report = audit_alignment(sample_factor_df, sample_target_df)
        assert isinstance(report, AuditReport)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_expression(self) -> None:
        """Test handling of empty expression."""
        analyzer = QlibExpressionAnalyzer()
        result = analyzer.analyze("")
        assert not result.has_bias

    def test_empty_code(self) -> None:
        """Test handling of empty code."""
        analyzer = PythonCodeAnalyzer()
        result = analyzer.analyze("")
        assert not result.has_bias

    def test_complex_nested_expression(self) -> None:
        """Test complex nested Qlib expression."""
        analyzer = QlibExpressionAnalyzer()
        expr = "Mean(Ref($close, 1) / Ref($close, 5), 20)"
        result = analyzer.analyze(expr)
        assert not result.has_bias  # All positive refs

    def test_case_insensitivity(self) -> None:
        """Test case-insensitive detection."""
        analyzer = QlibExpressionAnalyzer()

        # All caps
        result = analyzer.analyze("REF($CLOSE, -1)")
        assert result.has_bias

        # Mixed case
        result = analyzer.analyze("Ref($Close, -1)")
        assert result.has_bias

    def test_whitespace_handling(self) -> None:
        """Test handling of various whitespace."""
        analyzer = QlibExpressionAnalyzer()

        # Extra spaces
        result = analyzer.analyze("Ref(  $close  ,  -1  )")
        assert result.has_bias

    def test_detector_with_dataframe_returns(
        self, sample_returns: pd.Series
    ) -> None:
        """Test IC decay analysis with DataFrame returns."""
        np.random.seed(42)
        dates = sample_returns.index
        factor = pd.Series(np.random.randn(100), index=dates)
        returns_df = pd.DataFrame({"returns": sample_returns.values}, index=dates)

        detector = LookaheadBiasDetector()
        analysis = detector.ic_decay_analysis(factor, returns_df)

        assert isinstance(analysis, ICDecayAnalysis)


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enum values."""

    def test_detection_modes(self) -> None:
        """Test all detection modes are defined."""
        assert DetectionMode.STRICT.value == "strict"
        assert DetectionMode.LENIENT.value == "lenient"
        assert DetectionMode.AUDIT.value == "audit"

    def test_bias_types(self) -> None:
        """Test all bias types are defined."""
        assert BiasType.FUTURE_REFERENCE.value == "future_reference"
        assert BiasType.NEGATIVE_SHIFT.value == "negative_shift"
        assert BiasType.FORWARD_LOOKING.value == "forward_looking"
        assert BiasType.TEMPORAL_MISALIGNMENT.value == "temporal_misalignment"
        assert BiasType.IC_ANOMALY.value == "ic_anomaly"

    def test_severity_levels(self) -> None:
        """Test all severity levels are defined."""
        assert SeverityLevel.CRITICAL.value == "critical"
        assert SeverityLevel.WARNING.value == "warning"
        assert SeverityLevel.INFO.value == "info"
