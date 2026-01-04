"""Lookahead Bias Detection for IQFMP.

This module provides tools to detect lookahead bias (future data leakage)
in factor expressions, Python code, and data alignment.

Lookahead bias occurs when:
1. Future data is used in factor computation (e.g., Ref($close, -1))
2. Target/forward returns are used in feature engineering
3. Data alignment issues cause future information leakage

Detection methods:
- Static analysis of Qlib expressions
- AST analysis of Python code
- Temporal alignment auditing
- IC decay analysis

References:
- De Prado AFML Chapter 7: Cross-Validation in Finance
- Qlib expression documentation
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Configuration
# =============================================================================


class DetectionMode(Enum):
    """Lookahead detection behavior mode."""

    STRICT = "strict"  # Raise exception on detection
    LENIENT = "lenient"  # Warn but continue
    AUDIT = "audit"  # Only report, no warnings


class BiasType(Enum):
    """Types of lookahead bias detected."""

    FUTURE_REFERENCE = "future_reference"  # Direct future data access
    NEGATIVE_SHIFT = "negative_shift"  # Negative shift in pandas operations
    FORWARD_LOOKING = "forward_looking"  # Forward-looking operations
    TEMPORAL_MISALIGNMENT = "temporal_misalignment"  # Data alignment issues
    IC_ANOMALY = "ic_anomaly"  # Suspicious IC pattern


class SeverityLevel(Enum):
    """Severity of detected bias."""

    CRITICAL = "critical"  # Definite lookahead bias
    WARNING = "warning"  # Potential lookahead bias
    INFO = "info"  # Informational, may not be bias


# =============================================================================
# Detection Results
# =============================================================================


@dataclass
class BiasInstance:
    """Single instance of detected bias."""

    bias_type: BiasType
    severity: SeverityLevel
    location: str  # Expression, line number, or column name
    description: str
    evidence: str  # The problematic code/expression
    suggestion: str  # How to fix


@dataclass
class DetectionResult:
    """Result of lookahead bias detection."""

    has_bias: bool
    instances: list[BiasInstance] = field(default_factory=list)
    mode: DetectionMode = DetectionMode.LENIENT

    def __post_init__(self) -> None:
        """Validate consistency between has_bias and instances."""
        # has_bias should reflect presence of CRITICAL or WARNING instances
        # INFO-only instances are allowed with has_bias=False (informational only)
        significant_instances = [
            i for i in self.instances
            if i.severity in (SeverityLevel.CRITICAL, SeverityLevel.WARNING)
        ]

        if significant_instances and not self.has_bias:
            raise ValueError(
                "has_bias must be True when CRITICAL/WARNING instances are present. "
                f"Got has_bias=False with {len(significant_instances)} significant instances."
            )
        if self.has_bias and not significant_instances:
            raise ValueError(
                "has_bias=True requires at least one CRITICAL or WARNING instance. "
                "Set has_bias=False for INFO-only or empty instances list."
            )

    @property
    def critical_count(self) -> int:
        """Count of critical severity instances."""
        return sum(1 for i in self.instances if i.severity == SeverityLevel.CRITICAL)

    @property
    def warning_count(self) -> int:
        """Count of warning severity instances."""
        return sum(1 for i in self.instances if i.severity == SeverityLevel.WARNING)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "has_bias": self.has_bias,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "instances": [
                {
                    "type": i.bias_type.value,
                    "severity": i.severity.value,
                    "location": i.location,
                    "description": i.description,
                    "evidence": i.evidence,
                    "suggestion": i.suggestion,
                }
                for i in self.instances
            ],
        }


@dataclass
class AuditReport:
    """Comprehensive temporal alignment audit report."""

    is_aligned: bool
    detection_result: DetectionResult
    factor_date_range: tuple[str, str]
    target_date_range: tuple[str, str]
    overlap_days: int
    gap_days: int
    recommendations: list[str] = field(default_factory=list)


@dataclass
class ICDecayAnalysis:
    """IC decay analysis for lookahead detection."""

    ic_values: list[float]  # IC at lag 0, 1, 2, ...
    ic_half_life: float  # Periods until IC halves
    is_suspicious: bool  # True if IC_0 >> IC_1 (possible lookahead)
    suspicion_reason: str | None = None


# =============================================================================
# Qlib Expression Analyzer
# =============================================================================


class QlibExpressionAnalyzer:
    """Analyze Qlib expressions for lookahead bias.

    Detects patterns like:
    - Ref($close, -1)  # Future close price
    - Ref($volume, -N) where N > 0
    - Any negative period references
    """

    # Qlib operators that can access future data with negative periods
    TEMPORAL_OPERATORS = [
        "Ref",
        "Shift",
        "Delay",
        "Mean",
        "Std",
        "Sum",
        "Max",
        "Min",
        "Rolling",
        "Window",
    ]

    # Regex pattern to find function calls with negative numbers
    NEGATIVE_REF_PATTERN = re.compile(
        r"\b(Ref|Shift|Delay)\s*\(\s*[^,]+,\s*-\d+\s*\)",
        re.IGNORECASE,
    )

    # Pattern for any negative number in temporal context
    NEGATIVE_PERIOD_PATTERN = re.compile(
        r"\b(Mean|Std|Sum|Max|Min|Rolling|Window)\s*\([^)]*,\s*-\d+",
        re.IGNORECASE,
    )

    def analyze(self, expression: str) -> DetectionResult:
        """Analyze a Qlib expression for lookahead bias.

        Args:
            expression: Qlib factor expression string

        Returns:
            DetectionResult with any detected bias instances
        """
        instances: list[BiasInstance] = []

        # Check for Ref/Shift/Delay with negative periods
        for match in self.NEGATIVE_REF_PATTERN.finditer(expression):
            instances.append(
                BiasInstance(
                    bias_type=BiasType.FUTURE_REFERENCE,
                    severity=SeverityLevel.CRITICAL,
                    location=f"position {match.start()}-{match.end()}",
                    description="Negative reference period accesses future data",
                    evidence=match.group(),
                    suggestion="Use positive period (e.g., Ref($close, 1) for yesterday)",
                )
            )

        # Check for rolling operations with negative periods
        for match in self.NEGATIVE_PERIOD_PATTERN.finditer(expression):
            instances.append(
                BiasInstance(
                    bias_type=BiasType.FORWARD_LOOKING,
                    severity=SeverityLevel.CRITICAL,
                    location=f"position {match.start()}",
                    description="Rolling operation with negative period",
                    evidence=match.group() + "...",
                    suggestion="Use positive periods for historical lookback",
                )
            )

        # Check for common mistakes
        self._check_common_patterns(expression, instances)

        has_bias = any(
            i.severity in (SeverityLevel.CRITICAL, SeverityLevel.WARNING)
            for i in instances
        )

        return DetectionResult(has_bias=has_bias, instances=instances)

    def _check_common_patterns(
        self, expression: str, instances: list[BiasInstance]
    ) -> None:
        """Check for common lookahead patterns."""
        # Pattern: using future returns directly
        if re.search(r"\bRef\s*\(\s*\$return", expression, re.IGNORECASE):
            # This might be OK if period is positive
            if re.search(r"\bRef\s*\(\s*\$return[^,]*,\s*-", expression, re.IGNORECASE):
                instances.append(
                    BiasInstance(
                        bias_type=BiasType.FUTURE_REFERENCE,
                        severity=SeverityLevel.CRITICAL,
                        location="return reference",
                        description="Direct reference to future returns",
                        evidence="Ref($return, -N)",
                        suggestion="Target returns should only be used for labels, not features",
                    )
                )


# =============================================================================
# Python Code Analyzer
# =============================================================================


class PythonCodeAnalyzer(ast.NodeVisitor):
    """Analyze Python code for lookahead bias using AST.

    Detects patterns like:
    - df.shift(-1)  # Shifting forward
    - df['close'].pct_change(-1)
    - Any operations with negative periods
    """

    SHIFT_METHODS = ["shift", "pct_change", "diff", "rolling"]
    LOOKAHEAD_FUNCTIONS = ["future", "forward", "next"]

    def __init__(self) -> None:
        self.instances: list[BiasInstance] = []
        self.current_line: int = 0

    def analyze(self, code: str) -> DetectionResult:
        """Analyze Python code for lookahead bias.

        Args:
            code: Python source code string

        Returns:
            DetectionResult with any detected bias instances
        """
        self.instances = []

        try:
            tree = ast.parse(code)
            self.visit(tree)
        except SyntaxError as e:
            logger.warning(f"Failed to parse code for lookahead check: {e}")
            self.instances.append(
                BiasInstance(
                    bias_type=BiasType.FORWARD_LOOKING,
                    severity=SeverityLevel.INFO,
                    location="parse_error",
                    description=f"Could not parse code: {e}",
                    evidence=code[:100],
                    suggestion="Fix syntax error for proper analysis",
                )
            )

        has_bias = any(
            i.severity in (SeverityLevel.CRITICAL, SeverityLevel.WARNING)
            for i in self.instances
        )

        return DetectionResult(has_bias=has_bias, instances=self.instances)

    def visit_Call(self, node: ast.Call) -> Any:
        """Visit function/method calls."""
        self.current_line = node.lineno

        # Check for method calls like df.shift(-1)
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr

            if method_name in self.SHIFT_METHODS:
                self._check_negative_args(node, method_name)

            if method_name.lower() in self.LOOKAHEAD_FUNCTIONS:
                self.instances.append(
                    BiasInstance(
                        bias_type=BiasType.FORWARD_LOOKING,
                        severity=SeverityLevel.WARNING,
                        location=f"line {node.lineno}",
                        description=f"Suspicious method name: {method_name}",
                        evidence=f".{method_name}(...)",
                        suggestion="Review if this accesses future data",
                    )
                )

        self.generic_visit(node)

    def _check_negative_args(self, node: ast.Call, method_name: str) -> None:
        """Check if call has negative numeric arguments."""
        for arg in node.args:
            if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                if isinstance(arg.operand, ast.Constant) and isinstance(
                    arg.operand.value, (int, float)
                ):
                    value = -arg.operand.value
                    if value < 0:
                        self.instances.append(
                            BiasInstance(
                                bias_type=BiasType.NEGATIVE_SHIFT,
                                severity=SeverityLevel.CRITICAL,
                                location=f"line {node.lineno}",
                                description=f"{method_name}({value}) shifts into future",
                                evidence=f".{method_name}({value})",
                                suggestion=f"Use positive value: .{method_name}({abs(value)})",
                            )
                        )

            elif isinstance(arg, ast.Constant) and isinstance(
                arg.value, (int, float)
            ):
                if arg.value < 0:
                    self.instances.append(
                        BiasInstance(
                            bias_type=BiasType.NEGATIVE_SHIFT,
                            severity=SeverityLevel.CRITICAL,
                            location=f"line {node.lineno}",
                            description=f"{method_name}({arg.value}) shifts into future",
                            evidence=f".{method_name}({arg.value})",
                            suggestion=f"Use positive value: .{method_name}({abs(arg.value)})",
                        )
                    )


# =============================================================================
# Temporal Alignment Auditor
# =============================================================================


class TemporalAlignmentAuditor:
    """Audit temporal alignment between factor and target data.

    Ensures that:
    - Factor values at time T only use data from T-N to T
    - Target values at time T represent returns from T to T+1
    - No data leakage from future
    """

    def audit(
        self,
        factor_df: pd.DataFrame,
        target_df: pd.DataFrame,
        factor_timestamp_col: str | None = None,
        target_timestamp_col: str | None = None,
    ) -> AuditReport:
        """Audit temporal alignment between factor and target data.

        Args:
            factor_df: DataFrame with factor values
            target_df: DataFrame with target values (e.g., forward returns)
            factor_timestamp_col: Column name for factor timestamps (or use index)
            target_timestamp_col: Column name for target timestamps (or use index)

        Returns:
            AuditReport with alignment analysis
        """
        instances: list[BiasInstance] = []
        recommendations: list[str] = []

        # Get timestamps
        factor_times = self._get_timestamps(factor_df, factor_timestamp_col)
        target_times = self._get_timestamps(target_df, target_timestamp_col)

        if factor_times is None or target_times is None:
            return AuditReport(
                is_aligned=False,
                detection_result=DetectionResult(
                    has_bias=True,
                    instances=[
                        BiasInstance(
                            bias_type=BiasType.TEMPORAL_MISALIGNMENT,
                            severity=SeverityLevel.CRITICAL,
                            location="timestamps",
                            description="Could not extract timestamps from data",
                            evidence="Missing or invalid datetime index",
                            suggestion="Ensure data has datetime index or specify timestamp column",
                        )
                    ],
                ),
                factor_date_range=("unknown", "unknown"),
                target_date_range=("unknown", "unknown"),
                overlap_days=0,
                gap_days=0,
                recommendations=["Fix timestamp extraction"],
            )

        factor_min, factor_max = factor_times.min(), factor_times.max()
        target_min, target_max = target_times.min(), target_times.max()

        # Check for suspicious alignment
        # If target starts before factor, might have lookahead
        if target_min < factor_min:
            instances.append(
                BiasInstance(
                    bias_type=BiasType.TEMPORAL_MISALIGNMENT,
                    severity=SeverityLevel.WARNING,
                    location="date_range",
                    description="Target data starts before factor data",
                    evidence=f"Target starts {target_min}, Factor starts {factor_min}",
                    suggestion="Ensure target represents FUTURE returns from factor date",
                )
            )
            recommendations.append("Verify target represents forward returns")

        # Check for exact alignment (suspicious if perfect match)
        common_times = set(factor_times) & set(target_times)
        if len(common_times) == len(factor_times) == len(target_times):
            # Perfect alignment - might be OK but worth checking
            instances.append(
                BiasInstance(
                    bias_type=BiasType.TEMPORAL_MISALIGNMENT,
                    severity=SeverityLevel.INFO,
                    location="alignment",
                    description="Perfect alignment between factor and target",
                    evidence=f"{len(common_times)} matching timestamps",
                    suggestion="Verify target is properly shifted for prediction horizon",
                )
            )

        # Calculate overlap and gap
        overlap_start = max(factor_min, target_min)
        overlap_end = min(factor_max, target_max)

        if overlap_start <= overlap_end:
            overlap_days = (overlap_end - overlap_start).days
            gap_days = 0
        else:
            overlap_days = 0
            gap_days = (overlap_start - overlap_end).days

        has_bias = any(
            i.severity in (SeverityLevel.CRITICAL, SeverityLevel.WARNING)
            for i in instances
        )

        return AuditReport(
            is_aligned=not has_bias,
            detection_result=DetectionResult(has_bias=has_bias, instances=instances),
            factor_date_range=(str(factor_min), str(factor_max)),
            target_date_range=(str(target_min), str(target_max)),
            overlap_days=overlap_days,
            gap_days=gap_days,
            recommendations=recommendations,
        )

    def _get_timestamps(
        self, df: pd.DataFrame, col: str | None = None
    ) -> pd.DatetimeIndex | None:
        """Extract timestamps from DataFrame."""
        try:
            if col is not None and col in df.columns:
                return pd.DatetimeIndex(df[col])
            elif isinstance(df.index, pd.DatetimeIndex):
                return df.index
            else:
                # Try to convert index
                return pd.DatetimeIndex(df.index)
        except Exception as e:
            logger.warning(f"Failed to extract timestamps from DataFrame: {e}")
            return None


# =============================================================================
# IC Decay Analyzer
# =============================================================================


class ICDecayAnalyzer:
    """Analyze IC decay to detect potential lookahead bias.

    If IC at lag 0 is much higher than IC at lag 1+, it may indicate
    lookahead bias (the factor "knows" the immediate future).

    Healthy factor: IC decays gradually over lags
    Suspicious: IC_0 >> IC_1 (sudden drop after lag 0)
    """

    def __init__(
        self,
        max_lag: int = 10,
        suspicion_threshold: float = 2.0,
    ) -> None:
        """Initialize IC decay analyzer.

        Args:
            max_lag: Maximum lag to analyze
            suspicion_threshold: Ratio of IC_0/IC_1 that triggers suspicion
        """
        self.max_lag = max_lag
        self.suspicion_threshold = suspicion_threshold

    def analyze(
        self,
        factor: pd.Series,
        returns: pd.DataFrame | pd.Series,
    ) -> ICDecayAnalysis:
        """Analyze IC decay pattern.

        Args:
            factor: Factor values Series with datetime index
            returns: Price returns DataFrame or Series

        Returns:
            ICDecayAnalysis with decay pattern analysis
        """
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]

        ic_values: list[float] = []

        for lag in range(self.max_lag + 1):
            # Shift returns backward to get future returns at each lag
            # lag=0: same-day returns, lag=1: next-day returns, etc.
            lagged_returns = returns.shift(-lag)

            # Align and compute IC
            aligned = pd.concat([factor, lagged_returns], axis=1).dropna()
            if len(aligned) < 30:
                break

            # Rank IC (Spearman correlation)
            ic = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")
            ic_values.append(float(ic) if not np.isnan(ic) else 0.0)

        if len(ic_values) < 2:
            return ICDecayAnalysis(
                ic_values=ic_values,
                ic_half_life=float("inf"),
                is_suspicious=False,
                suspicion_reason="Insufficient data for IC decay analysis",
            )

        # Calculate half-life (when IC drops to half of IC_0)
        ic_half_life = self._calculate_half_life(ic_values)

        # Check for suspicious pattern
        is_suspicious = False
        suspicion_reason = None

        ic_0 = abs(ic_values[0])
        ic_1 = abs(ic_values[1]) if len(ic_values) > 1 else 0

        if ic_1 > 0 and ic_0 / ic_1 > self.suspicion_threshold:
            is_suspicious = True
            suspicion_reason = (
                f"IC drops sharply from lag 0 ({ic_0:.3f}) to lag 1 ({ic_1:.3f}). "
                f"Ratio: {ic_0/ic_1:.2f}x > {self.suspicion_threshold}x threshold. "
                "This may indicate lookahead bias."
            )

        # Also suspicious if IC_0 is very high (>0.3 is unusual)
        if ic_0 > 0.3:
            is_suspicious = True
            suspicion_reason = (
                suspicion_reason or ""
            ) + f" IC_0={ic_0:.3f} is unusually high (>0.3)."

        return ICDecayAnalysis(
            ic_values=ic_values,
            ic_half_life=ic_half_life,
            is_suspicious=is_suspicious,
            suspicion_reason=suspicion_reason,
        )

    def _calculate_half_life(self, ic_values: list[float]) -> float:
        """Calculate IC half-life (lags until IC halves)."""
        if not ic_values or ic_values[0] == 0:
            return float("inf")

        target = abs(ic_values[0]) / 2

        for lag, ic in enumerate(ic_values):
            if abs(ic) <= target:
                return float(lag)

        return float("inf")


# =============================================================================
# Main Detector Class
# =============================================================================


class LookaheadBiasDetector:
    """Main class for comprehensive lookahead bias detection.

    Combines all detection methods:
    - Qlib expression analysis
    - Python code AST analysis
    - Temporal alignment auditing
    - IC decay analysis

    Usage:
        detector = LookaheadBiasDetector(mode=DetectionMode.STRICT)

        # Check expression
        result = detector.check_qlib_expression("Ref($close, -1)")

        # Check code
        result = detector.check_python_code("df.shift(-1)")

        # Audit alignment
        report = detector.audit_temporal_alignment(factor_df, target_df)

        # IC decay analysis
        analysis = detector.ic_decay_analysis(factor_series, returns)
    """

    def __init__(self, mode: DetectionMode = DetectionMode.LENIENT) -> None:
        """Initialize detector.

        Args:
            mode: Detection behavior mode
        """
        self.mode = mode
        self.expression_analyzer = QlibExpressionAnalyzer()
        self.code_analyzer = PythonCodeAnalyzer()
        self.alignment_auditor = TemporalAlignmentAuditor()
        self.ic_analyzer = ICDecayAnalyzer()

    def check_qlib_expression(self, expression: str) -> DetectionResult:
        """Check a Qlib expression for lookahead bias.

        Args:
            expression: Qlib factor expression

        Returns:
            DetectionResult

        Raises:
            LookaheadBiasError: If mode is STRICT and bias is detected
        """
        result = self.expression_analyzer.analyze(expression)
        result.mode = self.mode

        self._handle_result(result, f"expression: {expression[:50]}...")

        return result

    def check_python_code(self, code: str) -> DetectionResult:
        """Check Python code for lookahead bias.

        Args:
            code: Python source code

        Returns:
            DetectionResult

        Raises:
            LookaheadBiasError: If mode is STRICT and bias is detected
        """
        result = self.code_analyzer.analyze(code)
        result.mode = self.mode

        self._handle_result(result, "Python code")

        return result

    def audit_temporal_alignment(
        self,
        factor_df: pd.DataFrame,
        target_df: pd.DataFrame,
        **kwargs: Any,
    ) -> AuditReport:
        """Audit temporal alignment between factor and target.

        Args:
            factor_df: Factor values DataFrame
            target_df: Target values DataFrame
            **kwargs: Additional arguments for auditor

        Returns:
            AuditReport

        Raises:
            LookaheadBiasError: If mode is STRICT and misalignment detected
        """
        report = self.alignment_auditor.audit(factor_df, target_df, **kwargs)
        report.detection_result.mode = self.mode

        self._handle_result(report.detection_result, "temporal alignment")

        return report

    def ic_decay_analysis(
        self,
        factor: pd.Series,
        returns: pd.DataFrame | pd.Series,
    ) -> ICDecayAnalysis:
        """Analyze IC decay for lookahead indicators.

        Args:
            factor: Factor values
            returns: Price returns

        Returns:
            ICDecayAnalysis
        """
        analysis = self.ic_analyzer.analyze(factor, returns)

        if analysis.is_suspicious and self.mode == DetectionMode.STRICT:
            raise LookaheadBiasError(
                f"IC decay analysis suspicious: {analysis.suspicion_reason}"
            )
        elif analysis.is_suspicious and self.mode == DetectionMode.LENIENT:
            logger.warning(
                f"IC decay analysis warning: {analysis.suspicion_reason}"
            )

        return analysis

    def _handle_result(self, result: DetectionResult, context: str) -> None:
        """Handle detection result based on mode."""
        if result.has_bias:
            message = f"Lookahead bias detected in {context}: {result.critical_count} critical, {result.warning_count} warnings"

            if self.mode == DetectionMode.STRICT:
                raise LookaheadBiasError(message)
            elif self.mode == DetectionMode.LENIENT:
                logger.warning(message)
                for instance in result.instances:
                    if instance.severity == SeverityLevel.CRITICAL:
                        logger.warning(
                            f"  {instance.bias_type.value}: {instance.description}"
                        )


class LookaheadBiasError(Exception):
    """Exception raised when lookahead bias is detected in STRICT mode."""

    pass


# =============================================================================
# Convenience Functions
# =============================================================================


def check_expression(
    expression: str,
    mode: DetectionMode = DetectionMode.LENIENT,
) -> DetectionResult:
    """Convenience function to check a Qlib expression.

    Args:
        expression: Qlib factor expression
        mode: Detection mode

    Returns:
        DetectionResult
    """
    detector = LookaheadBiasDetector(mode=mode)
    return detector.check_qlib_expression(expression)


def check_code(
    code: str,
    mode: DetectionMode = DetectionMode.LENIENT,
) -> DetectionResult:
    """Convenience function to check Python code.

    Args:
        code: Python source code
        mode: Detection mode

    Returns:
        DetectionResult
    """
    detector = LookaheadBiasDetector(mode=mode)
    return detector.check_python_code(code)


def audit_alignment(
    factor_df: pd.DataFrame,
    target_df: pd.DataFrame,
    mode: DetectionMode = DetectionMode.LENIENT,
) -> AuditReport:
    """Convenience function to audit temporal alignment.

    Args:
        factor_df: Factor values
        target_df: Target values
        mode: Detection mode

    Returns:
        AuditReport
    """
    detector = LookaheadBiasDetector(mode=mode)
    return detector.audit_temporal_alignment(factor_df, target_df)
