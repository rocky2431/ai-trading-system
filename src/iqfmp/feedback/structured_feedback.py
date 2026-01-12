"""Structured feedback for closed-loop factor mining.

This module defines the StructuredFeedback data structure that captures
evaluation results in a format consumable by LLMs for iterative improvement.

The feedback includes:
- Quantitative metrics (IC, IR, Sharpe, MaxDD)
- Failure classification (why the factor failed)
- Actionable suggestions (how to improve)
- Confidence scoring (for weighted decisions)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional

if TYPE_CHECKING:
    from iqfmp.evaluation.factor_evaluator import EvaluationResult, FactorMetrics


class MetricThreshold(NamedTuple):
    """Configuration for a metric threshold check."""

    reason: "FailureReason"
    check: Callable[["FactorMetrics"], bool]
    message: Callable[["FactorMetrics"], str]


class FailureReason(str, Enum):
    """Classification of factor failure reasons.

    Used to categorize why a factor failed evaluation, enabling
    targeted improvements in subsequent iterations.
    """

    LOW_IC = "low_ic"  # IC < 0.03
    LOW_IR = "low_ir"  # IR < 1.0
    HIGH_DRAWDOWN = "high_drawdown"  # MaxDD > 20%
    OVERFITTING = "overfitting"  # CV results inconsistent
    LOOKAHEAD_BIAS = "lookahead_bias"  # Time alignment issues
    INSTABILITY = "instability"  # Poor stability across periods
    DUPLICATE = "duplicate"  # Too similar to existing factor
    LOW_SHARPE = "low_sharpe"  # Sharpe ratio below threshold
    LOW_WIN_RATE = "low_win_rate"  # Win rate below 45%


# Data-driven metric threshold configuration
# Each entry: (FailureReason, check_function, message_function)
_METRIC_THRESHOLDS: list[MetricThreshold] = [
    MetricThreshold(
        FailureReason.LOW_IC,
        lambda m: m.ic < 0.03,
        lambda m: f"IC={m.ic:.4f}, need >= 0.03. Factor has weak predictive power.",
    ),
    MetricThreshold(
        FailureReason.LOW_IR,
        lambda m: m.ir < 1.0,
        lambda m: f"IR={m.ir:.2f}, need >= 1.0. Factor signal is inconsistent over time.",
    ),
    MetricThreshold(
        FailureReason.HIGH_DRAWDOWN,
        lambda m: m.max_drawdown > 0.2,
        lambda m: f"MaxDD={m.max_drawdown:.1%}, need <= 20%. Factor has excessive downside risk.",
    ),
    MetricThreshold(
        FailureReason.LOW_SHARPE,
        lambda m: m.sharpe_ratio < 1.5,
        lambda m: f"Sharpe={m.sharpe_ratio:.2f}, need >= 1.5. Risk-adjusted return is insufficient.",
    ),
    MetricThreshold(
        FailureReason.LOW_WIN_RATE,
        lambda m: m.win_rate < 0.45,
        lambda m: f"WinRate={m.win_rate:.1%}, need >= 45%. Factor loses too often.",
    ),
]


@dataclass
class StructuredFeedback:
    """Structured feedback from factor evaluation.

    Captures evaluation results in a format that can be injected into
    LLM prompts for iterative factor improvement.

    Attributes:
        factor_name: Name of the evaluated factor
        hypothesis: Original research hypothesis
        factor_code: Generated factor code (Qlib expression or Python)
        ic: Information Coefficient
        ir: Information Ratio
        sharpe: Sharpe Ratio
        max_drawdown: Maximum Drawdown (as decimal, e.g., 0.15 = 15%)
        passes_threshold: Whether factor passed dynamic threshold
        failure_reasons: List of classified failure reasons
        failure_details: Detailed explanation for each failure reason
        suggestions: Actionable improvement suggestions
        confidence: Confidence score (0-1) for weighting decisions
        trial_id: ID of the evaluation trial
        stability_score: Stability analysis score (if available)
        cv_consistency: Cross-validation consistency (if available)
    """

    factor_name: str
    hypothesis: str
    factor_code: str

    # Metrics
    ic: float
    ir: float
    sharpe: float
    max_drawdown: float
    passes_threshold: bool

    # Failure analysis
    failure_reasons: list[FailureReason] = field(default_factory=list)
    failure_details: dict[FailureReason, str] = field(default_factory=dict)

    # Improvement suggestions
    suggestions: list[str] = field(default_factory=list)

    # Metadata
    confidence: float = 0.5
    trial_id: Optional[str] = None
    stability_score: Optional[float] = None
    cv_consistency: Optional[float] = None
    win_rate: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate feedback data."""
        if not self.factor_name:
            raise ValueError("factor_name cannot be empty")
        if not self.hypothesis:
            raise ValueError("hypothesis cannot be empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")

    @classmethod
    def from_evaluation_result(
        cls,
        result: "EvaluationResult",
        hypothesis: str,
        factor_code: str,
    ) -> "StructuredFeedback":
        """Create structured feedback from an EvaluationResult.

        Args:
            result: The evaluation result from FactorEvaluator
            hypothesis: The research hypothesis that led to this factor
            factor_code: The generated factor code

        Returns:
            StructuredFeedback instance with classified failures and suggestions
        """
        failure_reasons: list[FailureReason] = []
        failure_details: dict[FailureReason, str] = {}

        # Classify failures based on metrics using data-driven configuration
        for threshold in _METRIC_THRESHOLDS:
            if threshold.check(result.metrics):
                failure_reasons.append(threshold.reason)
                failure_details[threshold.reason] = threshold.message(result.metrics)

        # Check CV consistency if available
        if result.cv_results:
            cv_sharpes = [
                cv.metrics.sharpe_ratio
                for cv in result.cv_results
                if cv.metrics.sharpe_ratio is not None
            ]
            if cv_sharpes:
                cv_std = _calculate_std(cv_sharpes)
                cv_mean = sum(cv_sharpes) / len(cv_sharpes)
                is_high_variance = cv_std > 0.5 * abs(cv_mean) if cv_mean != 0 else cv_std > 0.5
                if is_high_variance:
                    failure_reasons.append(FailureReason.OVERFITTING)
                    failure_details[FailureReason.OVERFITTING] = (
                        f"CV Sharpe std={cv_std:.2f}, high variance across folds. "
                        "Factor may be overfit to specific periods."
                    )

        # Check lookahead bias if available
        if result.lookahead_result and result.lookahead_result.has_bias:
            failure_reasons.append(FailureReason.LOOKAHEAD_BIAS)
            # Format bias instances for details
            bias_details = "; ".join(
                f"{inst.location}({inst.severity.value})"
                for inst in result.lookahead_result.instances[:3]
            ) if result.lookahead_result.instances else "Unknown"
            failure_details[FailureReason.LOOKAHEAD_BIAS] = (
                "Lookahead bias detected. Factor uses future information. "
                f"Instances: {bias_details}"
            )

        # Check stability if available
        if result.stability_report:
            stability_score = result.stability_report.overall_score
            # StabilityScore has a .value attribute
            score_value = stability_score.value if stability_score is not None else None
            if score_value is not None and score_value < 0.6:
                failure_reasons.append(FailureReason.INSTABILITY)
                failure_details[FailureReason.INSTABILITY] = (
                    f"Stability score={score_value:.2f}, need >= 0.6. "
                    "Factor performance varies significantly across periods."
                )

        # Generate suggestions based on failures
        suggestions = _generate_suggestions(failure_reasons)

        # Calculate confidence based on data quality
        confidence = _calculate_confidence(result, failure_reasons)

        # Extract stability score value if available
        stability_score_value: Optional[float] = None
        if result.stability_report and result.stability_report.overall_score:
            stability_score_value = result.stability_report.overall_score.value

        return cls(
            factor_name=result.factor_name,
            hypothesis=hypothesis,
            factor_code=factor_code,
            ic=result.metrics.ic,
            ir=result.metrics.ir,
            sharpe=result.metrics.sharpe_ratio,
            max_drawdown=result.metrics.max_drawdown,
            passes_threshold=result.passes_threshold,
            failure_reasons=failure_reasons,
            failure_details=failure_details,
            suggestions=suggestions,
            confidence=confidence,
            trial_id=result.trial_id,
            stability_score=stability_score_value,
            cv_consistency=_calculate_cv_consistency(result),
            win_rate=result.metrics.win_rate,
        )

    def to_prompt_context(self) -> str:
        """Convert feedback to LLM-consumable prompt context.

        Returns:
            Formatted string suitable for injection into LLM prompts
        """
        if self.passes_threshold:
            return self._format_success_context()
        return self._format_failure_context()

    def _format_success_context(self) -> str:
        """Format context for successful factors."""
        return f"""
## Previous Attempt: SUCCESS
Factor `{self.factor_name}` passed evaluation:
- IC: {self.ic:.4f}
- IR: {self.ir:.2f}
- Sharpe: {self.sharpe:.2f}
- MaxDD: {self.max_drawdown:.1%}

The factor meets all thresholds. Consider:
1. Exploring variations on this approach
2. Testing on different market conditions
3. Combining with complementary factors
"""

    def _format_failure_context(self) -> str:
        """Format context for failed factors."""
        # Format failure reasons
        reasons_text = "\n".join(
            f"- **{reason.value}**: {self.failure_details.get(reason, 'No details')}"
            for reason in self.failure_reasons
        )

        # Format suggestions
        suggestions_text = "\n".join(f"- {s}" for s in self.suggestions)

        return f"""
## Previous Attempt: FAILED
Factor `{self.factor_name}` did not pass evaluation.

### Metrics
- IC: {self.ic:.4f} (threshold: 0.03)
- IR: {self.ir:.2f} (threshold: 1.0)
- Sharpe: {self.sharpe:.2f}
- MaxDD: {self.max_drawdown:.1%}

### Failure Analysis
{reasons_text}

### Improvement Suggestions
{suggestions_text}

### Original Hypothesis
{self.hypothesis}

### Failed Factor Code
```
{self.factor_code}
```

IMPORTANT: Generate a DIFFERENT approach that addresses the identified issues.
Do NOT simply adjust parameters - try a fundamentally different strategy.
"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "factor_name": self.factor_name,
            "hypothesis": self.hypothesis,
            "factor_code": self.factor_code,
            "ic": self.ic,
            "ir": self.ir,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "passes_threshold": self.passes_threshold,
            "failure_reasons": [r.value for r in self.failure_reasons],
            "failure_details": {k.value: v for k, v in self.failure_details.items()},
            "suggestions": self.suggestions,
            "confidence": self.confidence,
            "trial_id": self.trial_id,
            "stability_score": self.stability_score,
            "cv_consistency": self.cv_consistency,
            "win_rate": self.win_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StructuredFeedback":
        """Create from dictionary."""
        return cls(
            factor_name=data["factor_name"],
            hypothesis=data["hypothesis"],
            factor_code=data["factor_code"],
            ic=data["ic"],
            ir=data["ir"],
            sharpe=data["sharpe"],
            max_drawdown=data["max_drawdown"],
            passes_threshold=data["passes_threshold"],
            failure_reasons=[FailureReason(r) for r in data.get("failure_reasons", [])],
            failure_details={
                FailureReason(k): v for k, v in data.get("failure_details", {}).items()
            },
            suggestions=data.get("suggestions", []),
            confidence=data.get("confidence", 0.5),
            trial_id=data.get("trial_id"),
            stability_score=data.get("stability_score"),
            cv_consistency=data.get("cv_consistency"),
            win_rate=data.get("win_rate"),
        )


def _calculate_std(values: list[float]) -> float:
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def _generate_suggestions(
    failure_reasons: list[FailureReason],
) -> list[str]:
    """Generate actionable suggestions based on failures."""
    suggestions: list[str] = []

    if FailureReason.LOW_IC in failure_reasons:
        suggestions.append(
            "Try using different data fields or combining multiple signals"
        )
        suggestions.append(
            "Consider longer lookback periods to capture more persistent patterns"
        )

    if FailureReason.LOW_IR in failure_reasons:
        suggestions.append(
            "Add smoothing or averaging to reduce signal noise"
        )
        suggestions.append(
            "Use rank-based transformations to improve signal consistency"
        )

    if FailureReason.HIGH_DRAWDOWN in failure_reasons:
        suggestions.append(
            "Add risk management conditions (e.g., volatility scaling)"
        )
        suggestions.append(
            "Consider using conditional signals that avoid high-risk periods"
        )

    if FailureReason.OVERFITTING in failure_reasons:
        suggestions.append(
            "Simplify the factor expression - fewer parameters means less overfitting"
        )
        suggestions.append(
            "Use more robust statistics (median instead of mean)"
        )

    if FailureReason.LOOKAHEAD_BIAS in failure_reasons:
        suggestions.append(
            "Ensure all data references use proper lag (e.g., Ref($field, -1))"
        )
        suggestions.append(
            "Check that no future information is used in calculations"
        )

    if FailureReason.INSTABILITY in failure_reasons:
        suggestions.append(
            "Test on multiple time periods before finalizing"
        )
        suggestions.append(
            "Consider market regime awareness in the factor design"
        )

    if FailureReason.LOW_SHARPE in failure_reasons:
        suggestions.append(
            "Focus on factors with higher expected returns or lower volatility"
        )
        suggestions.append(
            "Consider combining with volatility-targeting mechanisms"
        )

    if FailureReason.LOW_WIN_RATE in failure_reasons:
        suggestions.append(
            "Consider mean-reversion components for higher win rate"
        )
        suggestions.append(
            "Add confirmation signals to filter out false positives"
        )

    # Add generic suggestions if no specific ones
    if not suggestions:
        suggestions.append(
            "Try a completely different approach to the same market phenomenon"
        )
        suggestions.append(
            "Consider what successful similar factors have in common"
        )

    return suggestions


def _calculate_confidence(
    result: "EvaluationResult",
    failure_reasons: list[FailureReason],
) -> float:
    """Calculate confidence score for the feedback.

    Higher confidence means the feedback is more reliable for guiding
    subsequent iterations.
    """
    confidence = 0.5  # Base confidence

    # More data points increase confidence
    if hasattr(result, "cv_results") and result.cv_results:
        confidence += 0.1 * min(len(result.cv_results) / 5, 1.0)

    # Stability analysis increases confidence
    if result.stability_report:
        confidence += 0.1

    # Lookahead check increases confidence
    if result.lookahead_result:
        confidence += 0.1

    # More failure reasons = more certain about what's wrong
    confidence += 0.05 * min(len(failure_reasons), 3)

    return min(confidence, 1.0)


def _calculate_cv_consistency(result: "EvaluationResult") -> Optional[float]:
    """Calculate CV consistency score."""
    if not result.cv_results:
        return None

    cv_sharpes = [
        cv.metrics.sharpe_ratio
        for cv in result.cv_results
        if cv.metrics.sharpe_ratio is not None
    ]
    if len(cv_sharpes) < 2:
        return None

    mean = sum(cv_sharpes) / len(cv_sharpes)
    std = _calculate_std(cv_sharpes)

    if abs(mean) < 0.01:
        return 0.0

    # Coefficient of variation (lower is more consistent)
    cv = std / abs(mean)
    # Convert to consistency score (higher is better)
    return max(0.0, 1.0 - cv)
