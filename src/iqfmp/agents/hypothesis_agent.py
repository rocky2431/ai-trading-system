"""Hypothesis-driven Research Agent.

This module implements the core RD-Agent hypothesis-experiment-feedback loop.
Based on the RD-Agent paper architecture but adapted for cryptocurrency factor mining.

Key components:
- HypothesisGenerator: Generate trading hypotheses from market analysis
- HypothesisToCode: Convert hypotheses to executable factor code
- FeedbackAnalyzer: Analyze experiment results and provide feedback
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class HypothesisStatus(str, Enum):
    """Status of a hypothesis in the RD loop."""
    PENDING = "pending"  # Not yet tested
    TESTING = "testing"  # Currently being evaluated
    VALIDATED = "validated"  # Passed threshold
    REJECTED = "rejected"  # Failed threshold
    REFINED = "refined"  # Needs refinement based on feedback


class HypothesisFamily(str, Enum):
    """Categories of trading hypotheses."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    TREND = "trend"
    MICROSTRUCTURE = "microstructure"
    FUNDING = "funding"  # Crypto-specific
    SENTIMENT = "sentiment"
    CROSS_ASSET = "cross_asset"


@dataclass
class Hypothesis:
    """A trading hypothesis to be tested.

    Example:
        hypothesis = Hypothesis(
            name="RSI Oversold Reversal",
            description="Prices tend to revert when RSI drops below 30",
            family=HypothesisFamily.MEAN_REVERSION,
            rationale="RSI measures momentum exhaustion...",
            expected_ic=0.05,
        )
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    family: HypothesisFamily = HypothesisFamily.MOMENTUM
    rationale: str = ""  # Why this hypothesis should work

    # Expected metrics
    expected_ic: float = 0.03
    expected_direction: str = "long"  # long, short, long_short

    # Source information
    source: str = "generated"  # generated, user, literature
    literature_ref: Optional[str] = None

    # Status
    status: HypothesisStatus = HypothesisStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)

    # Factor code (filled after HypothesisToCode)
    factor_code: Optional[str] = None
    factor_name: Optional[str] = None

    # Experiment results (filled after evaluation)
    experiment_result: Optional[dict[str, Any]] = None
    actual_ic: Optional[float] = None
    passed_threshold: bool = False

    # Feedback (filled after FeedbackAnalyzer)
    feedback: Optional[str] = None
    refinement_suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "family": self.family.value,
            "rationale": self.rationale,
            "expected_ic": self.expected_ic,
            "expected_direction": self.expected_direction,
            "source": self.source,
            "status": self.status.value,
            "factor_code": self.factor_code,
            "actual_ic": self.actual_ic,
            "passed_threshold": self.passed_threshold,
            "feedback": self.feedback,
            "refinement_suggestions": self.refinement_suggestions,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# Hypothesis Templates - Pre-defined hypotheses for each family
# =============================================================================

HYPOTHESIS_TEMPLATES: dict[HypothesisFamily, list[dict[str, Any]]] = {
    HypothesisFamily.MOMENTUM: [
        {
            "name": "Price Momentum",
            "description": "Assets with positive past returns continue to outperform",
            "rationale": "Momentum reflects gradual information diffusion and investor underreaction",
            "expected_ic": 0.04,
            "expected_direction": "long_short",
        },
        {
            "name": "Volume Confirmation",
            "description": "Price movements confirmed by volume are more likely to persist",
            "rationale": "High volume indicates strong conviction behind price moves",
            "expected_ic": 0.03,
            "expected_direction": "long_short",
        },
    ],
    HypothesisFamily.MEAN_REVERSION: [
        {
            "name": "RSI Oversold Reversal",
            "description": "Prices tend to revert when RSI drops below 30",
            "rationale": "Extreme RSI values indicate exhaustion of selling pressure",
            "expected_ic": 0.05,
            "expected_direction": "long",
        },
        {
            "name": "Bollinger Band Mean Reversion",
            "description": "Prices touching lower Bollinger Band tend to revert to mean",
            "rationale": "Statistical mean reversion in normal distribution assumption",
            "expected_ic": 0.04,
            "expected_direction": "long_short",
        },
    ],
    HypothesisFamily.VOLATILITY: [
        {
            "name": "Low Volatility Premium",
            "description": "Low volatility assets provide better risk-adjusted returns",
            "rationale": "Investors overpay for lottery-like high volatility assets",
            "expected_ic": 0.03,
            "expected_direction": "long",
        },
        {
            "name": "Volatility Breakout",
            "description": "Low volatility periods precede directional moves",
            "rationale": "Consolidation builds energy for breakout moves",
            "expected_ic": 0.04,
            "expected_direction": "long_short",
        },
    ],
    HypothesisFamily.FUNDING: [
        {
            "name": "Funding Rate Mean Reversion",
            "description": "Extreme funding rates predict price reversals",
            "rationale": "High funding = crowded longs, negative funding = crowded shorts",
            "expected_ic": 0.06,
            "expected_direction": "long_short",
        },
        {
            "name": "Funding-Price Divergence",
            "description": "Divergence between funding and price signals momentum exhaustion",
            "rationale": "Price-sentiment divergence often precedes reversals",
            "expected_ic": 0.05,
            "expected_direction": "long_short",
        },
    ],
}


class HypothesisGenerator:
    """Generate trading hypotheses from market analysis."""

    def __init__(
        self,
        templates: Optional[dict[HypothesisFamily, list[dict]]] = None,
    ) -> None:
        """Initialize generator.

        Args:
            templates: Hypothesis templates by family
        """
        self.templates = templates or HYPOTHESIS_TEMPLATES
        self._generated_count = 0

    def generate_from_template(
        self,
        family: HypothesisFamily,
        variation: int = 0,
    ) -> Hypothesis:
        """Generate hypothesis from template.

        Args:
            family: Hypothesis family
            variation: Template variation index

        Returns:
            Generated Hypothesis
        """
        templates = self.templates.get(family, [])
        if not templates:
            raise ValueError(f"No templates for family: {family}")

        template = templates[variation % len(templates)]
        self._generated_count += 1

        return Hypothesis(
            name=template["name"],
            description=template["description"],
            family=family,
            rationale=template["rationale"],
            expected_ic=template["expected_ic"],
            expected_direction=template["expected_direction"],
            source="template",
        )

    def generate_from_analysis(
        self,
        market_data: pd.DataFrame,
        focus_family: Optional[HypothesisFamily] = None,
    ) -> list[Hypothesis]:
        """Generate hypotheses based on market data analysis.

        Args:
            market_data: OHLCV DataFrame
            focus_family: Optional family to focus on

        Returns:
            List of generated hypotheses
        """
        hypotheses = []

        # Analyze market conditions
        conditions = self._analyze_market_conditions(market_data)

        # Generate hypotheses based on conditions
        if conditions["trending"]:
            # In trending markets, test momentum hypotheses
            hypotheses.append(self.generate_from_template(HypothesisFamily.MOMENTUM, 0))

        if conditions["high_volatility"]:
            # High volatility: test volatility hypotheses
            hypotheses.append(self.generate_from_template(HypothesisFamily.VOLATILITY, 0))
        else:
            # Low volatility: test breakout hypothesis
            hypotheses.append(self.generate_from_template(HypothesisFamily.VOLATILITY, 1))

        if conditions["extreme_rsi"]:
            # Extreme RSI: test mean reversion
            hypotheses.append(self.generate_from_template(HypothesisFamily.MEAN_REVERSION, 0))

        # Always add funding hypothesis for crypto
        if "funding_rate" in market_data.columns or focus_family == HypothesisFamily.FUNDING:
            hypotheses.append(self.generate_from_template(HypothesisFamily.FUNDING, 0))

        return hypotheses

    def _analyze_market_conditions(
        self,
        df: pd.DataFrame,
    ) -> dict[str, bool]:
        """Analyze market conditions to guide hypothesis generation."""
        conditions = {
            "trending": False,
            "high_volatility": False,
            "extreme_rsi": False,
        }

        if len(df) < 20:
            return conditions

        # Check for trend
        returns_20d = df["close"].pct_change(20).iloc[-1]
        conditions["trending"] = abs(returns_20d) > 0.1  # 10% move in 20 days

        # Check volatility regime
        volatility = df["close"].pct_change().rolling(20).std().iloc[-1] * (252 ** 0.5)
        conditions["high_volatility"] = volatility > 0.5  # 50% annualized vol

        # Check RSI extremes
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        conditions["extreme_rsi"] = current_rsi < 30 or current_rsi > 70

        return conditions


class HypothesisToCode:
    """Convert trading hypotheses to executable factor code."""

    # Code templates for each family
    CODE_TEMPLATES: dict[str, str] = {
        "Price Momentum": '''
def {func_name}(df):
    """Price momentum factor - {period}d returns."""
    returns = df["close"].pct_change({period})
    factor = (returns - returns.rolling(60).mean()) / (returns.rolling(60).std() + 1e-10)
    return factor.fillna(0)
''',
        "RSI Oversold Reversal": '''
def {func_name}(df):
    """RSI mean reversion factor."""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    # Long when RSI oversold, short when overbought
    factor = 50 - rsi  # Negative = overbought, Positive = oversold
    factor = (factor - factor.rolling(60).mean()) / (factor.rolling(60).std() + 1e-10)
    return factor.fillna(0)
''',
        "Low Volatility Premium": '''
def {func_name}(df):
    """Low volatility premium factor."""
    import numpy as np
    returns = df["close"].pct_change()
    vol = returns.rolling(20).std() * np.sqrt(252)
    # Negative because low vol is good
    factor = -vol
    factor = (factor - factor.rolling(60).mean()) / (factor.rolling(60).std() + 1e-10)
    return factor.fillna(0)
''',
        "Volatility Breakout": '''
def {func_name}(df):
    """Volatility breakout factor - compression predicts expansion."""
    import numpy as np
    returns = df["close"].pct_change()
    vol_short = returns.rolling(5).std()
    vol_long = returns.rolling(20).std()
    # Low short-term vol relative to long-term = compression
    compression = -vol_short / (vol_long + 1e-10)
    # Direction based on price momentum
    momentum = np.sign(returns.rolling(5).mean())
    factor = compression * momentum
    factor = (factor - factor.rolling(60).mean()) / (factor.rolling(60).std() + 1e-10)
    return factor.fillna(0)
''',
        "Funding Rate Mean Reversion": '''
def {func_name}(df):
    """Funding rate mean reversion factor (crypto-specific)."""
    if "funding_rate" not in df.columns:
        return df["close"] * 0  # Return zeros if no funding data
    funding = df["funding_rate"]
    # Short when funding extreme positive, long when extreme negative
    factor = -funding
    factor = (factor - factor.rolling(60).mean()) / (factor.rolling(60).std() + 1e-10)
    return factor.fillna(0)
''',
        "Bollinger Band Mean Reversion": '''
def {func_name}(df):
    """Bollinger band mean reversion factor."""
    ma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    # Position within band (-1 to 1)
    bb_position = (df["close"] - ma20) / (2 * std20 + 1e-10)
    # Mean reversion: short when high, long when low
    factor = -bb_position
    return factor.fillna(0)
''',
        "Volume Confirmation": '''
def {func_name}(df):
    """Volume-confirmed momentum factor."""
    import numpy as np
    if "volume" not in df.columns:
        return df["close"] * 0
    returns = df["close"].pct_change()
    vol_ratio = df["volume"] / df["volume"].rolling(20).mean()
    # Momentum confirmed by high volume
    factor = returns.rolling(5).mean() * vol_ratio
    factor = (factor - factor.rolling(60).mean()) / (factor.rolling(60).std() + 1e-10)
    return factor.fillna(0)
''',
    }

    def __init__(self) -> None:
        """Initialize converter."""
        self._conversion_count = 0

    def convert(
        self,
        hypothesis: Hypothesis,
        params: Optional[dict[str, Any]] = None,
    ) -> str:
        """Convert hypothesis to factor code.

        Args:
            hypothesis: Hypothesis to convert
            params: Optional parameters for the factor

        Returns:
            Executable Python code string
        """
        params = params or {}

        # Look up code template
        template = self.CODE_TEMPLATES.get(hypothesis.name)

        if template is None:
            # Use generic template based on family
            template = self._get_family_template(hypothesis.family)

        # Generate function name
        func_name = self._generate_func_name(hypothesis)

        # Fill in template
        code = template.format(
            func_name=func_name,
            period=params.get("period", 20),
            **params,
        )

        self._conversion_count += 1
        hypothesis.factor_code = code
        hypothesis.factor_name = func_name

        return code

    def _generate_func_name(self, hypothesis: Hypothesis) -> str:
        """Generate function name from hypothesis."""
        # Clean name for function
        name = hypothesis.name.lower().replace(" ", "_").replace("-", "_")
        name = "".join(c for c in name if c.isalnum() or c == "_")
        return f"factor_{name}"

    def _get_family_template(self, family: HypothesisFamily) -> str:
        """Get generic template for a family."""
        templates = {
            HypothesisFamily.MOMENTUM: '''
def {func_name}(df):
    """Momentum factor."""
    returns = df["close"].pct_change({period})
    factor = (returns - returns.rolling(60).mean()) / (returns.rolling(60).std() + 1e-10)
    return factor.fillna(0)
''',
            HypothesisFamily.MEAN_REVERSION: '''
def {func_name}(df):
    """Mean reversion factor."""
    ma = df["close"].rolling({period}).mean()
    deviation = (df["close"] - ma) / ma
    factor = -deviation  # Short when above MA, long when below
    return factor.fillna(0)
''',
            HypothesisFamily.VOLATILITY: '''
def {func_name}(df):
    """Volatility factor."""
    import numpy as np
    returns = df["close"].pct_change()
    vol = returns.rolling({period}).std() * np.sqrt(252)
    factor = -vol  # Low vol premium
    factor = (factor - factor.rolling(60).mean()) / (factor.rolling(60).std() + 1e-10)
    return factor.fillna(0)
''',
        }
        return templates.get(family, templates[HypothesisFamily.MOMENTUM])


class FeedbackAnalyzer:
    """Analyze experiment results and provide feedback for hypothesis refinement."""

    def __init__(
        self,
        ic_threshold: float = 0.03,
        ir_threshold: float = 1.0,
    ) -> None:
        """Initialize analyzer.

        Args:
            ic_threshold: Minimum IC for passing
            ir_threshold: Minimum IR for passing
        """
        self.ic_threshold = ic_threshold
        self.ir_threshold = ir_threshold

    def analyze(
        self,
        hypothesis: Hypothesis,
        experiment_result: dict[str, Any],
    ) -> Hypothesis:
        """Analyze experiment results and update hypothesis with feedback.

        Args:
            hypothesis: Tested hypothesis
            experiment_result: Results from evaluation

        Returns:
            Updated hypothesis with feedback
        """
        metrics = experiment_result.get("metrics", {})
        hypothesis.experiment_result = experiment_result
        hypothesis.actual_ic = metrics.get("ic_mean", metrics.get("ic", 0))

        # Check if passed threshold
        ic = abs(hypothesis.actual_ic or 0)
        ir = metrics.get("ir", 0)
        hypothesis.passed_threshold = ic >= self.ic_threshold and ir >= self.ir_threshold

        # Update status
        if hypothesis.passed_threshold:
            hypothesis.status = HypothesisStatus.VALIDATED
            hypothesis.feedback = f"Hypothesis validated! IC={ic:.4f}, IR={ir:.2f}"
        else:
            hypothesis.status = HypothesisStatus.REJECTED
            hypothesis.feedback = self._generate_rejection_feedback(hypothesis, metrics)
            hypothesis.refinement_suggestions = self._generate_suggestions(hypothesis, metrics)

        return hypothesis

    def _generate_rejection_feedback(
        self,
        hypothesis: Hypothesis,
        metrics: dict[str, float],
    ) -> str:
        """Generate feedback explaining why hypothesis was rejected."""
        ic = abs(metrics.get("ic_mean", metrics.get("ic", 0)))
        ir = metrics.get("ir", 0)
        sharpe = metrics.get("sharpe", 0)

        feedback_parts = []

        if ic < self.ic_threshold:
            feedback_parts.append(
                f"IC ({ic:.4f}) below threshold ({self.ic_threshold}). "
                "Factor may not capture the intended relationship."
            )

        if ir < self.ir_threshold:
            feedback_parts.append(
                f"IR ({ir:.2f}) below threshold ({self.ir_threshold}). "
                "IC is inconsistent over time."
            )

        if sharpe < 0:
            feedback_parts.append(
                f"Negative Sharpe ratio ({sharpe:.2f}). "
                "Strategy loses money - consider reversing signal direction."
            )

        # Compare to expected
        expected = hypothesis.expected_ic
        if expected and ic < expected * 0.5:
            feedback_parts.append(
                f"Actual IC ({ic:.4f}) much lower than expected ({expected:.4f}). "
                "Hypothesis may be fundamentally flawed or market conditions changed."
            )

        return " ".join(feedback_parts)

    def _generate_suggestions(
        self,
        hypothesis: Hypothesis,
        metrics: dict[str, float],
    ) -> list[str]:
        """Generate refinement suggestions."""
        suggestions = []

        ic = abs(metrics.get("ic_mean", metrics.get("ic", 0)))
        ir = metrics.get("ir", 0)
        stability = metrics.get("stability", {})

        # IC-based suggestions
        if ic < self.ic_threshold * 0.5:
            suggestions.append("Consider combining with other factors for stronger signal")
            suggestions.append("Try different lookback periods (5d, 10d, 20d, 60d)")

        # IR-based suggestions
        if ir < 0.5:
            suggestions.append("Add regime filters to improve consistency")
            suggestions.append("Test on different market conditions (bull/bear)")

        # Stability-based suggestions
        if stability:
            regime = stability.get("regime_stability", {})
            if regime.get("consistency", 1) < 0.5:
                suggestions.append("Factor behaves differently in bull vs bear - add regime adaptation")

        # Family-specific suggestions
        if hypothesis.family == HypothesisFamily.MOMENTUM:
            suggestions.append("Try momentum with volume confirmation")
            suggestions.append("Test momentum at different frequencies")

        if hypothesis.family == HypothesisFamily.MEAN_REVERSION:
            suggestions.append("Add volatility filter - mean reversion works better in high vol")
            suggestions.append("Test with adaptive lookback based on volatility")

        return suggestions[:5]  # Limit to 5 suggestions


class HypothesisAgent:
    """Main agent for hypothesis-driven factor research.

    Implements the RD-Agent research loop:
    1. Generate hypotheses
    2. Convert to factor code
    3. Evaluate factors
    4. Analyze feedback
    5. Refine or generate new hypotheses
    """

    def __init__(
        self,
        generator: Optional[HypothesisGenerator] = None,
        converter: Optional[HypothesisToCode] = None,
        analyzer: Optional[FeedbackAnalyzer] = None,
    ) -> None:
        """Initialize agent.

        Args:
            generator: Hypothesis generator
            converter: Hypothesis to code converter
            analyzer: Feedback analyzer
        """
        self.generator = generator or HypothesisGenerator()
        self.converter = converter or HypothesisToCode()
        self.analyzer = analyzer or FeedbackAnalyzer()

        self._hypothesis_history: list[Hypothesis] = []
        self._iteration_count = 0

    def generate_hypotheses(
        self,
        market_data: pd.DataFrame,
        n_hypotheses: int = 5,
        focus_family: Optional[HypothesisFamily] = None,
    ) -> list[Hypothesis]:
        """Generate hypotheses for testing.

        Args:
            market_data: OHLCV DataFrame
            n_hypotheses: Number of hypotheses to generate
            focus_family: Optional family to focus on

        Returns:
            List of generated hypotheses
        """
        hypotheses = self.generator.generate_from_analysis(
            market_data, focus_family
        )

        # Add more from templates if needed
        while len(hypotheses) < n_hypotheses:
            for family in HypothesisFamily:
                if len(hypotheses) >= n_hypotheses:
                    break
                try:
                    h = self.generator.generate_from_template(
                        family, variation=len(hypotheses)
                    )
                    hypotheses.append(h)
                except ValueError:
                    continue

        return hypotheses[:n_hypotheses]

    def convert_to_factors(
        self,
        hypotheses: list[Hypothesis],
    ) -> list[tuple[Hypothesis, str]]:
        """Convert hypotheses to factor code.

        Args:
            hypotheses: List of hypotheses

        Returns:
            List of (hypothesis, code) tuples
        """
        results = []
        for h in hypotheses:
            code = self.converter.convert(h)
            results.append((h, code))
        return results

    def process_results(
        self,
        hypothesis: Hypothesis,
        experiment_result: dict[str, Any],
    ) -> Hypothesis:
        """Process experiment results and provide feedback.

        Args:
            hypothesis: Tested hypothesis
            experiment_result: Evaluation results

        Returns:
            Updated hypothesis with feedback
        """
        updated = self.analyzer.analyze(hypothesis, experiment_result)
        self._hypothesis_history.append(updated)
        return updated

    def get_validated_hypotheses(self) -> list[Hypothesis]:
        """Get all validated hypotheses."""
        return [
            h for h in self._hypothesis_history
            if h.status == HypothesisStatus.VALIDATED
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get research statistics."""
        total = len(self._hypothesis_history)
        validated = len(self.get_validated_hypotheses())

        return {
            "total_hypotheses_tested": total,
            "validated_count": validated,
            "rejection_rate": 1 - (validated / total) if total > 0 else 0,
            "by_family": self._count_by_family(),
            "avg_ic": self._average_ic(),
        }

    def _count_by_family(self) -> dict[str, int]:
        """Count hypotheses by family."""
        counts = {}
        for h in self._hypothesis_history:
            # Safely get family value (handle both enum and string)
            family = h.family.value if hasattr(h.family, 'value') else str(h.family)
            counts[family] = counts.get(family, 0) + 1
        return counts

    def _average_ic(self) -> float:
        """Calculate average IC of tested hypotheses."""
        ics = [h.actual_ic for h in self._hypothesis_history if h.actual_ic is not None]
        return sum(abs(ic) for ic in ics) / len(ics) if ics else 0.0
