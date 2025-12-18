"""Hypothesis-driven Research Agent.

This module implements the core RD-Agent hypothesis-experiment-feedback loop.
Based on the RD-Agent paper architecture but adapted for cryptocurrency factor mining.

Key components:
- HypothesisGenerator: Generate trading hypotheses from market analysis (LLM-powered)
- HypothesisToCode: Convert hypotheses to executable factor code (LLM-powered)
- FeedbackAnalyzer: Analyze experiment results and provide feedback (LLM-powered)

All components use frontend-configured LLM models via OpenRouter API.
Model configuration is loaded from ConfigService (see model_config.py).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Protocol
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# LLM Integration Protocol and System Prompts
# =============================================================================

class LLMProviderProtocol(Protocol):
    """Protocol for LLM provider interface."""

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Generate completion from the LLM."""
        ...


HYPOTHESIS_SYSTEM_PROMPT = """You are an expert quantitative researcher specializing in cryptocurrency and financial market factor research.

Your role is to generate creative and testable trading hypotheses based on:
1. Market microstructure patterns
2. Behavioral finance insights
3. Technical analysis principles
4. Cryptocurrency-specific phenomena (funding rates, liquidations, etc.)

For each hypothesis, you must provide:
- A clear, testable statement
- The underlying rationale (why this should work)
- Expected information coefficient (IC) based on historical patterns
- Expected direction (long, short, or long_short)

Focus on hypotheses that are:
1. Actionable - can be converted to trading signals
2. Testable - can be evaluated with historical data
3. Novel - not already widely exploited
4. Robust - likely to work across different market conditions

Output your hypotheses in JSON format."""

CODE_GENERATION_SYSTEM_PROMPT = """You are an expert Python developer specializing in quantitative finance.

Your task is to convert trading hypotheses into executable factor code that works with pandas DataFrames.

Requirements:
1. The function must accept a DataFrame with OHLCV columns: open, high, low, close, volume
2. Return a pandas Series of factor values (typically z-scored)
3. Handle edge cases (NaN, insufficient data)
4. Use only pandas, numpy - no external dependencies
5. Apply proper normalization (z-score with rolling mean/std)

Output format:
```python
def factor_name(df):
    \"\"\"Docstring explaining the factor.\"\"\"
    # Your implementation
    return factor_values
```"""

FEEDBACK_SYSTEM_PROMPT = """You are an expert quantitative researcher analyzing factor performance results.

Your task is to:
1. Explain why a factor succeeded or failed
2. Identify potential issues in the implementation
3. Suggest specific improvements or refinements
4. Recommend whether to continue iterating on this hypothesis

Consider:
- Information Coefficient (IC) - predictive power
- Information Ratio (IR) - IC consistency over time
- Sharpe Ratio - risk-adjusted returns
- Turnover - trading costs impact
- Regime stability - performance across market conditions

Provide actionable feedback that can improve the next iteration."""


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
    """Generate trading hypotheses from market analysis using LLM.

    Uses frontend-configured model from ConfigService via model_config.py.
    Falls back to template-based generation if LLM is unavailable.
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProviderProtocol] = None,
        templates: Optional[dict[HypothesisFamily, list[dict]]] = None,
    ) -> None:
        """Initialize generator.

        Args:
            llm_provider: LLM provider for AI-powered generation
            templates: Hypothesis templates by family (fallback)
        """
        self.llm_provider = llm_provider
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

    async def generate_with_llm(
        self,
        market_data: pd.DataFrame,
        n_hypotheses: int = 5,
        focus_family: Optional[HypothesisFamily] = None,
    ) -> list[Hypothesis]:
        """Generate hypotheses using LLM based on market data.

        Args:
            market_data: OHLCV DataFrame for context
            n_hypotheses: Number of hypotheses to generate
            focus_family: Optional family to focus on

        Returns:
            List of LLM-generated hypotheses
        """
        if self.llm_provider is None:
            logger.warning("No LLM provider, falling back to template generation")
            return self.generate_from_analysis(market_data, focus_family)

        # Get model configuration from ConfigService
        from iqfmp.agents.model_config import get_agent_full_config
        model_id, temperature, custom_system_prompt = get_agent_full_config("hypothesis")

        # Prepare market context for LLM
        conditions = self._analyze_market_conditions(market_data)
        market_summary = self._create_market_summary(market_data, conditions)

        # Build prompt
        prompt = f"""Analyze the following market conditions and generate {n_hypotheses} creative trading hypotheses.

Market Summary:
{market_summary}

{"Focus on " + focus_family.value + " related hypotheses." if focus_family else "Cover diverse hypothesis families."}

Generate exactly {n_hypotheses} hypotheses in the following JSON format:
```json
[
    {{
        "name": "Hypothesis Name",
        "description": "Clear description of the hypothesis",
        "family": "momentum|mean_reversion|volatility|volume|trend|microstructure|funding|sentiment|cross_asset",
        "rationale": "Why this hypothesis should work",
        "expected_ic": 0.03,
        "expected_direction": "long|short|long_short"
    }}
]
```"""

        try:
            # Use custom system prompt if configured, otherwise use default
            system_prompt = custom_system_prompt or HYPOTHESIS_SYSTEM_PROMPT

            response = await self.llm_provider.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model_id,
                temperature=temperature,
                max_tokens=2048,
            )

            # Parse LLM response
            hypotheses = self._parse_llm_response(response.content)
            self._generated_count += len(hypotheses)

            logger.info(f"LLM generated {len(hypotheses)} hypotheses using model {model_id}")
            return hypotheses

        except Exception as e:
            logger.error(f"LLM hypothesis generation failed: {e}. Falling back to templates.")
            return self.generate_from_analysis(market_data, focus_family)

    def _parse_llm_response(self, response_text: str) -> list[Hypothesis]:
        """Parse LLM response to extract hypotheses."""
        hypotheses = []

        # Extract JSON from response
        json_match = re.search(r'\[[\s\S]*\]', response_text)
        if not json_match:
            logger.warning("No JSON found in LLM response")
            return hypotheses

        try:
            data = json.loads(json_match.group())
            for item in data:
                family_str = item.get("family", "momentum")
                try:
                    family = HypothesisFamily(family_str)
                except ValueError:
                    family = HypothesisFamily.MOMENTUM

                hypothesis = Hypothesis(
                    name=item.get("name", "Unnamed Hypothesis"),
                    description=item.get("description", ""),
                    family=family,
                    rationale=item.get("rationale", ""),
                    expected_ic=float(item.get("expected_ic", 0.03)),
                    expected_direction=item.get("expected_direction", "long_short"),
                    source="llm",
                )
                hypotheses.append(hypothesis)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON: {e}")

        return hypotheses

    def _create_market_summary(
        self,
        df: pd.DataFrame,
        conditions: dict[str, bool],
    ) -> str:
        """Create market summary for LLM context."""
        if len(df) < 20:
            return "Insufficient data for analysis."

        # Calculate key metrics
        returns_1d = df["close"].pct_change().iloc[-1] * 100
        returns_7d = df["close"].pct_change(7).iloc[-1] * 100
        returns_30d = df["close"].pct_change(30).iloc[-1] * 100 if len(df) >= 30 else 0

        volatility = df["close"].pct_change().rolling(20).std().iloc[-1] * (252 ** 0.5) * 100

        # Volume analysis
        volume_ratio = 1.0
        if "volume" in df.columns:
            volume_ratio = df["volume"].iloc[-1] / df["volume"].rolling(20).mean().iloc[-1]

        summary_parts = [
            f"- 1-day return: {returns_1d:.2f}%",
            f"- 7-day return: {returns_7d:.2f}%",
            f"- 30-day return: {returns_30d:.2f}%",
            f"- Annualized volatility: {volatility:.1f}%",
            f"- Volume ratio (vs 20d avg): {volume_ratio:.2f}x",
            f"- Trending: {'Yes' if conditions['trending'] else 'No'}",
            f"- High volatility: {'Yes' if conditions['high_volatility'] else 'No'}",
            f"- Extreme RSI: {'Yes' if conditions['extreme_rsi'] else 'No'}",
        ]

        if "funding_rate" in df.columns:
            funding = df["funding_rate"].iloc[-1] * 100
            summary_parts.append(f"- Current funding rate: {funding:.4f}%")

        return "\n".join(summary_parts)

    def generate_from_analysis(
        self,
        market_data: pd.DataFrame,
        focus_family: Optional[HypothesisFamily] = None,
    ) -> list[Hypothesis]:
        """Generate hypotheses based on market data analysis (template fallback).

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
            hypotheses.append(self.generate_from_template(HypothesisFamily.MOMENTUM, 0))

        if conditions["high_volatility"]:
            hypotheses.append(self.generate_from_template(HypothesisFamily.VOLATILITY, 0))
        else:
            hypotheses.append(self.generate_from_template(HypothesisFamily.VOLATILITY, 1))

        if conditions["extreme_rsi"]:
            hypotheses.append(self.generate_from_template(HypothesisFamily.MEAN_REVERSION, 0))

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
        conditions["trending"] = abs(returns_20d) > 0.1

        # Check volatility regime
        volatility = df["close"].pct_change().rolling(20).std().iloc[-1] * (252 ** 0.5)
        conditions["high_volatility"] = volatility > 0.5

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
    """Convert trading hypotheses to executable factor code using LLM.

    Uses frontend-configured model from ConfigService via model_config.py.
    Falls back to template-based conversion if LLM is unavailable.
    """

    # Code templates for each family (fallback)
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

    def __init__(
        self,
        llm_provider: Optional[LLMProviderProtocol] = None,
    ) -> None:
        """Initialize converter.

        Args:
            llm_provider: LLM provider for AI-powered code generation
        """
        self.llm_provider = llm_provider
        self._conversion_count = 0

    async def convert_with_llm(
        self,
        hypothesis: Hypothesis,
        params: Optional[dict[str, Any]] = None,
    ) -> str:
        """Convert hypothesis to factor code using LLM.

        Args:
            hypothesis: Hypothesis to convert
            params: Optional parameters for the factor

        Returns:
            Executable Python code string
        """
        if self.llm_provider is None:
            logger.warning("No LLM provider, falling back to template conversion")
            return self.convert(hypothesis, params)

        # Get model configuration from ConfigService
        from iqfmp.agents.model_config import get_agent_full_config
        model_id, temperature, custom_system_prompt = get_agent_full_config("factor_generation")

        # Generate function name
        func_name = self._generate_func_name(hypothesis)

        # Build prompt
        prompt = f"""Convert the following trading hypothesis into executable Python factor code.

Hypothesis:
- Name: {hypothesis.name}
- Description: {hypothesis.description}
- Family: {hypothesis.family.value}
- Rationale: {hypothesis.rationale}
- Expected Direction: {hypothesis.expected_direction}

Requirements:
1. Function name must be: {func_name}
2. Input: pandas DataFrame with columns: open, high, low, close, volume
3. Output: pandas Series of factor values (z-scored for normalization)
4. Handle edge cases: NaN values, insufficient data
5. Use only pandas, numpy - no external dependencies

Return ONLY the Python code in a code block:
```python
def {func_name}(df):
    # Your implementation
    return factor_values
```"""

        try:
            # Use custom system prompt if configured, otherwise use default
            system_prompt = custom_system_prompt or CODE_GENERATION_SYSTEM_PROMPT

            response = await self.llm_provider.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model_id,
                temperature=temperature,
                max_tokens=2048,
            )

            # Parse code from response
            code = self._extract_code(response.content, func_name)

            self._conversion_count += 1
            hypothesis.factor_code = code
            hypothesis.factor_name = func_name

            logger.info(f"LLM generated code for '{hypothesis.name}' using model {model_id}")
            return code

        except Exception as e:
            logger.error(f"LLM code generation failed: {e}. Falling back to template.")
            return self.convert(hypothesis, params)

    def _extract_code(self, response_text: str, func_name: str) -> str:
        """Extract Python code from LLM response."""
        # Try to find code block
        code_match = re.search(r'```python\s*(.*?)```', response_text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Try to find function definition directly
        func_match = re.search(rf'(def {func_name}\(.*?\):.*)', response_text, re.DOTALL)
        if func_match:
            return func_match.group(1).strip()

        # Return the whole response as a fallback
        return response_text.strip()

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
    """Analyze experiment results and provide feedback for hypothesis refinement using LLM.

    Uses frontend-configured model from ConfigService via model_config.py.
    Falls back to template-based analysis if LLM is unavailable.
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProviderProtocol] = None,
        ic_threshold: float = 0.03,
        ir_threshold: float = 1.0,
    ) -> None:
        """Initialize analyzer.

        Args:
            llm_provider: LLM provider for AI-powered analysis
            ic_threshold: Minimum IC for passing
            ir_threshold: Minimum IR for passing
        """
        self.llm_provider = llm_provider
        self.ic_threshold = ic_threshold
        self.ir_threshold = ir_threshold

    async def analyze_with_llm(
        self,
        hypothesis: Hypothesis,
        experiment_result: dict[str, Any],
    ) -> Hypothesis:
        """Analyze experiment results using LLM and update hypothesis with feedback.

        Args:
            hypothesis: Tested hypothesis
            experiment_result: Results from evaluation

        Returns:
            Updated hypothesis with LLM-generated feedback
        """
        # First, do the basic analysis
        metrics = experiment_result.get("metrics", {})
        hypothesis.experiment_result = experiment_result
        hypothesis.actual_ic = metrics.get("ic_mean", metrics.get("ic", 0))

        # Check if passed threshold
        ic = abs(hypothesis.actual_ic or 0)
        ir = metrics.get("ir", 0)
        hypothesis.passed_threshold = ic >= self.ic_threshold and ir >= self.ir_threshold

        if hypothesis.passed_threshold:
            hypothesis.status = HypothesisStatus.VALIDATED
            hypothesis.feedback = f"Hypothesis validated! IC={ic:.4f}, IR={ir:.2f}"
            return hypothesis

        # For rejected hypotheses, use LLM for detailed analysis
        if self.llm_provider is None:
            hypothesis.status = HypothesisStatus.REJECTED
            hypothesis.feedback = self._generate_rejection_feedback(hypothesis, metrics)
            hypothesis.refinement_suggestions = self._generate_suggestions(hypothesis, metrics)
            return hypothesis

        # Get model configuration from ConfigService
        from iqfmp.agents.model_config import get_agent_full_config
        model_id, temperature, custom_system_prompt = get_agent_full_config("factor_evaluation")

        # Build analysis prompt
        prompt = f"""Analyze the following factor performance results and provide detailed feedback.

Hypothesis:
- Name: {hypothesis.name}
- Description: {hypothesis.description}
- Family: {hypothesis.family.value}
- Rationale: {hypothesis.rationale}
- Expected IC: {hypothesis.expected_ic}
- Expected Direction: {hypothesis.expected_direction}

Experiment Results:
- Actual IC: {ic:.4f} (threshold: {self.ic_threshold})
- Information Ratio (IR): {ir:.2f} (threshold: {self.ir_threshold})
- Sharpe Ratio: {metrics.get('sharpe', 'N/A')}
- Max Drawdown: {metrics.get('max_drawdown', 'N/A')}
- Turnover: {metrics.get('turnover', 'N/A')}

Stability Metrics:
{json.dumps(metrics.get('stability', {}), indent=2)}

Factor Code:
```python
{hypothesis.factor_code or 'Not available'}
```

Provide:
1. A detailed explanation of why this factor failed (2-3 sentences)
2. 3-5 specific, actionable improvement suggestions

Format your response as JSON:
```json
{{
    "feedback": "Your detailed analysis...",
    "suggestions": ["Suggestion 1", "Suggestion 2", "Suggestion 3"]
}}
```"""

        try:
            # Use custom system prompt if configured, otherwise use default
            system_prompt = custom_system_prompt or FEEDBACK_SYSTEM_PROMPT

            response = await self.llm_provider.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model_id,
                temperature=temperature,
                max_tokens=1024,
            )

            # Parse LLM response
            feedback_data = self._parse_feedback_response(response.content)

            hypothesis.status = HypothesisStatus.REJECTED
            hypothesis.feedback = feedback_data.get("feedback", self._generate_rejection_feedback(hypothesis, metrics))
            hypothesis.refinement_suggestions = feedback_data.get("suggestions", self._generate_suggestions(hypothesis, metrics))

            logger.info(f"LLM analyzed '{hypothesis.name}' using model {model_id}")
            return hypothesis

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}. Falling back to template.")
            hypothesis.status = HypothesisStatus.REJECTED
            hypothesis.feedback = self._generate_rejection_feedback(hypothesis, metrics)
            hypothesis.refinement_suggestions = self._generate_suggestions(hypothesis, metrics)
            return hypothesis

    def _parse_feedback_response(self, response_text: str) -> dict[str, Any]:
        """Parse LLM feedback response."""
        # Try to extract JSON
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: extract text manually
        return {
            "feedback": response_text[:500],
            "suggestions": [],
        }

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
    """Main agent for hypothesis-driven factor research using LLM.

    Implements the RD-Agent research loop with LLM-powered components:
    1. Generate hypotheses (LLM or template-based)
    2. Convert to factor code (LLM or template-based)
    3. Evaluate factors
    4. Analyze feedback (LLM or template-based)
    5. Refine or generate new hypotheses

    All LLM-powered components use frontend-configured models via ConfigService.
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProviderProtocol] = None,
        generator: Optional[HypothesisGenerator] = None,
        converter: Optional[HypothesisToCode] = None,
        analyzer: Optional[FeedbackAnalyzer] = None,
    ) -> None:
        """Initialize agent with LLM support.

        Args:
            llm_provider: LLM provider for AI-powered operations
            generator: Hypothesis generator (uses llm_provider if not provided)
            converter: Hypothesis to code converter (uses llm_provider if not provided)
            analyzer: Feedback analyzer (uses llm_provider if not provided)
        """
        self.llm_provider = llm_provider
        self.generator = generator or HypothesisGenerator(llm_provider=llm_provider)
        self.converter = converter or HypothesisToCode(llm_provider=llm_provider)
        self.analyzer = analyzer or FeedbackAnalyzer(llm_provider=llm_provider)

        self._hypothesis_history: list[Hypothesis] = []
        self._iteration_count = 0

    async def generate_hypotheses_with_llm(
        self,
        market_data: pd.DataFrame,
        n_hypotheses: int = 5,
        focus_family: Optional[HypothesisFamily] = None,
    ) -> list[Hypothesis]:
        """Generate hypotheses using LLM.

        Args:
            market_data: OHLCV DataFrame
            n_hypotheses: Number of hypotheses to generate
            focus_family: Optional family to focus on

        Returns:
            List of LLM-generated hypotheses
        """
        hypotheses = await self.generator.generate_with_llm(
            market_data, n_hypotheses, focus_family
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

        self._iteration_count += 1
        return hypotheses[:n_hypotheses]

    async def convert_to_factors_with_llm(
        self,
        hypotheses: list[Hypothesis],
    ) -> list[tuple[Hypothesis, str]]:
        """Convert hypotheses to factor code using LLM.

        Args:
            hypotheses: List of hypotheses

        Returns:
            List of (hypothesis, code) tuples
        """
        results = []
        for h in hypotheses:
            code = await self.converter.convert_with_llm(h)
            results.append((h, code))
        return results

    async def process_results_with_llm(
        self,
        hypothesis: Hypothesis,
        experiment_result: dict[str, Any],
    ) -> Hypothesis:
        """Process experiment results using LLM and provide feedback.

        Args:
            hypothesis: Tested hypothesis
            experiment_result: Evaluation results

        Returns:
            Updated hypothesis with LLM-generated feedback
        """
        updated = await self.analyzer.analyze_with_llm(hypothesis, experiment_result)
        self._hypothesis_history.append(updated)
        return updated

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
