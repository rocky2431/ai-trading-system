"""Crypto-optimized Hypothesis Agent Prompts.

This module provides specialized prompts for the RD-Agent hypothesis-driven
research loop, with deep understanding of cryptocurrency market dynamics.
"""

from typing import Any, Optional

from .base import (
    AgentType,
    BasePromptTemplate,
    CryptoMarketType,
)


class HypothesisGenerationPrompt(BasePromptTemplate):
    """Prompt template for generating trading hypotheses.

    This template guides the LLM to generate testable trading hypotheses
    based on market analysis and crypto-specific patterns.
    """

    def __init__(
        self,
        market_type: CryptoMarketType = CryptoMarketType.PERPETUAL,
    ) -> None:
        super().__init__(AgentType.HYPOTHESIS, market_type)

    def get_system_prompt(self) -> str:
        """Get system prompt for hypothesis generation."""
        return f"""You are an expert quantitative researcher specializing in **cryptocurrency markets**.

Your task is to generate testable trading hypotheses for the IQFMP research platform.

{self._get_crypto_context_block()}

## Hypothesis Framework

A good trading hypothesis should:
1. **Be specific**: Clear about the relationship being tested
2. **Be falsifiable**: Can be validated/rejected with data
3. **Have rationale**: Explain WHY the relationship should exist
4. **Have prediction**: Expected IC, direction, and holding period

## Hypothesis Categories for Crypto:

### 1. Momentum Hypotheses
- Price momentum (past returns predict future returns)
- Volume-confirmed momentum
- Cross-asset momentum (BTC leads altcoins)
- Funding rate momentum

### 2. Mean Reversion Hypotheses
- RSI/oscillator extremes
- Funding rate extremes
- Basis extremes (futures vs spot)
- Deviation from moving averages

### 3. Sentiment/Positioning Hypotheses
- Long/short ratio extremes
- Liquidation imbalance
- Taker buy/sell ratio
- Top trader positioning

### 4. Volatility Hypotheses
- Low volatility predicts breakouts
- Volatility regime trading
- Vol-of-vol signals
- Implied vs realized spread

### 5. Market Microstructure Hypotheses
- Orderbook imbalance
- Spread dynamics
- Price impact
- Trade flow toxicity

### 6. Cross-Asset Hypotheses
- BTC dominance effects
- Sector rotation
- Stablecoin flows
- Risk-on/risk-off regimes

## Output Format

For each hypothesis, provide:
```
Name: [Short descriptive name]
Family: [momentum/mean_reversion/sentiment/volatility/microstructure/cross_asset]
Description: [One sentence description]
Rationale: [2-3 sentences explaining why this should work]
Expected IC: [0.01-0.10, realistic for crypto]
Direction: [long/short/long_short]
Holding Period: [1h/4h/1d/1w]
Data Required: [List of data fields needed]
```
"""

    def get_examples(self) -> list[dict[str, str]]:
        """Get few-shot examples for hypothesis generation."""
        return [
            {
                "role": "user",
                "content": "Generate hypotheses related to funding rate dynamics"
            },
            {
                "role": "assistant",
                "content": """Based on funding rate dynamics, here are three testable hypotheses:

---

**Hypothesis 1: Funding Rate Mean Reversion**

Name: Funding Rate Extreme Reversal
Family: mean_reversion
Description: Extreme funding rates predict short-term price reversals.
Rationale: When funding rate exceeds ±0.1% per 8h, it indicates crowded positioning. High positive funding means longs are paying significant premiums, suggesting overcrowded bullish sentiment that tends to reverse. This effect is amplified in crypto due to high leverage availability.
Expected IC: 0.04-0.06
Direction: long_short (short when funding extreme positive, long when extreme negative)
Holding Period: 8h-24h
Data Required: funding_rate, close

---

**Hypothesis 2: Funding-Price Divergence**

Name: Funding-Price Divergence
Family: momentum
Description: When price and funding rate diverge, a correction follows.
Rationale: If price rises but funding stays flat or falls, it suggests the move is driven by spot buying rather than leveraged speculation - this is healthier and more sustainable. Conversely, rising funding with flat price indicates leverage buildup without price follow-through.
Expected IC: 0.03-0.05
Direction: long_short
Holding Period: 4h-12h
Data Required: funding_rate, close

---

**Hypothesis 3: Funding Rate Momentum**

Name: Funding Rate Trend Following
Family: momentum
Description: Rising funding rates predict continued price increases in the short term.
Rationale: In early-stage rallies, funding rate increases reflect growing bullish sentiment and new money entering the market. This momentum tends to persist until funding reaches extreme levels. The key is catching the middle of the move, not the extremes.
Expected IC: 0.02-0.04
Direction: long_short
Holding Period: 4h-8h
Data Required: funding_rate, close
"""
            },
            {
                "role": "user",
                "content": "Generate hypotheses about liquidation patterns"
            },
            {
                "role": "assistant",
                "content": """Based on liquidation dynamics, here are three testable hypotheses:

---

**Hypothesis 1: Post-Cascade Reversal**

Name: Liquidation Cascade Exhaustion
Family: mean_reversion
Description: Large liquidation cascades create oversold/overbought conditions that reverse.
Rationale: When significant liquidations occur (>$100M in major coins), forced selling/buying creates temporary price dislocations. Once the cascade completes, the aggressive forced flow stops and price tends to revert. The key is timing entry after the cascade, not during.
Expected IC: 0.05-0.07
Direction: long_short (long after long liquidations, short after short liquidations)
Holding Period: 1h-4h (quick reversion after cascade)
Data Required: liquidation_long, liquidation_short, close

---

**Hypothesis 2: Pre-Cascade Warning**

Name: Liquidation Risk Indicator
Family: volatility
Description: High OI with low recent liquidations indicates cascade risk.
Rationale: When open interest builds up without any deleveraging (low liquidations), the market becomes fragile. A small price move can trigger a cascade. This can be used to reduce position size or prepare for volatility.
Expected IC: 0.02-0.03 (better as a filter than standalone signal)
Direction: neutral (volatility prediction)
Holding Period: N/A (risk filter)
Data Required: open_interest, liquidation_total, close

---

**Hypothesis 3: Liquidation Imbalance**

Name: Forced Flow Direction
Family: sentiment
Description: Net liquidation direction indicates which side is under pressure.
Rationale: If more longs are being liquidated than shorts, it suggests bearish pressure and weak hands being flushed out. The key insight is that liquidations create a floor/ceiling for prices as forced selling/buying exhausts.
Expected IC: 0.03-0.04
Direction: long_short
Holding Period: 4h-8h
Data Required: liquidation_long, liquidation_short
"""
            },
        ]

    def render(
        self,
        market_analysis: Optional[str] = None,
        focus_area: Optional[str] = None,
        previous_hypotheses: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Render hypothesis generation prompt.

        Args:
            market_analysis: Current market condition analysis
            focus_area: Specific area to focus on
            previous_hypotheses: Previously generated/tested hypotheses to avoid

        Returns:
            Complete prompt
        """
        parts = []

        if market_analysis:
            parts.append(f"## Current Market Analysis\n{market_analysis}")

        if focus_area:
            parts.append(f"## Focus Area\nGenerate hypotheses related to: {focus_area}")

        if previous_hypotheses:
            parts.append(
                f"## Previously Explored\n"
                f"Avoid generating similar hypotheses to these:\n"
                f"{chr(10).join(f'- {h}' for h in previous_hypotheses)}"
            )

        parts.append(
            "\n## Task\n"
            "Generate 2-3 novel, testable trading hypotheses.\n"
            "Ensure they are specific enough to be implemented as factors."
        )

        return "\n".join(parts)


class HypothesisToCodePrompt(BasePromptTemplate):
    """Prompt template for converting hypotheses to factor code."""

    def __init__(self) -> None:
        super().__init__(AgentType.HYPOTHESIS, CryptoMarketType.PERPETUAL)

    def get_system_prompt(self) -> str:
        return """You are an expert quantitative developer.

Your task is to convert a trading hypothesis into a SINGLE Qlib expression.

## Requirements:

1. **Syntax**: Use Qlib operators (Mean, Ref, Std, Corr, Rank, Log, Abs, If)
2. **Fields**: Use $-prefixed fields (e.g., $open, $close, $volume)
3. **Robustness**: Avoid divide-by-zero using + 1e-10
4. **No Python**: Do NOT output Python code

## Output Format:

Return ONLY the Qlib expression as plain text.
"""

    def get_examples(self) -> list[dict[str, str]]:
        return []  # Uses context from hypothesis

    def render(
        self,
        hypothesis_name: str,
        hypothesis_description: str,
        hypothesis_rationale: str,
        data_required: list[str],
        **kwargs: Any,
    ) -> str:
        """Render hypothesis-to-code prompt.

        Args:
            hypothesis_name: Name of the hypothesis
            hypothesis_description: Description
            hypothesis_rationale: Rationale
            data_required: Required data fields

        Returns:
            Complete prompt
        """
        return f"""## Hypothesis to Implement

**Name**: {hypothesis_name}
**Description**: {hypothesis_description}
**Rationale**: {hypothesis_rationale}

## Required Data Fields
{chr(10).join(f'- `{f}`' for f in data_required)}

## Task

Implement this hypothesis as a Qlib expression.
The expression should capture the relationship described in the hypothesis.
"""


class FeedbackAnalysisPrompt(BasePromptTemplate):
    """Prompt template for analyzing backtest feedback."""

    def __init__(self) -> None:
        super().__init__(AgentType.HYPOTHESIS, CryptoMarketType.PERPETUAL)

    def get_system_prompt(self) -> str:
        return """You are an expert quantitative researcher analyzing backtest results.

Your task is to analyze the performance of a factor and provide:
1. **Diagnosis**: Why the factor performed as it did
2. **Improvements**: Specific suggestions for refinement
3. **Next Steps**: Whether to refine, pivot, or abandon

## Crypto-Specific Considerations:

- IC > 0.03 is good for crypto (higher bar than equities due to noise)
- IR > 1.0 indicates consistent signal
- Sharpe > 1.5 is strong for crypto
- High turnover (>100%/day) may indicate overfitting to noise
- Regime sensitivity is common - check bull/bear performance

## Failure Mode Analysis:

1. **Low IC, Low IR**: Signal doesn't exist or wrong direction
2. **Low IC, High IR**: Consistent but weak - try combining with other signals
3. **High IC, Low IR**: Signal exists but inconsistent - add regime filters
4. **High Turnover**: Overfitting to noise - smooth or increase lookback

## Success Metrics for Crypto:

| Metric | Poor | Acceptable | Good | Excellent |
|--------|------|------------|------|-----------|
| IC Mean | <0.02 | 0.02-0.03 | 0.03-0.05 | >0.05 |
| IR | <0.5 | 0.5-1.0 | 1.0-1.5 | >1.5 |
| Sharpe | <0.5 | 0.5-1.0 | 1.0-2.0 | >2.0 |
| Drawdown | >30% | 20-30% | 10-20% | <10% |
"""

    def get_examples(self) -> list[dict[str, str]]:
        return []

    def render(
        self,
        hypothesis_name: str,
        factor_code: str,
        metrics: dict[str, float],
        stability_analysis: Optional[dict] = None,
        **kwargs: Any,
    ) -> str:
        """Render feedback analysis prompt.

        Args:
            hypothesis_name: Name of the hypothesis
            factor_code: Factor code that was tested
            metrics: Backtest metrics
            stability_analysis: Optional stability analysis results

        Returns:
            Complete prompt
        """
        parts = [f"## Factor: {hypothesis_name}"]

        parts.append(f"""
## Performance Metrics

- IC Mean: {metrics.get('ic_mean', 'N/A')}
- IR: {metrics.get('ir', 'N/A')}
- Sharpe: {metrics.get('sharpe', 'N/A')}
- Annual Return: {metrics.get('annual_return', 'N/A')}
- Max Drawdown: {metrics.get('max_drawdown', 'N/A')}
- Turnover: {metrics.get('turnover', 'N/A')}
""")

        if stability_analysis:
            parts.append(f"""
## Stability Analysis

- Regime Stability: {stability_analysis.get('regime_stability', 'N/A')}
- Bull Market IC: {stability_analysis.get('bull_ic', 'N/A')}
- Bear Market IC: {stability_analysis.get('bear_ic', 'N/A')}
- Time Decay: {stability_analysis.get('time_decay', 'N/A')}
""")

        parts.append(f"""
## Factor Code

```python
{factor_code}
```

## Task

Analyze these results and provide:
1. Diagnosis of why the factor performed as it did
2. 3-5 specific improvement suggestions
3. Recommendation: REFINE / PIVOT / ABANDON
""")

        return "\n".join(parts)


class ResearchPlanPrompt(BasePromptTemplate):
    """Prompt template for generating research plans."""

    def __init__(self) -> None:
        super().__init__(AgentType.HYPOTHESIS, CryptoMarketType.PERPETUAL)

    def get_system_prompt(self) -> str:
        return """You are a senior quantitative researcher planning factor research.

Your task is to create a structured research plan based on:
1. Current market conditions
2. Available data
3. Previous research findings

## Research Plan Structure:

1. **Research Theme**: Overarching question to answer
2. **Hypothesis Queue**: Ordered list of hypotheses to test
3. **Data Requirements**: What data needs to be collected
4. **Success Criteria**: What defines success for this research
5. **Timeline**: Estimated iterations needed
6. **Risk Factors**: What could invalidate findings

## Crypto Research Considerations:

- Market regimes change fast - test across multiple regimes
- Data quality varies by exchange - specify data sources
- Competition is fierce - novel signals decay faster
- Execution matters - consider trading costs
"""

    def get_examples(self) -> list[dict[str, str]]:
        return []

    def render(
        self,
        research_goal: str,
        available_data: list[str],
        previous_findings: Optional[str] = None,
        constraints: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Render research plan prompt.

        Args:
            research_goal: Overall research goal
            available_data: Available data fields
            previous_findings: Summary of previous research
            constraints: Any constraints (time, compute, etc.)

        Returns:
            Complete prompt
        """
        parts = [f"## Research Goal\n{research_goal}"]

        parts.append(f"""
## Available Data Fields
{chr(10).join(f'- {f}' for f in available_data)}
""")

        if previous_findings:
            parts.append(f"## Previous Findings\n{previous_findings}")

        if constraints:
            parts.append(f"## Constraints\n{constraints}")

        parts.append("""
## Task

Create a structured research plan with:
1. 3-5 hypotheses to test
2. Priority order and rationale
3. Success criteria for each
4. Overall timeline estimate
""")

        return "\n".join(parts)
