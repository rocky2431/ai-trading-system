"""Crypto-optimized Factor Generation Prompts.

This module provides specialized prompts for generating cryptocurrency
trading factors using LLM, with deep understanding of:
- Crypto market microstructure
- Perpetual futures dynamics
- Funding rate mechanics
- Liquidation patterns
- On-chain metrics
"""

from typing import Any, Optional

from .base import (
    AgentType,
    BasePromptTemplate,
    CryptoMarketType,
)


class FactorGenerationPrompt(BasePromptTemplate):
    """Crypto-optimized prompt template for factor generation.

    This template is designed to generate high-quality quantitative factors
    for cryptocurrency trading, with special attention to:
    1. Crypto-specific data fields (funding, OI, liquidations)
    2. 24/7 market dynamics
    3. High volatility handling
    4. Cross-exchange considerations
    """

    def __init__(
        self,
        market_type: CryptoMarketType = CryptoMarketType.PERPETUAL,
        include_advanced_fields: bool = True,
    ) -> None:
        """Initialize factor generation prompt.

        Args:
            market_type: Target market type
            include_advanced_fields: Include all advanced data fields
        """
        super().__init__(AgentType.FACTOR_GENERATION, market_type)
        self.include_advanced_fields = include_advanced_fields

    def get_system_prompt(self) -> str:
        """Get crypto-optimized system prompt for factor generation.

        Design Philosophy:
        - Provide ONLY syntax and operators, NO hardcoded indicator formulas
        - LLM should research and implement indicators itself
        - Intelligent feedback loop will guide LLM if indicators are missing
        """
        return f"""You are an expert quantitative factor developer specializing in **cryptocurrency markets**.

Your task is to generate **Qlib expression** factors that implement the user's hypothesis.

{self._get_crypto_context_block()}

## Available Data Fields (use $ prefix)

**ONLY these 5 fields are available. DO NOT use any other fields.**

- `$open` - Opening price
- `$high` - Highest price
- `$low` - Lowest price
- `$close` - Closing price
- `$volume` - Trading volume

If you need returns: `$close / Ref($close, -1) - 1`

## Qlib Expression Operators

You MUST use Qlib expression syntax. DO NOT write Python functions.

### Time-Series Operators:
- `Ref($field, -N)` - Reference N periods ago
- `Mean($field, N)` - Rolling mean (SMA)
- `Std($field, N)` - Rolling standard deviation
- `Sum($field, N)` - Rolling sum
- `Max($field, N)` - Rolling maximum (or element-wise: Max(a, b))
- `Min($field, N)` - Rolling minimum (or element-wise: Min(a, b))
- `Delta($field, N)` - Change over N periods
- `EMA($field, N)` - Exponential moving average
- `WMA($field, N)` - Weighted moving average

### Technical Indicators:
- `RSI($field, N)` - Relative Strength Index
- `MACD($field, fast, slow, signal)` - MACD histogram

### Math & Logic:
- `Abs(expr)`, `Log(expr)`, `Sign(expr)` - Math functions
- `Rank($field)` - Cross-sectional rank
- `Corr($f1, $f2, N)`, `Cov($f1, $f2, N)` - Correlation/Covariance
- `If(condition, true_val, false_val)` - Conditional logic
- `+`, `-`, `*`, `/`, `>`, `<` - Arithmetic and comparison

## Your Task

**CRITICAL**: You MUST implement ALL indicators mentioned in the user's request.

If the user mentions indicators like WR, SSL, MACD, Zigzag, Bollinger, etc.:
1. Research the indicator's mathematical formula
2. Translate it to Qlib expression syntax using available operators
3. Combine multiple indicators if requested

**Do NOT skip any indicator.** If you're unsure how to implement one, make your best attempt.
The system will provide feedback if your implementation is incomplete.

## Output Format

Return ONLY a single Qlib expression. No Python code, no markdown, no explanation before.

Example output:
(EMA($close, 12) - EMA($close, 26)) / $close

You may add a brief comment after the expression starting with #
"""

    def get_examples(self) -> list[dict[str, str]]:
        """Get crypto-specific few-shot examples using Qlib expressions."""
        return [
            {
                "role": "user",
                "content": "Create a momentum factor"
            },
            {
                "role": "assistant",
                "content": """Ref($close, -20) / $close - 1
# 20-period momentum: current price relative to 20 periods ago"""
            },
            {
                "role": "user",
                "content": "Create a mean reversion factor"
            },
            {
                "role": "assistant",
                "content": """($close - Mean($close, 20)) / Std($close, 20)
# Z-score mean reversion: how many std devs price is from 20-period mean"""
            },
            {
                "role": "user",
                "content": "Create a volume surge factor"
            },
            {
                "role": "assistant",
                "content": """$volume / Mean($volume, 20)
# Volume ratio: current volume relative to 20-period average"""
            },
            {
                "role": "user",
                "content": "Create a volatility factor"
            },
            {
                "role": "assistant",
                "content": """Std($close, 20) / Std($close, 60)
# Volatility ratio: short-term vs long-term volatility regime"""
            },
            {
                "role": "user",
                "content": "Create an RSI-based factor"
            },
            {
                "role": "assistant",
                "content": """RSI($close, 14)
# 14-period RSI: values > 70 overbought, < 30 oversold"""
            },
            {
                "role": "user",
                "content": "Create a MACD factor"
            },
            {
                "role": "assistant",
                "content": """MACD($close, 12, 26, 9)
# MACD histogram: difference between MACD line and signal line"""
            },
            {
                "role": "user",
                "content": "Create a price range factor"
            },
            {
                "role": "assistant",
                "content": """($high - $low) / $close
# Intraday price range normalized by close price"""
            },
            {
                "role": "user",
                "content": "Create a trend strength factor"
            },
            {
                "role": "assistant",
                "content": """Abs(Delta($close, 10)) / Mean(Abs(Delta($close, 1)), 10)
# Trend strength: 10-day move relative to average daily moves"""
            },
            {
                "role": "user",
                "content": "Create a volume-price correlation factor"
            },
            {
                "role": "assistant",
                "content": """Corr($close, $volume, 20)
# 20-period correlation between price and volume"""
            },
            {
                "role": "user",
                "content": "Create a short-term reversal factor"
            },
            {
                "role": "assistant",
                "content": """(Mean($close, 5) - Mean($close, 20)) / Std($close, 20)
# Short-term vs long-term MA difference, normalized by volatility"""
            },
            {
                "role": "user",
                "content": "Create a breakout factor"
            },
            {
                "role": "assistant",
                "content": """($close - Max($high, 20)) / Std($close, 20)
# Distance from 20-period high, negative means below recent highs"""
            },
            {
                "role": "user",
                "content": "Create an EMA crossover factor"
            },
            {
                "role": "assistant",
                "content": """(EMA($close, 12) - EMA($close, 26)) / $close
# EMA difference normalized by price, positive = bullish crossover"""
            },
        ]

    def render(
        self,
        user_request: str,
        factor_family: Optional[str] = None,
        include_examples: bool = True,
        extra_context: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Render the complete prompt.

        Args:
            user_request: User's natural language request
            factor_family: Optional factor family (momentum, mean_reversion, etc.)
            include_examples: Whether to include few-shot examples
            extra_context: Additional context to include

        Returns:
            Complete rendered prompt
        """
        parts = []

        # Add user request
        parts.append(f"## User Request\n{user_request}")

        # Add factor family context if specified
        if factor_family:
            family_context = self._get_family_context(factor_family)
            parts.append(f"\n## Factor Family Context\n{family_context}")

        # Add extra context if provided
        if extra_context:
            parts.append(f"\n## Additional Context\n{extra_context}")

        # Add instruction - MUST match system prompt (Qlib expression, NOT Python)
        parts.append(
            "\n## Task\n"
            "Generate a **Qlib expression** that implements the requested factor.\n"
            "Output ONLY the expression, no Python code, no markdown code blocks.\n"
            "You may add a brief comment after the expression starting with #"
        )

        return "\n".join(parts)

    def _get_family_context(self, family: str) -> str:
        """Get context for specific factor families."""
        family_contexts = {
            "momentum": """
**Momentum Factors for Crypto:**
- Price momentum (returns over various horizons)
- Volume-confirmed momentum
- Funding rate momentum
- Open interest momentum
- Cross-asset momentum (BTC leading altcoins)

Key considerations:
- Crypto momentum often reverses faster than equities
- Use shorter lookbacks (5-20 periods) vs equities (20-60)
- Volume confirmation is crucial due to wash trading
""",
            "mean_reversion": """
**Mean Reversion Factors for Crypto:**
- RSI-based (14-period standard)
- Bollinger Band deviation
- Funding rate extremes
- Distance from moving averages
- Basis (futures-spot spread)

Key considerations:
- Works better in high-volatility regimes
- Extreme readings in crypto are MORE extreme than equities
- Funding rate extremes are strong mean reversion signals
- Use wider bands (3+ std) for crypto
""",
            "volatility": """
**Volatility Factors for Crypto:**
- Realized volatility ratio (short/long)
- Volatility regime detection
- Implied vs realized (if options data available)
- Volatility term structure
- ATR-based signals

Key considerations:
- Crypto annualized vol typically 50-100%+ (vs 15-20% equities)
- Vol clusters are stronger and longer in crypto
- Low vol periods often precede large moves
- Consider using intraday data for vol estimation
""",
            "funding": """
**Funding Rate Factors for Crypto (Perpetual Futures):**
- Absolute funding rate level
- Funding rate momentum/trend
- Funding rate extremes (mean reversion)
- Funding-price divergence
- Cross-exchange funding spread

Key considerations:
- Funding paid every 8 hours
- Typical range: -0.1% to +0.1% per 8h
- Extreme: > ±0.3% per 8h
- High positive funding = crowded longs (bearish contrarian signal)
- Funding arbitrage can suppress the signal
""",
            "sentiment": """
**Sentiment/Positioning Factors for Crypto:**
- Long/short ratio
- Top trader positioning
- Taker buy/sell ratio
- Liquidation imbalance
- Social sentiment (if available)

Key considerations:
- Retail sentiment is often wrong at extremes
- Top trader positioning is more informative
- Taker ratio shows aggressive buying/selling
- Liquidation cascades create short-term opportunities
""",
            "microstructure": """
**Market Microstructure Factors for Crypto:**
- Bid-ask spread
- Orderbook imbalance
- Depth ratio
- Trade flow toxicity
- Price impact estimation

Key considerations:
- Spread and depth vary significantly across exchanges
- Weekend liquidity drops 30-50%
- Market makers adjust faster in crypto
- High-frequency signals may not survive latency
""",
        }
        return family_contexts.get(family.lower(), f"Factor family: {family}")


class FactorRefinementPrompt(BasePromptTemplate):
    """Prompt template for refining existing factors based on feedback."""

    def __init__(self) -> None:
        super().__init__(AgentType.FACTOR_GENERATION, CryptoMarketType.PERPETUAL)

    def get_system_prompt(self) -> str:
        return """You are an expert quantitative factor developer specializing in cryptocurrency markets.

Your task is to REFINE an existing factor based on backtest feedback.

## Refinement Strategies:

1. **Low IC (< 0.03)**:
   - Try different lookback periods
   - Add volume/volatility filters
   - Combine with complementary signals
   - Check if signal direction is correct

2. **Low IR (< 1.0)**:
   - Add regime filters (only trade in appropriate conditions)
   - Use adaptive parameters
   - Add cross-validation across time periods

3. **High Turnover**:
   - Add smoothing (rolling average of signal)
   - Increase threshold for position changes
   - Use longer lookback periods

4. **Regime Sensitivity**:
   - Add volatility regime filter
   - Use bull/bear market detection
   - Consider funding rate regime

## Output Format:
Return the REFINED Qlib expression ONLY. No Python code, no markdown code blocks.
You may add a brief comment after the expression starting with # to explain changes.
"""

    def get_examples(self) -> list[dict[str, str]]:
        return []  # Refinement uses context from previous results

    def render(
        self,
        original_code: str,
        feedback: dict[str, Any],
        suggestions: list[str],
        **kwargs: Any,
    ) -> str:
        """Render refinement prompt.

        Args:
            original_code: Original factor code
            feedback: Backtest feedback metrics
            suggestions: Suggested improvements

        Returns:
            Complete refinement prompt
        """
        return f"""## Original Factor Expression

{original_code}

## Backtest Feedback

- IC (mean): {feedback.get('ic_mean', 'N/A'):.4f}
- IR: {feedback.get('ir', 'N/A'):.2f}
- Sharpe: {feedback.get('sharpe', 'N/A'):.2f}
- Turnover: {feedback.get('turnover', 'N/A'):.2%}
- Max Drawdown: {feedback.get('max_drawdown', 'N/A'):.2%}

## Suggested Improvements

{chr(10).join(f'- {s}' for s in suggestions)}

## Task

Refine the factor to address the issues identified in the feedback.
Focus on the most impactful improvements first.
"""
