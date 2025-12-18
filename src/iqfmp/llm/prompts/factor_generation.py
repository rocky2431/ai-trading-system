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
        """Get crypto-optimized system prompt for factor generation."""
        return f"""You are an expert quantitative factor developer specializing in **cryptocurrency markets**.

Your task is to generate Python factor code for the Qlib-compatible IQFMP system.

{self._get_crypto_context_block()}

{self._get_available_fields_block(include_all=self.include_advanced_fields)}

## Factor Development Guidelines

### Code Standards:
1. **Function signature**: `def factor_name(df: pd.DataFrame) -> pd.Series`
2. **Input**: DataFrame with columns from Available Data Fields
3. **Output**: pd.Series with same index, values typically z-scored or normalized
4. **Dependencies**: Only use `pandas`, `numpy` (pre-imported)
5. **NaN handling**: Use `.fillna(0)` or appropriate method
6. **Lookback**: Be explicit about lookback periods

### Crypto-Specific Best Practices:
1. **No overnight gaps**: Crypto trades 24/7, no gap handling needed
2. **Funding rate**: Available every 8 hours, use rolling mean for intraday
3. **Volume**: Use `quote_volume` (USDT) for cross-asset comparisons
4. **Volatility**: Crypto vol is ~5-10x equity vol, adjust thresholds
5. **Regime detection**: Bull/bear/ranging markets behave very differently
6. **Exchange hours**: Consider Asia (00:00-08:00 UTC) vs US (13:00-21:00 UTC)

### Factor Normalization:
- Z-score: `(x - x.rolling(60).mean()) / (x.rolling(60).std() + 1e-10)`
- Rank: `x.rank(pct=True) - 0.5`
- Min-max: `(x - x.min()) / (x.max() - x.min() + 1e-10)`

### Output Format:
Return ONLY the Python code in a ```python code block.
Include a detailed docstring explaining:
- What the factor captures
- Why it might predict returns in crypto
- Any crypto-specific considerations
"""

    def get_examples(self) -> list[dict[str, str]]:
        """Get crypto-specific few-shot examples."""
        return [
            {
                "role": "user",
                "content": "Create a factor that captures funding rate momentum"
            },
            {
                "role": "assistant",
                "content": """```python
def funding_rate_momentum(df: pd.DataFrame) -> pd.Series:
    \"\"\"Funding Rate Momentum Factor.

    Captures the trend in perpetual futures funding rates.

    Crypto-specific insight:
    - Positive funding = longs pay shorts (bullish sentiment, crowded longs)
    - Rising funding = increasing bullish sentiment
    - BUT extreme positive funding often precedes reversals

    Signal interpretation:
    - Positive momentum in funding → Short-term bullish, but watch for extremes
    - We use z-score to identify extreme readings

    Args:
        df: DataFrame with 'funding_rate' column

    Returns:
        Z-scored funding rate momentum
    \"\"\"
    if 'funding_rate' not in df.columns:
        return pd.Series(0, index=df.index)

    funding = df['funding_rate']

    # 3-period momentum (24 hours for 8h funding)
    funding_momentum = funding.diff(3)

    # Z-score for normalization
    mean = funding_momentum.rolling(30).mean()
    std = funding_momentum.rolling(30).std()
    z_score = (funding_momentum - mean) / (std + 1e-10)

    return z_score.fillna(0)
```"""
            },
            {
                "role": "user",
                "content": "Create a factor based on liquidation imbalance"
            },
            {
                "role": "assistant",
                "content": """```python
def liquidation_pressure(df: pd.DataFrame) -> pd.Series:
    \"\"\"Liquidation Pressure Factor.

    Measures the imbalance between long and short liquidations.

    Crypto-specific insight:
    - Crypto markets have high leverage (up to 125x)
    - Liquidation cascades can cause sharp moves
    - Net long liquidations → potential buying opportunity (forced sellers exhausted)
    - Net short liquidations → potential shorting opportunity (forced buyers exhausted)

    Signal interpretation:
    - Positive value = more longs liquidated → contrarian long signal
    - Negative value = more shorts liquidated → contrarian short signal
    - Works best after the cascade completes (use lagged values)

    Args:
        df: DataFrame with 'liquidation_long' and 'liquidation_short' columns

    Returns:
        Normalized liquidation pressure signal
    \"\"\"
    import numpy as np

    if 'liquidation_long' not in df.columns or 'liquidation_short' not in df.columns:
        return pd.Series(0, index=df.index)

    liq_long = df['liquidation_long'].fillna(0)
    liq_short = df['liquidation_short'].fillna(0)
    liq_total = liq_long + liq_short + 1e-10

    # Net liquidation imbalance: positive = more longs liquidated
    imbalance = (liq_long - liq_short) / liq_total

    # Rolling sum for smoother signal (last 6 periods)
    rolling_imbalance = imbalance.rolling(6).sum()

    # Z-score normalization
    mean = rolling_imbalance.rolling(48).mean()
    std = rolling_imbalance.rolling(48).std()
    z_score = (rolling_imbalance - mean) / (std + 1e-10)

    # Lag by 1 period - trade AFTER the cascade
    signal = z_score.shift(1)

    return signal.fillna(0)
```"""
            },
            {
                "role": "user",
                "content": "Create a factor that combines volume and price for crypto"
            },
            {
                "role": "assistant",
                "content": """```python
def volume_price_confirmation(df: pd.DataFrame) -> pd.Series:
    \"\"\"Volume-Price Confirmation Factor.

    Measures whether price moves are confirmed by volume.

    Crypto-specific insight:
    - Crypto volume can be manipulated (wash trading)
    - Use quote_volume (USDT) for cross-asset comparison
    - High volume on up-moves = genuine buying pressure
    - High volume on down-moves = genuine selling pressure
    - Low volume moves tend to reverse

    Signal interpretation:
    - Positive = upward price move with high volume (bullish confirmation)
    - Negative = downward price move with high volume (bearish confirmation)
    - Near zero = unconfirmed move (potential reversal)

    Args:
        df: DataFrame with 'close' and 'volume' (or 'quote_volume') columns

    Returns:
        Volume-confirmed momentum signal
    \"\"\"
    import numpy as np

    # Use quote_volume if available (more reliable cross-asset)
    vol_col = 'quote_volume' if 'quote_volume' in df.columns else 'volume'

    if vol_col not in df.columns:
        return pd.Series(0, index=df.index)

    # Price return (5-period for crypto's higher frequency)
    returns = df['close'].pct_change(5)

    # Volume ratio: current vs 20-period average
    vol = df[vol_col]
    vol_ratio = vol / vol.rolling(20).mean()

    # Log volume ratio to reduce impact of volume spikes
    vol_signal = np.log1p(vol_ratio - 1)

    # Combine: direction from returns, magnitude from volume
    raw_signal = np.sign(returns) * vol_signal.abs()

    # Smooth and normalize
    smoothed = raw_signal.rolling(3).mean()
    mean = smoothed.rolling(60).mean()
    std = smoothed.rolling(60).std()
    z_score = (smoothed - mean) / (std + 1e-10)

    return z_score.fillna(0)
```"""
            },
            {
                "role": "user",
                "content": "Create a factor based on open interest changes"
            },
            {
                "role": "assistant",
                "content": """```python
def open_interest_momentum(df: pd.DataFrame) -> pd.Series:
    \"\"\"Open Interest Momentum Factor.

    Tracks changes in futures open interest to gauge market participation.

    Crypto-specific insight:
    - Rising OI + rising price = new longs entering (bullish)
    - Rising OI + falling price = new shorts entering (bearish)
    - Falling OI + rising price = short covering (weak rally)
    - Falling OI + falling price = long liquidation (weak decline)

    The factor captures conviction behind price moves.

    Signal interpretation:
    - High positive = rising OI confirms price direction
    - High negative = falling OI suggests weak move
    - Combined with price direction for final signal

    Args:
        df: DataFrame with 'close' and 'open_interest' columns

    Returns:
        OI-confirmed momentum signal
    \"\"\"
    import numpy as np

    if 'open_interest' not in df.columns:
        return pd.Series(0, index=df.index)

    oi = df['open_interest']
    price = df['close']

    # OI change (percentage)
    oi_pct_change = oi.pct_change(3)

    # Price direction
    price_direction = np.sign(price.pct_change(3))

    # Raw signal: OI change magnitude * price direction
    # Positive when OI rising with price, negative when OI falling
    raw_signal = oi_pct_change * price_direction

    # Z-score normalization
    mean = raw_signal.rolling(48).mean()
    std = raw_signal.rolling(48).std()
    z_score = (raw_signal - mean) / (std + 1e-10)

    return z_score.fillna(0)
```"""
            },
            {
                "role": "user",
                "content": "Create a crypto volatility regime factor"
            },
            {
                "role": "assistant",
                "content": """```python
def volatility_regime(df: pd.DataFrame) -> pd.Series:
    \"\"\"Volatility Regime Factor.

    Classifies the current volatility regime for regime-adaptive trading.

    Crypto-specific insight:
    - Crypto volatility is 5-10x higher than equities
    - Mean reversion works better in high vol regimes
    - Momentum works better in trending low-vol regimes
    - Extremely low vol often precedes explosive moves

    Regime definitions (annualized vol):
    - Low: < 40% (consolidation, breakout imminent)
    - Normal: 40-80% (typical crypto conditions)
    - High: 80-120% (elevated, good for mean reversion)
    - Extreme: > 120% (crisis/euphoria, reduced position sizing)

    Signal interpretation:
    - Factor returns the normalized vol regime indicator
    - Positive = above-normal volatility
    - Negative = below-normal volatility (breakout watch)

    Args:
        df: DataFrame with 'close' column

    Returns:
        Volatility regime indicator
    \"\"\"
    import numpy as np

    price = df['close']
    returns = price.pct_change()

    # Realized volatility (20-period, annualized for crypto = sqrt(365*24))
    vol_20 = returns.rolling(20).std() * np.sqrt(365 * 24)

    # Long-term vol for regime comparison (60-period)
    vol_60 = returns.rolling(60).std() * np.sqrt(365 * 24)

    # Regime indicator: current vs long-term
    vol_ratio = vol_20 / (vol_60 + 1e-10)

    # Log transform for better distribution
    log_ratio = np.log(vol_ratio + 1e-10)

    # Z-score normalize
    mean = log_ratio.rolling(120).mean()
    std = log_ratio.rolling(120).std()
    z_score = (log_ratio - mean) / (std + 1e-10)

    return z_score.fillna(0)
```"""
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

        # Add instruction
        parts.append(
            "\n## Task\n"
            "Generate a Python factor function following the guidelines in the system prompt.\n"
            "Ensure the factor is optimized for cryptocurrency markets."
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
Return the REFINED Python code in a ```python code block.
Explain what changes were made and why.
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
        return f"""## Original Factor Code

```python
{original_code}
```

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
