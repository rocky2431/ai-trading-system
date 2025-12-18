"""Crypto-optimized Strategy and Risk Prompts.

This module provides specialized prompts for strategy generation
and risk management in cryptocurrency markets.
"""

from typing import Any, Optional

from .base import (
    AgentType,
    BasePromptTemplate,
    CryptoMarketType,
)


class StrategyGenerationPrompt(BasePromptTemplate):
    """Prompt template for generating trading strategies from factors."""

    def __init__(
        self,
        market_type: CryptoMarketType = CryptoMarketType.PERPETUAL,
    ) -> None:
        super().__init__(AgentType.STRATEGY, market_type)

    def get_system_prompt(self) -> str:
        return f"""You are an expert algorithmic trading strategist specializing in cryptocurrency markets.

Your task is to combine validated factors into executable trading strategies.

{self._get_crypto_context_block()}

## Strategy Design Principles for Crypto:

### 1. Position Sizing
- Kelly Criterion adjusted for crypto vol (typically 0.25-0.5 Kelly)
- Maximum position size: 5-10% of portfolio per asset
- Aggregate leverage: Never exceed 3x
- Reduce size in extreme volatility regimes

### 2. Entry/Exit Logic
- Avoid market orders in low liquidity (use limit orders)
- Size into positions gradually (TWAP-style)
- Use trailing stops rather than fixed stops
- Consider funding rate timing for perpetuals

### 3. Risk Controls
- Per-trade stop loss: 2-5% of position
- Daily loss limit: 3-5% of portfolio
- Weekly loss limit: 10% of portfolio
- Max drawdown circuit breaker: 20%

### 4. Execution Considerations
- Slippage estimation: 0.05-0.2% per trade
- Consider maker rebates (limit orders)
- Watch for spread widening during volatility
- Exchange-specific quirks (rate limits, etc.)

## Strategy Components:

1. **Signal Generation**: How factors are combined
2. **Position Sizing**: How much to trade
3. **Entry Rules**: When to enter positions
4. **Exit Rules**: When to close positions
5. **Risk Limits**: Maximum exposure and loss limits

## Output Format:

```yaml
strategy:
  name: [Strategy Name]
  type: [momentum/mean_reversion/hybrid/market_neutral]
  timeframe: [1h/4h/1d]

signals:
  factors:
    - name: [factor_name]
      weight: [0.0-1.0]
  combination: [equal_weight/ic_weighted/ml_combined]
  threshold: [entry threshold]

position_sizing:
  method: [equal_weight/volatility_inverse/kelly]
  max_position: [% of portfolio]
  leverage: [1x-3x]

entry:
  condition: [signal condition]
  order_type: [limit/market/twap]
  timing: [immediate/scheduled]

exit:
  take_profit: [% or signal condition]
  stop_loss: [%]
  trailing_stop: [true/false]
  max_holding: [periods]

risk:
  daily_loss_limit: [%]
  max_positions: [n]
  correlation_limit: [0.0-1.0]
```
"""

    def get_examples(self) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": "Design a strategy using funding rate and momentum factors"
            },
            {
                "role": "assistant",
                "content": """```yaml
strategy:
  name: Funding-Momentum Hybrid
  type: hybrid
  timeframe: 4h
  description: >
    Combines funding rate mean reversion with momentum confirmation.
    Goes long when funding is negative (shorts paying) and momentum is positive.
    This captures reversals in crowded short positioning.

signals:
  factors:
    - name: funding_rate_zscore
      weight: 0.6
      direction: contrarian  # Short when positive, long when negative
    - name: price_momentum_12h
      weight: 0.4
      direction: trend  # Long when positive
  combination: weighted_average
  threshold: 0.5  # Combined signal must exceed this

position_sizing:
  method: volatility_inverse
  max_position: 5%
  base_leverage: 2x
  volatility_scalar: 0.5  # Reduce in high vol
  min_size: 0.1%

entry:
  condition: combined_signal > threshold AND volatility_regime != extreme
  order_type: limit
  limit_offset: 0.02%  # Place limit slightly inside market
  timing: at_funding  # Enter just after funding payment

exit:
  take_profit:
    method: trailing
    activation: 2%  # Start trailing after 2% profit
    trail_distance: 1.5%
  stop_loss: 3%
  time_stop: 16h  # Max 2 funding periods
  signal_exit: true  # Exit if signal reverses

risk:
  daily_loss_limit: 4%
  max_positions: 3  # Max concurrent positions
  correlation_limit: 0.7  # Don't hold highly correlated assets
  funding_check: true  # Avoid negative carry
```

**Rationale:**
1. **Entry timing**: Entering after funding payment captures the reversal as shorts start covering
2. **Momentum confirmation**: Prevents shorting into strong trends even with high funding
3. **Volatility scaling**: Reduces position in high vol to avoid being stopped out
4. **Trailing stop**: Captures extended moves while protecting profits
5. **Time stop**: Limits exposure to adverse funding payments
"""
            },
        ]

    def render(
        self,
        validated_factors: list[dict[str, Any]],
        target_sharpe: float = 1.5,
        risk_budget: float = 0.10,
        **kwargs: Any,
    ) -> str:
        """Render strategy generation prompt.

        Args:
            validated_factors: List of validated factors with metrics
            target_sharpe: Target Sharpe ratio
            risk_budget: Maximum risk budget (annual vol)

        Returns:
            Complete prompt
        """
        factor_summary = "\n".join(
            f"- {f['name']}: IC={f.get('ic', 0):.3f}, IR={f.get('ir', 0):.2f}, Family={f.get('family', 'unknown')}"
            for f in validated_factors
        )

        return f"""## Validated Factors

{factor_summary}

## Constraints

- Target Sharpe: {target_sharpe}
- Risk Budget: {risk_budget:.0%} annualized volatility
- Execution: Binance/OKX perpetual futures
- Minimum trade size: $100

## Task

Design a trading strategy that:
1. Combines the validated factors effectively
2. Includes proper position sizing and risk management
3. Accounts for crypto-specific considerations
4. Is implementable in production
"""


class RiskManagementPrompt(BasePromptTemplate):
    """Prompt template for risk management decisions."""

    def __init__(self) -> None:
        super().__init__(AgentType.RISK, CryptoMarketType.PERPETUAL)

    def get_system_prompt(self) -> str:
        return """You are a risk manager for a cryptocurrency trading operation.

Your task is to analyze portfolio risk and provide actionable recommendations.

## Crypto-Specific Risk Factors:

### Market Risk
- Crypto volatility is 5-10x equities
- Correlation spikes during crashes (everything drops together)
- 24/7 markets mean no overnight hedging
- Liquidity can evaporate rapidly

### Execution Risk
- Exchange outages during volatility
- Rate limits can prevent position closing
- Slippage increases exponentially with size
- Different prices across exchanges

### Leverage Risk
- Liquidation cascades
- Funding rate costs can compound
- Mark price vs last price differences
- Maintenance margin requirements

### Counterparty Risk
- Exchange solvency (remember FTX)
- Stablecoin depegging
- Smart contract risks for DeFi
- API key security

## Risk Metrics to Monitor:

| Metric | Warning | Critical |
|--------|---------|----------|
| Portfolio VaR (95%, 1d) | > 5% | > 10% |
| Expected Shortfall | > 8% | > 15% |
| Gross Leverage | > 2x | > 3x |
| Net Exposure | > ±50% | > ±80% |
| Largest Position | > 20% | > 30% |
| Daily PnL Drawdown | > 3% | > 5% |

## Response Format:

1. **Risk Assessment**: Overall risk level (LOW/MEDIUM/HIGH/CRITICAL)
2. **Key Concerns**: Top 3 risk factors
3. **Recommendations**: Specific actions to take
4. **Limits**: Any limits that should be tightened/relaxed
"""

    def get_examples(self) -> list[dict[str, str]]:
        return []

    def render(
        self,
        portfolio_state: dict[str, Any],
        market_conditions: dict[str, Any],
        recent_pnl: list[float],
        **kwargs: Any,
    ) -> str:
        """Render risk management prompt.

        Args:
            portfolio_state: Current portfolio positions and metrics
            market_conditions: Current market conditions
            recent_pnl: Recent PnL history

        Returns:
            Complete prompt
        """
        return f"""## Current Portfolio State

**Positions:**
{self._format_positions(portfolio_state.get('positions', []))}

**Metrics:**
- Net Exposure: {portfolio_state.get('net_exposure', 0):.1%}
- Gross Leverage: {portfolio_state.get('gross_leverage', 0):.2f}x
- Portfolio VaR (95%, 1d): {portfolio_state.get('var_95', 0):.2%}
- Expected Shortfall: {portfolio_state.get('es', 0):.2%}

## Market Conditions

- BTC Volatility (30d): {market_conditions.get('btc_vol', 0):.0%}
- Funding Rates: {market_conditions.get('avg_funding', 0):.4%}
- OI Change (24h): {market_conditions.get('oi_change', 0):.1%}
- Fear & Greed Index: {market_conditions.get('fear_greed', 50)}

## Recent Performance

- Today: {recent_pnl[-1] if recent_pnl else 0:.2%}
- Week: {sum(recent_pnl[-7:]) if len(recent_pnl) >= 7 else sum(recent_pnl):.2%}
- Max Drawdown (7d): {min(recent_pnl[-7:]) if recent_pnl else 0:.2%}

## Task

Analyze the current risk profile and provide:
1. Overall risk assessment
2. Top concerns
3. Specific recommendations
4. Any limit adjustments needed
"""

    def _format_positions(self, positions: list[dict]) -> str:
        """Format positions for display."""
        if not positions:
            return "No positions"

        lines = []
        for p in positions:
            lines.append(
                f"- {p.get('symbol', 'UNKNOWN')}: "
                f"{p.get('size', 0):+.2f} ({p.get('weight', 0):.1%} of portfolio), "
                f"PnL: {p.get('pnl', 0):+.2%}"
            )
        return "\n".join(lines)


class BacktestAnalysisPrompt(BasePromptTemplate):
    """Prompt template for analyzing backtest results."""

    def __init__(self) -> None:
        super().__init__(AgentType.BACKTEST, CryptoMarketType.PERPETUAL)

    def get_system_prompt(self) -> str:
        return """You are an expert quantitative analyst reviewing backtest results.

Your task is to identify potential issues and validate strategy robustness.

## Overfitting Detection:

### Red Flags:
1. Sharpe in-sample >> out-of-sample
2. Performance degrades over time
3. Very high turnover (>200%/day)
4. Thin performance in specific periods
5. Excessive parameters

### Validation Checks:
1. Walk-forward analysis
2. Monte Carlo simulation
3. Regime-specific performance
4. Transaction cost sensitivity
5. Slippage sensitivity

## Crypto-Specific Validation:

1. **Bull/Bear Split**: Performance should be reasonable in both
2. **High Vol Periods**: Should not blow up in 2021-2022 volatility
3. **Funding Impact**: Does funding rate affect PnL significantly?
4. **Liquidity Constraints**: Are positions realistic given volume?
5. **Exchange Risk**: How would 2022 FTX event affect this?

## Response Format:

1. **Overall Assessment**: ROBUST / NEEDS WORK / LIKELY OVERFIT
2. **Strengths**: What's working well
3. **Weaknesses**: Concerns identified
4. **Robustness Score**: 1-10
5. **Recommendations**: Specific improvements
"""

    def get_examples(self) -> list[dict[str, str]]:
        return []

    def render(
        self,
        backtest_results: dict[str, Any],
        validation_results: dict[str, Any],
        strategy_description: str,
        **kwargs: Any,
    ) -> str:
        """Render backtest analysis prompt.

        Args:
            backtest_results: Full backtest results
            validation_results: Walk-forward and other validation
            strategy_description: Description of the strategy

        Returns:
            Complete prompt
        """
        return f"""## Strategy Description

{strategy_description}

## Backtest Results (In-Sample)

- Period: {backtest_results.get('start_date', 'N/A')} to {backtest_results.get('end_date', 'N/A')}
- Annual Return: {backtest_results.get('annual_return', 0):.1%}
- Sharpe Ratio: {backtest_results.get('sharpe', 0):.2f}
- Max Drawdown: {backtest_results.get('max_drawdown', 0):.1%}
- Win Rate: {backtest_results.get('win_rate', 0):.0%}
- Avg Trade PnL: {backtest_results.get('avg_pnl', 0):.2%}
- Turnover: {backtest_results.get('turnover', 0):.0%}/day

## Validation Results (Out-of-Sample)

- OOS Sharpe: {validation_results.get('oos_sharpe', 0):.2f}
- OOS Return: {validation_results.get('oos_return', 0):.1%}
- Walk-Forward Efficiency: {validation_results.get('wfe', 0):.0%}
- Monte Carlo p-value: {validation_results.get('mc_pvalue', 1):.3f}

## Regime Analysis

- Bull Market Sharpe: {validation_results.get('bull_sharpe', 0):.2f}
- Bear Market Sharpe: {validation_results.get('bear_sharpe', 0):.2f}
- High Vol Sharpe: {validation_results.get('high_vol_sharpe', 0):.2f}
- Low Vol Sharpe: {validation_results.get('low_vol_sharpe', 0):.2f}

## Task

Analyze these results and provide:
1. Overall robustness assessment
2. Identified strengths and weaknesses
3. Overfitting risk level
4. Specific recommendations for improvement
"""
