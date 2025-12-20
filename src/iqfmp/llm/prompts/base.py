"""Base classes for prompt templates.

This module provides the foundation for modular, crypto-optimized prompt templates.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class PromptRole(str, Enum):
    """Role types for prompts."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class AgentType(str, Enum):
    """Types of agents in the system."""
    FACTOR_GENERATION = "factor_generation"
    HYPOTHESIS = "hypothesis"
    STRATEGY = "strategy"
    RISK = "risk"
    BACKTEST = "backtest"


class CryptoMarketType(str, Enum):
    """Cryptocurrency market types."""
    SPOT = "spot"
    PERPETUAL = "perpetual"
    FUTURES = "futures"
    OPTIONS = "options"


@dataclass
class CryptoDataFields:
    """Available crypto-specific data fields for factor generation.

    This defines all data fields available in the IQFMP system,
    optimized for cryptocurrency trading.
    """

    # Core OHLCV fields
    CORE_FIELDS: list[str] = field(default_factory=lambda: [
        "open", "high", "low", "close", "volume",
        "quote_volume",  # Volume in quote currency (USDT)
        "trades_count",  # Number of trades
    ])

    # Perpetual futures specific
    PERPETUAL_FIELDS: list[str] = field(default_factory=lambda: [
        "funding_rate",  # 8-hour funding rate
        "funding_rate_predicted",  # Next funding rate estimate
        "mark_price",  # Mark price for liquidation calc
        "index_price",  # Spot index price
        "open_interest",  # Total open interest (USD)
        "open_interest_change",  # OI change over period
    ])

    # Orderbook/depth fields
    ORDERBOOK_FIELDS: list[str] = field(default_factory=lambda: [
        "bid_price",  # Best bid
        "ask_price",  # Best ask
        "bid_volume",  # Bid depth
        "ask_volume",  # Ask depth
        "spread",  # Bid-ask spread
        "depth_imbalance",  # (bid_vol - ask_vol) / total
    ])

    # Sentiment/positioning fields
    SENTIMENT_FIELDS: list[str] = field(default_factory=lambda: [
        "long_short_ratio",  # Long/short account ratio
        "top_trader_long_short_ratio",  # Top traders ratio
        "taker_buy_volume",  # Taker buy volume
        "taker_sell_volume",  # Taker sell volume
        "taker_buy_ratio",  # Buy/(Buy+Sell)
    ])

    # Liquidation fields
    LIQUIDATION_FIELDS: list[str] = field(default_factory=lambda: [
        "liquidation_long",  # Long liquidation volume
        "liquidation_short",  # Short liquidation volume
        "liquidation_total",  # Total liquidation volume
    ])

    # On-chain fields (for major coins)
    ONCHAIN_FIELDS: list[str] = field(default_factory=lambda: [
        "exchange_inflow",  # Tokens flowing into exchanges
        "exchange_outflow",  # Tokens flowing out
        "exchange_netflow",  # Net flow
        "whale_transactions",  # Large transaction count
        "active_addresses",  # Active address count
    ])

    def get_all_fields(self) -> list[str]:
        """Get all available fields."""
        return (
            self.CORE_FIELDS +
            self.PERPETUAL_FIELDS +
            self.ORDERBOOK_FIELDS +
            self.SENTIMENT_FIELDS +
            self.LIQUIDATION_FIELDS +
            self.ONCHAIN_FIELDS
        )

    def get_field_descriptions(self) -> dict[str, str]:
        """Get descriptions for all fields."""
        return {
            # Core
            "open": "Opening price of the candle",
            "high": "Highest price during the candle",
            "low": "Lowest price during the candle",
            "close": "Closing price of the candle",
            "volume": "Trading volume in base currency",
            "quote_volume": "Trading volume in quote currency (USDT)",
            "trades_count": "Number of individual trades",

            # Perpetual
            "funding_rate": "Perpetual funding rate (8-hour), positive = longs pay shorts",
            "funding_rate_predicted": "Predicted next funding rate",
            "mark_price": "Mark price used for liquidation calculations",
            "index_price": "Underlying spot index price",
            "open_interest": "Total open interest in USD",
            "open_interest_change": "Change in open interest over period",

            # Orderbook
            "bid_price": "Best bid price",
            "ask_price": "Best ask price",
            "bid_volume": "Total bid depth near best bid",
            "ask_volume": "Total ask depth near best ask",
            "spread": "Bid-ask spread as percentage",
            "depth_imbalance": "Orderbook imbalance: (bid - ask) / (bid + ask)",

            # Sentiment
            "long_short_ratio": "Ratio of long to short accounts",
            "top_trader_long_short_ratio": "Top traders long/short ratio",
            "taker_buy_volume": "Aggressive buy volume",
            "taker_sell_volume": "Aggressive sell volume",
            "taker_buy_ratio": "Taker buy / (buy + sell), >0.5 = net buying",

            # Liquidation
            "liquidation_long": "Long position liquidation volume",
            "liquidation_short": "Short position liquidation volume",
            "liquidation_total": "Total liquidation volume",

            # On-chain
            "exchange_inflow": "Tokens flowing into exchanges (potential sell pressure)",
            "exchange_outflow": "Tokens flowing out of exchanges (accumulation)",
            "exchange_netflow": "Net token flow to exchanges",
            "whale_transactions": "Count of large transactions (>$100k)",
            "active_addresses": "Number of active addresses on-chain",
        }


@dataclass
class CryptoMarketContext:
    """Context about crypto market characteristics.

    Used to provide LLM with understanding of crypto-specific dynamics.
    """

    # Market structure
    is_24_7: bool = True
    has_funding: bool = True  # Perpetual futures
    has_leverage: bool = True
    max_leverage: int = 125

    # Volatility characteristics
    typical_daily_vol: float = 0.05  # 5% typical daily move
    extreme_vol_threshold: float = 0.15  # 15%+ considered extreme

    # Crypto-specific patterns
    patterns: list[str] = field(default_factory=lambda: [
        "Funding rate arbitrage: High positive funding → short pressure",
        "Liquidation cascades: Large OI + low funding → cascade risk",
        "Weekend low liquidity: Reduced depth, higher spreads",
        "Asia/US session patterns: Different trading behaviors",
        "Whale accumulation: Exchange outflows + low price = bullish",
        "Stablecoin flows: USDT supply increase = potential inflows",
        "Derivatives premium: Perp > spot = bullish sentiment",
        "Fear and Greed extremes: Contrarian signals",
    ])

    # Exchange-specific notes
    exchange_notes: dict[str, str] = field(default_factory=lambda: {
        "binance": "Largest volume, most representative orderbook",
        "okx": "Strong Asia presence, reliable OI data",
        "bybit": "Retail-heavy, useful sentiment indicator",
        "deribit": "Options market leader, implied vol signals",
    })


class BasePromptTemplate(ABC):
    """Abstract base class for prompt templates.

    P2 Fix: Added version tracking for prompt templates.
    """

    # Class-level version tracking (override in subclasses)
    VERSION: str = "1.0.0"
    PROMPT_ID: str = "base_template"

    def __init__(
        self,
        agent_type: AgentType,
        market_type: CryptoMarketType = CryptoMarketType.PERPETUAL,
    ) -> None:
        """Initialize template.

        Args:
            agent_type: Type of agent using this template
            market_type: Target market type
        """
        self.agent_type = agent_type
        self.market_type = market_type
        self.data_fields = CryptoDataFields()
        self.market_context = CryptoMarketContext()

    @property
    def version(self) -> str:
        """Get prompt template version."""
        return self.VERSION

    @property
    def prompt_id(self) -> str:
        """Get unique prompt template identifier."""
        return self.PROMPT_ID

    def get_version_info(self) -> dict[str, str]:
        """Get version information for tracking.

        P2 Fix: Returns version info for logging and tracking.
        """
        return {
            "prompt_id": self.prompt_id,
            "version": self.version,
            "agent_type": self.agent_type.value,
            "market_type": self.market_type.value,
        }

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt."""
        pass

    @abstractmethod
    def get_examples(self) -> list[dict[str, str]]:
        """Get few-shot examples."""
        pass

    @abstractmethod
    def render(self, **kwargs: Any) -> str:
        """Render the complete prompt with context."""
        pass

    def _get_crypto_context_block(self) -> str:
        """Get crypto market context as a prompt block."""
        return f"""
## Cryptocurrency Market Context

**Market Structure:**
- 24/7 trading (no market close)
- High leverage available (up to {self.market_context.max_leverage}x)
- Funding rate mechanism in perpetuals
- Cross-exchange arbitrage opportunities

**Typical Dynamics:**
- Daily volatility: ~{self.market_context.typical_daily_vol*100:.0f}%
- Extreme moves: >{self.market_context.extreme_vol_threshold*100:.0f}% in single day
- Weekend liquidity: 30-50% lower than weekdays
- Asian session (00:00-08:00 UTC): Often accumulation phase
- US session (13:00-21:00 UTC): Higher volume, larger moves

**Key Patterns to Consider:**
{chr(10).join(f'- {p}' for p in self.market_context.patterns)}
"""

    def _get_available_fields_block(self, include_all: bool = False) -> str:
        """Get available data fields as a prompt block."""
        fields = self.data_fields.get_field_descriptions()

        if not include_all:
            # Default to core + perpetual fields
            core = self.data_fields.CORE_FIELDS
            perp = self.data_fields.PERPETUAL_FIELDS
            allowed = set(core + perp)
            fields = {k: v for k, v in fields.items() if k in allowed}

        field_list = "\n".join(f"- `{k}`: {v}" for k, v in sorted(fields.items()))
        return f"""
## Available Data Fields

{field_list}
"""
