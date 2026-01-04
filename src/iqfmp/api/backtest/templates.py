"""Strategy Templates - Pre-configured trading strategy templates.

P1-2: Provides hardcoded templates for quick strategy creation.
"""

from dataclasses import dataclass, field
from enum import Enum


class StrategyCategory(Enum):
    """Strategy category types."""

    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    MULTI_FACTOR = "multi_factor"
    CRYPTO = "crypto"


class RiskLevel(Enum):
    """Strategy risk level."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class WeightingMethod(Enum):
    """Factor weighting methods."""

    EQUAL = "equal"
    IC_WEIGHTED = "ic_weighted"
    VOL_INVERSE = "vol_inverse"
    CUSTOM = "custom"


class RebalanceFrequency(Enum):
    """Rebalance frequency options."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class StrategyTemplate:
    """Strategy template definition."""

    id: str
    name: str
    description: str
    category: str  # momentum, mean_reversion, multi_factor, crypto
    risk_level: str  # conservative, moderate, aggressive

    # Strategy configuration
    factors: list[str]
    factor_descriptions: dict[str, str]
    weighting_method: str
    rebalance_frequency: str
    max_positions: int
    long_only: bool

    # Risk parameters
    max_drawdown: float
    position_size_limit: float
    stop_loss_enabled: bool
    stop_loss_threshold: float | None = None

    # Expected performance
    expected_sharpe: float = 0.0
    expected_annual_return: float = 0.0
    expected_max_drawdown: float = 0.0

    # Metadata
    tags: list[str] = field(default_factory=list)
    suitable_for: list[str] = field(default_factory=list)
    not_suitable_for: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate template fields at construction time."""
        # Validate category
        valid_categories = [c.value for c in StrategyCategory]
        if self.category not in valid_categories:
            raise ValueError(
                f"Invalid category '{self.category}'. "
                f"Must be one of: {valid_categories}"
            )

        # Validate risk level
        valid_risk_levels = [r.value for r in RiskLevel]
        if self.risk_level not in valid_risk_levels:
            raise ValueError(
                f"Invalid risk_level '{self.risk_level}'. "
                f"Must be one of: {valid_risk_levels}"
            )

        # Validate weighting method
        valid_methods = [w.value for w in WeightingMethod]
        if self.weighting_method not in valid_methods:
            raise ValueError(
                f"Invalid weighting_method '{self.weighting_method}'. "
                f"Must be one of: {valid_methods}"
            )

        # Validate rebalance frequency
        valid_frequencies = [f.value for f in RebalanceFrequency]
        if self.rebalance_frequency not in valid_frequencies:
            raise ValueError(
                f"Invalid rebalance_frequency '{self.rebalance_frequency}'. "
                f"Must be one of: {valid_frequencies}"
            )

        # Validate numeric ranges
        if self.max_positions <= 0:
            raise ValueError(f"max_positions must be > 0, got {self.max_positions}")

        if not 0 < self.position_size_limit <= 1:
            raise ValueError(
                f"position_size_limit must be in (0, 1], got {self.position_size_limit}"
            )

        if not 0 < self.max_drawdown <= 1:
            raise ValueError(
                f"max_drawdown must be in (0, 1], got {self.max_drawdown}"
            )

        # Validate stop loss consistency
        if self.stop_loss_enabled and self.stop_loss_threshold is None:
            raise ValueError(
                "stop_loss_threshold is required when stop_loss_enabled=True"
            )

        if self.stop_loss_threshold is not None and not 0 < self.stop_loss_threshold < 1:
            raise ValueError(
                f"stop_loss_threshold must be in (0, 1), got {self.stop_loss_threshold}"
            )

        # Validate factors match descriptions
        if set(self.factors) != set(self.factor_descriptions.keys()):
            missing = set(self.factors) - set(self.factor_descriptions.keys())
            extra = set(self.factor_descriptions.keys()) - set(self.factors)
            raise ValueError(
                f"factors and factor_descriptions keys must match. "
                f"Missing descriptions: {missing}. Extra descriptions: {extra}"
            )


# Pre-defined templates
STRATEGY_TEMPLATES: list[StrategyTemplate] = [
    # ==================== Momentum Strategies ====================
    StrategyTemplate(
        id="momentum_basic",
        name="Basic Momentum",
        description=(
            "Simple price momentum strategy using 20-day and 60-day returns. "
            "Suitable for trending markets with moderate volatility."
        ),
        category="momentum",
        risk_level="moderate",
        factors=["momentum_20d", "momentum_60d"],
        factor_descriptions={
            "momentum_20d": "20-day price momentum (return over last 20 trading days)",
            "momentum_60d": "60-day price momentum (return over last 60 trading days)",
        },
        weighting_method="equal",
        rebalance_frequency="weekly",
        max_positions=20,
        long_only=True,
        max_drawdown=0.15,
        position_size_limit=0.1,
        stop_loss_enabled=False,
        expected_sharpe=0.8,
        expected_annual_return=0.12,
        expected_max_drawdown=0.18,
        tags=["beginner-friendly", "trend-following", "long-only"],
        suitable_for=["bull markets", "trending assets", "liquid stocks"],
        not_suitable_for=["sideways markets", "high volatility periods"],
    ),
    StrategyTemplate(
        id="momentum_enhanced",
        name="Enhanced Momentum",
        description=(
            "Advanced momentum strategy with volume confirmation and volatility adjustment. "
            "Uses multi-timeframe momentum with risk scaling."
        ),
        category="momentum",
        risk_level="moderate",
        factors=[
            "momentum_20d",
            "momentum_60d",
            "volume_momentum",
            "volatility_adjusted_momentum",
        ],
        factor_descriptions={
            "momentum_20d": "20-day price momentum",
            "momentum_60d": "60-day price momentum",
            "volume_momentum": "Volume-weighted momentum indicator",
            "volatility_adjusted_momentum": "Momentum normalized by rolling volatility",
        },
        weighting_method="ic_weighted",
        rebalance_frequency="weekly",
        max_positions=30,
        long_only=True,
        max_drawdown=0.2,
        position_size_limit=0.08,
        stop_loss_enabled=True,
        stop_loss_threshold=0.1,
        expected_sharpe=1.1,
        expected_annual_return=0.18,
        expected_max_drawdown=0.22,
        tags=["intermediate", "trend-following", "risk-adjusted"],
        suitable_for=["trending markets", "institutional portfolios"],
        not_suitable_for=["low liquidity assets", "crisis periods"],
    ),
    # ==================== Mean Reversion Strategies ====================
    StrategyTemplate(
        id="mean_reversion_basic",
        name="Value Mean Reversion",
        description=(
            "Classic mean reversion strategy buying oversold assets and selling overbought ones. "
            "Uses RSI and Bollinger Band deviation."
        ),
        category="mean_reversion",
        risk_level="moderate",
        factors=["rsi_14d", "bollinger_deviation", "price_to_ma_50"],
        factor_descriptions={
            "rsi_14d": "14-day Relative Strength Index (inverted - buy low RSI)",
            "bollinger_deviation": "Deviation from Bollinger Bands (buy when below lower band)",
            "price_to_ma_50": "Price relative to 50-day moving average",
        },
        weighting_method="equal",
        rebalance_frequency="daily",
        max_positions=25,
        long_only=False,
        max_drawdown=0.12,
        position_size_limit=0.08,
        stop_loss_enabled=True,
        stop_loss_threshold=0.05,
        expected_sharpe=0.9,
        expected_annual_return=0.1,
        expected_max_drawdown=0.15,
        tags=["contrarian", "short-term", "market-neutral-possible"],
        suitable_for=["sideways markets", "range-bound assets"],
        not_suitable_for=["strongly trending markets", "momentum periods"],
    ),
    StrategyTemplate(
        id="statistical_arbitrage",
        name="Statistical Arbitrage",
        description=(
            "Pairs trading and statistical mean reversion using cointegration and z-score signals. "
            "Market neutral approach."
        ),
        category="mean_reversion",
        risk_level="conservative",
        factors=["pairs_zscore", "cointegration_residual", "sector_relative_value"],
        factor_descriptions={
            "pairs_zscore": "Z-score of price spread between correlated pairs",
            "cointegration_residual": "Residual from cointegration relationship",
            "sector_relative_value": "Value relative to sector peers",
        },
        weighting_method="vol_inverse",
        rebalance_frequency="daily",
        max_positions=40,
        long_only=False,
        max_drawdown=0.08,
        position_size_limit=0.05,
        stop_loss_enabled=True,
        stop_loss_threshold=0.03,
        expected_sharpe=1.5,
        expected_annual_return=0.08,
        expected_max_drawdown=0.1,
        tags=["market-neutral", "pairs-trading", "low-volatility"],
        suitable_for=["all market conditions", "institutional", "hedging"],
        not_suitable_for=["retail with limited capital", "illiquid markets"],
    ),
    # ==================== Multi-Factor Strategies ====================
    StrategyTemplate(
        id="quality_value_momentum",
        name="Quality-Value-Momentum",
        description=(
            "Balanced multi-factor approach combining quality, value, and momentum factors. "
            "Classic quant allocation strategy."
        ),
        category="multi_factor",
        risk_level="moderate",
        factors=["quality_score", "value_composite", "momentum_12m", "low_volatility"],
        factor_descriptions={
            "quality_score": "Composite quality score (ROE, margins, stability)",
            "value_composite": "Value score (P/E, P/B, EV/EBITDA)",
            "momentum_12m": "12-month price momentum with 1-month skip",
            "low_volatility": "Inverse volatility factor",
        },
        weighting_method="ic_weighted",
        rebalance_frequency="monthly",
        max_positions=50,
        long_only=True,
        max_drawdown=0.18,
        position_size_limit=0.05,
        stop_loss_enabled=False,
        expected_sharpe=1.0,
        expected_annual_return=0.14,
        expected_max_drawdown=0.2,
        tags=["diversified", "long-term", "factor-investing"],
        suitable_for=["long-term investors", "pension funds", "endowments"],
        not_suitable_for=["short-term traders", "high-frequency"],
    ),
    StrategyTemplate(
        id="risk_parity_factors",
        name="Risk Parity Factors",
        description=(
            "Risk parity allocation across multiple factors. "
            "Equal risk contribution from each factor exposure."
        ),
        category="multi_factor",
        risk_level="conservative",
        factors=[
            "momentum_composite",
            "value_composite",
            "size_factor",
            "quality_factor",
            "volatility_factor",
        ],
        factor_descriptions={
            "momentum_composite": "Combined short and long-term momentum",
            "value_composite": "Multi-metric value score",
            "size_factor": "Market cap factor (small cap premium)",
            "quality_factor": "Financial quality metrics",
            "volatility_factor": "Low volatility anomaly",
        },
        weighting_method="vol_inverse",
        rebalance_frequency="monthly",
        max_positions=100,
        long_only=True,
        max_drawdown=0.12,
        position_size_limit=0.03,
        stop_loss_enabled=False,
        expected_sharpe=1.2,
        expected_annual_return=0.1,
        expected_max_drawdown=0.14,
        tags=["risk-parity", "diversified", "institutional"],
        suitable_for=["risk-averse investors", "institutional mandates"],
        not_suitable_for=["return-maximizers", "concentrated portfolios"],
    ),
    # ==================== Crypto-Specific Strategies ====================
    StrategyTemplate(
        id="crypto_trend_following",
        name="Crypto Trend Following",
        description=(
            "Trend-following strategy optimized for cryptocurrency markets. "
            "Uses breakout signals with volatility-based position sizing."
        ),
        category="crypto",
        risk_level="aggressive",
        factors=[
            "crypto_momentum_7d",
            "crypto_momentum_30d",
            "volume_breakout",
            "volatility_regime",
        ],
        factor_descriptions={
            "crypto_momentum_7d": "7-day crypto momentum (24/7 trading adjusted)",
            "crypto_momentum_30d": "30-day crypto momentum",
            "volume_breakout": "Volume breakout signal",
            "volatility_regime": "Current volatility regime indicator",
        },
        weighting_method="vol_inverse",
        rebalance_frequency="daily",
        max_positions=10,
        long_only=False,
        max_drawdown=0.35,
        position_size_limit=0.15,
        stop_loss_enabled=True,
        stop_loss_threshold=0.15,
        expected_sharpe=0.7,
        expected_annual_return=0.5,
        expected_max_drawdown=0.4,
        tags=["crypto", "high-risk", "trend-following"],
        suitable_for=["crypto-native traders", "high risk tolerance"],
        not_suitable_for=["conservative investors", "regulated funds"],
    ),
    StrategyTemplate(
        id="crypto_mean_reversion",
        name="Crypto Mean Reversion",
        description=(
            "Short-term mean reversion for crypto markets. "
            "Captures overreaction bounces with tight risk controls."
        ),
        category="crypto",
        risk_level="aggressive",
        factors=["crypto_rsi_4h", "funding_rate", "orderbook_imbalance"],
        factor_descriptions={
            "crypto_rsi_4h": "4-hour RSI for crypto (faster timeframe)",
            "funding_rate": "Perpetual futures funding rate (sentiment indicator)",
            "orderbook_imbalance": "Order book bid/ask imbalance",
        },
        weighting_method="equal",
        rebalance_frequency="daily",
        max_positions=5,
        long_only=False,
        max_drawdown=0.25,
        position_size_limit=0.2,
        stop_loss_enabled=True,
        stop_loss_threshold=0.08,
        expected_sharpe=0.8,
        expected_annual_return=0.4,
        expected_max_drawdown=0.3,
        tags=["crypto", "short-term", "mean-reversion"],
        suitable_for=["active traders", "crypto exchanges"],
        not_suitable_for=["passive investors", "low risk tolerance"],
    ),
    StrategyTemplate(
        id="crypto_defi_yield",
        name="DeFi Yield Optimizer",
        description=(
            "Yield-focused strategy for DeFi protocols. "
            "Allocates to highest risk-adjusted yield opportunities."
        ),
        category="crypto",
        risk_level="aggressive",
        factors=["yield_score", "protocol_tvl", "smart_contract_risk", "il_risk"],
        factor_descriptions={
            "yield_score": "Annualized yield adjusted for token emissions",
            "protocol_tvl": "Protocol total value locked (security proxy)",
            "smart_contract_risk": "Smart contract audit and risk score",
            "il_risk": "Impermanent loss risk assessment",
        },
        weighting_method="custom",
        rebalance_frequency="weekly",
        max_positions=8,
        long_only=True,
        max_drawdown=0.4,
        position_size_limit=0.2,
        stop_loss_enabled=True,
        stop_loss_threshold=0.2,
        expected_sharpe=0.6,
        expected_annual_return=0.6,
        expected_max_drawdown=0.45,
        tags=["defi", "yield-farming", "crypto-native"],
        suitable_for=["DeFi experienced", "yield seekers"],
        not_suitable_for=["traditional investors", "regulatory-constrained"],
    ),
]


def get_all_templates() -> list[StrategyTemplate]:
    """Get all available strategy templates."""
    return STRATEGY_TEMPLATES


def get_template_by_id(template_id: str) -> StrategyTemplate | None:
    """Get a template by ID."""
    for template in STRATEGY_TEMPLATES:
        if template.id == template_id:
            return template
    return None


def get_templates_by_category(category: str) -> list[StrategyTemplate]:
    """Get templates filtered by category."""
    return [t for t in STRATEGY_TEMPLATES if t.category == category]


def get_templates_by_risk_level(risk_level: str) -> list[StrategyTemplate]:
    """Get templates filtered by risk level."""
    return [t for t in STRATEGY_TEMPLATES if t.risk_level == risk_level]


def search_templates(query: str) -> list[StrategyTemplate]:
    """Search templates by name, description, or tags."""
    query = query.lower()
    results = []
    for template in STRATEGY_TEMPLATES:
        if (
            query in template.name.lower()
            or query in template.description.lower()
            or any(query in tag.lower() for tag in template.tags)
        ):
            results.append(template)
    return results
