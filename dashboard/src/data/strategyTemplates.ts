/**
 * Strategy Templates - Pre-configured trading strategy templates
 *
 * Categories:
 * - Momentum: Price momentum and trend-following strategies
 * - Mean Reversion: Counter-trend and value strategies
 * - Multi-Factor: Combined factor approaches
 * - Crypto-Specific: Cryptocurrency-optimized strategies
 */

export type StrategyCategory =
  | "momentum"
  | "mean_reversion"
  | "multi_factor"
  | "crypto";

export type WeightingMethod = "equal" | "ic_weighted" | "vol_inverse" | "custom";

export type RebalanceFrequency = "daily" | "weekly" | "monthly" | "quarterly";

export type RiskLevel = "conservative" | "moderate" | "aggressive";

/**
 * Historical reference data for strategy performance.
 * IMPORTANT: Past performance is not indicative of future results.
 * These values are derived from backtests and should be used for
 * reference purposes only, not as predictions.
 */
export interface HistoricalReference {
  sharpe: number;
  annualReturn: number;
  maxDrawdown: number;
  dataSource: string;      // e.g., "2020-2023 backtest on S&P 500"
  lastUpdated: string;     // ISO date string
  disclaimer: string;      // Required legal disclaimer
}

export interface StrategyTemplate {
  id: string;
  name: string;
  description: string;
  category: StrategyCategory;
  riskLevel: RiskLevel;

  // Strategy configuration
  factors: string[];
  factorDescriptions: Record<string, string>;
  weightingMethod: WeightingMethod;
  rebalanceFrequency: RebalanceFrequency;
  maxPositions: number;
  longOnly: boolean;

  // Risk parameters
  maxDrawdown: number; // Target max drawdown (decimal)
  positionSizeLimit: number; // Max position size (decimal)
  stopLossEnabled: boolean;
  stopLossThreshold?: number;

  // Historical performance reference (NOT predictions)
  // See HistoricalReference interface for important disclaimers
  historicalReference: HistoricalReference;

  // Metadata
  tags: string[];
  suitableFor: string[];
  notSuitableFor: string[];
  createdAt: string;
  updatedAt: string;
}

export const STRATEGY_TEMPLATES: StrategyTemplate[] = [
  // ==================== Momentum Strategies ====================
  {
    id: "momentum_basic",
    name: "Basic Momentum",
    description:
      "Simple price momentum strategy using 20-day and 60-day returns. Suitable for trending markets with moderate volatility.",
    category: "momentum",
    riskLevel: "moderate",

    factors: ["momentum_20d", "momentum_60d"],
    factorDescriptions: {
      momentum_20d: "20-day price momentum (return over last 20 trading days)",
      momentum_60d: "60-day price momentum (return over last 60 trading days)",
    },
    weightingMethod: "equal",
    rebalanceFrequency: "weekly",
    maxPositions: 20,
    longOnly: true,

    maxDrawdown: 0.15,
    positionSizeLimit: 0.1,
    stopLossEnabled: false,

    historicalReference: {
      sharpe: 0.8,
      annualReturn: 0.12,
      maxDrawdown: 0.18,
      dataSource: "2018-2023 backtest on US large-cap equities",
      lastUpdated: "2024-01-15",
      disclaimer: "Past performance is not indicative of future results. Actual results may vary significantly.",
    },

    tags: ["beginner-friendly", "trend-following", "long-only"],
    suitableFor: ["bull markets", "trending assets", "liquid stocks"],
    notSuitableFor: ["sideways markets", "high volatility periods"],
    createdAt: "2024-01-01",
    updatedAt: "2024-12-01",
  },
  {
    id: "momentum_enhanced",
    name: "Enhanced Momentum",
    description:
      "Advanced momentum strategy with volume confirmation and volatility adjustment. Uses multi-timeframe momentum with risk scaling.",
    category: "momentum",
    riskLevel: "moderate",

    factors: [
      "momentum_20d",
      "momentum_60d",
      "volume_momentum",
      "volatility_adjusted_momentum",
    ],
    factorDescriptions: {
      momentum_20d: "20-day price momentum",
      momentum_60d: "60-day price momentum",
      volume_momentum: "Volume-weighted momentum indicator",
      volatility_adjusted_momentum: "Momentum normalized by rolling volatility",
    },
    weightingMethod: "ic_weighted",
    rebalanceFrequency: "weekly",
    maxPositions: 30,
    longOnly: true,

    maxDrawdown: 0.2,
    positionSizeLimit: 0.08,
    stopLossEnabled: true,
    stopLossThreshold: 0.1,

    historicalReference: {
      sharpe: 1.1,
      annualReturn: 0.18,
      maxDrawdown: 0.22,
      dataSource: "2018-2023 backtest on US large-cap equities",
      lastUpdated: "2024-01-15",
      disclaimer: "Past performance is not indicative of future results. Actual results may vary significantly.",
    },

    tags: ["intermediate", "trend-following", "risk-adjusted"],
    suitableFor: ["trending markets", "institutional portfolios"],
    notSuitableFor: ["low liquidity assets", "crisis periods"],
    createdAt: "2024-01-01",
    updatedAt: "2024-12-01",
  },

  // ==================== Mean Reversion Strategies ====================
  {
    id: "mean_reversion_basic",
    name: "Value Mean Reversion",
    description:
      "Classic mean reversion strategy buying oversold assets and selling overbought ones. Uses RSI and Bollinger Band deviation.",
    category: "mean_reversion",
    riskLevel: "moderate",

    factors: ["rsi_14d", "bollinger_deviation", "price_to_ma_50"],
    factorDescriptions: {
      rsi_14d: "14-day Relative Strength Index (inverted - buy low RSI)",
      bollinger_deviation:
        "Deviation from Bollinger Bands (buy when below lower band)",
      price_to_ma_50: "Price relative to 50-day moving average",
    },
    weightingMethod: "equal",
    rebalanceFrequency: "daily",
    maxPositions: 25,
    longOnly: false,

    maxDrawdown: 0.12,
    positionSizeLimit: 0.08,
    stopLossEnabled: true,
    stopLossThreshold: 0.05,

    historicalReference: {
      sharpe: 0.9,
      annualReturn: 0.1,
      maxDrawdown: 0.15,
      dataSource: "2018-2023 backtest on US equities",
      lastUpdated: "2024-01-15",
      disclaimer: "Past performance is not indicative of future results. Actual results may vary significantly.",
    },

    tags: ["contrarian", "short-term", "market-neutral-possible"],
    suitableFor: ["sideways markets", "range-bound assets"],
    notSuitableFor: ["strongly trending markets", "momentum periods"],
    createdAt: "2024-01-01",
    updatedAt: "2024-12-01",
  },
  {
    id: "statistical_arbitrage",
    name: "Statistical Arbitrage",
    description:
      "Pairs trading and statistical mean reversion using cointegration and z-score signals. Market neutral approach.",
    category: "mean_reversion",
    riskLevel: "conservative",

    factors: ["pairs_zscore", "cointegration_residual", "sector_relative_value"],
    factorDescriptions: {
      pairs_zscore: "Z-score of price spread between correlated pairs",
      cointegration_residual: "Residual from cointegration relationship",
      sector_relative_value: "Value relative to sector peers",
    },
    weightingMethod: "vol_inverse",
    rebalanceFrequency: "daily",
    maxPositions: 40,
    longOnly: false,

    maxDrawdown: 0.08,
    positionSizeLimit: 0.05,
    stopLossEnabled: true,
    stopLossThreshold: 0.03,

    historicalReference: {
      sharpe: 1.5,
      annualReturn: 0.08,
      maxDrawdown: 0.1,
      dataSource: "2018-2023 backtest on US equity pairs",
      lastUpdated: "2024-01-15",
      disclaimer: "Past performance is not indicative of future results. Actual results may vary significantly.",
    },

    tags: ["market-neutral", "pairs-trading", "low-volatility"],
    suitableFor: ["all market conditions", "institutional", "hedging"],
    notSuitableFor: ["retail with limited capital", "illiquid markets"],
    createdAt: "2024-01-01",
    updatedAt: "2024-12-01",
  },

  // ==================== Multi-Factor Strategies ====================
  {
    id: "quality_value_momentum",
    name: "Quality-Value-Momentum",
    description:
      "Balanced multi-factor approach combining quality, value, and momentum factors. Classic quant allocation strategy.",
    category: "multi_factor",
    riskLevel: "moderate",

    factors: [
      "quality_score",
      "value_composite",
      "momentum_12m",
      "low_volatility",
    ],
    factorDescriptions: {
      quality_score: "Composite quality score (ROE, margins, stability)",
      value_composite: "Value score (P/E, P/B, EV/EBITDA)",
      momentum_12m: "12-month price momentum with 1-month skip",
      low_volatility: "Inverse volatility factor",
    },
    weightingMethod: "ic_weighted",
    rebalanceFrequency: "monthly",
    maxPositions: 50,
    longOnly: true,

    maxDrawdown: 0.18,
    positionSizeLimit: 0.05,
    stopLossEnabled: false,

    historicalReference: {
      sharpe: 1.0,
      annualReturn: 0.14,
      maxDrawdown: 0.2,
      dataSource: "2015-2023 backtest on global developed equities",
      lastUpdated: "2024-01-15",
      disclaimer: "Past performance is not indicative of future results. Actual results may vary significantly.",
    },

    tags: ["diversified", "long-term", "factor-investing"],
    suitableFor: ["long-term investors", "pension funds", "endowments"],
    notSuitableFor: ["short-term traders", "high-frequency"],
    createdAt: "2024-01-01",
    updatedAt: "2024-12-01",
  },
  {
    id: "risk_parity_factors",
    name: "Risk Parity Factors",
    description:
      "Risk parity allocation across multiple factors. Equal risk contribution from each factor exposure.",
    category: "multi_factor",
    riskLevel: "conservative",

    factors: [
      "momentum_composite",
      "value_composite",
      "size_factor",
      "quality_factor",
      "volatility_factor",
    ],
    factorDescriptions: {
      momentum_composite: "Combined short and long-term momentum",
      value_composite: "Multi-metric value score",
      size_factor: "Market cap factor (small cap premium)",
      quality_factor: "Financial quality metrics",
      volatility_factor: "Low volatility anomaly",
    },
    weightingMethod: "vol_inverse",
    rebalanceFrequency: "monthly",
    maxPositions: 100,
    longOnly: true,

    maxDrawdown: 0.12,
    positionSizeLimit: 0.03,
    stopLossEnabled: false,

    historicalReference: {
      sharpe: 1.2,
      annualReturn: 0.1,
      maxDrawdown: 0.14,
      dataSource: "2015-2023 backtest on global multi-asset",
      lastUpdated: "2024-01-15",
      disclaimer: "Past performance is not indicative of future results. Actual results may vary significantly.",
    },

    tags: ["risk-parity", "diversified", "institutional"],
    suitableFor: ["risk-averse investors", "institutional mandates"],
    notSuitableFor: ["return-maximizers", "concentrated portfolios"],
    createdAt: "2024-01-01",
    updatedAt: "2024-12-01",
  },

  // ==================== Crypto-Specific Strategies ====================
  {
    id: "crypto_trend_following",
    name: "Crypto Trend Following",
    description:
      "Trend-following strategy optimized for cryptocurrency markets. Uses breakout signals with volatility-based position sizing.",
    category: "crypto",
    riskLevel: "aggressive",

    factors: [
      "crypto_momentum_7d",
      "crypto_momentum_30d",
      "volume_breakout",
      "volatility_regime",
    ],
    factorDescriptions: {
      crypto_momentum_7d: "7-day crypto momentum (24/7 trading adjusted)",
      crypto_momentum_30d: "30-day crypto momentum",
      volume_breakout: "Volume breakout signal",
      volatility_regime: "Current volatility regime indicator",
    },
    weightingMethod: "vol_inverse",
    rebalanceFrequency: "daily",
    maxPositions: 10,
    longOnly: false,

    maxDrawdown: 0.35,
    positionSizeLimit: 0.15,
    stopLossEnabled: true,
    stopLossThreshold: 0.15,

    historicalReference: {
      sharpe: 0.7,
      annualReturn: 0.5,
      maxDrawdown: 0.4,
      dataSource: "2020-2023 backtest on top 10 cryptocurrencies",
      lastUpdated: "2024-01-15",
      disclaimer: "Past performance is not indicative of future results. Crypto markets are highly volatile.",
    },

    tags: ["crypto", "high-risk", "trend-following"],
    suitableFor: ["crypto-native traders", "high risk tolerance"],
    notSuitableFor: ["conservative investors", "regulated funds"],
    createdAt: "2024-01-01",
    updatedAt: "2024-12-01",
  },
  {
    id: "crypto_mean_reversion",
    name: "Crypto Mean Reversion",
    description:
      "Short-term mean reversion for crypto markets. Captures overreaction bounces with tight risk controls.",
    category: "crypto",
    riskLevel: "aggressive",

    factors: ["crypto_rsi_4h", "funding_rate", "orderbook_imbalance"],
    factorDescriptions: {
      crypto_rsi_4h: "4-hour RSI for crypto (faster timeframe)",
      funding_rate: "Perpetual futures funding rate (sentiment indicator)",
      orderbook_imbalance: "Order book bid/ask imbalance",
    },
    weightingMethod: "equal",
    rebalanceFrequency: "daily",
    maxPositions: 5,
    longOnly: false,

    maxDrawdown: 0.25,
    positionSizeLimit: 0.2,
    stopLossEnabled: true,
    stopLossThreshold: 0.08,

    historicalReference: {
      sharpe: 0.8,
      annualReturn: 0.4,
      maxDrawdown: 0.3,
      dataSource: "2021-2023 backtest on BTC/ETH perpetuals",
      lastUpdated: "2024-01-15",
      disclaimer: "Past performance is not indicative of future results. Crypto markets are highly volatile.",
    },

    tags: ["crypto", "short-term", "mean-reversion"],
    suitableFor: ["active traders", "crypto exchanges"],
    notSuitableFor: ["passive investors", "low risk tolerance"],
    createdAt: "2024-01-01",
    updatedAt: "2024-12-01",
  },
  {
    id: "crypto_defi_yield",
    name: "DeFi Yield Optimizer",
    description:
      "Yield-focused strategy for DeFi protocols. Allocates to highest risk-adjusted yield opportunities.",
    category: "crypto",
    riskLevel: "aggressive",

    factors: ["yield_score", "protocol_tvl", "smart_contract_risk", "il_risk"],
    factorDescriptions: {
      yield_score: "Annualized yield adjusted for token emissions",
      protocol_tvl: "Protocol total value locked (security proxy)",
      smart_contract_risk: "Smart contract audit and risk score",
      il_risk: "Impermanent loss risk assessment",
    },
    weightingMethod: "custom",
    rebalanceFrequency: "weekly",
    maxPositions: 8,
    longOnly: true,

    maxDrawdown: 0.4,
    positionSizeLimit: 0.2,
    stopLossEnabled: true,
    stopLossThreshold: 0.2,

    historicalReference: {
      sharpe: 0.6,
      annualReturn: 0.6,
      maxDrawdown: 0.45,
      dataSource: "2021-2023 backtest on top DeFi protocols",
      lastUpdated: "2024-01-15",
      disclaimer: "Past performance is not indicative of future results. DeFi yields are highly variable and subject to smart contract risk.",
    },

    tags: ["defi", "yield-farming", "crypto-native"],
    suitableFor: ["DeFi experienced", "yield seekers"],
    notSuitableFor: ["traditional investors", "regulatory-constrained"],
    createdAt: "2024-01-01",
    updatedAt: "2024-12-01",
  },
];

// Helper functions

export function getTemplatesByCategory(
  category: StrategyCategory
): StrategyTemplate[] {
  return STRATEGY_TEMPLATES.filter((t) => t.category === category);
}

export function getTemplatesByRiskLevel(
  riskLevel: RiskLevel
): StrategyTemplate[] {
  return STRATEGY_TEMPLATES.filter((t) => t.riskLevel === riskLevel);
}

export function getTemplateById(id: string): StrategyTemplate | undefined {
  return STRATEGY_TEMPLATES.find((t) => t.id === id);
}

export function searchTemplates(query: string): StrategyTemplate[] {
  const lowerQuery = query.toLowerCase();
  return STRATEGY_TEMPLATES.filter(
    (t) =>
      t.name.toLowerCase().includes(lowerQuery) ||
      t.description.toLowerCase().includes(lowerQuery) ||
      t.tags.some((tag) => tag.toLowerCase().includes(lowerQuery))
  );
}

export const CATEGORY_LABELS: Record<StrategyCategory, string> = {
  momentum: "Momentum",
  mean_reversion: "Mean Reversion",
  multi_factor: "Multi-Factor",
  crypto: "Crypto-Specific",
};

export const RISK_LEVEL_LABELS: Record<RiskLevel, string> = {
  conservative: "Conservative",
  moderate: "Moderate",
  aggressive: "Aggressive",
};

export const RISK_LEVEL_COLORS: Record<RiskLevel, string> = {
  conservative: "text-green-400",
  moderate: "text-yellow-400",
  aggressive: "text-red-400",
};
