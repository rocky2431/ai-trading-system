"""Strategy module for IQFMP.

Provides tools for strategy generation and management:
- Strategy templates and code generation
- Strategy validation
- Weighting schemes
- Position management

IMPORTANT: Backtesting has been moved to iqfmp.core.crypto_backtest
==================================================================
The old BacktestEngine has been removed. Use the new unified engine:

    from iqfmp.core.crypto_backtest import (
        CryptoQlibBacktest,
        CryptoBacktestConfig,
        CryptoBacktestResult,
    )

    config = CryptoBacktestConfig(initial_capital=100000, leverage=10)
    engine = CryptoQlibBacktest(config)
    result = engine.run(data, signals, "ETHUSDT")

The new engine provides:
- Qlib C++ expression engine for factor computation
- Funding Rate settlement (8h cycles)
- MarginCalculator for liquidation detection
- Multi-asset portfolio support
"""

from iqfmp.strategy.generator import (
    GeneratedStrategy,
    InvalidStrategyError,
    StrategyConfig,
    StrategyGenerator,
    StrategyTemplate,
    StrategyValidator,
    ValidationResult,
    WeightingScheme,
)
from iqfmp.strategy.position import (
    CloseResult,
    FixedSizer,
    FixedTakeProfit,
    InsufficientFundsError,
    InvalidPositionError,
    KellySizer,
    PercentStopLoss,
    Position,
    PositionConfig,
    PositionManager,
    PositionSide,
    PositionSizer,
    PositionStatus,
    PriceStopLoss,
    RiskParitySizer,
    StopLoss,
    TakeProfit,
    TimeStopLoss,
    TrailingStopLoss,
    TrailingTakeProfit,
)

__all__ = [
    # Generator
    "GeneratedStrategy",
    "InvalidStrategyError",
    "StrategyConfig",
    "StrategyGenerator",
    "StrategyTemplate",
    "StrategyValidator",
    "ValidationResult",
    "WeightingScheme",
    # Position
    "CloseResult",
    "FixedSizer",
    "FixedTakeProfit",
    "InsufficientFundsError",
    "InvalidPositionError",
    "KellySizer",
    "PercentStopLoss",
    "Position",
    "PositionConfig",
    "PositionManager",
    "PositionSide",
    "PositionSizer",
    "PositionStatus",
    "PriceStopLoss",
    "RiskParitySizer",
    "StopLoss",
    "TakeProfit",
    "TimeStopLoss",
    "TrailingStopLoss",
    "TrailingTakeProfit",
]
