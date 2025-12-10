"""Strategy module for IQFMP.

Provides tools for strategy generation and management:
- Strategy templates and code generation
- Strategy validation
- Weighting schemes
- Position management
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
