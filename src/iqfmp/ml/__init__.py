"""IQFMP Machine Learning Module.

This module provides:
- Bayesian hyperparameter optimization (Optuna)
- Model wrappers for LightGBM, CatBoost, XGBoost, TabNet
- Strategy optimization framework
"""

from .optimizer import (
    # Config and results
    OptimizationConfig,
    OptimizationResult,
    OptimizationMetric,
    SamplerType,
    PrunerType,
    # Base class
    BaseModelOptimizer,
    # Model-specific optimizers
    LightGBMOptimizer,
    CatBoostOptimizer,
    XGBoostOptimizer,
    TabNetOptimizer,
    StrategyOptimizer,
    # Factory functions
    create_optimizer,
    quick_optimize,
    # Availability flags
    OPTUNA_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    CATBOOST_AVAILABLE,
    XGBOOST_AVAILABLE,
    TABNET_AVAILABLE,
)

__all__ = [
    # Config and results
    "OptimizationConfig",
    "OptimizationResult",
    "OptimizationMetric",
    "SamplerType",
    "PrunerType",
    # Base class
    "BaseModelOptimizer",
    # Model-specific optimizers
    "LightGBMOptimizer",
    "CatBoostOptimizer",
    "XGBoostOptimizer",
    "TabNetOptimizer",
    "StrategyOptimizer",
    # Factory functions
    "create_optimizer",
    "quick_optimize",
    # Availability flags
    "OPTUNA_AVAILABLE",
    "LIGHTGBM_AVAILABLE",
    "CATBOOST_AVAILABLE",
    "XGBOOST_AVAILABLE",
    "TABNET_AVAILABLE",
]
