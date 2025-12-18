"""Optuna-based Bayesian Hyperparameter Optimization.

This module provides Bayesian optimization for ML model hyperparameters
using Optuna, with support for:
- LightGBM
- CatBoost
- XGBoost
- TabNet
- Custom models

Features:
- Multi-objective optimization (Sharpe, IC, drawdown)
- Early stopping and pruning
- Cross-validation
- Walk-forward optimization
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Type

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.samplers import TPESampler, CmaEsSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False


logger = logging.getLogger(__name__)


class OptimizationMetric(str, Enum):
    """Optimization objectives."""
    SHARPE = "sharpe"
    IC = "ic"
    IR = "ir"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    RMSE = "rmse"
    MAE = "mae"
    AUC = "auc"


class SamplerType(str, Enum):
    """Optuna sampler types."""
    TPE = "tpe"  # Tree-structured Parzen Estimator (default, good for most)
    CMAES = "cmaes"  # Covariance Matrix Adaptation Evolution Strategy
    RANDOM = "random"  # Random sampling (baseline)


class PrunerType(str, Enum):
    """Optuna pruner types."""
    MEDIAN = "median"  # Prune if worse than median
    HYPERBAND = "hyperband"  # Hyperband pruning
    NONE = "none"


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    n_trials: int = 100
    n_jobs: int = 1
    timeout: Optional[float] = None  # seconds
    metric: OptimizationMetric = OptimizationMetric.SHARPE
    direction: str = "maximize"  # "maximize" or "minimize"
    sampler: SamplerType = SamplerType.TPE
    pruner: PrunerType = PrunerType.MEDIAN
    cv_folds: int = 5
    early_stopping_rounds: int = 50
    seed: int = 42
    study_name: Optional[str] = None
    storage: Optional[str] = None  # SQLite/PostgreSQL URL for persistence


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    best_params: dict[str, Any]
    best_value: float
    best_trial_number: int
    n_trials: int
    optimization_history: list[float]
    param_importance: dict[str, float] = field(default_factory=dict)
    all_trials: list[dict] = field(default_factory=list)


class BaseModelOptimizer(ABC):
    """Abstract base class for model-specific hyperparameter optimization."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self._study: Optional["optuna.Study"] = None

    @abstractmethod
    def get_param_space(self, trial: "optuna.Trial") -> dict[str, Any]:
        """Define the hyperparameter search space.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of hyperparameters
        """
        pass

    @abstractmethod
    def train_and_evaluate(
        self,
        params: dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Train model and return evaluation metric.

        Args:
            params: Hyperparameters
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Evaluation metric value
        """
        pass

    def _create_study(self) -> "optuna.Study":
        """Create Optuna study with configured sampler and pruner."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Install with: pip install optuna")

        # Create sampler
        if self.config.sampler == SamplerType.TPE:
            sampler = TPESampler(seed=self.config.seed)
        elif self.config.sampler == SamplerType.CMAES:
            sampler = CmaEsSampler(seed=self.config.seed)
        else:
            sampler = optuna.samplers.RandomSampler(seed=self.config.seed)

        # Create pruner
        if self.config.pruner == PrunerType.MEDIAN:
            pruner = MedianPruner()
        elif self.config.pruner == PrunerType.HYPERBAND:
            pruner = HyperbandPruner()
        else:
            pruner = optuna.pruners.NopPruner()

        study = optuna.create_study(
            study_name=self.config.study_name or "iqfmp_optimization",
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner,
            storage=self.config.storage,
            load_if_exists=True,
        )

        return study

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> OptimizationResult:
        """Run hyperparameter optimization.

        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional, will use CV if not provided)
            y_val: Validation targets
            callback: Optional callback(trial_number, best_value)

        Returns:
            OptimizationResult with best parameters and history
        """
        self._study = self._create_study()

        def objective(trial: "optuna.Trial") -> float:
            params = self.get_param_space(trial)

            if X_val is not None and y_val is not None:
                # Use provided validation set
                score = self.train_and_evaluate(params, X, y, X_val, y_val)
            else:
                # Use cross-validation
                score = self._cross_validate(params, X, y)

            if callback:
                callback(trial.number, self._study.best_value if self._study.best_trial else score)

            return score

        # Run optimization
        self._study.optimize(
            objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
            timeout=self.config.timeout,
            show_progress_bar=True,
        )

        # Get parameter importance
        try:
            importance = optuna.importance.get_param_importances(self._study)
        except Exception:
            importance = {}

        # Collect all trial results
        all_trials = [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in self._study.trials
        ]

        return OptimizationResult(
            best_params=self._study.best_params,
            best_value=self._study.best_value,
            best_trial_number=self._study.best_trial.number,
            n_trials=len(self._study.trials),
            optimization_history=[t.value for t in self._study.trials if t.value is not None],
            param_importance=importance,
            all_trials=all_trials,
        )

    def _cross_validate(
        self,
        params: dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Run time-series cross-validation.

        Args:
            params: Hyperparameters
            X: Features
            y: Targets

        Returns:
            Average validation score
        """
        n_samples = len(X)
        fold_size = n_samples // (self.config.cv_folds + 1)

        scores = []
        for fold in range(self.config.cv_folds):
            train_end = fold_size * (fold + 1)
            val_start = train_end
            val_end = min(val_start + fold_size, n_samples)

            if val_end <= val_start:
                continue

            X_train = X[:train_end]
            y_train = y[:train_end]
            X_val = X[val_start:val_end]
            y_val = y[val_start:val_end]

            score = self.train_and_evaluate(params, X_train, y_train, X_val, y_val)
            scores.append(score)

        return np.mean(scores) if scores else 0.0


class LightGBMOptimizer(BaseModelOptimizer):
    """LightGBM hyperparameter optimizer."""

    def __init__(
        self,
        config: OptimizationConfig,
        task: str = "regression",  # "regression" or "classification"
    ):
        super().__init__(config)
        self.task = task

    def get_param_space(self, trial: "optuna.Trial") -> dict[str, Any]:
        """Define LightGBM hyperparameter search space."""
        return {
            "objective": "regression" if self.task == "regression" else "binary",
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 8, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "verbose": -1,
            "n_jobs": -1,
            "random_state": self.config.seed,
        }

    def train_and_evaluate(
        self,
        params: dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Train LightGBM and evaluate."""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")

        model = lgb.LGBMRegressor(**params) if self.task == "regression" else lgb.LGBMClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)],
        )

        predictions = model.predict(X_val)
        return self._calculate_metric(y_val, predictions)

    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the optimization metric."""
        if self.config.metric == OptimizationMetric.RMSE:
            return -np.sqrt(np.mean((y_true - y_pred) ** 2))  # Negative for maximization
        elif self.config.metric == OptimizationMetric.MAE:
            return -np.mean(np.abs(y_true - y_pred))
        elif self.config.metric == OptimizationMetric.IC:
            return np.corrcoef(y_true, y_pred)[0, 1]
        else:
            # Default: correlation as proxy for predictive power
            return np.corrcoef(y_true, y_pred)[0, 1]


class CatBoostOptimizer(BaseModelOptimizer):
    """CatBoost hyperparameter optimizer."""

    def __init__(
        self,
        config: OptimizationConfig,
        task: str = "regression",
    ):
        super().__init__(config)
        self.task = task

    def get_param_space(self, trial: "optuna.Trial") -> dict[str, Any]:
        """Define CatBoost hyperparameter search space."""
        return {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "verbose": 0,
            "random_seed": self.config.seed,
            "allow_writing_files": False,
        }

    def train_and_evaluate(
        self,
        params: dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Train CatBoost and evaluate."""
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed. Install with: pip install catboost")

        model = CatBoostRegressor(**params) if self.task == "regression" else CatBoostClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose=0,
        )

        predictions = model.predict(X_val)
        return self._calculate_metric(y_val, predictions)

    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the optimization metric."""
        if self.config.metric == OptimizationMetric.RMSE:
            return -np.sqrt(np.mean((y_true - y_pred) ** 2))
        elif self.config.metric == OptimizationMetric.MAE:
            return -np.mean(np.abs(y_true - y_pred))
        elif self.config.metric == OptimizationMetric.IC:
            return np.corrcoef(y_true, y_pred)[0, 1]
        else:
            return np.corrcoef(y_true, y_pred)[0, 1]


class XGBoostOptimizer(BaseModelOptimizer):
    """XGBoost hyperparameter optimizer."""

    def __init__(
        self,
        config: OptimizationConfig,
        task: str = "regression",
    ):
        super().__init__(config)
        self.task = task

    def get_param_space(self, trial: "optuna.Trial") -> dict[str, Any]:
        """Define XGBoost hyperparameter search space."""
        return {
            "objective": "reg:squarederror" if self.task == "regression" else "binary:logistic",
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "verbosity": 0,
            "n_jobs": -1,
            "random_state": self.config.seed,
        }

    def train_and_evaluate(
        self,
        params: dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Train XGBoost and evaluate."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")

        model = xgb.XGBRegressor(**params) if self.task == "regression" else xgb.XGBClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        predictions = model.predict(X_val)
        return self._calculate_metric(y_val, predictions)

    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the optimization metric."""
        if self.config.metric == OptimizationMetric.RMSE:
            return -np.sqrt(np.mean((y_true - y_pred) ** 2))
        elif self.config.metric == OptimizationMetric.MAE:
            return -np.mean(np.abs(y_true - y_pred))
        elif self.config.metric == OptimizationMetric.IC:
            return np.corrcoef(y_true, y_pred)[0, 1]
        else:
            return np.corrcoef(y_true, y_pred)[0, 1]


class TabNetOptimizer(BaseModelOptimizer):
    """TabNet hyperparameter optimizer for tabular deep learning."""

    def __init__(
        self,
        config: OptimizationConfig,
        task: str = "regression",
    ):
        super().__init__(config)
        self.task = task

    def get_param_space(self, trial: "optuna.Trial") -> dict[str, Any]:
        """Define TabNet hyperparameter search space."""
        n_d = trial.suggest_int("n_d", 8, 64)
        return {
            "n_d": n_d,
            "n_a": n_d,  # Usually set equal to n_d
            "n_steps": trial.suggest_int("n_steps", 3, 10),
            "gamma": trial.suggest_float("gamma", 1.0, 2.0),
            "n_independent": trial.suggest_int("n_independent", 1, 5),
            "n_shared": trial.suggest_int("n_shared", 1, 5),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
            "momentum": trial.suggest_float("momentum", 0.01, 0.4),
            "mask_type": trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
            "seed": self.config.seed,
            "verbose": 0,
        }

    def train_and_evaluate(
        self,
        params: dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Train TabNet and evaluate."""
        if not TABNET_AVAILABLE:
            raise ImportError("TabNet is not installed. Install with: pip install pytorch-tabnet")

        # Reshape y if needed
        y_train_reshaped = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
        y_val_reshaped = y_val.reshape(-1, 1) if y_val.ndim == 1 else y_val

        model = TabNetRegressor(**params) if self.task == "regression" else TabNetClassifier(**params)

        model.fit(
            X_train, y_train_reshaped,
            eval_set=[(X_val, y_val_reshaped)],
            max_epochs=200,
            patience=self.config.early_stopping_rounds,
            batch_size=1024,
            virtual_batch_size=128,
        )

        predictions = model.predict(X_val)
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        return self._calculate_metric(y_val, predictions)

    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the optimization metric."""
        if self.config.metric == OptimizationMetric.RMSE:
            return -np.sqrt(np.mean((y_true - y_pred) ** 2))
        elif self.config.metric == OptimizationMetric.MAE:
            return -np.mean(np.abs(y_true - y_pred))
        elif self.config.metric == OptimizationMetric.IC:
            return np.corrcoef(y_true, y_pred)[0, 1]
        else:
            return np.corrcoef(y_true, y_pred)[0, 1]


class StrategyOptimizer(BaseModelOptimizer):
    """Optimizer for trading strategy hyperparameters."""

    def __init__(
        self,
        config: OptimizationConfig,
        strategy_fn: Callable[[dict, pd.DataFrame], dict],
        data: pd.DataFrame,
    ):
        """Initialize strategy optimizer.

        Args:
            config: Optimization configuration
            strategy_fn: Function that takes (params, data) and returns metrics dict
            data: Historical data for backtesting
        """
        super().__init__(config)
        self.strategy_fn = strategy_fn
        self.data = data
        self._param_definitions: list[dict] = []

    def add_param(
        self,
        name: str,
        param_type: str,
        low: float,
        high: float,
        **kwargs,
    ):
        """Add a parameter to the search space.

        Args:
            name: Parameter name
            param_type: "int", "float", "log_float", "categorical"
            low: Lower bound (or list for categorical)
            high: Upper bound
            **kwargs: Additional arguments
        """
        self._param_definitions.append({
            "name": name,
            "type": param_type,
            "low": low,
            "high": high,
            **kwargs,
        })

    def get_param_space(self, trial: "optuna.Trial") -> dict[str, Any]:
        """Build parameter space from definitions."""
        params = {}
        for p in self._param_definitions:
            if p["type"] == "int":
                params[p["name"]] = trial.suggest_int(p["name"], int(p["low"]), int(p["high"]))
            elif p["type"] == "float":
                params[p["name"]] = trial.suggest_float(p["name"], p["low"], p["high"])
            elif p["type"] == "log_float":
                params[p["name"]] = trial.suggest_float(p["name"], p["low"], p["high"], log=True)
            elif p["type"] == "categorical":
                params[p["name"]] = trial.suggest_categorical(p["name"], p["low"])  # low is choices list
        return params

    def train_and_evaluate(
        self,
        params: dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Run strategy backtest and return metric."""
        # For strategy optimization, we don't use X/y but the full data
        metrics = self.strategy_fn(params, self.data)

        # Extract the configured metric
        metric_name = self.config.metric.value
        if metric_name in metrics:
            return metrics[metric_name]
        elif "sharpe" in metrics:
            return metrics["sharpe"]
        else:
            return 0.0

    def optimize_strategy(
        self,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> OptimizationResult:
        """Run strategy optimization (doesn't need X/y).

        Args:
            callback: Optional callback(trial_number, best_value)

        Returns:
            OptimizationResult
        """
        # Use dummy arrays since we use the strategy_fn directly
        dummy = np.zeros((10, 1))
        return self.optimize(dummy, dummy[:, 0], callback=callback)


# =============================================================================
# Factory Functions
# =============================================================================


def create_optimizer(
    model_type: str,
    config: Optional[OptimizationConfig] = None,
    **kwargs,
) -> BaseModelOptimizer:
    """Factory function to create an optimizer.

    Args:
        model_type: "lightgbm", "catboost", "xgboost", or "tabnet"
        config: Optimization configuration
        **kwargs: Additional arguments for the optimizer

    Returns:
        Configured optimizer instance
    """
    if config is None:
        config = OptimizationConfig()

    model_type_lower = model_type.lower()
    if model_type_lower == "lightgbm":
        return LightGBMOptimizer(config, **kwargs)
    elif model_type_lower == "catboost":
        return CatBoostOptimizer(config, **kwargs)
    elif model_type_lower == "xgboost":
        return XGBoostOptimizer(config, **kwargs)
    elif model_type_lower == "tabnet":
        return TabNetOptimizer(config, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported: lightgbm, catboost, xgboost, tabnet")


def quick_optimize(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    metric: OptimizationMetric = OptimizationMetric.IC,
) -> dict[str, Any]:
    """Quick hyperparameter optimization with sensible defaults.

    Args:
        model_type: "lightgbm" or "catboost"
        X: Features
        y: Targets
        n_trials: Number of optimization trials
        metric: Optimization metric

    Returns:
        Best hyperparameters
    """
    config = OptimizationConfig(
        n_trials=n_trials,
        metric=metric,
        early_stopping_rounds=30,
    )

    optimizer = create_optimizer(model_type, config)
    result = optimizer.optimize(X, y)

    logger.info(f"Best {metric.value}: {result.best_value:.4f}")
    logger.info(f"Best params: {result.best_params}")

    return result.best_params
