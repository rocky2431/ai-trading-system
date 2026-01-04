"""Optuna-based Backtest Parameter Optimization.

This module provides hyperparameter optimization for backtest configurations,
bridging UnifiedBacktestRunner with Optuna for automated parameter tuning.

Features:
- Multi-metric optimization (Sharpe, Calmar, Return, IC)
- Configurable search spaces for all backtest parameters
- Real-time progress tracking via Redis
- PostgreSQL persistence for study resumption
- Pruning for early stopping of unpromising trials
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

if TYPE_CHECKING:
    import redis

try:
    import optuna
    from optuna.pruners import HyperbandPruner, MedianPruner, PercentilePruner
    from optuna.samplers import CmaEsSampler, GridSampler, RandomSampler, TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None  # type: ignore[assignment]

from iqfmp.core.unified_backtest import (
    BacktestMode,
    CryptoBacktestResult,
    UnifiedBacktestParams,
    UnifiedBacktestRunner,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Configuration
# =============================================================================


class BacktestOptimizationMetric(str, Enum):
    """Optimization objective metrics for backtest."""

    SHARPE = "sharpe"
    CALMAR = "calmar"  # Annualized Return / Max Drawdown
    TOTAL_RETURN = "total_return"
    ANNUAL_RETURN = "annual_return"
    SORTINO = "sortino"
    MAX_DRAWDOWN = "max_drawdown"  # Minimize
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"


class BacktestSamplerType(str, Enum):
    """Optuna sampler types."""

    TPE = "tpe"  # Tree-structured Parzen Estimator (default)
    CMAES = "cmaes"  # Covariance Matrix Adaptation Evolution Strategy
    RANDOM = "random"  # Random sampling (baseline)
    GRID = "grid"  # Grid search (exhaustive)


class BacktestPrunerType(str, Enum):
    """Optuna pruner types."""

    MEDIAN = "median"
    HYPERBAND = "hyperband"
    PERCENTILE = "percentile"
    NONE = "none"


@dataclass
class BacktestSearchSpace:
    """Definition of a single parameter search space."""

    name: str
    param_type: str  # "float", "int", "categorical"
    bounds: tuple[float, float] | list[Any]
    log_scale: bool = False
    step: float | None = None
    enabled: bool = True  # Allow disabling specific parameters

    def __post_init__(self) -> None:
        """Validate search space at construction time."""
        self.validate()

    def validate(self) -> None:
        """Validate search space configuration."""
        if not self.enabled:
            return

        if self.param_type in ("float", "int"):
            if not isinstance(self.bounds, tuple) or len(self.bounds) != 2:
                raise ValueError(f"{self.name}: bounds must be (low, high) tuple")
            if self.bounds[0] >= self.bounds[1]:
                raise ValueError(f"{self.name}: low must be less than high")
        elif self.param_type == "categorical":
            if not isinstance(self.bounds, list) or len(self.bounds) < 2:
                raise ValueError(f"{self.name}: categorical needs at least 2 choices")
        else:
            raise ValueError(f"Unknown param_type: {self.param_type}")


@dataclass
class BacktestOptimizationConfig:
    """Configuration for backtest optimization."""

    # Optimization settings
    n_trials: int = 100
    n_jobs: int = 1
    timeout: float | None = None  # seconds
    metric: BacktestOptimizationMetric = BacktestOptimizationMetric.SHARPE
    direction: str = "maximize"  # "maximize" or "minimize"
    sampler: BacktestSamplerType = BacktestSamplerType.TPE
    pruner: BacktestPrunerType = BacktestPrunerType.MEDIAN

    # Search spaces (override defaults)
    custom_search_spaces: list[BacktestSearchSpace] = field(default_factory=list)

    # Study settings
    study_name: str | None = None
    storage: str | None = None  # SQLite/PostgreSQL URL

    # Callbacks
    seed: int = 42
    show_progress_bar: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.n_trials < 1:
            raise ValueError("n_trials must be at least 1")
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be at least 1")
        if self.direction not in ("maximize", "minimize"):
            raise ValueError("direction must be 'maximize' or 'minimize'")

        # Validate custom search spaces
        for space in self.custom_search_spaces:
            space.validate()


@dataclass
class BacktestTrialResult:
    """Result of a single optimization trial."""

    trial_number: int
    params: dict[str, Any]
    metric_value: float
    metric_name: str
    duration_seconds: float
    status: str = "completed"  # completed, failed, pruned
    backtest_result: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestOptimizationResult:
    """Complete result of backtest optimization."""

    optimization_id: str
    best_params: dict[str, Any]
    best_value: float
    best_trial_number: int
    n_trials_completed: int
    n_trials_total: int
    metric_name: str
    direction: str
    optimization_history: list[float]
    param_importance: dict[str, float]
    all_trials: list[BacktestTrialResult]
    started_at: datetime
    completed_at: datetime
    duration_seconds: float


# =============================================================================
# Default Search Spaces
# =============================================================================


def get_default_search_spaces(mode: BacktestMode) -> list[BacktestSearchSpace]:
    """Get default search spaces based on backtest mode."""
    # Common parameters
    common = [
        BacktestSearchSpace(
            name="commission_rate",
            param_type="float",
            bounds=(0.0001, 0.005),
            log_scale=True,
        ),
        BacktestSearchSpace(
            name="slippage_rate",
            param_type="float",
            bounds=(0.0, 0.005),
        ),
        BacktestSearchSpace(
            name="max_position_pct",
            param_type="float",
            bounds=(0.3, 1.0),
        ),
    ]

    # Mode-specific parameters
    if mode == BacktestMode.CRYPTO:
        crypto_params = [
            BacktestSearchSpace(
                name="leverage",
                param_type="int",
                bounds=(1, 10),
            ),
            BacktestSearchSpace(
                name="funding_enabled",
                param_type="categorical",
                bounds=[True, False],
            ),
        ]
        return common + crypto_params

    elif mode in (BacktestMode.QLIB, BacktestMode.NESTED):
        qlib_params = [
            BacktestSearchSpace(
                name="topk",
                param_type="int",
                bounds=(10, 100),
                step=5,
            ),
            BacktestSearchSpace(
                name="n_drop",
                param_type="int",
                bounds=(1, 20),
            ),
        ]
        return common + qlib_params

    # Standard mode
    return common


# =============================================================================
# BacktestOptimizer
# =============================================================================


class BacktestOptimizer:
    """Optuna-based backtest parameter optimization.

    This class bridges UnifiedBacktestRunner with Optuna to enable
    automated hyperparameter tuning for backtest configurations.

    Example:
        >>> optimizer = BacktestOptimizer(
        ...     signals=factor_signals,
        ...     price_data=ohlcv_data,
        ...     config=BacktestOptimizationConfig(n_trials=50),
        ... )
        >>> result = optimizer.optimize()
        >>> print(f"Best Sharpe: {result.best_value:.3f}")
        >>> print(f"Best params: {result.best_params}")
    """

    def __init__(
        self,
        signals: pd.DataFrame,
        price_data: pd.DataFrame,
        config: BacktestOptimizationConfig | None = None,
        base_params: UnifiedBacktestParams | None = None,
        funding_rates: pd.DataFrame | None = None,
        redis_client: redis.Redis | None = None,  # type: ignore[name-defined]
    ) -> None:
        """Initialize optimizer.

        Args:
            signals: Factor signals DataFrame (datetime index, asset columns)
            price_data: OHLCV price data
            config: Optimization configuration
            base_params: Base backtest params (non-optimized values)
            funding_rates: Optional funding rate data for crypto mode
            redis_client: Optional Redis client for progress tracking
        """
        if not OPTUNA_AVAILABLE:
            raise RuntimeError(
                "Optuna is required for backtest optimization. "
                "Install with: pip install optuna"
            )

        self.signals = signals
        self.price_data = price_data
        self.config = config or BacktestOptimizationConfig()
        self.base_params = base_params or UnifiedBacktestParams()
        self.funding_rates = funding_rates
        self.redis_client = redis_client

        # Initialize runner
        self._runner = UnifiedBacktestRunner()

        # Generate optimization ID
        self._optimization_id = str(uuid.uuid4())[:8]

        # Build search spaces
        self._search_spaces = self._build_search_spaces()

        # Track trials
        self._trial_results: list[BacktestTrialResult] = []
        self._start_time: datetime | None = None

    def _build_search_spaces(self) -> list[BacktestSearchSpace]:
        """Build search spaces from defaults and custom overrides."""
        # Get defaults for mode
        defaults = get_default_search_spaces(self.base_params.mode)

        # Apply custom overrides
        if self.config.custom_search_spaces:
            custom_names = {s.name for s in self.config.custom_search_spaces}
            # Keep defaults not overridden
            filtered = [s for s in defaults if s.name not in custom_names]
            return filtered + self.config.custom_search_spaces

        return defaults

    def _create_sampler(self) -> Any:
        """Create Optuna sampler based on config."""
        if self.config.sampler == BacktestSamplerType.TPE:
            return TPESampler(seed=self.config.seed)
        elif self.config.sampler == BacktestSamplerType.CMAES:
            return CmaEsSampler(seed=self.config.seed)
        elif self.config.sampler == BacktestSamplerType.GRID:
            # Build grid search space from search spaces
            search_space: dict[str, list[Any]] = {}
            for space in self._search_spaces:
                if not space.enabled:
                    continue
                if space.param_type == "categorical":
                    search_space[space.name] = list(space.bounds)
                elif space.param_type in ("float", "int"):
                    low, high = space.bounds  # type: ignore
                    step = space.step or (high - low) / 10
                    if space.param_type == "int":
                        search_space[space.name] = list(
                            range(int(low), int(high) + 1, int(step) or 1)
                        )
                    else:
                        import numpy as np

                        search_space[space.name] = list(
                            np.arange(low, high + step, step)
                        )
            return GridSampler(search_space, seed=self.config.seed)
        else:
            return RandomSampler(seed=self.config.seed)

    def _create_pruner(self) -> Any:
        """Create Optuna pruner based on config."""
        if self.config.pruner == BacktestPrunerType.MEDIAN:
            return MedianPruner()
        elif self.config.pruner == BacktestPrunerType.HYPERBAND:
            return HyperbandPruner()
        elif self.config.pruner == BacktestPrunerType.PERCENTILE:
            return PercentilePruner(percentile=25.0)
        else:
            return None

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest parameters from search spaces."""
        params: dict[str, Any] = {}

        for space in self._search_spaces:
            if not space.enabled:
                continue

            if space.param_type == "float":
                low, high = space.bounds  # type: ignore
                params[space.name] = trial.suggest_float(
                    space.name,
                    low,
                    high,
                    log=space.log_scale,
                )
            elif space.param_type == "int":
                low, high = space.bounds  # type: ignore
                step = int(space.step) if space.step else 1
                params[space.name] = trial.suggest_int(
                    space.name,
                    int(low),
                    int(high),
                    step=step,
                )
            elif space.param_type == "categorical":
                params[space.name] = trial.suggest_categorical(
                    space.name,
                    space.bounds,  # type: ignore
                )

        return params

    def _build_params(self, suggested: dict[str, Any]) -> UnifiedBacktestParams:
        """Build UnifiedBacktestParams from suggested values."""
        # Start with base params as dict
        base_dict = {
            "start_time": self.base_params.start_time,
            "end_time": self.base_params.end_time,
            "initial_capital": self.base_params.initial_capital,
            "mode": self.base_params.mode,
            "nested_config": self.base_params.nested_config,
            "benchmark": self.base_params.benchmark,
            "strict_cv_mode": self.base_params.strict_cv_mode,
        }

        # Override with suggested params
        for key, value in suggested.items():
            if hasattr(UnifiedBacktestParams, key):
                base_dict[key] = value

        return UnifiedBacktestParams(**base_dict)

    def _extract_metric(
        self,
        result: CryptoBacktestResult | dict[str, Any],
    ) -> float:
        """Extract optimization metric from backtest result."""
        metric = self.config.metric

        # Handle CryptoBacktestResult
        if isinstance(result, CryptoBacktestResult):
            if metric == BacktestOptimizationMetric.SHARPE:
                return result.sharpe_ratio
            elif metric == BacktestOptimizationMetric.CALMAR:
                if result.max_drawdown > 0:
                    return result.annualized_return / result.max_drawdown
                return 0.0
            elif metric == BacktestOptimizationMetric.TOTAL_RETURN:
                return result.total_return
            elif metric == BacktestOptimizationMetric.ANNUAL_RETURN:
                return result.annualized_return
            elif metric == BacktestOptimizationMetric.MAX_DRAWDOWN:
                return -result.max_drawdown  # Negate for maximization
            else:
                return result.sharpe_ratio

        # Handle dict result
        result_dict = result
        if metric == BacktestOptimizationMetric.SHARPE:
            return float(result_dict.get("sharpe_ratio", 0.0))
        elif metric == BacktestOptimizationMetric.CALMAR:
            ret = float(result_dict.get("annual_return", 0.0))
            dd = float(result_dict.get("max_drawdown", 1.0))
            return ret / dd if dd > 0 else 0.0
        elif metric == BacktestOptimizationMetric.TOTAL_RETURN:
            return float(result_dict.get("total_return", 0.0))
        elif metric == BacktestOptimizationMetric.ANNUAL_RETURN:
            return float(result_dict.get("annual_return", 0.0))
        elif metric == BacktestOptimizationMetric.MAX_DRAWDOWN:
            return -float(result_dict.get("max_drawdown", 0.0))
        elif metric == BacktestOptimizationMetric.WIN_RATE:
            return float(result_dict.get("win_rate", 0.0))
        elif metric == BacktestOptimizationMetric.PROFIT_FACTOR:
            return float(result_dict.get("profit_factor", 0.0))

        return 0.0

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        import time

        start = time.time()

        # Suggest parameters
        suggested = self._suggest_params(trial)

        # Build params
        params = self._build_params(suggested)

        # Run backtest
        try:
            result = self._runner.run(
                signals=self.signals,
                data=self.price_data,
                params=params,
                funding_rates=self.funding_rates,
            )

            # Extract metric
            metric_value = self._extract_metric(result)

            # Record trial
            duration = time.time() - start
            trial_result = BacktestTrialResult(
                trial_number=trial.number,
                params=suggested,
                metric_value=metric_value,
                metric_name=self.config.metric.value,
                duration_seconds=duration,
                status="completed",
                backtest_result=result if isinstance(result, dict) else result.__dict__,
            )
            self._trial_results.append(trial_result)

            # Update progress in Redis
            self._update_progress(trial.number + 1, metric_value)

            return metric_value

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")

            # Record failed trial
            duration = time.time() - start
            trial_result = BacktestTrialResult(
                trial_number=trial.number,
                params=suggested,
                metric_value=float("-inf"),
                metric_name=self.config.metric.value,
                duration_seconds=duration,
                status="failed",
            )
            self._trial_results.append(trial_result)

            # Return worst possible value
            return float("-inf") if self.config.direction == "maximize" else float("inf")

    def _update_progress(self, completed: int, best_value: float) -> None:
        """Update progress in Redis if available."""
        if self.redis_client is None:
            return

        try:
            progress_data = {
                "optimization_id": self._optimization_id,
                "completed": completed,
                "total": self.config.n_trials,
                "progress_pct": completed / self.config.n_trials * 100,
                "best_value": best_value,
                "metric": self.config.metric.value,
                "updated_at": datetime.now().isoformat(),
            }
            self.redis_client.set(
                f"backtest_optimization:{self._optimization_id}:progress",
                json.dumps(progress_data),
                ex=3600,  # 1 hour expiry
            )
        except Exception as e:
            logger.warning(f"Failed to update Redis progress: {e}")

    def optimize(
        self,
        callback: Callable[[int, float], None] | None = None,
    ) -> BacktestOptimizationResult:
        """Run optimization.

        Args:
            callback: Optional callback(trial_number, best_value) called after each trial

        Returns:
            BacktestOptimizationResult with best params and all trial data
        """
        self._start_time = datetime.now()
        self._trial_results = []

        # Create study
        study_name = self.config.study_name or f"backtest_opt_{self._optimization_id}"

        study = optuna.create_study(
            study_name=study_name,
            direction=self.config.direction,
            sampler=self._create_sampler(),
            pruner=self._create_pruner(),
            storage=self.config.storage,
            load_if_exists=True,
        )

        # Create callback wrapper if provided
        callbacks = []
        if callback:

            def _callback(study: optuna.Study, trial: optuna.Trial) -> None:
                callback(trial.number, study.best_value)

            callbacks.append(_callback)

        # Run optimization
        study.optimize(
            self._objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
            timeout=self.config.timeout,
            show_progress_bar=self.config.show_progress_bar,
            callbacks=callbacks or None,
        )

        # Extract results
        completed_at = datetime.now()
        duration = (completed_at - self._start_time).total_seconds()

        # Get parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
        except Exception as e:
            logger.warning(f"Failed to calculate parameter importance: {e}")
            importance = {}

        # Build result
        return BacktestOptimizationResult(
            optimization_id=self._optimization_id,
            best_params=study.best_params,
            best_value=study.best_value,
            best_trial_number=study.best_trial.number,
            n_trials_completed=len(self._trial_results),
            n_trials_total=self.config.n_trials,
            metric_name=self.config.metric.value,
            direction=self.config.direction,
            optimization_history=[t.metric_value for t in self._trial_results],
            param_importance=importance,
            all_trials=self._trial_results,
            started_at=self._start_time,
            completed_at=completed_at,
            duration_seconds=duration,
        )

    def get_search_space_info(self) -> list[dict[str, Any]]:
        """Get information about current search spaces."""
        return [
            {
                "name": s.name,
                "type": s.param_type,
                "bounds": s.bounds,
                "log_scale": s.log_scale,
                "step": s.step,
                "enabled": s.enabled,
            }
            for s in self._search_spaces
        ]


# =============================================================================
# Convenience Functions
# =============================================================================


def optimize_backtest(
    signals: pd.DataFrame,
    price_data: pd.DataFrame,
    n_trials: int = 50,
    metric: str = "sharpe",
    base_params: UnifiedBacktestParams | None = None,
    funding_rates: pd.DataFrame | None = None,
    storage: str | None = None,
) -> BacktestOptimizationResult:
    """Convenience function for backtest optimization.

    Args:
        signals: Factor signals DataFrame
        price_data: OHLCV price data
        n_trials: Number of optimization trials
        metric: Optimization metric ("sharpe", "calmar", "total_return", etc.)
        base_params: Base backtest parameters
        funding_rates: Optional funding rates for crypto
        storage: Optional PostgreSQL/SQLite URL for persistence

    Returns:
        BacktestOptimizationResult
    """
    config = BacktestOptimizationConfig(
        n_trials=n_trials,
        metric=BacktestOptimizationMetric(metric),
        storage=storage,
    )

    optimizer = BacktestOptimizer(
        signals=signals,
        price_data=price_data,
        config=config,
        base_params=base_params,
        funding_rates=funding_rates,
    )

    return optimizer.optimize()


def create_crypto_optimization_config(
    n_trials: int = 100,
    optimize_leverage: bool = True,
    max_leverage: int = 10,
) -> BacktestOptimizationConfig:
    """Create optimization config for crypto backtests.

    Args:
        n_trials: Number of trials
        optimize_leverage: Whether to optimize leverage
        max_leverage: Maximum leverage to try

    Returns:
        BacktestOptimizationConfig
    """
    custom_spaces = []

    if optimize_leverage:
        custom_spaces.append(
            BacktestSearchSpace(
                name="leverage",
                param_type="int",
                bounds=(1, max_leverage),
            )
        )

    return BacktestOptimizationConfig(
        n_trials=n_trials,
        metric=BacktestOptimizationMetric.SHARPE,
        custom_search_spaces=custom_spaces,
    )
