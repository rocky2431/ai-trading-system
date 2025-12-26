"""Signal Converter for bridging pandas factors to Qlib backtest.

This module provides the critical conversion layer between:
- pandas DataFrame factors (from LLM generation)
- Qlib Dataset format (for Qlib backtest engine)

The SignalConverter ensures compatibility between LLM-generated factor code
(which outputs pandas DataFrames) and Qlib's backtest infrastructure
(which expects Dataset format with MultiIndex).

Phase 3 Enhancement: ML-based signal generation using LightGBM.
Instead of simple Z-Score normalization, use LightGBM to predict
forward returns from factor values.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

# ML imports - lazy loaded
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# =============================================================================
# P1 OPTIMIZATION: Qlib Model Zoo Integration
# Use Qlib's native LGBModel instead of raw LightGBM for:
# - Model versioning and persistence
# - Consistent interface with Qlib ecosystem
# - Built-in cross-validation support
# =============================================================================
try:
    from qlib.contrib.model.gbdt import LGBModel as QlibLGBModel
    from qlib.data.dataset.handler import DataHandlerLP
    QLIB_MODEL_AVAILABLE = True
except ImportError:
    QLIB_MODEL_AVAILABLE = False
    QlibLGBModel = None
    DataHandlerLP = None

# =============================================================================
# P2: Optuna for Hyperparameter Optimization
# Supports: bayesian (TPE), random, grid, genetic (NSGA-II)
# =============================================================================
try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler, GridSampler, NSGAIISampler
    OPTUNA_AVAILABLE = True
    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None


class DataFrameDatasetH:
    """Adapter to wrap pandas DataFrame as Qlib DatasetH-compatible object.

    This allows using Qlib's LGBModel.fit() with plain DataFrames
    instead of requiring full Qlib data infrastructure.

    Usage:
        dataset = DataFrameDatasetH(X_train, y_train, X_val, y_val)
        model = QlibLGBModel(...)
        model.fit(dataset)
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ):
        self._X_train = X_train
        self._y_train = y_train
        self._X_val = X_val
        self._y_val = y_val

        # Build segments dict (required by Qlib LGBModel)
        self.segments = {"train": ("train_start", "train_end")}
        if X_val is not None and len(X_val) > 0:
            self.segments["valid"] = ("valid_start", "valid_end")

    def prepare(
        self,
        segment: str,
        col_set: list = None,
        data_key: Any = None,
    ) -> pd.DataFrame:
        """Prepare data for a segment (Qlib DatasetH interface).

        Args:
            segment: "train" or "valid"
            col_set: Column set to return ["feature", "label"] or ["feature"]
            data_key: Ignored (for Qlib compatibility)

        Returns:
            DataFrame with MultiIndex columns: ("feature", col_names) and ("label", "label")
        """
        if segment == "train":
            X, y = self._X_train, self._y_train
        elif segment == "valid":
            if self._X_val is None:
                raise ValueError("No validation data provided")
            X, y = self._X_val, self._y_val
        else:
            raise ValueError(f"Unknown segment: {segment}")

        # Build MultiIndex columns like Qlib expects
        # feature columns: ("feature", col_name)
        # label column: ("label", "label")
        feature_cols = pd.MultiIndex.from_tuples(
            [("feature", col) for col in X.columns]
        )
        X_multi = X.copy()
        X_multi.columns = feature_cols

        if col_set == ["feature"]:
            return X_multi

        # Add label column
        label_df = pd.DataFrame(
            {("label", "label"): y.values},
            index=X.index
        )

        return pd.concat([X_multi, label_df], axis=1)

logger = logging.getLogger(__name__)


@dataclass
class SignalConfig:
    """Configuration for signal conversion."""

    # Normalization
    normalize_method: str = "zscore"  # zscore, minmax, rank, none
    clip_std: float = 3.0  # Clip outliers beyond N std

    # Signal generation
    signal_threshold: float = 0.0  # Threshold for long/short
    top_k: Optional[int] = None  # Top-k selection

    # Position sizing
    position_scale: float = 1.0
    max_position: float = 0.1  # Max position per asset

    # ==========================================================================
    # Phase 3: ML-based signal generation (LightGBM)
    # ==========================================================================
    ml_signal_enabled: bool = False  # Enable ML-based signal generation
    ml_model_type: str = "lightgbm"  # Model type: lightgbm, catboost, xgboost

    # P1 OPTIMIZATION: Use Qlib Model Zoo instead of raw LightGBM
    # Benefits: model versioning, persistence, consistent Qlib interface
    use_qlib_model: bool = True  # Prefer Qlib LGBModel over raw lgb
    ml_lookback_window: int = 60  # Window for feature engineering
    ml_forward_period: int = 5  # Forward period for target calculation
    ml_train_ratio: float = 0.7  # Train/test split ratio
    ml_n_estimators: int = 100  # Number of trees
    ml_max_depth: int = 6  # Maximum tree depth
    ml_learning_rate: float = 0.1  # Learning rate
    ml_min_samples: int = 100  # Minimum samples required for ML
    ml_retrain_frequency: int = 20  # Retrain every N days (0 = train once)
    ml_early_stopping_rounds: int = 20  # Early stopping rounds for LightGBM

    # ==========================================================================
    # P2: Hyperparameter Optimization with Optuna
    # Supports: bayesian (TPE), random, grid, genetic (NSGA-II)
    # ==========================================================================
    ml_optimization_method: str = "none"  # none, bayesian, random, grid, genetic
    ml_optimization_trials: int = 20  # Number of optimization trials (for bayesian/random/genetic)
    ml_optimization_timeout: int = 300  # Timeout in seconds for optimization
    ml_optimization_metric: str = "ic"  # Metric to optimize: ic, sharpe, mse

    # Default LightGBM params (can be overridden)
    ml_params: dict = field(default_factory=lambda: {
        "objective": "regression",
        "boosting_type": "gbdt",
        "verbose": -1,
        "n_jobs": -1,
    })


class SignalConverter:
    """Convert pandas factors to trading signals and Qlib Dataset.

    This is the bridge between LLM-generated factor code and Qlib backtest.

    Usage:
        converter = SignalConverter(config)

        # From factor values to normalized signal
        signal = converter.to_signal(factor_values)

        # From signal to Qlib-compatible Dataset
        dataset = converter.to_qlib_dataset(signal, instruments)

    Phase 3 Enhancement:
        # Enable ML-based signal generation
        config = SignalConfig(ml_signal_enabled=True)
        converter = SignalConverter(config)
        signal = converter.to_signal(factor_values, price_data=df)
    """

    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()

        # Phase 3: ML model state
        self._ml_model: Optional[Any] = None
        self._ml_feature_names: list[str] = []
        self._ml_last_train_idx: int = 0
        self._ml_train_count: int = 0

    def normalize(self, factor: pd.Series) -> pd.Series:
        """Normalize factor values.

        Args:
            factor: Raw factor values

        Returns:
            Normalized factor values
        """
        if self.config.normalize_method == "zscore":
            mean = factor.mean()
            std = factor.std()
            if std == 0 or pd.isna(std):
                return pd.Series(0, index=factor.index)
            normalized = (factor - mean) / std
            # Clip outliers
            return normalized.clip(-self.config.clip_std, self.config.clip_std)

        elif self.config.normalize_method == "minmax":
            min_val = factor.min()
            max_val = factor.max()
            if max_val == min_val:
                return pd.Series(0.5, index=factor.index)
            return (factor - min_val) / (max_val - min_val)

        elif self.config.normalize_method == "rank":
            return factor.rank(pct=True)

        else:  # none
            return factor

    # ==========================================================================
    # Phase 3: ML-based Signal Generation
    # ==========================================================================

    def _build_ml_features(
        self,
        factor: pd.Series,
        price_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build features for ML model from factor and price data.

        Features include:
        - Raw factor value
        - Factor rolling statistics (mean, std, min, max)
        - Factor momentum (diff, pct_change)
        - Price-based features (returns, volatility)

        Args:
            factor: Factor values
            price_data: OHLCV data with 'close' column

        Returns:
            Feature DataFrame aligned with factor index
        """
        window = self.config.ml_lookback_window
        features = pd.DataFrame(index=factor.index)

        # Factor-based features
        features["factor_raw"] = factor
        features["factor_zscore"] = (factor - factor.rolling(window).mean()) / factor.rolling(window).std()
        features["factor_rank"] = factor.rolling(window).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
        features["factor_ma_ratio"] = factor / factor.rolling(window).mean()
        features["factor_momentum"] = factor.diff(5)
        features["factor_momentum_10"] = factor.diff(10)

        # Price-based features (if available)
        if "close" in price_data.columns:
            close = price_data["close"]
            # Align close to factor index
            close = close.reindex(factor.index)

            features["return_1d"] = close.pct_change(1)
            features["return_5d"] = close.pct_change(5)
            features["return_10d"] = close.pct_change(10)
            features["volatility_20d"] = close.pct_change().rolling(20).std()
            features["ma_ratio_20"] = close / close.rolling(20).mean()
            features["ma_ratio_60"] = close / close.rolling(60).mean()

            if "volume" in price_data.columns:
                volume = price_data["volume"].reindex(factor.index)
                features["volume_ma_ratio"] = volume / volume.rolling(20).mean()

        self._ml_feature_names = list(features.columns)
        return features

    def _calculate_target(
        self,
        price_data: pd.DataFrame,
        index: pd.Index,
    ) -> pd.Series:
        """Calculate forward returns as ML target.

        Args:
            price_data: OHLCV data with 'close' column
            index: Index to align target with

        Returns:
            Forward returns Series
        """
        if "close" not in price_data.columns:
            raise ValueError("price_data must contain 'close' column for ML target")

        close = price_data["close"]
        forward_period = self.config.ml_forward_period

        # Forward return
        forward_return = close.shift(-forward_period) / close - 1
        return forward_return.reindex(index)

    def _train_ml_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """Train ML model on features and target.

        P1 OPTIMIZATION: Supports both Qlib LGBModel and raw LightGBM.
        Qlib LGBModel is preferred for better ecosystem integration.

        P2 ENHANCEMENT: Supports hyperparameter optimization via Optuna.
        When ml_optimization_method != "none", optimizes hyperparams before training.

        Args:
            X: Feature DataFrame
            y: Target Series (forward returns)
        """
        # Check availability
        use_qlib = self.config.use_qlib_model and QLIB_MODEL_AVAILABLE
        if not use_qlib and not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "Neither Qlib Model Zoo nor LightGBM is available. "
                "Install with: pip install qlib lightgbm"
            )

        # Prepare data - remove NaN
        valid_mask = X.notna().all(axis=1) & y.notna()
        X_clean = X.loc[valid_mask]
        y_clean = y.loc[valid_mask]

        if len(X_clean) < self.config.ml_min_samples:
            logger.warning(
                f"Insufficient samples for ML training: {len(X_clean)} < {self.config.ml_min_samples}. "
                "Falling back to Z-Score."
            )
            self._ml_model = None
            return

        # Train/test split (time-series aware)
        train_size = int(len(X_clean) * self.config.ml_train_ratio)
        X_train = X_clean.iloc[:train_size]
        y_train = y_clean.iloc[:train_size]
        X_val = X_clean.iloc[train_size:]
        y_val = y_clean.iloc[train_size:]

        # =====================================================================
        # P2: Hyperparameter optimization with Optuna (if enabled)
        # =====================================================================
        optimized_params = None
        if self.config.ml_optimization_method != "none":
            if len(X_val) > 10:
                optimized_params = self._optimize_hyperparameters(
                    X_train, y_train, X_val, y_val
                )
                logger.info(f"Using optimized params: {optimized_params}")
            else:
                logger.warning(
                    "Insufficient validation data for optimization, using defaults"
                )

        # =====================================================================
        # P1: Use Qlib LGBModel if available (preferred)
        # =====================================================================
        if use_qlib:
            self._train_with_qlib_model(X_train, y_train, X_val, y_val, optimized_params)
        else:
            self._train_with_raw_lightgbm(X_train, y_train, X_val, y_val, optimized_params)

        self._ml_train_count += 1

    def _train_with_qlib_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        optimized_params: Optional[dict] = None,
    ) -> None:
        """Train using Qlib's native LGBModel (P1 optimization).

        Uses DataFrameDatasetH adapter to bridge pandas DataFrame to Qlib's
        DatasetH interface, enabling full Qlib LGBModel functionality:
        - Native Qlib model interface
        - Built-in early stopping
        - Consistent with Qlib Model Zoo patterns
        - Model serialization compatible with Qlib workflow

        P2: Now accepts optimized_params from Optuna optimization.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            optimized_params: Optional dict of optimized hyperparameters
        """
        opt_info = " with optimized params" if optimized_params else ""
        logger.info(f"Training with Qlib LGBModel (P1 - full Qlib integration){opt_info}")

        # Create DatasetH-compatible wrapper for our DataFrames
        dataset = DataFrameDatasetH(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val if len(X_val) > 10 else None,
            y_val=y_val if len(X_val) > 10 else None,
        )

        # Determine parameters - use optimized if available
        if optimized_params:
            num_boost_round = optimized_params.get("n_estimators", self.config.ml_n_estimators)
            max_depth = optimized_params.get("max_depth", self.config.ml_max_depth)
            learning_rate = optimized_params.get("learning_rate", self.config.ml_learning_rate)
            num_leaves = optimized_params.get("num_leaves", 31)
        else:
            num_boost_round = self.config.ml_n_estimators
            max_depth = self.config.ml_max_depth
            learning_rate = self.config.ml_learning_rate
            num_leaves = 31

        # Create Qlib LGBModel with parameters
        model = QlibLGBModel(
            loss="mse",
            early_stopping_rounds=self.config.ml_early_stopping_rounds if len(X_val) > 10 else None,
            num_boost_round=num_boost_round,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            verbosity=-1,
        )

        # Train using Qlib's native fit() method
        # Note: Qlib LGBModel.fit() may call R.log_metrics() which requires qlib.init()
        # We catch this error as it only affects logging, not the trained model
        evals_result = {}
        try:
            model.fit(dataset, evals_result=evals_result, verbose_eval=0)
        except AttributeError as e:
            if "qlib.init()" in str(e):
                # Model training succeeded, only R.log_metrics() failed
                # The model is already trained and usable
                logger.debug(
                    "Qlib workflow not initialized (R.log_metrics failed). "
                    "Model training succeeded."
                )
            else:
                raise

        self._ml_model = model
        self._ml_model_type = "qlib"  # Full Qlib LGBModel

        # Log feature importance using Qlib's interface
        try:
            # Qlib LGBModel stores underlying model in .model attribute
            if hasattr(model, 'model') and model.model is not None:
                importance = model.model.feature_importance()
                top_features = sorted(
                    zip(self._ml_feature_names, importance),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                logger.info(
                    f"Qlib LGBModel trained (iteration {self._ml_train_count + 1}). "
                    f"Top features: {top_features}"
                )
        except Exception as e:
            logger.debug(f"Could not get feature importance: {e}")

    def _train_with_raw_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        optimized_params: Optional[dict] = None,
    ) -> None:
        """Train using raw LightGBM (fallback).

        Used when Qlib Model Zoo is not available.

        P2: Now accepts optimized_params from Optuna optimization.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            optimized_params: Optional dict of optimized hyperparameters
        """
        opt_info = " with optimized params" if optimized_params else ""
        logger.info(f"Training with raw LightGBM (fallback){opt_info}")

        # Build model params - use optimized if available, otherwise defaults
        if optimized_params:
            params = {
                **optimized_params,
                "random_state": 42,
            }
        else:
            params = {
                **self.config.ml_params,
                "n_estimators": self.config.ml_n_estimators,
                "max_depth": self.config.ml_max_depth,
                "learning_rate": self.config.ml_learning_rate,
                "random_state": 42,
            }

        # Train model
        model = lgb.LGBMRegressor(**params)

        if len(X_val) > 10:
            model.fit(
                X_train.values, y_train.values,
                eval_set=[(X_val.values, y_val.values)],
                callbacks=[lgb.early_stopping(stopping_rounds=self.config.ml_early_stopping_rounds, verbose=False)],
            )
        else:
            model.fit(X_train.values, y_train.values)

        self._ml_model = model
        self._ml_model_type = "raw"

        # Log feature importance
        importance = dict(zip(self._ml_feature_names, model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(
            f"Raw LightGBM trained (iteration {self._ml_train_count + 1}). "
            f"Top features: {top_features}"
        )

    # =========================================================================
    # P2: Hyperparameter Optimization with Optuna
    # =========================================================================

    def _optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict:
        """Optimize LightGBM hyperparameters using Optuna.

        Supports multiple optimization strategies:
        - bayesian: Tree-structured Parzen Estimator (TPE) - Default, most efficient
        - random: Random search - Fast baseline
        - grid: Grid search - Exhaustive but slow
        - genetic: NSGA-II evolutionary algorithm - Good for multi-objective

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target

        Returns:
            Optimized hyperparameters dict
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, using default hyperparameters")
            return self._get_default_params()

        method = self.config.ml_optimization_method
        if method == "none":
            return self._get_default_params()

        logger.info(f"Starting hyperparameter optimization with method='{method}'")

        # Select sampler based on method
        sampler = self._get_optuna_sampler(method)

        # Create Optuna study
        study = optuna.create_study(
            direction="maximize" if self.config.ml_optimization_metric in ["ic", "sharpe"] else "minimize",
            sampler=sampler,
        )

        # Define objective function
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                **self.config.ml_params,  # Base params (objective, boosting_type, etc.)
            }

            # Train model
            model = lgb.LGBMRegressor(**params)
            if len(X_val) > 10:
                model.fit(
                    X_train.values, y_train.values,
                    eval_set=[(X_val.values, y_val.values)],
                    callbacks=[lgb.early_stopping(stopping_rounds=self.config.ml_early_stopping_rounds, verbose=False)],
                )
            else:
                model.fit(X_train.values, y_train.values)

            # Evaluate based on metric
            y_pred = model.predict(X_val.values)
            return self._calculate_optimization_metric(y_val.values, y_pred)

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.config.ml_optimization_trials,
            timeout=self.config.ml_optimization_timeout,
            show_progress_bar=False,
        )

        logger.info(
            f"Optimization complete: best {self.config.ml_optimization_metric}={study.best_value:.4f}, "
            f"trials={len(study.trials)}"
        )

        # Return best params merged with base params
        best_params = {**self.config.ml_params, **study.best_params}
        return best_params

    def _get_optuna_sampler(self, method: str):
        """Get Optuna sampler based on optimization method.

        Args:
            method: One of 'bayesian', 'random', 'grid', 'genetic'

        Returns:
            Optuna sampler instance
        """
        if method == "bayesian":
            return TPESampler(seed=42)
        elif method == "random":
            return RandomSampler(seed=42)
        elif method == "grid":
            # Grid sampler requires search space - use default grid
            search_space = {
                "n_estimators": [50, 100, 150, 200],
                "max_depth": [3, 5, 7, 9],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "num_leaves": [15, 31, 47, 63],
            }
            return GridSampler(search_space, seed=42)
        elif method == "genetic":
            return NSGAIISampler(seed=42)
        else:
            logger.warning(f"Unknown optimization method '{method}', using bayesian")
            return TPESampler(seed=42)

    def _calculate_optimization_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate optimization metric.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Metric value (higher is better for ic/sharpe, lower for mse)
        """
        metric = self.config.ml_optimization_metric

        if metric == "ic":
            # Information Coefficient (rank correlation)
            from scipy.stats import spearmanr
            ic, _ = spearmanr(y_true, y_pred)
            return ic if not np.isnan(ic) else 0.0

        elif metric == "sharpe":
            # Simplified Sharpe-like metric: mean / std of prediction-aligned returns
            # Positive predictions aligned with positive returns
            aligned_returns = y_true * np.sign(y_pred)
            if np.std(aligned_returns) > 0:
                return np.mean(aligned_returns) / np.std(aligned_returns)
            return 0.0

        elif metric == "mse":
            # Mean Squared Error (lower is better, return negative for maximization)
            mse = np.mean((y_true - y_pred) ** 2)
            return -mse  # Negative because Optuna maximizes

        else:
            logger.warning(f"Unknown metric '{metric}', using IC")
            from scipy.stats import spearmanr
            ic, _ = spearmanr(y_true, y_pred)
            return ic if not np.isnan(ic) else 0.0

    def _get_default_params(self) -> dict:
        """Get default LightGBM parameters from config."""
        return {
            **self.config.ml_params,
            "n_estimators": self.config.ml_n_estimators,
            "max_depth": self.config.ml_max_depth,
            "learning_rate": self.config.ml_learning_rate,
        }

    def _generate_ml_signal(
        self,
        factor: pd.Series,
        price_data: pd.DataFrame,
    ) -> pd.Series:
        """Generate trading signal using ML model.

        Args:
            factor: Factor values
            price_data: OHLCV data

        Returns:
            ML-predicted signal (-1 to 1)
        """
        # Build features
        features = self._build_ml_features(factor, price_data)
        target = self._calculate_target(price_data, factor.index)

        # Check if we need to (re)train
        current_idx = len(factor)
        should_train = (
            self._ml_model is None or
            (self.config.ml_retrain_frequency > 0 and
             current_idx - self._ml_last_train_idx >= self.config.ml_retrain_frequency)
        )

        if should_train:
            # Train on historical data (exclude last forward_period for target)
            train_end = max(0, len(factor) - self.config.ml_forward_period)
            X_train = features.iloc[:train_end]
            y_train = target.iloc[:train_end]
            self._train_ml_model(X_train, y_train)
            self._ml_last_train_idx = current_idx

        # If training failed, fallback to Z-Score
        if self._ml_model is None:
            logger.warning("ML model not available, falling back to Z-Score normalization")
            return self._generate_zscore_signal(factor)

        # Predict on all data
        valid_mask = features.notna().all(axis=1)
        predictions = pd.Series(0.0, index=factor.index)

        if valid_mask.sum() > 0:
            X_pred = features.loc[valid_mask]

            # P1: Handle different model types
            if self._ml_model_type == "qlib":
                # Qlib LGBModel: use underlying lgb.Booster directly for prediction
                # This avoids creating DatasetH for inference
                pred_values = self._ml_model.model.predict(X_pred.values)
            else:
                # Raw LightGBM: sklearn-style predict with numpy arrays
                pred_values = self._ml_model.predict(X_pred.values)

            predictions.loc[valid_mask] = pred_values

        # Normalize predictions to signal range (-1 to 1)
        # Use Z-Score normalization on predictions
        mean_pred = predictions.mean()
        std_pred = predictions.std()
        if std_pred > 0:
            signal = (predictions - mean_pred) / std_pred
            signal = signal.clip(-self.config.clip_std, self.config.clip_std)
            # Scale to -1 to 1
            signal = signal / self.config.clip_std
        else:
            signal = predictions

        return signal

    def _generate_zscore_signal(self, factor: pd.Series) -> pd.Series:
        """Generate signal using Z-Score normalization (original method).

        Args:
            factor: Factor values

        Returns:
            Z-Score normalized signal
        """
        normalized = self.normalize(factor)

        if self.config.top_k:
            k = min(self.config.top_k, len(factor) // 2)
            signal = pd.Series(0.0, index=factor.index)
            if k > 0:
                signal.loc[normalized.nlargest(k).index] = 1.0
                signal.loc[normalized.nsmallest(k).index] = -1.0
            return signal
        else:
            threshold = self.config.signal_threshold
            signal = pd.Series(0.0, index=factor.index)
            signal[normalized > threshold] = normalized[normalized > threshold]
            signal[normalized < -threshold] = normalized[normalized < -threshold]
            return signal.clip(-1, 1)

    # ==========================================================================
    # Original Methods (updated for ML integration)
    # ==========================================================================

    def to_signal(
        self,
        factor: Union[pd.Series, pd.DataFrame],
        normalize: bool = True,
        price_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Convert factor values to trading signal.

        Args:
            factor: Factor values (can be DataFrame or Series)
            normalize: Whether to normalize first
            price_data: OHLCV data (required for ML signal generation)

        Returns:
            Trading signal (-1 to 1)

        Phase 3: If ml_signal_enabled=True and price_data is provided,
        uses LightGBM to predict forward returns and convert to signal.
        """
        # Handle DataFrame input - extract first column or 'value' column
        if isinstance(factor, pd.DataFrame):
            if "value" in factor.columns:
                factor = factor["value"]
            elif "score" in factor.columns:
                factor = factor["score"]
            elif "factor" in factor.columns:
                factor = factor["factor"]
            else:
                factor = factor.iloc[:, 0]

        # Drop NaN values for calculation
        factor = factor.dropna()

        # ======================================================================
        # Phase 3: ML-based signal generation
        # ======================================================================
        if self.config.ml_signal_enabled:
            if price_data is None:
                logger.warning(
                    "ML signal enabled but no price_data provided. "
                    "Falling back to Z-Score normalization."
                )
            elif not LIGHTGBM_AVAILABLE:
                logger.warning(
                    "ML signal enabled but LightGBM not installed. "
                    "Install with: pip install lightgbm. "
                    "Falling back to Z-Score normalization."
                )
            else:
                # Use ML-based signal generation
                return self._generate_ml_signal(factor, price_data)

        # ======================================================================
        # Original Z-Score based signal generation
        # ======================================================================
        if normalize:
            factor = self.normalize(factor)

        if self.config.top_k:
            # Top-k selection: long top k, short bottom k
            k = min(self.config.top_k, len(factor) // 2)
            signal = pd.Series(0.0, index=factor.index)

            if k > 0:
                # Top k = long
                top_k_idx = factor.nlargest(k).index
                signal.loc[top_k_idx] = 1.0

                # Bottom k = short
                bottom_k_idx = factor.nsmallest(k).index
                signal.loc[bottom_k_idx] = -1.0

            return signal
        else:
            # Threshold-based: above threshold = long, below = short
            threshold = self.config.signal_threshold
            signal = pd.Series(0.0, index=factor.index)
            signal[factor > threshold] = factor[factor > threshold]
            signal[factor < -threshold] = factor[factor < -threshold]
            return signal.clip(-1, 1)

    def to_position(self, signal: pd.Series) -> pd.Series:
        """Convert signal to position weights.

        Args:
            signal: Trading signal

        Returns:
            Position weights (scaled and bounded)
        """
        position = signal * self.config.position_scale
        return position.clip(-self.config.max_position, self.config.max_position)

    def to_qlib_format(
        self,
        factor_df: pd.DataFrame,
        datetime_col: str = "datetime",
        instrument_col: str = "instrument",
    ) -> pd.DataFrame:
        """Convert pandas DataFrame to Qlib-compatible format.

        Qlib expects MultiIndex: (datetime, instrument)

        Args:
            factor_df: Factor DataFrame with datetime, instrument columns
            datetime_col: Name of datetime column
            instrument_col: Name of instrument column

        Returns:
            DataFrame with Qlib-compatible MultiIndex
        """
        df = factor_df.copy()

        # Handle 'symbol' as alias for 'instrument'
        if instrument_col not in df.columns and "symbol" in df.columns:
            df[instrument_col] = df["symbol"]

        # Ensure datetime column
        if datetime_col in df.columns:
            df[datetime_col] = pd.to_datetime(df[datetime_col])

        # Set MultiIndex
        if datetime_col in df.columns and instrument_col in df.columns:
            df = df.set_index([datetime_col, instrument_col])
            df.index.names = ["datetime", "instrument"]

        return df.sort_index()

    def create_prediction_dataset(
        self,
        signal: Union[pd.Series, pd.DataFrame],
        instruments: list[str],
        start_time: str,
        end_time: str,
    ) -> "QlibPredictionDataset":
        """Create a Qlib-compatible prediction dataset.

        This creates a minimal dataset structure that can be used
        with Qlib's backtest engine.

        Args:
            signal: Trading signal
            instruments: List of instrument codes
            start_time: Start datetime
            end_time: End datetime

        Returns:
            QlibPredictionDataset compatible with Qlib backtest
        """
        return QlibPredictionDataset(
            signal=signal,
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
        )

    def from_factor_output(
        self,
        factor_output: dict[str, Any],
        price_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Convert factor execution output to trading signal.

        Args:
            factor_output: Output from factor execution containing:
                - factor_values: The computed factor values
                - metadata: Optional metadata
            price_data: Optional price data for context

        Returns:
            Trading signal Series
        """
        factor_values = factor_output.get("factor_values")

        if factor_values is None:
            raise ValueError("factor_output must contain 'factor_values'")

        if isinstance(factor_values, dict):
            factor_values = pd.Series(factor_values)
        elif isinstance(factor_values, pd.DataFrame):
            if "value" in factor_values.columns:
                factor_values = factor_values["value"]
            else:
                factor_values = factor_values.iloc[:, 0]

        return self.to_signal(factor_values)


class QlibPredictionDataset:
    """Minimal Qlib-compatible prediction dataset.

    This class provides the minimal interface required by Qlib's
    backtest engine without requiring full Qlib data infrastructure.

    It wraps pandas DataFrame/Series signals and provides the necessary
    methods for Qlib backtest compatibility.
    """

    def __init__(
        self,
        signal: Union[pd.Series, pd.DataFrame],
        instruments: list[str],
        start_time: str,
        end_time: str,
    ):
        self.signal = signal
        self.instruments = instruments
        self.start_time = pd.Timestamp(start_time)
        self.end_time = pd.Timestamp(end_time)
        self._prepared = False

    def prepare(self, *args: Any, **kwargs: Any) -> "QlibPredictionDataset":
        """Prepare dataset (Qlib interface compatibility)."""
        self._prepared = True
        return self

    def get_segments(self) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
        """Get data segments (Qlib interface)."""
        return {
            "train": (self.start_time, self.end_time),
            "test": (self.start_time, self.end_time),
        }

    def __getitem__(self, key: Any) -> Union[pd.Series, pd.DataFrame, float]:
        """Get prediction for date/instrument (Qlib interface)."""
        if isinstance(self.signal, pd.DataFrame):
            if key in self.signal.index:
                return self.signal.loc[key]
        elif isinstance(self.signal, pd.Series):
            if key in self.signal.index:
                return self.signal.loc[key]
        return self.signal

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame (utility method)."""
        if isinstance(self.signal, pd.Series):
            return self.signal.to_frame(name="score")
        return self.signal

    @property
    def is_prepared(self) -> bool:
        """Check if dataset is prepared."""
        return self._prepared


# Factory function
def create_signal_converter(
    normalize: str = "zscore",
    top_k: Optional[int] = None,
    max_position: float = 0.1,
) -> SignalConverter:
    """Create a configured SignalConverter.

    Args:
        normalize: Normalization method
        top_k: Top-k selection (None for threshold-based)
        max_position: Maximum position per asset

    Returns:
        Configured SignalConverter
    """
    config = SignalConfig(
        normalize_method=normalize,
        top_k=top_k,
        max_position=max_position,
    )
    return SignalConverter(config)
