"""ML-based Factor Combiner using LightGBM and XGBoost.

This module provides factor combination using gradient boosting models,
allowing multiple individual factors to be combined into a more powerful
composite signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

# Check for optional dependencies
LIGHTGBM_AVAILABLE = False
XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None


class ModelType(str, Enum):
    """Available model types for factor combination."""
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    LINEAR = "linear"  # Fallback


@dataclass
class CombinerConfig:
    """Configuration for factor combiner."""

    model_type: ModelType = ModelType.LIGHTGBM

    # Training parameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_samples_leaf: int = 20

    # Validation
    n_splits: int = 5
    test_size: int = 252  # ~1 year of daily data

    # Feature engineering
    use_lagged_features: bool = True
    lag_periods: list[int] = field(default_factory=lambda: [1, 5, 20])

    # Regularization
    l1_ratio: float = 0.0
    l2_ratio: float = 0.1

    # Output
    prediction_type: str = "regression"  # regression, classification


@dataclass
class CombinerResult:
    """Result from factor combination."""

    combined_factor: pd.Series
    feature_importance: dict[str, float]
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    model_type: str
    config_used: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_importance": self.feature_importance,
            "train_metrics": self.train_metrics,
            "test_metrics": self.test_metrics,
            "model_type": self.model_type,
            "config": self.config_used,
        }

    def get_top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top N important features."""
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return sorted_features[:n]


class FactorCombiner:
    """Combine multiple factors using ML models."""

    def __init__(self, config: Optional[CombinerConfig] = None) -> None:
        """Initialize combiner.

        Args:
            config: Combiner configuration
        """
        self.config = config or CombinerConfig()
        self.model: Any = None
        self.feature_names: list[str] = []
        self._is_fitted = False

        # Validate model availability
        if self.config.model_type == ModelType.LIGHTGBM and not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available, falling back to Linear")
            self.config.model_type = ModelType.LINEAR
        elif self.config.model_type == ModelType.XGBOOST and not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, falling back to Linear")
            self.config.model_type = ModelType.LINEAR

    def fit(
        self,
        factors_df: pd.DataFrame,
        target: pd.Series,
        validation_split: float = 0.2,
    ) -> CombinerResult:
        """Fit the combiner model.

        Args:
            factors_df: DataFrame with factor values (columns = factors)
            target: Target variable (forward returns)
            validation_split: Fraction of data for validation

        Returns:
            CombinerResult with model info and metrics
        """
        # Prepare features
        X, y, feature_names = self._prepare_features(factors_df, target)
        self.feature_names = feature_names

        # Handle NaN
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < 100:
            raise ValueError("Insufficient data for training (need at least 100 samples)")

        # Time-based split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Train model
        self.model = self._create_model()
        train_metrics = self._fit_model(X_train, y_train, X_test, y_test)

        self._is_fitted = True

        # Evaluate
        test_metrics = self._evaluate(X_test, y_test)

        # Get feature importance
        importance = self._get_feature_importance()

        # Generate combined factor
        combined = self.predict(factors_df)

        return CombinerResult(
            combined_factor=combined,
            feature_importance=importance,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            model_type=self.config.model_type.value,
            config_used=self._config_to_dict(),
        )

    def predict(self, factors_df: pd.DataFrame) -> pd.Series:
        """Generate combined factor predictions.

        Args:
            factors_df: DataFrame with factor values

        Returns:
            Series with combined factor values
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X, _, _ = self._prepare_features(factors_df, None)

        # Handle NaN - fill with column mean for prediction
        X = X.fillna(X.mean())

        if self.config.model_type == ModelType.LINEAR:
            predictions = X @ self.model
        else:
            predictions = self.model.predict(X)

        return pd.Series(predictions, index=factors_df.index, name="combined_factor")

    def _prepare_features(
        self,
        factors_df: pd.DataFrame,
        target: Optional[pd.Series],
    ) -> tuple[pd.DataFrame, Optional[pd.Series], list[str]]:
        """Prepare features from factor DataFrame."""
        features = factors_df.copy()
        feature_names = list(factors_df.columns)

        # Add lagged features if configured
        if self.config.use_lagged_features:
            for col in factors_df.columns:
                for lag in self.config.lag_periods:
                    lag_col = f"{col}_lag{lag}"
                    features[lag_col] = factors_df[col].shift(lag)
                    feature_names.append(lag_col)

        return features, target, feature_names

    def _create_model(self) -> Any:
        """Create the ML model based on configuration."""
        if self.config.model_type == ModelType.LIGHTGBM:
            return lgb.LGBMRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_child_samples=self.config.min_samples_leaf,
                reg_alpha=self.config.l1_ratio,
                reg_lambda=self.config.l2_ratio,
                n_jobs=-1,
                verbose=-1,
            )

        elif self.config.model_type == ModelType.XGBOOST:
            return xgb.XGBRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_child_weight=self.config.min_samples_leaf,
                reg_alpha=self.config.l1_ratio,
                reg_lambda=self.config.l2_ratio,
                n_jobs=-1,
                verbosity=0,
            )

        else:
            # Linear fallback - just use correlation-weighted combination
            return None

    def _fit_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict[str, float]:
        """Fit the model and return training metrics."""
        if self.config.model_type == ModelType.LINEAR:
            # Simple IC-weighted linear combination
            weights = []
            for col in X_train.columns:
                valid = ~(X_train[col].isna() | y_train.isna())
                if valid.sum() > 10:
                    ic = X_train.loc[valid, col].corr(y_train[valid])
                    weights.append(ic if not np.isnan(ic) else 0.0)
                else:
                    weights.append(0.0)

            # Normalize weights
            total = sum(abs(w) for w in weights)
            if total > 0:
                weights = [w / total for w in weights]

            self.model = np.array(weights)

            # Calculate train IC
            train_pred = X_train @ self.model
            train_ic = train_pred.corr(y_train)

            return {
                "train_ic": float(train_ic) if not np.isnan(train_ic) else 0.0,
            }

        # Gradient boosting models
        if self.config.model_type == ModelType.LIGHTGBM:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(20, verbose=False)],
            )
        else:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=False,
            )

        # Training metrics
        train_pred = self.model.predict(X_train)
        train_ic = pd.Series(train_pred).corr(y_train)

        return {
            "train_ic": float(train_ic) if not np.isnan(train_ic) else 0.0,
            "n_estimators_used": getattr(self.model, "n_estimators_", self.config.n_estimators),
        }

    def _evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, float]:
        """Evaluate model on test set."""
        if self.config.model_type == ModelType.LINEAR:
            pred = X_test @ self.model
        else:
            pred = self.model.predict(X_test)

        pred = pd.Series(pred, index=y_test.index)

        # IC
        ic = pred.corr(y_test)

        # Rank IC
        from scipy import stats
        rank_ic, _ = stats.spearmanr(pred.values, y_test.values)

        # MSE
        mse = ((pred - y_test) ** 2).mean()

        return {
            "test_ic": float(ic) if not np.isnan(ic) else 0.0,
            "test_rank_ic": float(rank_ic) if not np.isnan(rank_ic) else 0.0,
            "test_mse": float(mse),
        }

    def _get_feature_importance(self) -> dict[str, float]:
        """Get feature importance from model."""
        if self.config.model_type == ModelType.LINEAR:
            return {
                name: float(weight)
                for name, weight in zip(self.feature_names, self.model)
            }

        importance = self.model.feature_importances_
        return {
            name: float(imp)
            for name, imp in zip(self.feature_names, importance)
        }

    def _config_to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_type": self.config.model_type.value,
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "use_lagged_features": self.config.use_lagged_features,
        }


class EnsembleCombiner:
    """Ensemble multiple combiners for robust factor combination."""

    def __init__(
        self,
        combiners: Optional[list[FactorCombiner]] = None,
    ) -> None:
        """Initialize ensemble combiner.

        Args:
            combiners: List of FactorCombiner instances
        """
        if combiners is None:
            # Default: one of each available type
            combiners = []

            if LIGHTGBM_AVAILABLE:
                combiners.append(FactorCombiner(CombinerConfig(model_type=ModelType.LIGHTGBM)))

            if XGBOOST_AVAILABLE:
                combiners.append(FactorCombiner(CombinerConfig(model_type=ModelType.XGBOOST)))

            # Always include linear as fallback
            combiners.append(FactorCombiner(CombinerConfig(model_type=ModelType.LINEAR)))

        self.combiners = combiners
        self.weights: list[float] = []
        self._is_fitted = False

    def fit(
        self,
        factors_df: pd.DataFrame,
        target: pd.Series,
        weight_by_performance: bool = True,
    ) -> dict[str, CombinerResult]:
        """Fit all combiners.

        Args:
            factors_df: DataFrame with factor values
            target: Target variable
            weight_by_performance: Weight combiners by test IC

        Returns:
            Dictionary of results by model type
        """
        results = {}
        test_ics = []

        for combiner in self.combiners:
            try:
                result = combiner.fit(factors_df, target)
                results[combiner.config.model_type.value] = result
                test_ics.append(abs(result.test_metrics.get("test_ic", 0)))
            except Exception as e:
                logger.warning(f"Combiner {combiner.config.model_type} failed: {e}")
                test_ics.append(0)

        # Calculate ensemble weights
        if weight_by_performance and sum(test_ics) > 0:
            total = sum(test_ics)
            self.weights = [ic / total for ic in test_ics]
        else:
            self.weights = [1.0 / len(self.combiners)] * len(self.combiners)

        self._is_fitted = True
        return results

    def predict(self, factors_df: pd.DataFrame) -> pd.Series:
        """Generate ensemble predictions.

        Args:
            factors_df: DataFrame with factor values

        Returns:
            Weighted average of combiner predictions
        """
        if not self._is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        predictions = []
        for combiner, weight in zip(self.combiners, self.weights):
            if combiner._is_fitted:
                pred = combiner.predict(factors_df)
                predictions.append(pred * weight)

        if not predictions:
            raise ValueError("No fitted combiners available")

        return sum(predictions)


# =============================================================================
# Factory functions
# =============================================================================

def create_lightgbm_combiner(**kwargs) -> FactorCombiner:
    """Create LightGBM-based combiner."""
    config = CombinerConfig(model_type=ModelType.LIGHTGBM, **kwargs)
    return FactorCombiner(config)


def create_xgboost_combiner(**kwargs) -> FactorCombiner:
    """Create XGBoost-based combiner."""
    config = CombinerConfig(model_type=ModelType.XGBOOST, **kwargs)
    return FactorCombiner(config)


def create_linear_combiner(**kwargs) -> FactorCombiner:
    """Create IC-weighted linear combiner."""
    config = CombinerConfig(model_type=ModelType.LINEAR, **kwargs)
    return FactorCombiner(config)


def get_available_model_types() -> list[str]:
    """Get list of available model types."""
    available = ["linear"]  # Always available

    if LIGHTGBM_AVAILABLE:
        available.append("lightgbm")

    if XGBOOST_AVAILABLE:
        available.append("xgboost")

    return available
