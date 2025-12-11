"""Walk-Forward Validation Framework for Factor Mining.

This module implements robust out-of-sample validation to prevent overfitting:
- Rolling window train/test splits
- IC degradation analysis
- Deflated Sharpe Ratio (DSR) calculation
- IC half-life estimation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import numpy as np
import pandas as pd
from scipy import stats

from iqfmp.evaluation.factor_evaluator import MetricsCalculator, FactorMetrics


class InsufficientDataError(Exception):
    """Raised when there's not enough data for walk-forward validation."""

    pass


@dataclass
class WalkForwardConfig:
    """Configuration for Walk-Forward validation."""

    # Window settings
    window_size: int = 252  # Training window in periods (e.g., days)
    step_size: int = 63  # Step size for rolling (e.g., quarter)
    min_train_samples: int = 126  # Minimum training samples
    min_test_samples: int = 21  # Minimum test samples

    # Column names
    date_column: str = "date"
    symbol_column: str = "symbol"
    factor_column: str = "factor_value"
    return_column: str = "forward_return"

    # Thresholds
    max_ic_degradation: float = 0.5  # Max acceptable IC degradation ratio
    min_oos_ic: float = 0.02  # Minimum out-of-sample IC
    min_ic_consistency: float = 0.6  # Minimum IC consistency score

    # IC decay settings
    detect_ic_decay: bool = True
    max_half_life: int = 60  # Maximum acceptable IC half-life in periods

    # DSR settings
    use_deflated_sharpe: bool = True
    num_trials: int = 100  # Number of trials for DSR adjustment

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "window_size": self.window_size,
            "step_size": self.step_size,
            "min_train_samples": self.min_train_samples,
            "min_test_samples": self.min_test_samples,
            "max_ic_degradation": self.max_ic_degradation,
            "min_oos_ic": self.min_oos_ic,
            "min_ic_consistency": self.min_ic_consistency,
            "detect_ic_decay": self.detect_ic_decay,
            "max_half_life": self.max_half_life,
            "use_deflated_sharpe": self.use_deflated_sharpe,
            "num_trials": self.num_trials,
        }


@dataclass
class WindowResult:
    """Result from a single walk-forward window."""

    window_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_ic: float
    test_ic: float  # Out-of-sample IC
    train_sharpe: float
    test_sharpe: float
    train_samples: int
    test_samples: int

    @property
    def ic_degradation(self) -> float:
        """Calculate IC degradation ratio."""
        if abs(self.train_ic) < 1e-6:
            return 1.0
        return 1.0 - (self.test_ic / self.train_ic)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "window": self.window_index,
            "train_ic": self.train_ic,
            "oos_ic": self.test_ic,
            "degradation": self.ic_degradation,
            "start_date": self.train_start.isoformat() if self.train_start else None,
            "end_date": self.test_end.isoformat() if self.test_end else None,
        }


@dataclass
class WalkForwardResult:
    """Complete result of Walk-Forward validation."""

    # Core metrics
    avg_train_ic: float = 0.0
    avg_oos_ic: float = 0.0  # Key: out-of-sample IC
    ic_degradation: float = 0.0  # OOS degradation ratio
    oos_ir: float = 0.0  # Out-of-sample IR

    # Distribution stats
    min_oos_ic: float = 0.0
    max_oos_ic: float = 0.0
    oos_ic_std: float = 0.0

    # Robustness verdict
    ic_consistency: float = 0.0  # IC stability score (0-1)
    passes_robustness: bool = False  # Passes robustness test?

    # IC decay
    ic_decay_rate: float = 0.0  # Exponential decay rate
    predicted_half_life: int = 999  # Predicted IC half-life in periods

    # Deflated Sharpe
    raw_sharpe: float = 0.0
    deflated_sharpe: float = 0.0

    # Detailed results
    window_results: list[WindowResult] = field(default_factory=list)
    n_windows: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for API response."""
        return {
            "avg_train_ic": round(self.avg_train_ic, 4),
            "avg_oos_ic": round(self.avg_oos_ic, 4),
            "ic_degradation": round(self.ic_degradation, 4),
            "oos_ir": round(self.oos_ir, 4),
            "min_oos_ic": round(self.min_oos_ic, 4),
            "max_oos_ic": round(self.max_oos_ic, 4),
            "oos_ic_std": round(self.oos_ic_std, 4),
            "ic_consistency": round(self.ic_consistency, 4),
            "passes_robustness": self.passes_robustness,
            "ic_decay_rate": round(self.ic_decay_rate, 6),
            "predicted_half_life": self.predicted_half_life,
            "raw_sharpe": round(self.raw_sharpe, 4),
            "deflated_sharpe": round(self.deflated_sharpe, 4),
            "n_windows": self.n_windows,
            "window_results": [w.to_dict() for w in self.window_results],
        }

    def get_diagnosis(self) -> str:
        """Generate diagnosis string."""
        issues = []

        if self.ic_degradation > 0.5:
            issues.append("High IC degradation (>50%) - likely overfitting")
        if self.avg_oos_ic < 0.02:
            issues.append("Low out-of-sample IC (<2%) - weak predictive power")
        if self.ic_consistency < 0.6:
            issues.append("Low IC consistency - unstable performance")
        if self.predicted_half_life < 30:
            issues.append(f"Short IC half-life ({self.predicted_half_life} periods) - rapid decay")
        if self.deflated_sharpe < 1.0 and self.raw_sharpe > 1.5:
            issues.append("Significant Sharpe deflation - multiple testing concern")

        if not issues:
            return "Factor shows robust out-of-sample performance"
        return "; ".join(issues)


class WalkForwardValidator:
    """Walk-Forward Validation engine for factor evaluation.

    Implements rolling window validation to assess true out-of-sample
    performance and detect overfitting.
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None) -> None:
        """Initialize validator.

        Args:
            config: Validation configuration
        """
        self.config = config or WalkForwardConfig()
        self.calculator = MetricsCalculator()

    def validate(
        self,
        data: pd.DataFrame,
        factor_column: Optional[str] = None,
        return_column: Optional[str] = None,
    ) -> WalkForwardResult:
        """Run walk-forward validation on factor data.

        Args:
            data: DataFrame with factor values and returns
            factor_column: Override factor column name
            return_column: Override return column name

        Returns:
            WalkForwardResult with all validation metrics
        """
        factor_col = factor_column or self.config.factor_column
        return_col = return_column or self.config.return_column
        date_col = self.config.date_column

        # Validate data
        self._validate_data(data, factor_col, return_col, date_col)

        # Prepare data
        df = self._prepare_data(data, date_col)

        # Generate windows
        windows = self._generate_windows(df, date_col)

        if len(windows) == 0:
            raise InsufficientDataError(
                f"Not enough data for walk-forward validation. "
                f"Need at least {self.config.window_size + self.config.step_size} periods."
            )

        # Evaluate each window
        window_results = []
        for i, (train_df, test_df, train_dates, test_dates) in enumerate(windows):
            result = self._evaluate_window(
                i, train_df, test_df, train_dates, test_dates, factor_col, return_col
            )
            window_results.append(result)

        # Aggregate results
        result = self._aggregate_results(window_results)

        # Calculate IC decay
        if self.config.detect_ic_decay and len(window_results) >= 3:
            decay_rate, half_life = self._estimate_ic_decay(window_results)
            result.ic_decay_rate = decay_rate
            result.predicted_half_life = half_life

        # Calculate Deflated Sharpe if enabled
        if self.config.use_deflated_sharpe:
            result.deflated_sharpe = self._calculate_deflated_sharpe(
                result.raw_sharpe, len(window_results)
            )

        # Determine if passes robustness
        result.passes_robustness = self._check_robustness(result)

        return result

    def _validate_data(
        self,
        data: pd.DataFrame,
        factor_col: str,
        return_col: str,
        date_col: str,
    ) -> None:
        """Validate input data."""
        if data.empty:
            raise InsufficientDataError("Input data is empty")

        required_cols = [factor_col, return_col]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        if date_col not in data.columns and not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError(f"Missing date column '{date_col}' and index is not DatetimeIndex")

        # Check minimum data
        min_required = self.config.window_size + self.config.min_test_samples
        if len(data) < min_required:
            raise InsufficientDataError(
                f"Need at least {min_required} rows, got {len(data)}"
            )

    def _prepare_data(self, data: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Prepare data for validation."""
        df = data.copy()

        # Ensure date column
        if date_col in df.columns:
            df["_date"] = pd.to_datetime(df[date_col])
        elif isinstance(df.index, pd.DatetimeIndex):
            df["_date"] = df.index
        else:
            raise ValueError("Cannot determine date column")

        # Sort by date
        df = df.sort_values("_date").reset_index(drop=True)

        return df

    def _generate_windows(
        self, df: pd.DataFrame, date_col: str
    ) -> list[tuple[pd.DataFrame, pd.DataFrame, tuple, tuple]]:
        """Generate rolling train/test windows.

        Returns list of (train_df, test_df, train_dates, test_dates)
        """
        windows = []
        n = len(df)

        window_size = self.config.window_size
        step_size = self.config.step_size
        min_test = self.config.min_test_samples

        start = 0

        while start + window_size + min_test <= n:
            train_end = start + window_size
            test_end = min(train_end + step_size, n)

            if test_end - train_end < min_test:
                break

            train_df = df.iloc[start:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()

            train_dates = (
                train_df["_date"].iloc[0],
                train_df["_date"].iloc[-1],
            )
            test_dates = (
                test_df["_date"].iloc[0],
                test_df["_date"].iloc[-1],
            )

            windows.append((train_df, test_df, train_dates, test_dates))

            start += step_size

        return windows

    def _evaluate_window(
        self,
        window_idx: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_dates: tuple,
        test_dates: tuple,
        factor_col: str,
        return_col: str,
    ) -> WindowResult:
        """Evaluate a single train/test window."""
        # Calculate train metrics
        train_ic = self.calculator.calculate_rank_ic(
            train_df[factor_col], train_df[return_col]
        )
        train_sharpe = self.calculator.calculate_sharpe_ratio(train_df[return_col])

        # Calculate test (OOS) metrics
        test_ic = self.calculator.calculate_rank_ic(
            test_df[factor_col], test_df[return_col]
        )
        test_sharpe = self.calculator.calculate_sharpe_ratio(test_df[return_col])

        return WindowResult(
            window_index=window_idx,
            train_start=train_dates[0],
            train_end=train_dates[1],
            test_start=test_dates[0],
            test_end=test_dates[1],
            train_ic=train_ic,
            test_ic=test_ic,
            train_sharpe=train_sharpe,
            test_sharpe=test_sharpe,
            train_samples=len(train_df),
            test_samples=len(test_df),
        )

    def _aggregate_results(self, window_results: list[WindowResult]) -> WalkForwardResult:
        """Aggregate results from all windows."""
        if not window_results:
            return WalkForwardResult()

        train_ics = [w.train_ic for w in window_results]
        test_ics = [w.test_ic for w in window_results]
        train_sharpes = [w.train_sharpe for w in window_results]
        test_sharpes = [w.test_sharpe for w in window_results]

        avg_train_ic = np.mean(train_ics)
        avg_oos_ic = np.mean(test_ics)

        # IC degradation
        if abs(avg_train_ic) > 1e-6:
            ic_degradation = 1.0 - (avg_oos_ic / avg_train_ic)
        else:
            ic_degradation = 1.0

        # OOS IR
        oos_ic_std = np.std(test_ics) if len(test_ics) > 1 else 1.0
        if oos_ic_std > 1e-6:
            oos_ir = avg_oos_ic / oos_ic_std
        else:
            oos_ir = 0.0

        # IC consistency: ratio of positive IC windows
        positive_ic_count = sum(1 for ic in test_ics if ic > 0)
        ic_consistency = positive_ic_count / len(test_ics) if test_ics else 0.0

        # Raw Sharpe (average of test sharpes)
        raw_sharpe = np.mean(test_sharpes) if test_sharpes else 0.0

        return WalkForwardResult(
            avg_train_ic=float(avg_train_ic),
            avg_oos_ic=float(avg_oos_ic),
            ic_degradation=float(ic_degradation),
            oos_ir=float(oos_ir),
            min_oos_ic=float(np.min(test_ics)),
            max_oos_ic=float(np.max(test_ics)),
            oos_ic_std=float(oos_ic_std),
            ic_consistency=float(ic_consistency),
            raw_sharpe=float(raw_sharpe),
            window_results=window_results,
            n_windows=len(window_results),
        )

    def _estimate_ic_decay(
        self, window_results: list[WindowResult]
    ) -> tuple[float, int]:
        """Estimate IC decay rate and half-life.

        Uses exponential decay model: IC(t) = IC(0) * exp(-λt)
        Half-life = ln(2) / λ

        Returns:
            (decay_rate, half_life_periods)
        """
        if len(window_results) < 3:
            return 0.0, 999

        # Use test ICs for decay estimation
        test_ics = np.array([abs(w.test_ic) for w in window_results])

        # Filter out zeros for log
        valid_mask = test_ics > 1e-6
        if valid_mask.sum() < 3:
            return 0.0, 999

        t = np.arange(len(test_ics))[valid_mask]
        log_ics = np.log(test_ics[valid_mask])

        # Linear regression on log(IC) vs t
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(t, log_ics)

            # Decay rate is negative of slope
            decay_rate = -slope if slope < 0 else 0.0

            # Half-life
            if decay_rate > 1e-6:
                half_life = int(np.log(2) / decay_rate)
                half_life = min(half_life, 999)  # Cap at 999
            else:
                half_life = 999

            return decay_rate, half_life

        except Exception:
            return 0.0, 999

    def _calculate_deflated_sharpe(
        self, raw_sharpe: float, n_windows: int
    ) -> float:
        """Calculate Deflated Sharpe Ratio (DSR).

        Adjusts Sharpe ratio for multiple testing / overfitting.

        Based on Bailey & Lopez de Prado (2014):
        DSR = (SR - E[max(SR)]) / σ[max(SR)]

        Simplified approximation for practical use.
        """
        if n_windows <= 1:
            return raw_sharpe

        # Number of trials (from config or based on windows)
        n_trials = max(self.config.num_trials, n_windows * 10)

        # Expected maximum Sharpe under null hypothesis
        # E[max] ≈ (1 - γ) * Φ^(-1)(1 - 1/N) + γ * Φ^(-1)(1 - 1/(N*e))
        # where γ ≈ 0.5772 (Euler-Mascheroni constant)

        gamma = 0.5772156649
        try:
            term1 = (1 - gamma) * stats.norm.ppf(1 - 1 / n_trials)
            term2 = gamma * stats.norm.ppf(1 - 1 / (n_trials * np.e))
            expected_max_sr = term1 + term2

            # Variance of max Sharpe
            var_max_sr = (np.pi ** 2 / 6) * (1 / n_trials)
            std_max_sr = np.sqrt(var_max_sr) if var_max_sr > 0 else 1.0

            # Deflated Sharpe
            deflated = (raw_sharpe - expected_max_sr) / std_max_sr if std_max_sr > 0 else raw_sharpe

            # Convert back to Sharpe-like scale
            # This is a probability that SR is not due to chance
            p_value = 1 - stats.norm.cdf(deflated)

            # Return adjusted Sharpe (scaled by significance)
            significance_multiplier = max(0, 1 - 2 * p_value)  # 0 if p > 0.5, 1 if p < 0
            return raw_sharpe * significance_multiplier

        except Exception:
            return raw_sharpe * 0.8  # Conservative 20% haircut as fallback

    def _check_robustness(self, result: WalkForwardResult) -> bool:
        """Check if factor passes robustness criteria."""
        checks = [
            result.ic_degradation <= self.config.max_ic_degradation,
            result.avg_oos_ic >= self.config.min_oos_ic,
            result.ic_consistency >= self.config.min_ic_consistency,
        ]

        # Optional: IC half-life check
        if self.config.detect_ic_decay:
            checks.append(result.predicted_half_life >= self.config.max_half_life)

        return all(checks)


class WalkForwardPipeline:
    """Pipeline for running Walk-Forward validation on multiple factors."""

    def __init__(self, config: Optional[WalkForwardConfig] = None) -> None:
        """Initialize pipeline."""
        self.config = config or WalkForwardConfig()
        self.validator = WalkForwardValidator(config=self.config)

    def validate_factors(
        self,
        data: pd.DataFrame,
        factor_columns: list[str],
        return_column: str,
    ) -> dict[str, WalkForwardResult]:
        """Validate multiple factors.

        Args:
            data: DataFrame with all factor values and returns
            factor_columns: List of factor column names
            return_column: Return column name

        Returns:
            Dictionary mapping factor names to validation results
        """
        results = {}

        for factor_col in factor_columns:
            try:
                result = self.validator.validate(
                    data=data,
                    factor_column=factor_col,
                    return_column=return_column,
                )
                results[factor_col] = result
            except Exception as e:
                # Create failed result
                results[factor_col] = WalkForwardResult(
                    passes_robustness=False,
                )

        return results

    def get_summary(self, results: dict[str, WalkForwardResult]) -> dict[str, Any]:
        """Generate summary of validation results."""
        total = len(results)
        passed = sum(1 for r in results.values() if r.passes_robustness)

        avg_oos_ics = [r.avg_oos_ic for r in results.values() if r.avg_oos_ic != 0]
        avg_degradations = [r.ic_degradation for r in results.values()]

        # Find best factor
        best_factor = None
        best_oos_ic = -float("inf")
        for name, r in results.items():
            if r.passes_robustness and r.avg_oos_ic > best_oos_ic:
                best_oos_ic = r.avg_oos_ic
                best_factor = name

        return {
            "total_validated": total,
            "passed_count": passed,
            "pass_rate": passed / total if total > 0 else 0,
            "avg_oos_ic": np.mean(avg_oos_ics) if avg_oos_ics else 0,
            "avg_ic_degradation": np.mean(avg_degradations) if avg_degradations else 0,
            "best_factor": best_factor,
            "best_oos_ic": best_oos_ic if best_factor else 0,
        }
