"""Factor Evaluator for comprehensive factor analysis.

This module integrates:
- CryptoCVSplitter for multi-dimensional validation
- ResearchLedger for trial tracking
- StabilityAnalyzer for robustness analysis

Provides complete factor evaluation with IC/IR/Sharpe metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional
import numpy as np
import pandas as pd
from scipy import stats

from iqfmp.evaluation.research_ledger import (
    ResearchLedger,
    TrialRecord,
    MemoryStorage,
)
from iqfmp.evaluation.stability_analyzer import (
    StabilityAnalyzer,
    StabilityReport,
    StabilityConfig,
)


class InvalidFactorError(Exception):
    """Raised when factor data is invalid."""

    pass


class EvaluationFailedError(Exception):
    """Raised when evaluation fails."""

    pass


@dataclass
class EvaluationConfig:
    """Configuration for factor evaluation."""

    # Column names
    date_column: str = "date"
    symbol_column: str = "symbol"
    factor_column: str = "factor_value"
    return_column: str = "forward_return"
    market_cap_column: str = "market_cap"

    # Evaluation options
    use_cv_splits: bool = False
    run_stability_analysis: bool = False
    min_periods: int = 20

    # Thresholds
    ic_threshold: float = 0.03
    ir_threshold: float = 1.0

    # Annualization
    annualization_factor: float = 252.0  # Daily data


@dataclass
class FactorMetrics:
    """Metrics calculated for a factor."""

    ic: float = 0.0
    rank_ic: float = 0.0
    ir: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    turnover: float = 0.0
    ic_std: float = 0.0
    ic_skew: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "ic": self.ic,
            "rank_ic": self.rank_ic,
            "ir": self.ir,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "turnover": self.turnover,
            "ic_std": self.ic_std,
            "ic_skew": self.ic_skew,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FactorMetrics:
        """Create from dictionary."""
        return cls(
            ic=data.get("ic", 0.0),
            rank_ic=data.get("rank_ic", 0.0),
            ir=data.get("ir", 0.0),
            sharpe_ratio=data.get("sharpe_ratio", 0.0),
            max_drawdown=data.get("max_drawdown", 0.0),
            win_rate=data.get("win_rate", 0.0),
            turnover=data.get("turnover", 0.0),
            ic_std=data.get("ic_std", 0.0),
            ic_skew=data.get("ic_skew", 0.0),
        )

    def is_significant(
        self, ic_threshold: float = 0.03, ir_threshold: float = 1.0
    ) -> bool:
        """Check if metrics meet significance thresholds."""
        return abs(self.ic) >= ic_threshold and self.ir >= ir_threshold


class MetricsCalculator:
    """Calculator for factor metrics."""

    def calculate_ic(
        self, factor_values: pd.Series, returns: pd.Series
    ) -> float:
        """Calculate Information Coefficient (Pearson correlation)."""
        if len(factor_values) < 3:
            return 0.0

        # Handle NaN values
        mask = ~(factor_values.isna() | returns.isna())
        if mask.sum() < 3:
            return 0.0

        corr = factor_values[mask].corr(returns[mask])
        return float(corr) if not np.isnan(corr) else 0.0

    def calculate_rank_ic(
        self, factor_values: pd.Series, returns: pd.Series
    ) -> float:
        """Calculate Rank IC (Spearman correlation)."""
        if len(factor_values) < 3:
            return 0.0

        mask = ~(factor_values.isna() | returns.isna())
        if mask.sum() < 3:
            return 0.0

        corr, _ = stats.spearmanr(
            factor_values[mask].values, returns[mask].values
        )
        return float(corr) if not np.isnan(corr) else 0.0

    def calculate_ir(self, ic_series: pd.Series) -> float:
        """Calculate Information Ratio from IC series."""
        if len(ic_series) < 2:
            return 0.0

        ic_mean = ic_series.mean()
        ic_std = ic_series.std()

        if ic_std == 0 or np.isnan(ic_std):
            return 0.0

        return float(ic_mean / ic_std)

    def calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.0
    ) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free_rate / 252
        mean_return = excess_returns.mean()
        std_return = excess_returns.std()

        if std_return == 0 or np.isnan(std_return):
            return 0.0

        # Annualize
        sharpe = (mean_return / std_return) * np.sqrt(252)
        return float(sharpe) if not np.isnan(sharpe) else 0.0

    def calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(cumulative_returns) < 2:
            return 0.0

        # Running maximum
        running_max = cumulative_returns.cummax()

        # Drawdown at each point
        drawdown = (running_max - cumulative_returns) / running_max

        # Handle division by zero
        drawdown = drawdown.replace([np.inf, -np.inf], 0)
        drawdown = drawdown.fillna(0)

        return float(drawdown.max())

    def calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate."""
        if len(returns) == 0:
            return 0.0

        wins = (returns > 0).sum()
        return float(wins / len(returns))

    def calculate_turnover(
        self, positions_t0: pd.Series, positions_t1: pd.Series
    ) -> float:
        """Calculate turnover between two periods."""
        if len(positions_t0) == 0 or len(positions_t1) == 0:
            return 0.0

        # Align series
        common_idx = positions_t0.index.intersection(positions_t1.index)
        if len(common_idx) == 0:
            return 0.0

        diff = (positions_t1[common_idx] - positions_t0[common_idx]).abs()
        return float(diff.sum() / 2)


@dataclass
class CVResult:
    """Result from a single CV split."""

    split_name: str
    metrics: FactorMetrics
    data_points: int


@dataclass
class FactorReport:
    """Complete factor evaluation report."""

    factor_name: str
    factor_family: str
    metrics: FactorMetrics
    grade: str
    passes_threshold: bool
    threshold_used: float
    stability_report: Optional[StabilityReport] = None
    cv_results: Optional[list[CVResult]] = None
    recommendations: list[str] = field(default_factory=list)
    evaluation_date: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "factor_name": self.factor_name,
            "factor_family": self.factor_family,
            "metrics": self.metrics.to_dict(),
            "grade": self.grade,
            "passes_threshold": self.passes_threshold,
            "threshold_used": self.threshold_used,
            "recommendations": self.recommendations,
            "evaluation_date": self.evaluation_date,
        }

        if self.stability_report:
            result["stability"] = self.stability_report.to_dict()

        if self.cv_results:
            result["cv_results"] = [
                {"split": r.split_name, "metrics": r.metrics.to_dict()}
                for r in self.cv_results
            ]

        return result

    def get_summary(self) -> str:
        """Generate text summary."""
        lines = [
            f"Factor Evaluation Report: {self.factor_name}",
            "=" * 50,
            f"Family: {self.factor_family}",
            f"Grade: {self.grade}",
            f"Passes Threshold: {'Yes' if self.passes_threshold else 'No'}",
            "",
            "Metrics:",
            f"  IC: {self.metrics.ic:.4f}",
            f"  Rank IC: {self.metrics.rank_ic:.4f}",
            f"  IR: {self.metrics.ir:.2f}",
            f"  Sharpe: {self.metrics.sharpe_ratio:.2f}",
            f"  Max DD: {self.metrics.max_drawdown:.2%}",
            f"  Win Rate: {self.metrics.win_rate:.2%}",
            "",
        ]

        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


@dataclass
class EvaluationResult:
    """Result of factor evaluation."""

    factor_name: str
    factor_family: str
    metrics: FactorMetrics
    passes_threshold: bool
    threshold_used: float
    trial_id: str
    cv_results: Optional[list[CVResult]] = None
    stability_report: Optional[StabilityReport] = None

    def generate_report(self) -> FactorReport:
        """Generate full report from result."""
        grade = self._calculate_grade()
        recommendations = self._generate_recommendations()

        return FactorReport(
            factor_name=self.factor_name,
            factor_family=self.factor_family,
            metrics=self.metrics,
            grade=grade,
            passes_threshold=self.passes_threshold,
            threshold_used=self.threshold_used,
            stability_report=self.stability_report,
            cv_results=self.cv_results,
            recommendations=recommendations,
        )

    def _calculate_grade(self) -> str:
        """Calculate grade based on metrics."""
        score = 0.0

        # IC contribution (0-30 points)
        score += min(30, abs(self.metrics.ic) * 500)

        # IR contribution (0-30 points)
        score += min(30, self.metrics.ir * 15)

        # Sharpe contribution (0-20 points)
        score += min(20, max(0, self.metrics.sharpe_ratio) * 10)

        # Win rate contribution (0-10 points)
        score += max(0, (self.metrics.win_rate - 0.5)) * 100

        # Stability contribution (0-10 points)
        if self.stability_report:
            score += self.stability_report.overall_score.value * 10

        if score >= 80:
            return "A"
        elif score >= 65:
            return "B"
        elif score >= 50:
            return "C"
        elif score >= 35:
            return "D"
        else:
            return "F"

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on metrics."""
        recommendations = []

        if abs(self.metrics.ic) < 0.03:
            recommendations.append(
                "IC is low - consider adding more features or adjusting calculation"
            )

        if self.metrics.ir < 1.0:
            recommendations.append(
                "IR is below 1.0 - factor may have inconsistent predictive power"
            )

        if self.metrics.max_drawdown > 0.2:
            recommendations.append(
                "High max drawdown - implement position sizing or stop-loss"
            )

        if self.metrics.win_rate < 0.45:
            recommendations.append(
                "Low win rate - factor may need refinement"
            )

        if not self.passes_threshold:
            recommendations.append(
                "Factor does not meet dynamic threshold - may be overfitted"
            )

        if not recommendations:
            recommendations.append(
                "Factor shows promising characteristics"
            )

        return recommendations


class FactorEvaluator:
    """Main factor evaluator integrating all components."""

    def __init__(
        self,
        ledger: Optional[ResearchLedger] = None,
        config: Optional[EvaluationConfig] = None,
    ) -> None:
        """Initialize evaluator."""
        self.ledger = ledger or ResearchLedger(storage=MemoryStorage())
        self.config = config or EvaluationConfig()
        self.calculator = MetricsCalculator()
        self.stability_analyzer = StabilityAnalyzer(
            StabilityConfig(
                date_column=self.config.date_column,
                symbol_column=self.config.symbol_column,
                factor_column=self.config.factor_column,
                return_column=self.config.return_column,
                market_cap_column=self.config.market_cap_column,
            )
        )

    def evaluate(
        self,
        factor_name: str,
        factor_family: str,
        data: pd.DataFrame,
    ) -> EvaluationResult:
        """Evaluate a factor.

        Args:
            factor_name: Name of the factor
            factor_family: Family/category of the factor
            data: DataFrame with factor values and returns

        Returns:
            EvaluationResult with all metrics and analysis
        """
        # Validate inputs
        self._validate_inputs(factor_name, factor_family, data)

        # Prepare data
        df = self._prepare_data(data)

        # Calculate metrics
        metrics = self._calculate_metrics(df)

        # Run CV splits if enabled
        cv_results = None
        if self.config.use_cv_splits:
            cv_results = self._run_cv_evaluation(df)

        # Run stability analysis if enabled
        stability_report = None
        if self.config.run_stability_analysis:
            try:
                stability_report = self.stability_analyzer.analyze(df)
            except Exception:
                pass  # Stability analysis is optional

        # Record to ledger
        trial = TrialRecord(
            factor_name=factor_name,
            factor_family=factor_family,
            sharpe_ratio=metrics.sharpe_ratio,
            ic_mean=metrics.ic,
            ir=metrics.ir,
        )
        trial_id = self.ledger.record(trial)

        # Check threshold
        threshold_result = self.ledger.check_significance(metrics.sharpe_ratio)

        return EvaluationResult(
            factor_name=factor_name,
            factor_family=factor_family,
            metrics=metrics,
            passes_threshold=threshold_result.passes,
            threshold_used=threshold_result.threshold,
            trial_id=trial_id,
            cv_results=cv_results,
            stability_report=stability_report,
        )

    def _validate_inputs(
        self, factor_name: str, factor_family: str, data: pd.DataFrame
    ) -> None:
        """Validate inputs."""
        if not factor_name or not factor_name.strip():
            raise InvalidFactorError("Factor name cannot be empty")

        if data.empty:
            raise InvalidFactorError("Input data is empty")

        # Check required columns
        required = [self.config.factor_column, self.config.return_column]
        for col in required:
            if col not in data.columns:
                raise InvalidFactorError(f"Missing required column: {col}")

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for evaluation."""
        df = data.copy()

        # Handle date column or index
        if self.config.date_column in df.columns:
            df["_date"] = pd.to_datetime(df[self.config.date_column])
        elif isinstance(df.index, pd.DatetimeIndex):
            df["_date"] = df.index
            df[self.config.date_column] = df.index
        else:
            df["_date"] = pd.date_range(
                start="2022-01-01", periods=len(df), freq="D"
            )
            df[self.config.date_column] = df["_date"]

        return df

    def _calculate_metrics(self, df: pd.DataFrame) -> FactorMetrics:
        """Calculate all metrics for the factor."""
        factor_col = self.config.factor_column
        return_col = self.config.return_column

        # Overall IC
        ic = self.calculator.calculate_ic(df[factor_col], df[return_col])
        rank_ic = self.calculator.calculate_rank_ic(df[factor_col], df[return_col])

        # Calculate IC series by period for IR
        df["_period"] = df["_date"].dt.to_period("M")
        ic_series = []

        for period, group in df.groupby("_period"):
            if len(group) >= 5:
                period_ic = self.calculator.calculate_ic(
                    group[factor_col], group[return_col]
                )
                if not np.isnan(period_ic):
                    ic_series.append(period_ic)

        ic_series = pd.Series(ic_series)
        ir = self.calculator.calculate_ir(ic_series)

        # Calculate IC statistics
        ic_std = float(ic_series.std()) if len(ic_series) > 1 else 0.0
        ic_skew = float(ic_series.skew()) if len(ic_series) > 2 else 0.0

        # Calculate returns-based metrics
        returns = df[return_col]
        cumulative = (1 + returns).cumprod()

        sharpe = self.calculator.calculate_sharpe_ratio(returns)
        max_dd = self.calculator.calculate_max_drawdown(cumulative)
        win_rate = self.calculator.calculate_win_rate(returns)

        return FactorMetrics(
            ic=ic,
            rank_ic=rank_ic,
            ir=ir,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            ic_std=ic_std,
            ic_skew=ic_skew,
        )

    def _run_cv_evaluation(self, df: pd.DataFrame) -> list[CVResult]:
        """Run cross-validation evaluation."""
        results = []

        # Simple time-based split
        n = len(df)
        split_size = n // 3

        splits = [
            ("train", df.iloc[:split_size]),
            ("valid", df.iloc[split_size : 2 * split_size]),
            ("test", df.iloc[2 * split_size :]),
        ]

        for name, split_df in splits:
            if len(split_df) >= self.config.min_periods:
                metrics = self._calculate_metrics(split_df)
                results.append(
                    CVResult(
                        split_name=name,
                        metrics=metrics,
                        data_points=len(split_df),
                    )
                )

        return results


class EvaluationPipeline:
    """Pipeline for batch factor evaluation."""

    def __init__(
        self,
        ledger: Optional[ResearchLedger] = None,
        config: Optional[EvaluationConfig] = None,
    ) -> None:
        """Initialize pipeline."""
        self.ledger = ledger or ResearchLedger(storage=MemoryStorage())
        self.config = config or EvaluationConfig()
        self.evaluator = FactorEvaluator(ledger=self.ledger, config=self.config)

    def evaluate_batch(
        self,
        factors: list[dict[str, str]],
        data: pd.DataFrame,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[EvaluationResult]:
        """Evaluate multiple factors.

        Args:
            factors: List of dicts with 'name' and 'family' keys
            data: DataFrame with factor values
            on_progress: Optional callback for progress updates

        Returns:
            List of EvaluationResult for each factor
        """
        results = []
        total = len(factors)

        for i, factor in enumerate(factors):
            name = factor["name"]
            family = factor["family"]

            try:
                result = self.evaluator.evaluate(
                    factor_name=name,
                    factor_family=family,
                    data=data,
                )
                results.append(result)
            except Exception as e:
                # Create failed result
                results.append(
                    EvaluationResult(
                        factor_name=name,
                        factor_family=family,
                        metrics=FactorMetrics(),
                        passes_threshold=False,
                        threshold_used=0.0,
                        trial_id="",
                    )
                )

            if on_progress:
                on_progress(i + 1, total, name)

        return results

    def get_summary(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Generate summary of batch evaluation.

        Args:
            results: List of evaluation results

        Returns:
            Summary dictionary
        """
        total = len(results)
        passed = sum(1 for r in results if r.passes_threshold)

        ic_values = [r.metrics.ic for r in results if r.metrics.ic != 0]
        ir_values = [r.metrics.ir for r in results if r.metrics.ir != 0]

        return {
            "total_evaluated": total,
            "passed_count": passed,
            "pass_rate": passed / total if total > 0 else 0,
            "avg_ic": np.mean(ic_values) if ic_values else 0,
            "avg_ir": np.mean(ir_values) if ir_values else 0,
            "best_factor": max(results, key=lambda r: r.metrics.ic).factor_name
            if results
            else None,
        }
