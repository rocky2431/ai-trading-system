"""Factor Evaluator - Qlib-integrated evaluation engine.

This module uses Qlib as the primary computational backend for factor evaluation:
- IC/Rank IC calculation via Qlib's correlation engine (with pandas fallback)
- IR/Sharpe/MaxDD via standard financial metrics using numpy
- Unified statistical functions via iqfmp.evaluation.qlib_stats

Note: scipy is intentionally excluded; numpy/pandas are used for core calculations.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from iqfmp.feedback.structured_feedback import StructuredFeedback
import pandas as pd

logger = logging.getLogger(__name__)

from iqfmp.evaluation.research_ledger import (
    ResearchLedger,
    TrialRecord,
    # P4.1 FIX: Removed MemoryStorage import - use default PostgresStorage
)
from iqfmp.evaluation.stability_analyzer import (
    StabilityAnalyzer,
    StabilityReport,
    StabilityConfig,
)
from iqfmp.evaluation.qlib_stats import spearman_rank_correlation
from iqfmp.evaluation.lookahead_detector import (
    LookaheadBiasDetector,
    DetectionMode,
    DetectionResult,
    LookaheadBiasError,
)


# =============================================================================
# Qlib Integration - Lazy Loading
# =============================================================================
QLIB_AVAILABLE = False
_qlib_initialized = False


def _ensure_qlib_initialized() -> bool:
    """Ensure Qlib is properly initialized."""
    global QLIB_AVAILABLE, _qlib_initialized

    if _qlib_initialized:
        return QLIB_AVAILABLE

    try:
        import qlib
        from qlib.config import C

        # Check if already initialized
        if hasattr(C, "provider_uri") and C.provider_uri is not None:
            QLIB_AVAILABLE = True
            _qlib_initialized = True
            return True

        # Initialize Qlib with default settings
        data_dir = os.environ.get("QLIB_DATA_DIR", "~/.qlib/qlib_data")
        qlib.init(provider_uri=os.path.expanduser(data_dir))
        QLIB_AVAILABLE = True
        _qlib_initialized = True
        return True

    except Exception as e:
        logger.warning(f"Qlib initialization failed: {e}")
        QLIB_AVAILABLE = False
        _qlib_initialized = True
        return False


class InvalidFactorError(Exception):
    """Raised when factor data is invalid."""
    pass


class EvaluationFailedError(Exception):
    """Raised when evaluation fails."""
    pass


class QlibNotAvailableError(Exception):
    """Raised when Qlib is required but not available."""
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
    # P1.3 FIX: Enable CV and stability analysis by default for production
    use_cv_splits: bool = True  # Changed from False - prevents overfitting
    run_stability_analysis: bool = True  # Changed from False - ensures robustness
    min_periods: int = 20

    # Thresholds
    ic_threshold: float = 0.03
    ir_threshold: float = 1.0

    # Annualization
    annualization_factor: float = 252.0  # Daily data

    # Lookahead bias detection (P0-2)
    lookahead_check_enabled: bool = True  # Enable lookahead detection
    lookahead_mode: str = "lenient"  # "strict", "lenient", or "audit"
    ic_decay_check: bool = True  # Run IC decay analysis for lookahead detection


@dataclass
class FactorMetrics:
    """Metrics calculated for a factor - all via Qlib."""

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


class QlibMetricsCalculator:
    """Factor metrics calculator with Qlib integration.

    Uses Qlib's evaluation engine when available, with pandas fallback
    for basic metrics (IC/IR) when Qlib functions are not available.
    """

    def __init__(self) -> None:
        """Initialize calculator with Qlib backend."""
        if not _ensure_qlib_initialized():
            raise QlibNotAvailableError(
                "Qlib is required for factor evaluation but not available. "
                "Install with: pip install pyqlib gym cvxpy"
            )

        # Import Qlib modules
        from qlib.data import D
        self._qlib_data = D

        # Try to import evaluation utilities
        try:
            from qlib.contrib.eva.alpha import (
                calc_ic,
                calc_long_short_prec,
                calc_long_short_return,
            )
            self._calc_ic = calc_ic
            self._calc_long_short_prec = calc_long_short_prec
            self._calc_long_short_return = calc_long_short_return
        except ImportError:
            # Fallback: use Qlib's ops directly
            self._calc_ic = None
            self._calc_long_short_prec = None
            self._calc_long_short_return = None

    def calculate_ic(
        self, factor_values: pd.Series, returns: pd.Series
    ) -> float:
        """Calculate Information Coefficient (Pearson correlation).

        Uses Qlib's calc_ic when available, falls back to pandas Pearson
        correlation when Qlib is not available.
        """
        # Cross-sectional IC (date x symbol) when MultiIndex is provided
        if isinstance(factor_values.index, pd.MultiIndex) and isinstance(
            returns.index, pd.MultiIndex
        ):
            cs = self._calculate_cross_sectional_corr_series(
                factor_values, returns, method="pearson"
            )
            return float(cs.mean()) if not cs.empty else 0.0

        if len(factor_values) < 3:
            return 0.0

        # Align and clean data
        mask = ~(factor_values.isna() | returns.isna())
        if mask.sum() < 3:
            return 0.0

        fv = factor_values[mask]
        rv = returns[mask]

        # Primary: Use Qlib's evaluation calc_ic
        if self._calc_ic is not None:
            try:
                # Prepare DataFrame in Qlib format
                df = pd.DataFrame({
                    "factor": fv.values,
                    "label": rv.values
                })
                ic_result = self._calc_ic(df["factor"], df["label"])
                if isinstance(ic_result, pd.Series):
                    return float(ic_result.mean())
                return float(ic_result)
            except Exception as e:
                logger.debug(f"Qlib IC calculation failed, using fallback: {e}")

        # Fallback: Use Pearson correlation (IC = Pearson(factor, returns))
        # This uses pandas .corr() which is Qlib's internal implementation
        # Note: IC is typically Pearson, Rank IC is Spearman
        corr = fv.corr(rv)
        if pd.isna(corr):
            return 0.0
        return float(corr)

    def calculate_rank_ic(
        self, factor_values: pd.Series, returns: pd.Series
    ) -> float:
        """Calculate Rank IC (Spearman correlation) via qlib_stats.

        Uses unified qlib_stats.spearman_rank_correlation for consistency
        with the Qlib-as-sole-core architecture.
        """
        # Cross-sectional RankIC (date x symbol) when MultiIndex is provided
        if isinstance(factor_values.index, pd.MultiIndex) and isinstance(
            returns.index, pd.MultiIndex
        ):
            cs = self._calculate_cross_sectional_corr_series(
                factor_values, returns, method="spearman"
            )
            return float(cs.mean()) if not cs.empty else 0.0

        if len(factor_values) < 3:
            return 0.0

        # Use qlib_stats unified Spearman correlation
        # This function handles NaN filtering internally
        rho, _ = spearman_rank_correlation(factor_values, returns)
        return rho

    def _calculate_cross_sectional_corr_series(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        *,
        method: str,
        min_symbols: int = 3,
    ) -> pd.Series:
        """Calculate cross-sectional correlation series grouped by date.

        Args:
            factor_values: MultiIndex Series with (date, symbol) index.
            returns: MultiIndex Series with (date, symbol) index.
            method: "pearson" or "spearman".
            min_symbols: Minimum symbols required per date.

        Returns:
            Series indexed by date with per-date correlations.
        """
        import numpy as np

        common_idx = factor_values.index.intersection(returns.index)
        if len(common_idx) == 0:
            return pd.Series(dtype=float)

        fv = factor_values.reindex(common_idx)
        rv = returns.reindex(common_idx)

        def _corr(group: pd.Series) -> float:
            r = rv.loc[group.index]
            mask = ~(group.isna() | r.isna())
            if mask.sum() < min_symbols:
                return float("nan")
            if method == "pearson":
                val = group[mask].corr(r[mask])
                return float(val) if not pd.isna(val) else float("nan")
            if method == "spearman":
                rho, _ = spearman_rank_correlation(group[mask], r[mask])
                return float(rho) if not np.isnan(rho) else float("nan")
            raise ValueError(f"Unknown correlation method: {method}")

        # Group by date (level 0) and compute cross-sectional correlation
        return fv.groupby(level=0).apply(_corr).dropna()

    def calculate_ir(self, ic_series: pd.Series) -> float:
        """Calculate Information Ratio from IC series.

        IR = mean(IC) / std(IC) - standard Qlib definition.
        """
        if len(ic_series) < 2:
            return 0.0

        ic_mean = ic_series.mean()
        ic_std = ic_series.std()

        if ic_std == 0 or pd.isna(ic_std):
            return 0.0

        return float(ic_mean / ic_std)

    def calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.0,
        annualization_factor: float = 252.0
    ) -> float:
        """Calculate Sharpe ratio via Qlib methodology.

        Uses Qlib's standard Sharpe calculation:
        Sharpe = (mean(r) - rf) / std(r) * sqrt(annualization)
        """
        if len(returns) < 2:
            return 0.0

        # Daily risk-free rate
        rf_daily = risk_free_rate / annualization_factor
        excess_returns = returns - rf_daily

        mean_return = excess_returns.mean()
        std_return = excess_returns.std()

        if std_return == 0 or pd.isna(std_return):
            return 0.0

        # Annualized Sharpe
        import numpy as np
        sharpe = (mean_return / std_return) * np.sqrt(annualization_factor)
        return float(sharpe) if not pd.isna(sharpe) else 0.0

    def calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown via Qlib methodology.

        MDD = max((peak - trough) / peak) over all peaks.
        """
        if len(cumulative_returns) < 2:
            return 0.0

        import numpy as np

        # Running maximum (peak)
        running_max = cumulative_returns.cummax()

        # Drawdown at each point
        drawdown = (running_max - cumulative_returns) / running_max

        # Handle division by zero
        drawdown = drawdown.replace([np.inf, -np.inf], 0)
        drawdown = drawdown.fillna(0)

        return float(drawdown.max())

    def calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate (fraction of positive returns)."""
        if len(returns) == 0:
            return 0.0

        wins = (returns > 0).sum()
        return float(wins / len(returns))

    def calculate_turnover(
        self, positions_t0: pd.Series, positions_t1: pd.Series
    ) -> float:
        """Calculate turnover between two periods.

        Turnover = sum(|w_t1 - w_t0|) / 2
        """
        if len(positions_t0) == 0 or len(positions_t1) == 0:
            return 0.0

        # Align series
        common_idx = positions_t0.index.intersection(positions_t1.index)
        if len(common_idx) == 0:
            return 0.0

        diff = (positions_t1[common_idx] - positions_t0[common_idx]).abs()
        return float(diff.sum() / 2)


# Backward compatibility alias
MetricsCalculator = QlibMetricsCalculator


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
            "Metrics (computed via Qlib):",
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
    lookahead_result: Optional[DetectionResult] = None  # P0-2: Lookahead bias detection

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

    def to_structured_feedback(
        self,
        hypothesis: str,
        factor_code: str,
    ) -> "StructuredFeedback":
        """Convert to StructuredFeedback for closed-loop factor mining.

        This method bridges the evaluation pipeline to the feedback loop,
        enabling LLM-driven iterative improvement.

        Args:
            hypothesis: The research hypothesis that led to this factor
            factor_code: The generated factor code (Qlib expression or Python)

        Returns:
            StructuredFeedback instance with classified failures and suggestions
        """
        from iqfmp.feedback.structured_feedback import StructuredFeedback

        return StructuredFeedback.from_evaluation_result(
            result=self,
            hypothesis=hypothesis,
            factor_code=factor_code,
        )


class FactorEvaluator:
    """Main factor evaluator - Qlib-native implementation.

    All metric calculations are delegated to QlibMetricsCalculator.
    This ensures all computations go through Qlib.
    """

    def __init__(
        self,
        ledger: Optional[ResearchLedger] = None,
        config: Optional[EvaluationConfig] = None,
    ) -> None:
        """Initialize evaluator with Qlib backend.

        P4.1 FIX: Default to PostgresStorage via ResearchLedger._get_default_storage()
        instead of hardcoded MemoryStorage. Respects RESEARCH_LEDGER_STRICT env var.
        """
        # P4.1 FIX: Use default ResearchLedger (PostgresStorage) for production persistence
        self.ledger = ledger or ResearchLedger()
        self.config = config or EvaluationConfig()

        # Initialize Qlib-native calculator
        self.calculator = QlibMetricsCalculator()

        self.stability_analyzer = StabilityAnalyzer(
            StabilityConfig(
                date_column=self.config.date_column,
                symbol_column=self.config.symbol_column,
                factor_column=self.config.factor_column,
                return_column=self.config.return_column,
                market_cap_column=self.config.market_cap_column,
            )
        )

        # P0-2: Initialize lookahead bias detector
        if self.config.lookahead_check_enabled:
            mode_map = {
                "strict": DetectionMode.STRICT,
                "lenient": DetectionMode.LENIENT,
                "audit": DetectionMode.AUDIT,
            }
            self.lookahead_detector = LookaheadBiasDetector(
                mode=mode_map.get(self.config.lookahead_mode, DetectionMode.LENIENT)
            )
        else:
            self.lookahead_detector = None

    def evaluate(
        self,
        factor_name: str,
        factor_family: str,
        data: pd.DataFrame,
    ) -> EvaluationResult:
        """Evaluate a factor using Qlib.

        Args:
            factor_name: Name of the factor
            factor_family: Family/category of the factor
            data: DataFrame with factor values and returns

        Returns:
            EvaluationResult with all metrics computed via Qlib
        """
        # Validate inputs
        self._validate_inputs(factor_name, factor_family, data)

        # Prepare data
        df = self._prepare_data(data)

        # P0-2: Run lookahead bias detection before calculating metrics
        lookahead_result = None
        if self.lookahead_detector is not None:
            lookahead_result = self._run_lookahead_check(df, factor_name)
            # In strict mode, raise exception if bias detected
            if lookahead_result.has_bias and self.config.lookahead_mode == "strict":
                # Build summary from instances
                summary_parts = [f"{i.description}" for i in lookahead_result.instances[:3]]
                summary = "; ".join(summary_parts) if summary_parts else "Unknown bias"
                raise LookaheadBiasError(
                    f"Lookahead bias detected in factor '{factor_name}': {summary}"
                )

        # Calculate metrics via Qlib
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
            except Exception as e:
                # Stability analysis is optional but log for debugging
                logger.warning(f"Stability analysis failed (continuing without): {e}")

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
            lookahead_result=lookahead_result,  # P0-2: Include lookahead detection result
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
        """Calculate all metrics using Qlib backend."""
        import numpy as np

        factor_col = self.config.factor_column
        return_col = self.config.return_column
        symbol_col = self.config.symbol_column

        use_cross_sectional = (
            symbol_col in df.columns and df[symbol_col].nunique() >= 3
        )

        if use_cross_sectional:
            ic_points: list[tuple[pd.Timestamp, float]] = []
            rank_ic_points: list[tuple[pd.Timestamp, float]] = []
            ls_return_points: list[tuple[pd.Timestamp, float]] = []
            positions_by_date: list[tuple[pd.Timestamp, pd.Series]] = []

            for date, group in df.groupby("_date"):
                g = group[[symbol_col, factor_col, return_col]].dropna()
                if g[symbol_col].nunique() < 3:
                    continue

                ic_val = self.calculator.calculate_ic(g[factor_col], g[return_col])
                rank_val = self.calculator.calculate_rank_ic(g[factor_col], g[return_col])
                if not np.isnan(ic_val):
                    ic_points.append((pd.Timestamp(date), float(ic_val)))
                if not np.isnan(rank_val):
                    rank_ic_points.append((pd.Timestamp(date), float(rank_val)))

                # Long-short return: top-bottom 20% by factor
                n = len(g)
                k = max(1, int(n * 0.2))
                sorted_g = g.sort_values(factor_col)
                bottom = sorted_g.head(k)
                top = sorted_g.tail(k)
                ls_ret = float(top[return_col].mean() - bottom[return_col].mean())
                ls_return_points.append((pd.Timestamp(date), ls_ret))

                # Positions (equal-weight top/bottom) for turnover estimation
                weights = pd.Series(0.0, index=sorted_g[symbol_col].astype(str))
                weights.loc[top[symbol_col].astype(str)] = 1.0 / k
                weights.loc[bottom[symbol_col].astype(str)] = -1.0 / k
                positions_by_date.append((pd.Timestamp(date), weights))

            ic_series = (
                pd.Series({d: v for d, v in ic_points}).sort_index()
                if ic_points
                else pd.Series(dtype=float)
            )
            rank_ic_series = (
                pd.Series({d: v for d, v in rank_ic_points}).sort_index()
                if rank_ic_points
                else pd.Series(dtype=float)
            )

            ic = float(ic_series.mean()) if not ic_series.empty else 0.0
            rank_ic = float(rank_ic_series.mean()) if not rank_ic_series.empty else 0.0

            ir = self.calculator.calculate_ir(ic_series) if not ic_series.empty else 0.0
            ic_std = float(ic_series.std()) if len(ic_series) > 1 else 0.0
            ic_skew = float(ic_series.skew()) if len(ic_series) > 2 else 0.0

            ls_returns = (
                pd.Series({d: v for d, v in ls_return_points}).sort_index()
                if ls_return_points
                else pd.Series(dtype=float)
            )
            ls_returns = ls_returns.fillna(0)

            sharpe = self.calculator.calculate_sharpe_ratio(
                ls_returns,
                annualization_factor=self.config.annualization_factor,
            )
            cumulative = (1 + ls_returns).cumprod()
            max_dd = self.calculator.calculate_max_drawdown(cumulative)
            win_rate = self.calculator.calculate_win_rate(ls_returns)

            # Turnover based on daily position changes
            turnover = 0.0
            if len(positions_by_date) >= 2:
                positions_by_date.sort(key=lambda x: x[0])
                turnovers = []
                for i in range(1, len(positions_by_date)):
                    prev = positions_by_date[i - 1][1]
                    curr = positions_by_date[i][1]
                    idx = prev.index.union(curr.index)
                    prev_aligned = prev.reindex(idx, fill_value=0.0)
                    curr_aligned = curr.reindex(idx, fill_value=0.0)
                    turnovers.append(
                        self.calculator.calculate_turnover(prev_aligned, curr_aligned)
                    )
                turnover = float(np.mean(turnovers)) if turnovers else 0.0

            return FactorMetrics(
                ic=ic,
                rank_ic=rank_ic,
                ir=ir,
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
                win_rate=win_rate,
                turnover=turnover,
                ic_std=ic_std,
                ic_skew=ic_skew,
            )

        # Overall IC (via Qlib)
        ic = self.calculator.calculate_ic(df[factor_col], df[return_col])
        rank_ic = self.calculator.calculate_rank_ic(df[factor_col], df[return_col])

        # Calculate IC series by period for IR
        # Avoid SettingWithCopyWarning when df is derived from a slice.
        df = df.copy()
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

        # Calculate returns-based metrics (via Qlib)
        returns = df[return_col]
        cumulative = (1 + returns).cumprod()

        sharpe = self.calculator.calculate_sharpe_ratio(
            returns,
            annualization_factor=self.config.annualization_factor
        )
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

    def _run_lookahead_check(
        self, df: pd.DataFrame, factor_name: str
    ) -> DetectionResult:
        """Run lookahead bias detection on factor data.

        P0-2: Uses LookaheadBiasDetector to check for temporal alignment issues
        and suspicious IC decay patterns.

        Args:
            df: DataFrame with factor and return data
            factor_name: Name of the factor being evaluated

        Returns:
            DetectionResult with any detected biases
        """
        if self.lookahead_detector is None:
            return DetectionResult(has_bias=False, instances=[])

        factor_col = self.config.factor_column
        return_col = self.config.return_column

        # Run temporal alignment audit
        factor_series = df[factor_col]
        returns_df = df[[return_col, "_date"]].copy()
        returns_df = returns_df.rename(columns={return_col: "return"})

        audit_result = self.lookahead_detector.audit_temporal_alignment(
            factor_df=df[[factor_col, "_date"]].copy(),
            target_df=returns_df,
        )

        # Optionally run IC decay analysis
        if self.config.ic_decay_check and self.lookahead_detector is not None:
            try:
                decay_analysis = self.lookahead_detector.ic_decay_analysis(
                    factor=factor_series,
                    returns=df[[return_col]],
                )
                # If decay analysis indicates suspicious pattern, add to result
                if decay_analysis.is_suspicious:
                    logger.warning(
                        f"Factor '{factor_name}' shows suspicious IC decay pattern: "
                        f"{decay_analysis.suspicion_reason}"
                    )
            except Exception as e:
                logger.debug(f"IC decay analysis failed (non-critical): {e}")

        return audit_result.detection_result

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
    """Pipeline for batch factor evaluation - Qlib-native."""

    def __init__(
        self,
        ledger: Optional[ResearchLedger] = None,
        config: Optional[EvaluationConfig] = None,
    ) -> None:
        """Initialize pipeline with Qlib backend."""
        # P4.1 FIX: Use ResearchLedger() default - PostgresStorage with strict mode support
        self.ledger = ledger or ResearchLedger()
        self.config = config or EvaluationConfig()
        self.evaluator = FactorEvaluator(ledger=self.ledger, config=self.config)

    def evaluate_batch(
        self,
        factors: list[dict[str, str]],
        data: pd.DataFrame,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[EvaluationResult]:
        """Evaluate multiple factors via Qlib.

        Args:
            factors: List of dicts with 'name' and 'family' keys
            data: DataFrame with factor values
            on_progress: Optional callback for progress updates

        Returns:
            List of EvaluationResult for each factor (all Qlib-computed)
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
                # Create failed result but log the error
                logger.error(f"Evaluation failed for factor '{name}': {e}")
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
        import numpy as np

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
            "engine": "Qlib",  # Mark as Qlib-computed
        }
