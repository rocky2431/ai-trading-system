"""Tests for FactorEvaluator (Task 13).

Six-dimensional test coverage:
1. Functional: Metrics calculation, evaluation pipeline
2. Boundary: Edge cases for data sizes and values
3. Exception: Error handling for invalid inputs
4. Performance: Evaluation speed
5. Security: Data integrity
6. Compatibility: Different data formats
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any
from unittest.mock import Mock, patch

from iqfmp.evaluation.factor_evaluator import (
    FactorEvaluator,
    FactorMetrics,
    FactorReport,
    EvaluationConfig,
    EvaluationResult,
    MetricsCalculator,
    EvaluationPipeline,
    InvalidFactorError,
    EvaluationFailedError,
)
from iqfmp.evaluation.research_ledger import ResearchLedger, MemoryStorage


@pytest.fixture
def sample_factor_data() -> pd.DataFrame:
    """Create sample factor data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=365, freq="D")
    symbols = ["BTC", "ETH", "SOL", "DOGE", "XRP"]

    data = []
    for date in dates:
        for symbol in symbols:
            factor_value = np.random.randn()
            # Create correlated forward return
            forward_return = factor_value * 0.01 + np.random.randn() * 0.02
            data.append({
                "date": date,
                "symbol": symbol,
                "factor_value": factor_value,
                "forward_return": forward_return,
                "market_cap": np.random.uniform(1e9, 1e12),
            })

    return pd.DataFrame(data)


@pytest.fixture
def evaluator() -> FactorEvaluator:
    """Create factor evaluator with memory storage."""
    ledger = ResearchLedger(storage=MemoryStorage())
    return FactorEvaluator(ledger=ledger)


@pytest.fixture
def metrics_calculator() -> MetricsCalculator:
    """Create metrics calculator."""
    return MetricsCalculator()


class TestMetricsCalculator:
    """Tests for metrics calculation."""

    def test_calculate_ic(
        self, metrics_calculator: MetricsCalculator, sample_factor_data: pd.DataFrame
    ) -> None:
        """Test IC calculation."""
        df = sample_factor_data.set_index(["date", "symbol"])
        ic = metrics_calculator.calculate_ic(
            df["factor_value"],
            df["forward_return"],
        )

        assert isinstance(ic, float)
        assert -1 <= ic <= 1

    def test_calculate_rank_ic(
        self, metrics_calculator: MetricsCalculator, sample_factor_data: pd.DataFrame
    ) -> None:
        """Test Rank IC calculation."""
        df = sample_factor_data.set_index(["date", "symbol"])
        rank_ic = metrics_calculator.calculate_rank_ic(
            df["factor_value"],
            df["forward_return"],
        )

        assert isinstance(rank_ic, float)
        assert -1 <= rank_ic <= 1

    def test_calculate_ir(
        self, metrics_calculator: MetricsCalculator
    ) -> None:
        """Test IR calculation from IC series."""
        ic_series = pd.Series([0.05, 0.03, 0.04, 0.06, 0.02, 0.05])
        ir = metrics_calculator.calculate_ir(ic_series)

        expected_ir = ic_series.mean() / ic_series.std()
        assert ir == pytest.approx(expected_ir, rel=0.01)

    def test_calculate_sharpe_ratio(
        self, metrics_calculator: MetricsCalculator
    ) -> None:
        """Test Sharpe ratio calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01, -0.02])
        sharpe = metrics_calculator.calculate_sharpe_ratio(returns)

        assert isinstance(sharpe, float)

    def test_calculate_max_drawdown(
        self, metrics_calculator: MetricsCalculator
    ) -> None:
        """Test max drawdown calculation."""
        cumulative_returns = pd.Series([1.0, 1.1, 1.05, 0.95, 1.0, 1.15])
        max_dd = metrics_calculator.calculate_max_drawdown(cumulative_returns)

        assert max_dd >= 0
        assert max_dd <= 1

    def test_calculate_win_rate(
        self, metrics_calculator: MetricsCalculator
    ) -> None:
        """Test win rate calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01, -0.02])
        win_rate = metrics_calculator.calculate_win_rate(returns)

        assert 0 <= win_rate <= 1
        assert win_rate == pytest.approx(4/6, rel=0.01)

    def test_calculate_turnover(
        self, metrics_calculator: MetricsCalculator
    ) -> None:
        """Test turnover calculation."""
        positions_t0 = pd.Series([1.0, 0.5, 0.0, -0.5])
        positions_t1 = pd.Series([0.5, 0.0, 0.5, -1.0])

        turnover = metrics_calculator.calculate_turnover(positions_t0, positions_t1)

        assert turnover >= 0


class TestFactorMetrics:
    """Tests for FactorMetrics data structure."""

    def test_create_metrics(self) -> None:
        """Test creating metrics."""
        metrics = FactorMetrics(
            ic=0.05,
            rank_ic=0.04,
            ir=1.5,
            sharpe_ratio=1.2,
            max_drawdown=0.15,
            win_rate=0.55,
            turnover=0.3,
        )

        assert metrics.ic == 0.05
        assert metrics.ir == 1.5
        assert metrics.sharpe_ratio == 1.2

    def test_metrics_to_dict(self) -> None:
        """Test metrics serialization."""
        metrics = FactorMetrics(
            ic=0.05,
            rank_ic=0.04,
            ir=1.5,
            sharpe_ratio=1.2,
        )

        d = metrics.to_dict()
        assert "ic" in d
        assert "ir" in d
        assert d["ic"] == 0.05

    def test_metrics_from_dict(self) -> None:
        """Test metrics deserialization."""
        data = {
            "ic": 0.05,
            "rank_ic": 0.04,
            "ir": 1.5,
            "sharpe_ratio": 1.2,
        }

        metrics = FactorMetrics.from_dict(data)
        assert metrics.ic == 0.05
        assert metrics.ir == 1.5

    def test_metrics_is_significant(self) -> None:
        """Test significance check."""
        good_metrics = FactorMetrics(ic=0.05, ir=2.0, sharpe_ratio=1.5)
        bad_metrics = FactorMetrics(ic=0.01, ir=0.5, sharpe_ratio=0.3)

        assert good_metrics.is_significant(ic_threshold=0.03, ir_threshold=1.0)
        assert not bad_metrics.is_significant(ic_threshold=0.03, ir_threshold=1.0)


class TestFactorEvaluator:
    """Tests for FactorEvaluator."""

    def test_evaluate_factor(
        self, evaluator: FactorEvaluator, sample_factor_data: pd.DataFrame
    ) -> None:
        """Test basic factor evaluation."""
        result = evaluator.evaluate(
            factor_name="test_factor",
            factor_family="momentum",
            data=sample_factor_data,
        )

        assert isinstance(result, EvaluationResult)
        assert result.factor_name == "test_factor"
        assert result.metrics is not None

    def test_evaluate_with_cv_splits(
        self, evaluator: FactorEvaluator, sample_factor_data: pd.DataFrame
    ) -> None:
        """Test evaluation with cross-validation splits."""
        config = EvaluationConfig(use_cv_splits=True)
        evaluator_with_cv = FactorEvaluator(
            ledger=ResearchLedger(storage=MemoryStorage()),
            config=config,
        )

        result = evaluator_with_cv.evaluate(
            factor_name="test_factor",
            factor_family="momentum",
            data=sample_factor_data,
        )

        assert result.cv_results is not None
        assert len(result.cv_results) > 0

    def test_evaluate_records_to_ledger(
        self, evaluator: FactorEvaluator, sample_factor_data: pd.DataFrame
    ) -> None:
        """Test that evaluation records trial to ledger."""
        evaluator.evaluate(
            factor_name="test_factor",
            factor_family="momentum",
            data=sample_factor_data,
        )

        assert evaluator.ledger.get_trial_count() == 1

    def test_evaluate_with_stability_analysis(
        self, evaluator: FactorEvaluator, sample_factor_data: pd.DataFrame
    ) -> None:
        """Test evaluation with stability analysis."""
        config = EvaluationConfig(run_stability_analysis=True)
        evaluator_with_stability = FactorEvaluator(
            ledger=ResearchLedger(storage=MemoryStorage()),
            config=config,
        )

        result = evaluator_with_stability.evaluate(
            factor_name="test_factor",
            factor_family="momentum",
            data=sample_factor_data,
        )

        assert result.stability_report is not None

    def test_evaluate_checks_threshold(
        self, evaluator: FactorEvaluator, sample_factor_data: pd.DataFrame
    ) -> None:
        """Test that evaluation checks dynamic threshold."""
        result = evaluator.evaluate(
            factor_name="test_factor",
            factor_family="momentum",
            data=sample_factor_data,
        )

        assert result.passes_threshold is not None
        assert result.threshold_used > 0

    def test_multiple_evaluations_increase_threshold(
        self, evaluator: FactorEvaluator, sample_factor_data: pd.DataFrame
    ) -> None:
        """Test that multiple evaluations increase threshold."""
        # First evaluation
        result1 = evaluator.evaluate(
            factor_name="factor_1",
            factor_family="momentum",
            data=sample_factor_data,
        )
        threshold1 = result1.threshold_used

        # Multiple more evaluations
        for i in range(10):
            evaluator.evaluate(
                factor_name=f"factor_{i+2}",
                factor_family="momentum",
                data=sample_factor_data,
            )

        # Last evaluation
        result_last = evaluator.evaluate(
            factor_name="factor_last",
            factor_family="momentum",
            data=sample_factor_data,
        )
        threshold_last = result_last.threshold_used

        assert threshold_last > threshold1


class TestFactorReport:
    """Tests for FactorReport."""

    def test_create_report(
        self, evaluator: FactorEvaluator, sample_factor_data: pd.DataFrame
    ) -> None:
        """Test report creation."""
        result = evaluator.evaluate(
            factor_name="test_factor",
            factor_family="momentum",
            data=sample_factor_data,
        )

        report = result.generate_report()

        assert isinstance(report, FactorReport)
        assert report.factor_name == "test_factor"

    def test_report_to_dict(
        self, evaluator: FactorEvaluator, sample_factor_data: pd.DataFrame
    ) -> None:
        """Test report serialization."""
        result = evaluator.evaluate(
            factor_name="test_factor",
            factor_family="momentum",
            data=sample_factor_data,
        )

        report = result.generate_report()
        d = report.to_dict()

        assert "factor_name" in d
        assert "metrics" in d
        assert "evaluation_date" in d

    def test_report_summary(
        self, evaluator: FactorEvaluator, sample_factor_data: pd.DataFrame
    ) -> None:
        """Test report summary generation."""
        result = evaluator.evaluate(
            factor_name="test_factor",
            factor_family="momentum",
            data=sample_factor_data,
        )

        report = result.generate_report()
        summary = report.get_summary()

        assert isinstance(summary, str)
        assert "test_factor" in summary

    def test_report_grade(
        self, evaluator: FactorEvaluator, sample_factor_data: pd.DataFrame
    ) -> None:
        """Test report grade assignment."""
        result = evaluator.evaluate(
            factor_name="test_factor",
            factor_family="momentum",
            data=sample_factor_data,
        )

        report = result.generate_report()

        assert report.grade in ["A", "B", "C", "D", "F"]


class TestEvaluationPipeline:
    """Tests for batch evaluation pipeline."""

    def test_pipeline_creation(self) -> None:
        """Test pipeline creation."""
        ledger = ResearchLedger(storage=MemoryStorage())
        pipeline = EvaluationPipeline(ledger=ledger)

        assert pipeline is not None

    def test_pipeline_evaluate_batch(
        self, sample_factor_data: pd.DataFrame
    ) -> None:
        """Test batch evaluation."""
        ledger = ResearchLedger(storage=MemoryStorage())
        pipeline = EvaluationPipeline(ledger=ledger)

        factors = [
            {"name": "factor_1", "family": "momentum"},
            {"name": "factor_2", "family": "value"},
            {"name": "factor_3", "family": "volatility"},
        ]

        results = pipeline.evaluate_batch(
            factors=factors,
            data=sample_factor_data,
        )

        assert len(results) == 3
        assert all(isinstance(r, EvaluationResult) for r in results)

    def test_pipeline_with_callback(
        self, sample_factor_data: pd.DataFrame
    ) -> None:
        """Test pipeline with progress callback."""
        ledger = ResearchLedger(storage=MemoryStorage())
        pipeline = EvaluationPipeline(ledger=ledger)

        progress_calls = []

        def on_progress(current: int, total: int, factor_name: str) -> None:
            progress_calls.append((current, total, factor_name))

        factors = [
            {"name": "factor_1", "family": "momentum"},
            {"name": "factor_2", "family": "value"},
        ]

        pipeline.evaluate_batch(
            factors=factors,
            data=sample_factor_data,
            on_progress=on_progress,
        )

        assert len(progress_calls) == 2

    def test_pipeline_summary(
        self, sample_factor_data: pd.DataFrame
    ) -> None:
        """Test pipeline summary generation."""
        ledger = ResearchLedger(storage=MemoryStorage())
        pipeline = EvaluationPipeline(ledger=ledger)

        factors = [
            {"name": "factor_1", "family": "momentum"},
            {"name": "factor_2", "family": "value"},
        ]

        results = pipeline.evaluate_batch(
            factors=factors,
            data=sample_factor_data,
        )

        summary = pipeline.get_summary(results)

        assert "total_evaluated" in summary
        assert "passed_count" in summary


class TestEvaluationBoundary:
    """Boundary tests."""

    def test_minimum_data(
        self, evaluator: FactorEvaluator
    ) -> None:
        """Test with minimum required data."""
        np.random.seed(42)
        df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=30),
            "symbol": ["BTC"] * 30,
            "factor_value": np.random.randn(30),
            "forward_return": np.random.randn(30) * 0.02,
            "market_cap": [1e11] * 30,
        })

        result = evaluator.evaluate(
            factor_name="test_factor",
            factor_family="momentum",
            data=df,
        )

        assert result is not None

    def test_extreme_factor_values(
        self, evaluator: FactorEvaluator
    ) -> None:
        """Test with extreme factor values."""
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=100)

        df = pd.DataFrame({
            "date": dates,
            "symbol": ["BTC"] * 100,
            "factor_value": [1e10, -1e10] * 50,  # Extreme values
            "forward_return": np.random.randn(100) * 0.02,
            "market_cap": [1e11] * 100,
        })

        result = evaluator.evaluate(
            factor_name="extreme_factor",
            factor_family="momentum",
            data=df,
        )

        assert result is not None
        assert not np.isnan(result.metrics.ic)

    def test_zero_variance_factor(
        self, evaluator: FactorEvaluator
    ) -> None:
        """Test with zero variance factor."""
        df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=100),
            "symbol": ["BTC"] * 100,
            "factor_value": [1.0] * 100,  # Constant value
            "forward_return": np.random.randn(100) * 0.02,
            "market_cap": [1e11] * 100,
        })

        result = evaluator.evaluate(
            factor_name="constant_factor",
            factor_family="momentum",
            data=df,
        )

        # Should handle gracefully
        assert result is not None


class TestEvaluationException:
    """Exception handling tests."""

    def test_empty_data(
        self, evaluator: FactorEvaluator
    ) -> None:
        """Test with empty data."""
        df = pd.DataFrame()

        with pytest.raises(InvalidFactorError, match="empty"):
            evaluator.evaluate(
                factor_name="test",
                factor_family="momentum",
                data=df,
            )

    def test_missing_columns(
        self, evaluator: FactorEvaluator
    ) -> None:
        """Test with missing required columns."""
        df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=10),
            "symbol": ["BTC"] * 10,
            # Missing factor_value and forward_return
        })

        with pytest.raises(InvalidFactorError, match="column"):
            evaluator.evaluate(
                factor_name="test",
                factor_family="momentum",
                data=df,
            )

    def test_invalid_factor_name(
        self, evaluator: FactorEvaluator, sample_factor_data: pd.DataFrame
    ) -> None:
        """Test with invalid factor name."""
        with pytest.raises(InvalidFactorError, match="name"):
            evaluator.evaluate(
                factor_name="",
                factor_family="momentum",
                data=sample_factor_data,
            )


class TestEvaluationPerformance:
    """Performance tests."""

    def test_evaluation_speed(
        self, evaluator: FactorEvaluator
    ) -> None:
        """Test evaluation completion time."""
        import time

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=1000)
        symbols = [f"SYM_{i}" for i in range(20)]

        data = []
        for date in dates:
            for symbol in symbols:
                data.append({
                    "date": date,
                    "symbol": symbol,
                    "factor_value": np.random.randn(),
                    "forward_return": np.random.randn() * 0.02,
                    "market_cap": np.random.uniform(1e9, 1e12),
                })

        df = pd.DataFrame(data)

        start = time.time()
        evaluator.evaluate(
            factor_name="perf_test",
            factor_family="momentum",
            data=df,
        )
        elapsed = time.time() - start

        assert elapsed < 10.0, f"Evaluation took {elapsed}s"

    def test_batch_evaluation_speed(self) -> None:
        """Test batch evaluation speed."""
        import time

        np.random.seed(42)
        df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=365).repeat(5),
            "symbol": ["BTC", "ETH", "SOL", "DOGE", "XRP"] * 365,
            "factor_value": np.random.randn(365 * 5),
            "forward_return": np.random.randn(365 * 5) * 0.02,
            "market_cap": np.random.uniform(1e9, 1e12, 365 * 5),
        })

        ledger = ResearchLedger(storage=MemoryStorage())
        pipeline = EvaluationPipeline(ledger=ledger)

        factors = [{"name": f"factor_{i}", "family": "momentum"} for i in range(10)]

        start = time.time()
        pipeline.evaluate_batch(factors=factors, data=df)
        elapsed = time.time() - start

        assert elapsed < 30.0, f"Batch evaluation took {elapsed}s"


class TestEvaluationCompatibility:
    """Compatibility tests."""

    def test_datetime_index(
        self, evaluator: FactorEvaluator
    ) -> None:
        """Test with datetime index."""
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=100)

        df = pd.DataFrame({
            "symbol": ["BTC"] * 100,
            "factor_value": np.random.randn(100),
            "forward_return": np.random.randn(100) * 0.02,
            "market_cap": [1e11] * 100,
        }, index=dates)

        result = evaluator.evaluate(
            factor_name="test",
            factor_family="momentum",
            data=df,
        )

        assert result is not None

    def test_string_dates(
        self, evaluator: FactorEvaluator
    ) -> None:
        """Test with string date column."""
        np.random.seed(42)

        df = pd.DataFrame({
            "date": [f"2022-{m:02d}-01" for m in range(1, 13)] * 10,
            "symbol": ["BTC"] * 120,
            "factor_value": np.random.randn(120),
            "forward_return": np.random.randn(120) * 0.02,
            "market_cap": [1e11] * 120,
        })

        result = evaluator.evaluate(
            factor_name="test",
            factor_family="momentum",
            data=df,
        )

        assert result is not None

    def test_custom_column_names(
        self, evaluator: FactorEvaluator
    ) -> None:
        """Test with custom column names."""
        np.random.seed(42)

        df = pd.DataFrame({
            "timestamp": pd.date_range("2022-01-01", periods=100),
            "ticker": ["BTC"] * 100,
            "alpha": np.random.randn(100),
            "ret": np.random.randn(100) * 0.02,
            "mcap": [1e11] * 100,
        })

        config = EvaluationConfig(
            date_column="timestamp",
            symbol_column="ticker",
            factor_column="alpha",
            return_column="ret",
            market_cap_column="mcap",
        )

        evaluator_custom = FactorEvaluator(
            ledger=ResearchLedger(storage=MemoryStorage()),
            config=config,
        )

        result = evaluator_custom.evaluate(
            factor_name="test",
            factor_family="momentum",
            data=df,
        )

        assert result is not None
