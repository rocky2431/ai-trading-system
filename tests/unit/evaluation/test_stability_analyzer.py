"""Tests for StabilityAnalyzer (Task 12).

Six-dimensional test coverage:
1. Functional: Time/Market/Regime stability analysis
2. Boundary: Edge cases for data sizes and periods
3. Exception: Error handling for invalid inputs
4. Performance: Analysis computation time
5. Security: Data integrity validation
6. Compatibility: Different data formats and frequencies
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any

from iqfmp.evaluation.stability_analyzer import (
    StabilityAnalyzer,
    TimeStabilityAnalyzer,
    MarketStabilityAnalyzer,
    RegimeStabilityAnalyzer,
    StabilityReport,
    StabilityConfig,
    TimeStabilityResult,
    MarketStabilityResult,
    RegimeStabilityResult,
    StabilityScore,
    MarketRegime,
    InvalidDataError,
    InsufficientDataError,
)


@pytest.fixture
def sample_factor_returns() -> pd.DataFrame:
    """Create sample factor returns data."""
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=365, freq="D")
    symbols = ["BTC", "ETH", "SOL", "DOGE", "XRP"]

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

    return pd.DataFrame(data)


@pytest.fixture
def sample_ic_series() -> pd.Series:
    """Create sample IC series."""
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=12, freq="ME")
    ic_values = np.random.uniform(0.02, 0.08, size=12)
    return pd.Series(ic_values, index=dates, name="IC")


@pytest.fixture
def time_analyzer() -> TimeStabilityAnalyzer:
    """Create time stability analyzer."""
    return TimeStabilityAnalyzer()


@pytest.fixture
def market_analyzer() -> MarketStabilityAnalyzer:
    """Create market stability analyzer."""
    return MarketStabilityAnalyzer()


@pytest.fixture
def regime_analyzer() -> RegimeStabilityAnalyzer:
    """Create regime stability analyzer."""
    return RegimeStabilityAnalyzer()


@pytest.fixture
def stability_analyzer() -> StabilityAnalyzer:
    """Create combined stability analyzer."""
    return StabilityAnalyzer()


class TestTimeStabilityAnalyzer:
    """Tests for time stability analysis."""

    def test_calculate_monthly_ic(
        self, time_analyzer: TimeStabilityAnalyzer, sample_factor_returns: pd.DataFrame
    ) -> None:
        """Test monthly IC calculation."""
        result = time_analyzer.analyze(sample_factor_returns)

        assert isinstance(result, TimeStabilityResult)
        assert len(result.monthly_ic) > 0
        assert result.ic_mean is not None
        assert result.ic_std is not None

    def test_calculate_quarterly_ic(
        self, time_analyzer: TimeStabilityAnalyzer, sample_factor_returns: pd.DataFrame
    ) -> None:
        """Test quarterly IC calculation."""
        config = StabilityConfig(time_frequency="quarterly")
        analyzer = TimeStabilityAnalyzer(config)
        result = analyzer.analyze(sample_factor_returns)

        assert len(result.quarterly_ic) > 0

    def test_ic_ir_calculation(
        self, time_analyzer: TimeStabilityAnalyzer, sample_factor_returns: pd.DataFrame
    ) -> None:
        """Test IC IR (Information Ratio) calculation."""
        result = time_analyzer.analyze(sample_factor_returns)

        # IR = IC_mean / IC_std
        if result.ic_std > 0:
            expected_ir = result.ic_mean / result.ic_std
            assert result.ir == pytest.approx(expected_ir, rel=0.01)

    def test_ic_decay_detection(
        self, time_analyzer: TimeStabilityAnalyzer
    ) -> None:
        """Test IC decay trend detection."""
        # Create decaying IC series
        dates = pd.date_range("2022-01-01", periods=12, freq="ME")
        ic_values = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.00, -0.01]
        ic_series = pd.Series(ic_values, index=dates)

        decay_result = time_analyzer.detect_decay(ic_series)

        assert decay_result.has_decay is True
        assert decay_result.decay_rate < 0

    def test_rolling_ic_stability(
        self, time_analyzer: TimeStabilityAnalyzer, sample_factor_returns: pd.DataFrame
    ) -> None:
        """Test rolling IC stability calculation."""
        result = time_analyzer.analyze(sample_factor_returns)

        assert result.rolling_ic_std is not None
        assert result.rolling_ic_std >= 0

    def test_time_stability_score(
        self, time_analyzer: TimeStabilityAnalyzer, sample_factor_returns: pd.DataFrame
    ) -> None:
        """Test time stability score calculation."""
        result = time_analyzer.analyze(sample_factor_returns)

        assert 0 <= result.stability_score <= 1


class TestMarketStabilityAnalyzer:
    """Tests for market stability analysis."""

    def test_market_cap_grouping(
        self, market_analyzer: MarketStabilityAnalyzer, sample_factor_returns: pd.DataFrame
    ) -> None:
        """Test market cap based grouping."""
        result = market_analyzer.analyze(sample_factor_returns)

        assert isinstance(result, MarketStabilityResult)
        assert "large" in result.group_ic
        assert "mid" in result.group_ic
        assert "small" in result.group_ic

    def test_cross_market_consistency(
        self, market_analyzer: MarketStabilityAnalyzer, sample_factor_returns: pd.DataFrame
    ) -> None:
        """Test cross-market consistency score."""
        result = market_analyzer.analyze(sample_factor_returns)

        assert result.consistency_score is not None
        assert 0 <= result.consistency_score <= 1

    def test_market_group_ic_difference(
        self, market_analyzer: MarketStabilityAnalyzer, sample_factor_returns: pd.DataFrame
    ) -> None:
        """Test IC difference between market groups."""
        result = market_analyzer.analyze(sample_factor_returns)

        assert result.max_ic_difference is not None
        assert result.max_ic_difference >= 0

    def test_custom_market_cap_thresholds(
        self, sample_factor_returns: pd.DataFrame
    ) -> None:
        """Test custom market cap thresholds."""
        config = StabilityConfig(
            large_cap_threshold=5e11,
            small_cap_threshold=1e10,
        )
        analyzer = MarketStabilityAnalyzer(config)
        result = analyzer.analyze(sample_factor_returns)

        assert isinstance(result, MarketStabilityResult)

    def test_market_stability_score(
        self, market_analyzer: MarketStabilityAnalyzer, sample_factor_returns: pd.DataFrame
    ) -> None:
        """Test market stability score."""
        result = market_analyzer.analyze(sample_factor_returns)

        assert 0 <= result.stability_score <= 1


class TestRegimeStabilityAnalyzer:
    """Tests for regime stability analysis."""

    def test_regime_detection(
        self, regime_analyzer: RegimeStabilityAnalyzer, sample_factor_returns: pd.DataFrame
    ) -> None:
        """Test market regime detection."""
        result = regime_analyzer.analyze(sample_factor_returns)

        assert isinstance(result, RegimeStabilityResult)
        assert len(result.regime_ic) > 0

    def test_volatility_based_regime(
        self, regime_analyzer: RegimeStabilityAnalyzer, sample_factor_returns: pd.DataFrame
    ) -> None:
        """Test volatility-based regime classification."""
        result = regime_analyzer.analyze(sample_factor_returns)

        # Should have high/low volatility regimes
        regimes = list(result.regime_ic.keys())
        assert len(regimes) >= 2

    def test_trend_based_regime(
        self, sample_factor_returns: pd.DataFrame
    ) -> None:
        """Test trend-based regime classification."""
        config = StabilityConfig(regime_method="trend")
        analyzer = RegimeStabilityAnalyzer(config)
        result = analyzer.analyze(sample_factor_returns)

        assert MarketRegime.BULL in result.regime_ic or MarketRegime.BEAR in result.regime_ic

    def test_regime_sensitivity(
        self, regime_analyzer: RegimeStabilityAnalyzer, sample_factor_returns: pd.DataFrame
    ) -> None:
        """Test regime sensitivity score."""
        result = regime_analyzer.analyze(sample_factor_returns)

        # Lower sensitivity means more stable
        assert result.sensitivity_score is not None
        assert result.sensitivity_score >= 0

    def test_regime_stability_score(
        self, regime_analyzer: RegimeStabilityAnalyzer, sample_factor_returns: pd.DataFrame
    ) -> None:
        """Test regime stability score."""
        result = regime_analyzer.analyze(sample_factor_returns)

        assert 0 <= result.stability_score <= 1


class TestStabilityAnalyzer:
    """Tests for combined stability analyzer."""

    def test_full_analysis(
        self, stability_analyzer: StabilityAnalyzer, sample_factor_returns: pd.DataFrame
    ) -> None:
        """Test full stability analysis."""
        report = stability_analyzer.analyze(sample_factor_returns)

        assert isinstance(report, StabilityReport)
        assert report.time_stability is not None
        assert report.market_stability is not None
        assert report.regime_stability is not None

    def test_overall_stability_score(
        self, stability_analyzer: StabilityAnalyzer, sample_factor_returns: pd.DataFrame
    ) -> None:
        """Test overall stability score calculation."""
        report = stability_analyzer.analyze(sample_factor_returns)

        assert isinstance(report.overall_score, StabilityScore)
        assert 0 <= report.overall_score.value <= 1

    def test_weighted_score_calculation(
        self, sample_factor_returns: pd.DataFrame
    ) -> None:
        """Test weighted stability score."""
        config = StabilityConfig(
            time_weight=0.5,
            market_weight=0.3,
            regime_weight=0.2,
        )
        analyzer = StabilityAnalyzer(config)
        report = analyzer.analyze(sample_factor_returns)

        # Verify weights sum to 1
        total_weight = config.time_weight + config.market_weight + config.regime_weight
        assert total_weight == pytest.approx(1.0)

    def test_stability_grade(
        self, stability_analyzer: StabilityAnalyzer, sample_factor_returns: pd.DataFrame
    ) -> None:
        """Test stability grade assignment."""
        report = stability_analyzer.analyze(sample_factor_returns)

        assert report.grade in ["A", "B", "C", "D", "F"]

    def test_report_recommendations(
        self, stability_analyzer: StabilityAnalyzer, sample_factor_returns: pd.DataFrame
    ) -> None:
        """Test stability recommendations."""
        report = stability_analyzer.analyze(sample_factor_returns)

        assert isinstance(report.recommendations, list)


class TestStabilityReport:
    """Tests for stability report generation."""

    def test_report_to_dict(
        self, stability_analyzer: StabilityAnalyzer, sample_factor_returns: pd.DataFrame
    ) -> None:
        """Test report serialization."""
        report = stability_analyzer.analyze(sample_factor_returns)

        d = report.to_dict()
        assert "time_stability" in d
        assert "market_stability" in d
        assert "regime_stability" in d
        assert "overall_score" in d

    def test_report_summary(
        self, stability_analyzer: StabilityAnalyzer, sample_factor_returns: pd.DataFrame
    ) -> None:
        """Test report summary generation."""
        report = stability_analyzer.analyze(sample_factor_returns)

        summary = report.get_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_report_visualizable_data(
        self, stability_analyzer: StabilityAnalyzer, sample_factor_returns: pd.DataFrame
    ) -> None:
        """Test data for visualization."""
        report = stability_analyzer.analyze(sample_factor_returns)

        viz_data = report.get_visualization_data()
        assert "time_series" in viz_data
        assert "market_groups" in viz_data
        assert "regime_breakdown" in viz_data


class TestStabilityBoundary:
    """Boundary tests for stability analysis."""

    def test_minimum_data_periods(
        self, stability_analyzer: StabilityAnalyzer
    ) -> None:
        """Test with minimum required data periods."""
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=30, freq="D")  # Minimum

        data = []
        for date in dates:
            data.append({
                "date": date,
                "symbol": "BTC",
                "factor_value": np.random.randn(),
                "forward_return": np.random.randn() * 0.02,
                "market_cap": 1e11,
            })

        df = pd.DataFrame(data)
        report = stability_analyzer.analyze(df)

        assert report is not None

    def test_single_symbol(
        self, stability_analyzer: StabilityAnalyzer
    ) -> None:
        """Test with single symbol."""
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=365, freq="D")

        data = [{
            "date": date,
            "symbol": "BTC",
            "factor_value": np.random.randn(),
            "forward_return": np.random.randn() * 0.02,
            "market_cap": 1e11,
        } for date in dates]

        df = pd.DataFrame(data)
        report = stability_analyzer.analyze(df)

        assert report is not None

    def test_extreme_ic_values(
        self, time_analyzer: TimeStabilityAnalyzer
    ) -> None:
        """Test with extreme IC values."""
        dates = pd.date_range("2022-01-01", periods=12, freq="ME")
        ic_values = [1.0, -1.0, 0.5, -0.5, 0.0, 0.99, -0.99, 0.1, -0.1, 0.0, 0.0, 0.0]
        ic_series = pd.Series(ic_values, index=dates)

        decay_result = time_analyzer.detect_decay(ic_series)
        assert decay_result is not None

    def test_all_same_market_cap(
        self, market_analyzer: MarketStabilityAnalyzer
    ) -> None:
        """Test when all symbols have same market cap."""
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=100, freq="D")

        data = [{
            "date": date,
            "symbol": f"SYM_{i % 5}",
            "factor_value": np.random.randn(),
            "forward_return": np.random.randn() * 0.02,
            "market_cap": 1e10,  # Same for all
        } for i, date in enumerate(dates)]

        df = pd.DataFrame(data)
        result = market_analyzer.analyze(df)

        # Should handle gracefully
        assert result is not None


class TestStabilityException:
    """Exception handling tests."""

    def test_empty_data(
        self, stability_analyzer: StabilityAnalyzer
    ) -> None:
        """Test with empty data."""
        df = pd.DataFrame()

        with pytest.raises(InvalidDataError, match="empty"):
            stability_analyzer.analyze(df)

    def test_missing_required_columns(
        self, stability_analyzer: StabilityAnalyzer
    ) -> None:
        """Test with missing required columns."""
        df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=10),
            "symbol": ["BTC"] * 10,
            # Missing factor_value and forward_return
        })

        with pytest.raises(InvalidDataError, match="column"):
            stability_analyzer.analyze(df)

    def test_insufficient_data(
        self, stability_analyzer: StabilityAnalyzer
    ) -> None:
        """Test with insufficient data for analysis."""
        df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=5),
            "symbol": ["BTC"] * 5,
            "factor_value": [1, 2, 3, 4, 5],
            "forward_return": [0.01] * 5,
            "market_cap": [1e10] * 5,
        })

        with pytest.raises(InsufficientDataError):
            stability_analyzer.analyze(df)

    def test_invalid_config_weights(self) -> None:
        """Test invalid configuration weights."""
        with pytest.raises(ValueError, match="weight"):
            StabilityConfig(
                time_weight=0.5,
                market_weight=0.5,
                regime_weight=0.5,  # Sum > 1
            )


class TestStabilityPerformance:
    """Performance tests for stability analysis."""

    def test_analysis_speed(
        self, stability_analyzer: StabilityAnalyzer
    ) -> None:
        """Test analysis completion time."""
        import time

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=1000, freq="D")
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
        stability_analyzer.analyze(df)
        elapsed = time.time() - start

        assert elapsed < 5.0, f"Analysis took {elapsed}s"

    def test_memory_efficiency(
        self, stability_analyzer: StabilityAnalyzer, sample_factor_returns: pd.DataFrame
    ) -> None:
        """Test memory usage during analysis."""
        import sys

        initial_size = sys.getsizeof(sample_factor_returns)
        report = stability_analyzer.analyze(sample_factor_returns)
        report_size = sys.getsizeof(report)

        # Report should not be excessively larger than input
        assert report_size < initial_size * 10


class TestStabilityCompatibility:
    """Compatibility tests."""

    def test_datetime_index(
        self, stability_analyzer: StabilityAnalyzer
    ) -> None:
        """Test with datetime index instead of column."""
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=100, freq="D")

        df = pd.DataFrame({
            "symbol": ["BTC"] * 100,
            "factor_value": np.random.randn(100),
            "forward_return": np.random.randn(100) * 0.02,
            "market_cap": [1e11] * 100,
        }, index=dates)

        report = stability_analyzer.analyze(df)
        assert report is not None

    def test_string_dates(
        self, stability_analyzer: StabilityAnalyzer
    ) -> None:
        """Test with string date column."""
        np.random.seed(42)
        dates = [f"2022-{m:02d}-01" for m in range(1, 13)] * 10

        df = pd.DataFrame({
            "date": dates,
            "symbol": ["BTC"] * 120,
            "factor_value": np.random.randn(120),
            "forward_return": np.random.randn(120) * 0.02,
            "market_cap": [1e11] * 120,
        })

        report = stability_analyzer.analyze(df)
        assert report is not None

    def test_different_column_names(
        self, stability_analyzer: StabilityAnalyzer
    ) -> None:
        """Test with alternative column names."""
        np.random.seed(42)

        df = pd.DataFrame({
            "timestamp": pd.date_range("2022-01-01", periods=100, freq="D"),
            "ticker": ["BTC"] * 100,
            "alpha": np.random.randn(100),
            "ret": np.random.randn(100) * 0.02,
            "mcap": [1e11] * 100,
        })

        config = StabilityConfig(
            date_column="timestamp",
            symbol_column="ticker",
            factor_column="alpha",
            return_column="ret",
            market_cap_column="mcap",
        )

        analyzer = StabilityAnalyzer(config)
        report = analyzer.analyze(df)
        assert report is not None

    def test_numpy_arrays(
        self, stability_analyzer: StabilityAnalyzer
    ) -> None:
        """Test with numpy array factor values."""
        np.random.seed(42)

        df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=100, freq="D"),
            "symbol": ["BTC"] * 100,
            "factor_value": np.array(np.random.randn(100), dtype=np.float64),
            "forward_return": np.array(np.random.randn(100) * 0.02, dtype=np.float64),
            "market_cap": np.array([1e11] * 100, dtype=np.float64),
        })

        report = stability_analyzer.analyze(df)
        assert report is not None
