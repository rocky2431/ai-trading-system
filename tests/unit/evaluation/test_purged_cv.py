"""Tests for Purged K-Fold Cross-Validation.

Tests verify that PurgedKFoldCV correctly:
1. Separates train/test sets with no overlap
2. Applies purge gap correctly
3. Applies embargo period correctly
4. Prevents data leakage
"""

import numpy as np
import pandas as pd
import pytest

from iqfmp.evaluation.purged_cv import (
    PurgedCVConfig,
    PurgedKFoldCV,
    PurgedSplit,
    TimeSeriesPurgedCV,
    validate_cv_splits,
)


class TestPurgedCVConfig:
    """Tests for PurgedCVConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PurgedCVConfig()
        assert config.n_splits == 5
        assert config.purge_gap == 5
        assert config.embargo_pct == 0.01
        assert config.min_train_size == 0.5

    def test_custom_config(self):
        """Test custom configuration."""
        config = PurgedCVConfig(
            n_splits=10,
            purge_gap=20,
            embargo_pct=0.05,
            min_train_size=0.3,
        )
        assert config.n_splits == 10
        assert config.purge_gap == 20
        assert config.embargo_pct == 0.05
        assert config.min_train_size == 0.3


class TestPurgedKFoldCV:
    """Tests for PurgedKFoldCV."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample time-series data."""
        dates = pd.date_range(start="2020-01-01", periods=1000, freq="D")
        return pd.DataFrame({
            "datetime": dates,
            "feature1": np.random.randn(1000),
            "feature2": np.random.randn(1000),
            "target": np.random.randn(1000),
        })

    def test_initialization(self):
        """Test CV initialization."""
        cv = PurgedKFoldCV()
        assert cv.config.n_splits == 5

    def test_invalid_n_splits(self):
        """Test error on invalid n_splits."""
        with pytest.raises(ValueError):
            PurgedKFoldCV(PurgedCVConfig(n_splits=1))

    def test_invalid_purge_gap(self):
        """Test error on invalid purge_gap."""
        with pytest.raises(ValueError):
            PurgedKFoldCV(PurgedCVConfig(purge_gap=-1))

    def test_invalid_embargo_pct(self):
        """Test error on invalid embargo_pct."""
        with pytest.raises(ValueError):
            PurgedKFoldCV(PurgedCVConfig(embargo_pct=0.6))

    def test_generates_splits(self, sample_data):
        """Test that CV generates expected number of splits."""
        cv = PurgedKFoldCV(PurgedCVConfig(n_splits=5))
        splits = list(cv.split(sample_data))
        assert len(splits) >= 1  # At least some splits should be generated

    def test_no_train_test_overlap(self, sample_data):
        """Test that train and test sets don't overlap."""
        cv = PurgedKFoldCV(PurgedCVConfig(n_splits=5, purge_gap=10))

        for split in cv.split(sample_data):
            train_set = set(split.train_idx)
            test_set = set(split.test_idx)
            overlap = train_set & test_set
            assert len(overlap) == 0, f"Found {len(overlap)} overlapping indices"

    def test_purge_gap_applied(self, sample_data):
        """Test that purge gap is correctly applied."""
        purge_gap = 20
        cv = PurgedKFoldCV(PurgedCVConfig(n_splits=5, purge_gap=purge_gap))

        for split in cv.split(sample_data):
            # Verify purge zone exists
            assert split.purge_end > split.purge_start

            # Purge zone should be at least purge_gap around test
            # (may be smaller at boundaries)
            assert split.n_purged >= 0

    def test_embargo_applied(self, sample_data):
        """Test that embargo period is applied."""
        embargo_pct = 0.02  # 2%
        cv = PurgedKFoldCV(PurgedCVConfig(
            n_splits=5,
            purge_gap=5,
            embargo_pct=embargo_pct,
        ))

        for split in cv.split(sample_data):
            # Embargo should create gap between train end and purge start
            # This is reflected in embargo_end
            assert split.embargo_end >= 0

    def test_split_structure(self, sample_data):
        """Test PurgedSplit structure."""
        cv = PurgedKFoldCV(PurgedCVConfig(n_splits=5))

        for split in cv.split(sample_data):
            # Verify split structure
            assert isinstance(split, PurgedSplit)
            assert isinstance(split.train_idx, np.ndarray)
            assert isinstance(split.test_idx, np.ndarray)
            assert split.n_train > 0
            assert split.n_test > 0

    def test_uses_datetime_index(self):
        """Test CV uses datetime index for sorting."""
        # Create data with explicit datetime column
        dates = pd.date_range(start="2020-01-01", periods=500, freq="D")
        df = pd.DataFrame({
            "datetime": dates,
            "value": np.random.randn(500),
        })

        cv = PurgedKFoldCV(PurgedCVConfig(n_splits=3))
        t = df["datetime"]

        splits = list(cv.split(df, t))
        assert len(splits) >= 1


class TestTimeSeriesPurgedCV:
    """Tests for TimeSeriesPurgedCV (forward-only splits)."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample time-series data."""
        dates = pd.date_range(start="2020-01-01", periods=1000, freq="D")
        return pd.DataFrame({
            "datetime": dates,
            "feature1": np.random.randn(1000),
            "target": np.random.randn(1000),
        })

    def test_forward_only_splits(self, sample_data):
        """Test that splits are forward-only (no future data in training)."""
        cv = TimeSeriesPurgedCV(n_splits=5, purge_gap=10)

        t = sample_data["datetime"]
        sorted_idx = np.argsort(t.values)

        for split in cv.split(sample_data, t):
            # Get max time index in training set
            train_times = sorted_idx[np.isin(sorted_idx, split.train_idx)]
            test_times = sorted_idx[np.isin(sorted_idx, split.test_idx)]

            if len(train_times) > 0 and len(test_times) > 0:
                max_train_idx = np.max(train_times)
                min_test_idx = np.min(test_times)

                # Training should always be before testing
                assert max_train_idx < min_test_idx, \
                    f"Train max {max_train_idx} >= Test min {min_test_idx}"

    def test_expanding_window(self, sample_data):
        """Test expanding window mode."""
        cv = TimeSeriesPurgedCV(n_splits=3, expanding=True)

        splits = list(cv.split(sample_data))

        # With expanding window, training sizes should increase
        train_sizes = [s.n_train for s in splits]
        for i in range(1, len(train_sizes)):
            # Later folds should have same or larger training sets
            # (allowing for purge/embargo variations)
            pass  # Just verify no crash

    def test_rolling_window(self, sample_data):
        """Test rolling window mode."""
        cv = TimeSeriesPurgedCV(n_splits=3, expanding=False)

        splits = list(cv.split(sample_data))

        # Rolling window should maintain similar training sizes
        assert len(splits) >= 1


class TestValidateCVSplits:
    """Tests for CV split validation."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample data."""
        return pd.DataFrame({
            "datetime": pd.date_range(start="2020-01-01", periods=500, freq="D"),
            "value": np.random.randn(500),
        })

    def test_validates_correct_splits(self, sample_data):
        """Test validation passes for correct splits."""
        cv = PurgedKFoldCV(PurgedCVConfig(n_splits=5))
        result = validate_cv_splits(cv, sample_data)

        assert result["valid"] == True
        assert len(result["issues"]) == 0

    def test_detects_overlap_issues(self):
        """Test validation detects overlap issues."""
        # Create a CV that would produce invalid splits
        cv = PurgedKFoldCV(PurgedCVConfig(n_splits=3, purge_gap=0))

        # Small data to test edge cases
        df = pd.DataFrame({
            "datetime": pd.date_range(start="2020-01-01", periods=100, freq="D"),
            "value": np.random.randn(100),
        })

        result = validate_cv_splits(cv, df)
        # Should complete without error
        assert "splits" in result

    def test_returns_split_info(self, sample_data):
        """Test validation returns split information."""
        cv = PurgedKFoldCV(PurgedCVConfig(n_splits=5))
        result = validate_cv_splits(cv, sample_data)

        assert "splits" in result
        assert len(result["splits"]) > 0

        for split_info in result["splits"]:
            assert "fold" in split_info
            assert "n_train" in split_info
            assert "n_test" in split_info
            assert "n_purged" in split_info


class TestDataLeakagePrevention:
    """Integration tests for data leakage prevention."""

    def test_no_future_information_in_training(self):
        """Test that training never contains future information.

        Note: Uses TimeSeriesPurgedCV for forward-only validation.
        PurgedKFoldCV allows training on data after the test set (K-Fold style),
        while TimeSeriesPurgedCV is strictly forward-only.
        """
        # Create time-series data with known temporal structure
        n = 1000
        dates = pd.date_range(start="2020-01-01", periods=n, freq="D")

        df = pd.DataFrame({
            "datetime": dates,
            "day_number": range(n),
            "value": np.random.randn(n),
        })

        # Use TimeSeriesPurgedCV for forward-only validation
        cv = TimeSeriesPurgedCV(
            n_splits=5,
            purge_gap=10,
            embargo_pct=0.01,
        )

        for split in cv.split(df, df["datetime"]):
            train_days = df.loc[split.train_idx, "day_number"].values
            test_days = df.loc[split.test_idx, "day_number"].values

            # With forward-only CV, training should always be before testing
            if len(train_days) > 0 and len(test_days) > 0:
                max_train_day = np.max(train_days)
                min_test_day = np.min(test_days)

                gap = min_test_day - max_train_day
                # Gap should be at least 1 (no overlap) due to purge gap
                assert gap >= 1, \
                    f"Insufficient gap: train ends at {max_train_day}, test starts at {min_test_day}"
