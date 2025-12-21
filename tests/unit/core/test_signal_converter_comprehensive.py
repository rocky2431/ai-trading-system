"""Comprehensive tests for SignalConverter module.

Tests use real implementations - NO MOCKS per user requirement.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from iqfmp.core.signal_converter import (
    SignalConfig,
    SignalConverter,
    QlibPredictionDataset,
    create_signal_converter,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_factor_series() -> pd.Series:
    """Create sample factor series for testing."""
    np.random.seed(42)
    return pd.Series(
        np.random.randn(100),
        index=pd.date_range("2024-01-01", periods=100, freq="D"),
    )


@pytest.fixture
def sample_factor_df() -> pd.DataFrame:
    """Create sample factor DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    return pd.DataFrame({
        "datetime": dates,
        "instrument": ["BTCUSDT"] * 50,
        "value": np.random.randn(50),
        "score": np.random.randn(50),
    })


@pytest.fixture
def constant_factor() -> pd.Series:
    """Create constant factor series (edge case)."""
    return pd.Series([5.0] * 20, index=range(20))


# =============================================================================
# Test SignalConfig
# =============================================================================

class TestSignalConfig:
    """Tests for SignalConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SignalConfig()

        assert config.normalize_method == "zscore"
        assert config.clip_std == 3.0
        assert config.signal_threshold == 0.0
        assert config.top_k is None
        assert config.position_scale == 1.0
        assert config.max_position == 0.1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SignalConfig(
            normalize_method="minmax",
            clip_std=2.0,
            signal_threshold=0.5,
            top_k=10,
            position_scale=2.0,
            max_position=0.2,
        )

        assert config.normalize_method == "minmax"
        assert config.clip_std == 2.0
        assert config.signal_threshold == 0.5
        assert config.top_k == 10
        assert config.position_scale == 2.0
        assert config.max_position == 0.2


# =============================================================================
# Test SignalConverter Normalization
# =============================================================================

class TestSignalConverterNormalize:
    """Tests for SignalConverter normalization methods."""

    def test_zscore_normalization(self, sample_factor_series: pd.Series):
        """Test z-score normalization."""
        converter = SignalConverter(SignalConfig(normalize_method="zscore"))
        normalized = converter.normalize(sample_factor_series)

        assert isinstance(normalized, pd.Series)
        assert len(normalized) == len(sample_factor_series)
        # Z-score should have mean ~0 and std ~1 (before clipping)
        assert abs(normalized.mean()) < 0.1
        assert abs(normalized.std() - 1) < 0.2

    def test_zscore_clipping(self, sample_factor_series: pd.Series):
        """Test z-score clipping of outliers."""
        config = SignalConfig(normalize_method="zscore", clip_std=2.0)
        converter = SignalConverter(config)
        normalized = converter.normalize(sample_factor_series)

        assert normalized.max() <= 2.0
        assert normalized.min() >= -2.0

    def test_zscore_constant_series(self, constant_factor: pd.Series):
        """Test z-score with constant series (std=0)."""
        converter = SignalConverter(SignalConfig(normalize_method="zscore"))
        normalized = converter.normalize(constant_factor)

        # Constant series should return zeros
        assert (normalized == 0).all()

    def test_minmax_normalization(self, sample_factor_series: pd.Series):
        """Test min-max normalization."""
        converter = SignalConverter(SignalConfig(normalize_method="minmax"))
        normalized = converter.normalize(sample_factor_series)

        assert isinstance(normalized, pd.Series)
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_minmax_constant_series(self, constant_factor: pd.Series):
        """Test min-max with constant series (min=max)."""
        converter = SignalConverter(SignalConfig(normalize_method="minmax"))
        normalized = converter.normalize(constant_factor)

        # Constant series should return 0.5
        assert (normalized == 0.5).all()

    def test_rank_normalization(self, sample_factor_series: pd.Series):
        """Test rank normalization."""
        converter = SignalConverter(SignalConfig(normalize_method="rank"))
        normalized = converter.normalize(sample_factor_series)

        assert isinstance(normalized, pd.Series)
        # Rank percentile should be between 0 and 1
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_no_normalization(self, sample_factor_series: pd.Series):
        """Test no normalization (pass-through)."""
        converter = SignalConverter(SignalConfig(normalize_method="none"))
        normalized = converter.normalize(sample_factor_series)

        # Should return original values
        pd.testing.assert_series_equal(normalized, sample_factor_series)


# =============================================================================
# Test SignalConverter to_signal
# =============================================================================

class TestSignalConverterToSignal:
    """Tests for SignalConverter.to_signal method."""

    def test_to_signal_from_series(self, sample_factor_series: pd.Series):
        """Test converting Series to signal."""
        converter = SignalConverter()
        signal = converter.to_signal(sample_factor_series)

        assert isinstance(signal, pd.Series)
        assert signal.max() <= 1.0
        assert signal.min() >= -1.0

    def test_to_signal_from_dataframe_value_column(self, sample_factor_df: pd.DataFrame):
        """Test converting DataFrame with 'value' column."""
        converter = SignalConverter()
        signal = converter.to_signal(sample_factor_df)

        assert isinstance(signal, pd.Series)
        assert len(signal) <= len(sample_factor_df)  # May drop NaN

    def test_to_signal_from_dataframe_score_column(self):
        """Test converting DataFrame with 'score' column."""
        df = pd.DataFrame({
            "score": np.random.randn(50),
            "other": np.random.randn(50),
        })
        # Remove 'value' column to force 'score' selection
        converter = SignalConverter()
        signal = converter.to_signal(df[["score", "other"]])

        assert isinstance(signal, pd.Series)

    def test_to_signal_from_dataframe_factor_column(self):
        """Test converting DataFrame with 'factor' column."""
        df = pd.DataFrame({
            "factor": np.random.randn(50),
            "other": np.random.randn(50),
        })
        converter = SignalConverter()
        signal = converter.to_signal(df)

        assert isinstance(signal, pd.Series)

    def test_to_signal_from_dataframe_first_column(self):
        """Test converting DataFrame using first column as fallback."""
        df = pd.DataFrame({
            "alpha": np.random.randn(50),
            "beta": np.random.randn(50),
        })
        converter = SignalConverter()
        signal = converter.to_signal(df)

        assert isinstance(signal, pd.Series)

    def test_to_signal_without_normalization(self, sample_factor_series: pd.Series):
        """Test converting without normalization."""
        converter = SignalConverter()
        signal = converter.to_signal(sample_factor_series, normalize=False)

        assert isinstance(signal, pd.Series)

    def test_to_signal_with_top_k(self, sample_factor_series: pd.Series):
        """Test top-k selection mode."""
        config = SignalConfig(top_k=10)
        converter = SignalConverter(config)
        signal = converter.to_signal(sample_factor_series)

        # Should have exactly k longs and k shorts (or less if not enough data)
        longs = (signal == 1.0).sum()
        shorts = (signal == -1.0).sum()
        neutrals = (signal == 0.0).sum()

        assert longs == 10
        assert shorts == 10
        assert neutrals == len(signal) - 20

    def test_to_signal_with_top_k_small_dataset(self):
        """Test top-k with small dataset."""
        small_series = pd.Series([1.0, 2.0, 3.0], index=range(3))
        config = SignalConfig(top_k=5)  # k > len/2
        converter = SignalConverter(config)
        signal = converter.to_signal(small_series, normalize=False)

        # k should be min(5, 3//2) = 1
        assert (signal == 1.0).sum() == 1
        assert (signal == -1.0).sum() == 1

    def test_to_signal_threshold_based(self, sample_factor_series: pd.Series):
        """Test threshold-based signal generation."""
        config = SignalConfig(signal_threshold=0.5)
        converter = SignalConverter(config)
        signal = converter.to_signal(sample_factor_series)

        # Values between -0.5 and 0.5 should be 0
        # Values > 0.5 should be positive, < -0.5 should be negative
        assert signal.max() <= 1.0
        assert signal.min() >= -1.0


# =============================================================================
# Test SignalConverter to_position
# =============================================================================

class TestSignalConverterToPosition:
    """Tests for SignalConverter.to_position method."""

    def test_to_position_default_scaling(self):
        """Test position conversion with default scaling."""
        converter = SignalConverter()
        signal = pd.Series([1.0, -1.0, 0.5, -0.5, 0.0])
        position = converter.to_position(signal)

        # Default max_position is 0.1
        assert position.max() <= 0.1
        assert position.min() >= -0.1

    def test_to_position_custom_scaling(self):
        """Test position conversion with custom scaling."""
        config = SignalConfig(position_scale=2.0, max_position=0.2)
        converter = SignalConverter(config)
        signal = pd.Series([1.0, -1.0, 0.5])
        position = converter.to_position(signal)

        # position_scale=2.0 -> 1.0*2=2.0, clipped to max_position=0.2
        assert position.iloc[0] == 0.2
        assert position.iloc[1] == -0.2

    def test_to_position_clipping(self):
        """Test that positions are clipped to max_position."""
        config = SignalConfig(position_scale=10.0, max_position=0.05)
        converter = SignalConverter(config)
        signal = pd.Series([1.0, -1.0])
        position = converter.to_position(signal)

        assert position.iloc[0] == 0.05
        assert position.iloc[1] == -0.05


# =============================================================================
# Test SignalConverter to_qlib_format
# =============================================================================

class TestSignalConverterToQlibFormat:
    """Tests for SignalConverter.to_qlib_format method."""

    def test_to_qlib_format_basic(self, sample_factor_df: pd.DataFrame):
        """Test basic Qlib format conversion."""
        converter = SignalConverter()
        qlib_df = converter.to_qlib_format(sample_factor_df)

        assert isinstance(qlib_df.index, pd.MultiIndex)
        assert qlib_df.index.names == ["datetime", "instrument"]

    def test_to_qlib_format_with_symbol_column(self):
        """Test Qlib format conversion with 'symbol' instead of 'instrument'."""
        df = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=10),
            "symbol": ["ETHUSDT"] * 10,
            "value": np.random.randn(10),
        })
        converter = SignalConverter()
        qlib_df = converter.to_qlib_format(df)

        assert isinstance(qlib_df.index, pd.MultiIndex)
        assert qlib_df.index.names == ["datetime", "instrument"]

    def test_to_qlib_format_sorting(self):
        """Test that output is sorted by index."""
        df = pd.DataFrame({
            "datetime": ["2024-01-03", "2024-01-01", "2024-01-02"],
            "instrument": ["BTC", "BTC", "BTC"],
            "value": [1.0, 2.0, 3.0],
        })
        converter = SignalConverter()
        qlib_df = converter.to_qlib_format(df)

        # Should be sorted ascending
        dates = qlib_df.index.get_level_values("datetime")
        assert dates[0] < dates[1] < dates[2]


# =============================================================================
# Test SignalConverter from_factor_output
# =============================================================================

class TestSignalConverterFromFactorOutput:
    """Tests for SignalConverter.from_factor_output method."""

    def test_from_factor_output_series(self):
        """Test conversion from factor output with Series values."""
        converter = SignalConverter()
        factor_output = {
            "factor_values": pd.Series([1.0, 2.0, 3.0]),
        }
        signal = converter.from_factor_output(factor_output)

        assert isinstance(signal, pd.Series)

    def test_from_factor_output_dict(self):
        """Test conversion from factor output with dict values."""
        converter = SignalConverter()
        factor_output = {
            "factor_values": {"a": 1.0, "b": 2.0, "c": 3.0},
        }
        signal = converter.from_factor_output(factor_output)

        assert isinstance(signal, pd.Series)
        assert len(signal) == 3

    def test_from_factor_output_dataframe_with_value(self):
        """Test conversion from factor output with DataFrame (value column)."""
        converter = SignalConverter()
        factor_output = {
            "factor_values": pd.DataFrame({
                "value": [1.0, 2.0, 3.0],
                "other": [4.0, 5.0, 6.0],
            }),
        }
        signal = converter.from_factor_output(factor_output)

        assert isinstance(signal, pd.Series)

    def test_from_factor_output_dataframe_first_column(self):
        """Test conversion from factor output with DataFrame (first column)."""
        converter = SignalConverter()
        factor_output = {
            "factor_values": pd.DataFrame({
                "alpha": [1.0, 2.0, 3.0],
                "beta": [4.0, 5.0, 6.0],
            }),
        }
        signal = converter.from_factor_output(factor_output)

        assert isinstance(signal, pd.Series)

    def test_from_factor_output_missing_values(self):
        """Test error when factor_values is missing."""
        converter = SignalConverter()
        factor_output = {"metadata": {"name": "test"}}

        with pytest.raises(ValueError, match="must contain 'factor_values'"):
            converter.from_factor_output(factor_output)

    def test_from_factor_output_with_price_data(self):
        """Test conversion with price data context."""
        converter = SignalConverter()
        factor_output = {
            "factor_values": pd.Series([1.0, 2.0, 3.0]),
        }
        price_data = pd.DataFrame({
            "close": [100.0, 101.0, 102.0],
        })
        signal = converter.from_factor_output(factor_output, price_data)

        assert isinstance(signal, pd.Series)


# =============================================================================
# Test QlibPredictionDataset
# =============================================================================

class TestQlibPredictionDataset:
    """Tests for QlibPredictionDataset class."""

    def test_initialization_with_series(self):
        """Test initialization with Series signal."""
        signal = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-01", periods=3))
        dataset = QlibPredictionDataset(
            signal=signal,
            instruments=["BTCUSDT"],
            start_time="2024-01-01",
            end_time="2024-01-03",
        )

        assert dataset.instruments == ["BTCUSDT"]
        assert dataset.start_time == pd.Timestamp("2024-01-01")
        assert dataset.end_time == pd.Timestamp("2024-01-03")
        assert not dataset.is_prepared

    def test_initialization_with_dataframe(self):
        """Test initialization with DataFrame signal."""
        signal = pd.DataFrame({
            "score": [1.0, 2.0, 3.0],
        }, index=pd.date_range("2024-01-01", periods=3))
        dataset = QlibPredictionDataset(
            signal=signal,
            instruments=["BTCUSDT", "ETHUSDT"],
            start_time="2024-01-01",
            end_time="2024-01-03",
        )

        assert len(dataset.instruments) == 2

    def test_prepare(self):
        """Test prepare method."""
        signal = pd.Series([1.0, 2.0])
        dataset = QlibPredictionDataset(
            signal=signal,
            instruments=["BTC"],
            start_time="2024-01-01",
            end_time="2024-01-02",
        )

        assert not dataset.is_prepared
        result = dataset.prepare()
        assert dataset.is_prepared
        assert result is dataset  # Should return self

    def test_get_segments(self):
        """Test get_segments method."""
        signal = pd.Series([1.0])
        dataset = QlibPredictionDataset(
            signal=signal,
            instruments=["BTC"],
            start_time="2024-01-01",
            end_time="2024-12-31",
        )
        segments = dataset.get_segments()

        assert "train" in segments
        assert "test" in segments
        assert segments["train"][0] == pd.Timestamp("2024-01-01")
        assert segments["train"][1] == pd.Timestamp("2024-12-31")

    def test_getitem_series(self):
        """Test __getitem__ with Series signal."""
        dates = pd.date_range("2024-01-01", periods=3)
        signal = pd.Series([1.0, 2.0, 3.0], index=dates)
        dataset = QlibPredictionDataset(
            signal=signal,
            instruments=["BTC"],
            start_time="2024-01-01",
            end_time="2024-01-03",
        )

        assert dataset[dates[0]] == 1.0
        assert dataset[dates[1]] == 2.0

    def test_getitem_dataframe(self):
        """Test __getitem__ with DataFrame signal."""
        dates = pd.date_range("2024-01-01", periods=3)
        signal = pd.DataFrame({
            "score": [1.0, 2.0, 3.0],
            "value": [4.0, 5.0, 6.0],
        }, index=dates)
        dataset = QlibPredictionDataset(
            signal=signal,
            instruments=["BTC"],
            start_time="2024-01-01",
            end_time="2024-01-03",
        )

        result = dataset[dates[0]]
        assert isinstance(result, pd.Series)
        assert result["score"] == 1.0

    def test_getitem_key_not_found(self):
        """Test __getitem__ with key not in index."""
        signal = pd.Series([1.0], index=[pd.Timestamp("2024-01-01")])
        dataset = QlibPredictionDataset(
            signal=signal,
            instruments=["BTC"],
            start_time="2024-01-01",
            end_time="2024-01-01",
        )

        # Should return the full signal when key not found
        result = dataset["invalid_key"]
        assert isinstance(result, pd.Series)

    def test_to_dataframe_from_series(self):
        """Test to_dataframe with Series signal."""
        signal = pd.Series([1.0, 2.0, 3.0], name="factor")
        dataset = QlibPredictionDataset(
            signal=signal,
            instruments=["BTC"],
            start_time="2024-01-01",
            end_time="2024-01-03",
        )
        df = dataset.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "score" in df.columns

    def test_to_dataframe_from_dataframe(self):
        """Test to_dataframe with DataFrame signal."""
        signal = pd.DataFrame({
            "alpha": [1.0, 2.0],
            "beta": [3.0, 4.0],
        })
        dataset = QlibPredictionDataset(
            signal=signal,
            instruments=["BTC"],
            start_time="2024-01-01",
            end_time="2024-01-02",
        )
        df = dataset.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "alpha" in df.columns
        assert "beta" in df.columns


# =============================================================================
# Test create_signal_converter Factory
# =============================================================================

class TestCreateSignalConverter:
    """Tests for create_signal_converter factory function."""

    def test_default_factory(self):
        """Test factory with default parameters."""
        converter = create_signal_converter()

        assert isinstance(converter, SignalConverter)
        assert converter.config.normalize_method == "zscore"
        assert converter.config.top_k is None
        assert converter.config.max_position == 0.1

    def test_factory_with_parameters(self):
        """Test factory with custom parameters."""
        converter = create_signal_converter(
            normalize="minmax",
            top_k=5,
            max_position=0.2,
        )

        assert converter.config.normalize_method == "minmax"
        assert converter.config.top_k == 5
        assert converter.config.max_position == 0.2

    def test_factory_produces_working_converter(self):
        """Test that factory produces a working converter."""
        converter = create_signal_converter(normalize="rank", top_k=3)
        factor = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        signal = converter.to_signal(factor, normalize=False)

        # With top_k=3, should have 3 longs and 3 shorts
        assert (signal == 1.0).sum() == 3
        assert (signal == -1.0).sum() == 3


# =============================================================================
# Test Integration
# =============================================================================

class TestSignalConverterIntegration:
    """Integration tests for full signal conversion workflow."""

    def test_full_workflow(self, sample_factor_series: pd.Series):
        """Test complete workflow from factor to signal to position."""
        converter = create_signal_converter(normalize="zscore", max_position=0.15)

        # 1. Factor to signal
        signal = converter.to_signal(sample_factor_series)
        assert isinstance(signal, pd.Series)
        assert signal.max() <= 1.0
        assert signal.min() >= -1.0

        # 2. Signal to position
        position = converter.to_position(signal)
        assert position.max() <= 0.15
        assert position.min() >= -0.15

    def test_qlib_dataset_workflow(self, sample_factor_series: pd.Series):
        """Test Qlib dataset creation workflow."""
        converter = SignalConverter()
        signal = converter.to_signal(sample_factor_series)

        dataset = converter.create_prediction_dataset(
            signal=signal,
            instruments=["BTCUSDT"],
            start_time="2024-01-01",
            end_time="2024-04-10",
        )

        assert isinstance(dataset, QlibPredictionDataset)
        assert dataset.instruments == ["BTCUSDT"]

        # Prepare and verify
        dataset.prepare()
        assert dataset.is_prepared

        # Get segments
        segments = dataset.get_segments()
        assert "train" in segments

        # Convert to DataFrame
        df = dataset.to_dataframe()
        assert isinstance(df, pd.DataFrame)
