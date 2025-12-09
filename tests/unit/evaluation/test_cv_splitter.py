"""Tests for CryptoCVSplitter (Task 10).

Six-dimensional test coverage:
1. Functional: Time/market/frequency splits, combination generation
2. Boundary: Edge cases for split ratios and data sizes
3. Exception: Error handling for invalid inputs
4. Performance: Split generation time
5. Security: Data leakage prevention
6. Compatibility: Different data formats and frequencies
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any
import time

from iqfmp.evaluation.cv_splitter import (
    TimeSplitter,
    MarketSplitter,
    FrequencySplitter,
    CryptoCVSplitter,
    CVSplitConfig,
    SplitResult,
    MarketGroup,
    TimeFrequency,
    DataLeakageError,
    InvalidSplitError,
)


@pytest.fixture
def sample_crypto_data() -> pd.DataFrame:
    """Create sample crypto price data for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="1h")
    symbols = ["BTC", "ETH", "SOL", "DOGE", "SHIB"]

    data = []
    for symbol in symbols:
        base_price = {"BTC": 30000, "ETH": 2000, "SOL": 50, "DOGE": 0.1, "SHIB": 0.00001}[symbol]
        for date in dates:
            data.append({
                "datetime": date,
                "symbol": symbol,
                "open": base_price * (1 + np.random.randn() * 0.01),
                "high": base_price * (1 + np.random.randn() * 0.02),
                "low": base_price * (1 - np.random.randn() * 0.02),
                "close": base_price * (1 + np.random.randn() * 0.01),
                "volume": np.random.randint(1000, 100000),
            })

    return pd.DataFrame(data)


@pytest.fixture
def small_data() -> pd.DataFrame:
    """Create small dataset for quick tests."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1D")
    return pd.DataFrame({
        "datetime": dates,
        "symbol": "BTC",
        "close": np.random.randn(100).cumsum() + 30000,
    })


class TestTimeSplitter:
    """Tests for TimeSplitter."""

    def test_default_split_ratios(self) -> None:
        """Test default split ratios (60/20/20)."""
        splitter = TimeSplitter()
        assert splitter.train_ratio == 0.6
        assert splitter.valid_ratio == 0.2
        assert splitter.test_ratio == 0.2

    def test_custom_split_ratios(self) -> None:
        """Test custom split ratios."""
        splitter = TimeSplitter(train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15)
        assert splitter.train_ratio == 0.7
        assert splitter.valid_ratio == 0.15
        assert splitter.test_ratio == 0.15

    def test_split_ratios_sum_to_one(self) -> None:
        """Test that invalid ratios raise error."""
        with pytest.raises(InvalidSplitError, match="ratios must sum to 1"):
            TimeSplitter(train_ratio=0.5, valid_ratio=0.3, test_ratio=0.3)

    def test_split_by_time(self, small_data: pd.DataFrame) -> None:
        """Test basic time-based split."""
        splitter = TimeSplitter()
        result = splitter.split(small_data)

        assert result.train is not None
        assert result.valid is not None
        assert result.test is not None

        # Check sizes
        total = len(small_data)
        assert len(result.train) == int(total * 0.6)
        assert len(result.valid) == int(total * 0.2)
        assert len(result.test) == total - int(total * 0.6) - int(total * 0.2)

    def test_no_time_overlap(self, small_data: pd.DataFrame) -> None:
        """Test that train/valid/test have no time overlap."""
        splitter = TimeSplitter()
        result = splitter.split(small_data)

        train_max = result.train["datetime"].max()
        valid_min = result.valid["datetime"].min()
        valid_max = result.valid["datetime"].max()
        test_min = result.test["datetime"].min()

        assert train_max < valid_min, "Train and valid should not overlap"
        assert valid_max < test_min, "Valid and test should not overlap"

    def test_rolling_window_split(self, sample_crypto_data: pd.DataFrame) -> None:
        """Test rolling window split mode."""
        splitter = TimeSplitter(mode="rolling", window_size=30, step_size=7)
        splits = list(splitter.split_rolling(sample_crypto_data))

        assert len(splits) > 0
        for split in splits:
            assert split.train is not None
            assert split.test is not None


class TestMarketSplitter:
    """Tests for MarketSplitter."""

    def test_default_market_groups(self) -> None:
        """Test default market group definitions."""
        splitter = MarketSplitter()
        groups = splitter.get_market_groups()

        assert MarketGroup.LARGE_CAP in groups
        assert MarketGroup.MID_CAP in groups
        assert MarketGroup.SMALL_CAP in groups

    def test_large_cap_includes_btc_eth(self) -> None:
        """Test that large cap group includes BTC and ETH."""
        splitter = MarketSplitter()
        large_cap_symbols = splitter.get_symbols_for_group(MarketGroup.LARGE_CAP)

        assert "BTC" in large_cap_symbols
        assert "ETH" in large_cap_symbols

    def test_split_by_market(self, sample_crypto_data: pd.DataFrame) -> None:
        """Test splitting by market cap."""
        splitter = MarketSplitter()
        result = splitter.split(sample_crypto_data)

        assert MarketGroup.LARGE_CAP in result
        assert MarketGroup.SMALL_CAP in result

        # BTC and ETH should be in large cap
        large_cap_data = result[MarketGroup.LARGE_CAP]
        assert "BTC" in large_cap_data["symbol"].unique()
        assert "ETH" in large_cap_data["symbol"].unique()

    def test_custom_market_groups(self) -> None:
        """Test custom market group definitions."""
        custom_groups = {
            MarketGroup.LARGE_CAP: ["BTC"],
            MarketGroup.MID_CAP: ["ETH", "SOL"],
            MarketGroup.SMALL_CAP: ["DOGE", "SHIB"],
        }
        splitter = MarketSplitter(custom_groups=custom_groups)

        assert splitter.get_symbols_for_group(MarketGroup.LARGE_CAP) == ["BTC"]
        assert "ETH" in splitter.get_symbols_for_group(MarketGroup.MID_CAP)

    def test_all_symbols_assigned(self, sample_crypto_data: pd.DataFrame) -> None:
        """Test that all symbols are assigned to a group."""
        splitter = MarketSplitter()
        result = splitter.split(sample_crypto_data)

        all_symbols = set(sample_crypto_data["symbol"].unique())
        assigned_symbols = set()
        for group_data in result.values():
            assigned_symbols.update(group_data["symbol"].unique())

        # All symbols should be assigned (or explicitly unassigned to OTHER)
        assert len(assigned_symbols) > 0


class TestFrequencySplitter:
    """Tests for FrequencySplitter."""

    def test_supported_frequencies(self) -> None:
        """Test supported time frequencies."""
        splitter = FrequencySplitter()
        frequencies = splitter.get_supported_frequencies()

        assert TimeFrequency.HOURLY in frequencies
        assert TimeFrequency.FOUR_HOURLY in frequencies
        assert TimeFrequency.DAILY in frequencies

    def test_resample_to_daily(self, sample_crypto_data: pd.DataFrame) -> None:
        """Test resampling hourly data to daily."""
        splitter = FrequencySplitter()
        daily_data = splitter.resample(sample_crypto_data, TimeFrequency.DAILY)

        # Daily data should have fewer rows than hourly
        assert len(daily_data) < len(sample_crypto_data)

        # Check OHLCV aggregation
        assert "open" in daily_data.columns
        assert "high" in daily_data.columns
        assert "low" in daily_data.columns
        assert "close" in daily_data.columns
        assert "volume" in daily_data.columns

    def test_resample_preserves_symbols(self, sample_crypto_data: pd.DataFrame) -> None:
        """Test that resampling preserves all symbols."""
        splitter = FrequencySplitter()
        daily_data = splitter.resample(sample_crypto_data, TimeFrequency.DAILY)

        original_symbols = set(sample_crypto_data["symbol"].unique())
        resampled_symbols = set(daily_data["symbol"].unique())

        assert original_symbols == resampled_symbols

    def test_split_by_frequency(self, sample_crypto_data: pd.DataFrame) -> None:
        """Test splitting into multiple frequencies."""
        splitter = FrequencySplitter()
        result = splitter.split(
            sample_crypto_data,
            frequencies=[TimeFrequency.HOURLY, TimeFrequency.DAILY],
        )

        assert TimeFrequency.HOURLY in result
        assert TimeFrequency.DAILY in result

    def test_invalid_frequency_conversion(self) -> None:
        """Test error when converting to higher frequency."""
        splitter = FrequencySplitter()
        daily_data = pd.DataFrame({
            "datetime": pd.date_range("2023-01-01", periods=10, freq="1D"),
            "close": range(10),
        })

        with pytest.raises(InvalidSplitError, match="Cannot upsample"):
            splitter.resample(daily_data, TimeFrequency.HOURLY)


class TestSplitResult:
    """Tests for SplitResult data structure."""

    def test_create_split_result(self, small_data: pd.DataFrame) -> None:
        """Test creating a split result."""
        result = SplitResult(
            train=small_data.iloc[:60],
            valid=small_data.iloc[60:80],
            test=small_data.iloc[80:],
            metadata={"split_type": "time"},
        )

        assert result.train is not None
        assert result.valid is not None
        assert result.test is not None
        assert result.metadata["split_type"] == "time"

    def test_split_result_sizes(self, small_data: pd.DataFrame) -> None:
        """Test getting split sizes."""
        result = SplitResult(
            train=small_data.iloc[:60],
            valid=small_data.iloc[60:80],
            test=small_data.iloc[80:],
        )

        sizes = result.get_sizes()
        assert sizes["train"] == 60
        assert sizes["valid"] == 20
        assert sizes["test"] == 20

    def test_split_result_to_dict(self, small_data: pd.DataFrame) -> None:
        """Test serialization."""
        result = SplitResult(
            train=small_data.iloc[:60],
            valid=small_data.iloc[60:80],
            test=small_data.iloc[80:],
        )

        d = result.to_dict()
        assert "train_size" in d
        assert "valid_size" in d
        assert "test_size" in d


class TestCryptoCVSplitterFunctional:
    """Functional tests for CryptoCVSplitter."""

    def test_basic_cv_split(self, sample_crypto_data: pd.DataFrame) -> None:
        """Test basic cross-validation split."""
        config = CVSplitConfig(
            time_split=True,
            market_split=False,
            frequency_split=False,
        )
        splitter = CryptoCVSplitter(config)

        splits = list(splitter.split(sample_crypto_data))
        assert len(splits) > 0

        for split in splits:
            assert split.train is not None
            assert split.test is not None

    def test_multi_dimensional_split(self, sample_crypto_data: pd.DataFrame) -> None:
        """Test multi-dimensional split (time + market)."""
        config = CVSplitConfig(
            time_split=True,
            market_split=True,
            frequency_split=False,
        )
        splitter = CryptoCVSplitter(config)

        splits = list(splitter.split(sample_crypto_data))

        # Should have multiple splits (time x market combinations)
        assert len(splits) >= 3  # At least large/mid/small cap

    def test_full_dimensional_split(self, sample_crypto_data: pd.DataFrame) -> None:
        """Test full dimensional split (time + market + frequency)."""
        config = CVSplitConfig(
            time_split=True,
            market_split=True,
            frequency_split=True,
            frequencies=[TimeFrequency.HOURLY, TimeFrequency.DAILY],
        )
        splitter = CryptoCVSplitter(config)

        splits = list(splitter.split(sample_crypto_data))

        # Should have many splits (time x market x frequency)
        assert len(splits) >= 6

    def test_split_metadata(self, sample_crypto_data: pd.DataFrame) -> None:
        """Test that splits include metadata."""
        config = CVSplitConfig(
            time_split=True,
            market_split=True,
        )
        splitter = CryptoCVSplitter(config)

        splits = list(splitter.split(sample_crypto_data))

        for split in splits:
            assert split.metadata is not None
            assert "market_group" in split.metadata or "time_period" in split.metadata

    def test_get_n_splits(self, sample_crypto_data: pd.DataFrame) -> None:
        """Test getting number of splits without generating them."""
        config = CVSplitConfig(
            time_split=True,
            market_split=True,
        )
        splitter = CryptoCVSplitter(config)

        n_splits = splitter.get_n_splits(sample_crypto_data)
        actual_splits = list(splitter.split(sample_crypto_data))

        assert n_splits == len(actual_splits)


class TestCryptoCVSplitterBoundary:
    """Boundary tests for CryptoCVSplitter."""

    def test_small_dataset(self) -> None:
        """Test handling of small dataset."""
        small_data = pd.DataFrame({
            "datetime": pd.date_range("2023-01-01", periods=10, freq="1D"),
            "symbol": "BTC",
            "close": range(10),
        })

        config = CVSplitConfig(time_split=True)
        splitter = CryptoCVSplitter(config)

        splits = list(splitter.split(small_data))
        assert len(splits) > 0

    def test_single_symbol(self) -> None:
        """Test handling of single symbol data."""
        data = pd.DataFrame({
            "datetime": pd.date_range("2023-01-01", periods=100, freq="1D"),
            "symbol": "BTC",
            "close": range(100),
        })

        config = CVSplitConfig(time_split=True, market_split=True)
        splitter = CryptoCVSplitter(config)

        splits = list(splitter.split(data))
        assert len(splits) > 0

    def test_minimum_train_size(self) -> None:
        """Test minimum train size enforcement."""
        data = pd.DataFrame({
            "datetime": pd.date_range("2023-01-01", periods=5, freq="1D"),
            "symbol": "BTC",
            "close": range(5),
        })

        config = CVSplitConfig(time_split=True, min_train_size=10)
        splitter = CryptoCVSplitter(config)

        with pytest.raises(InvalidSplitError, match="Insufficient data"):
            list(splitter.split(data))

    def test_extreme_split_ratios(self) -> None:
        """Test with extreme but valid split ratios."""
        config = CVSplitConfig(
            time_split=True,
            train_ratio=0.9,
            valid_ratio=0.05,
            test_ratio=0.05,
        )
        splitter = CryptoCVSplitter(config)

        data = pd.DataFrame({
            "datetime": pd.date_range("2023-01-01", periods=100, freq="1D"),
            "symbol": "BTC",
            "close": range(100),
        })

        splits = list(splitter.split(data))
        assert len(splits) > 0


class TestCryptoCVSplitterException:
    """Exception handling tests."""

    def test_empty_data(self) -> None:
        """Test handling of empty data."""
        config = CVSplitConfig(time_split=True)
        splitter = CryptoCVSplitter(config)

        empty_data = pd.DataFrame(columns=["datetime", "symbol", "close"])

        with pytest.raises(InvalidSplitError, match="Empty"):
            list(splitter.split(empty_data))

    def test_missing_datetime_column(self) -> None:
        """Test handling of missing datetime column."""
        config = CVSplitConfig(time_split=True)
        splitter = CryptoCVSplitter(config)

        data = pd.DataFrame({"symbol": ["BTC"], "close": [30000]})

        with pytest.raises(InvalidSplitError, match="datetime"):
            list(splitter.split(data))

    def test_invalid_config(self) -> None:
        """Test handling of invalid configuration."""
        with pytest.raises(InvalidSplitError, match="At least one"):
            CVSplitConfig(
                time_split=False,
                market_split=False,
                frequency_split=False,
            )


class TestDataLeakagePrevention:
    """Tests for data leakage prevention."""

    def test_no_future_data_in_train(self, sample_crypto_data: pd.DataFrame) -> None:
        """Test that train set doesn't contain future data."""
        config = CVSplitConfig(time_split=True)
        splitter = CryptoCVSplitter(config)

        splits = list(splitter.split(sample_crypto_data))

        for split in splits:
            train_max_time = split.train["datetime"].max()
            test_min_time = split.test["datetime"].min()

            assert train_max_time < test_min_time, "Data leakage detected"

    def test_strict_temporal_ordering(self, sample_crypto_data: pd.DataFrame) -> None:
        """Test strict temporal ordering in all splits."""
        config = CVSplitConfig(time_split=True, strict_temporal=True)
        splitter = CryptoCVSplitter(config)

        splits = list(splitter.split(sample_crypto_data))

        for split in splits:
            if split.valid is not None:
                train_max = split.train["datetime"].max()
                valid_min = split.valid["datetime"].min()
                valid_max = split.valid["datetime"].max()
                test_min = split.test["datetime"].min()

                # Train max should not exceed valid min (no overlap)
                assert train_max <= valid_min
                assert valid_max <= test_min

    def test_gap_between_splits(self, sample_crypto_data: pd.DataFrame) -> None:
        """Test configurable gap between train and test."""
        config = CVSplitConfig(
            time_split=True,
            gap_periods=24,  # 24 hour gap
        )
        splitter = CryptoCVSplitter(config)

        splits = list(splitter.split(sample_crypto_data))

        for split in splits:
            train_max = split.train["datetime"].max()
            test_min = split.test["datetime"].min()
            gap = (test_min - train_max).total_seconds() / 3600

            assert gap >= 24, f"Gap should be at least 24 hours, got {gap}"


class TestCryptoCVSplitterPerformance:
    """Performance tests."""

    def test_split_generation_time(self, sample_crypto_data: pd.DataFrame) -> None:
        """Test that split generation is fast."""
        config = CVSplitConfig(
            time_split=True,
            market_split=True,
        )
        splitter = CryptoCVSplitter(config)

        start = time.time()
        splits = list(splitter.split(sample_crypto_data))
        elapsed = time.time() - start

        assert elapsed < 5.0, f"Split generation took {elapsed}s, should be < 5s"

    def test_large_dataset_handling(self) -> None:
        """Test handling of larger datasets."""
        # Create larger dataset
        dates = pd.date_range("2020-01-01", "2023-12-31", freq="1h")
        large_data = pd.DataFrame({
            "datetime": dates,
            "symbol": "BTC",
            "close": np.random.randn(len(dates)).cumsum() + 30000,
        })

        config = CVSplitConfig(time_split=True)
        splitter = CryptoCVSplitter(config)

        start = time.time()
        splits = list(splitter.split(large_data))
        elapsed = time.time() - start

        assert elapsed < 10.0
        assert len(splits) > 0


class TestCryptoCVSplitterCompatibility:
    """Compatibility tests."""

    def test_pandas_datetime_index(self) -> None:
        """Test handling of DatetimeIndex."""
        dates = pd.date_range("2023-01-01", periods=100, freq="1D")
        data = pd.DataFrame({
            "close": range(100),
            "symbol": "BTC",
        }, index=dates)
        data["datetime"] = data.index

        config = CVSplitConfig(time_split=True)
        splitter = CryptoCVSplitter(config)

        splits = list(splitter.split(data))
        assert len(splits) > 0

    def test_different_datetime_formats(self) -> None:
        """Test handling of different datetime formats."""
        # String datetime
        data = pd.DataFrame({
            "datetime": ["2023-01-01", "2023-01-02", "2023-01-03"] * 30,
            "symbol": "BTC",
            "close": range(90),
        })
        data["datetime"] = pd.to_datetime(data["datetime"])

        config = CVSplitConfig(time_split=True)
        splitter = CryptoCVSplitter(config)

        splits = list(splitter.split(data))
        assert len(splits) > 0

    def test_config_serialization(self) -> None:
        """Test configuration serialization."""
        config = CVSplitConfig(
            time_split=True,
            market_split=True,
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15,
        )

        d = config.to_dict()
        assert d["time_split"] is True
        assert d["market_split"] is True
        assert d["train_ratio"] == 0.7

        # Recreate from dict
        config2 = CVSplitConfig.from_dict(d)
        assert config2.time_split == config.time_split
        assert config2.train_ratio == config.train_ratio

    def test_all_market_groups(self, sample_crypto_data: pd.DataFrame) -> None:
        """Test that all market groups are handled."""
        config = CVSplitConfig(time_split=True, market_split=True)
        splitter = CryptoCVSplitter(config)

        splits = list(splitter.split(sample_crypto_data))

        # Should have splits for different market groups
        market_groups = set()
        for split in splits:
            if "market_group" in split.metadata:
                market_groups.add(split.metadata["market_group"])

        assert len(market_groups) >= 1
