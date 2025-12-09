"""Cross-Validation Splitter for Crypto Data.

Implements multi-dimensional data splitting for robust factor evaluation:
- Time-based splits (Train/Valid/Test)
- Market-based splits (Large/Mid/Small cap)
- Frequency-based splits (1h/4h/1d)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Optional
import pandas as pd
import numpy as np


class InvalidSplitError(Exception):
    """Raised when split configuration or data is invalid."""

    pass


class DataLeakageError(Exception):
    """Raised when data leakage is detected."""

    pass


class MarketGroup(Enum):
    """Market capitalization groups."""

    LARGE_CAP = "large_cap"
    MID_CAP = "mid_cap"
    SMALL_CAP = "small_cap"
    OTHER = "other"


class TimeFrequency(Enum):
    """Supported time frequencies."""

    HOURLY = "1h"
    FOUR_HOURLY = "4h"
    DAILY = "1d"
    WEEKLY = "1w"

    @property
    def pandas_freq(self) -> str:
        """Get pandas frequency string."""
        mapping = {
            TimeFrequency.HOURLY: "1h",
            TimeFrequency.FOUR_HOURLY: "4h",
            TimeFrequency.DAILY: "1D",
            TimeFrequency.WEEKLY: "1W",
        }
        return mapping[self]

    @property
    def hours(self) -> int:
        """Get frequency in hours."""
        mapping = {
            TimeFrequency.HOURLY: 1,
            TimeFrequency.FOUR_HOURLY: 4,
            TimeFrequency.DAILY: 24,
            TimeFrequency.WEEKLY: 168,
        }
        return mapping[self]


@dataclass
class SplitResult:
    """Result of a data split operation."""

    train: pd.DataFrame
    test: pd.DataFrame
    valid: Optional[pd.DataFrame] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_sizes(self) -> dict[str, int]:
        """Get sizes of each split."""
        sizes = {
            "train": len(self.train),
            "test": len(self.test),
        }
        if self.valid is not None:
            sizes["valid"] = len(self.valid)
        return sizes

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        sizes = self.get_sizes()
        return {
            "train_size": sizes["train"],
            "valid_size": sizes.get("valid", 0),
            "test_size": sizes["test"],
            "metadata": self.metadata,
        }


@dataclass
class CVSplitConfig:
    """Configuration for cross-validation splits."""

    time_split: bool = True
    market_split: bool = False
    frequency_split: bool = False

    # Time split ratios
    train_ratio: float = 0.6
    valid_ratio: float = 0.2
    test_ratio: float = 0.2

    # Rolling window settings
    rolling_mode: bool = False
    window_size: int = 30
    step_size: int = 7

    # Minimum sizes
    min_train_size: int = 10

    # Frequencies for frequency split
    frequencies: list[TimeFrequency] = field(
        default_factory=lambda: [TimeFrequency.HOURLY, TimeFrequency.DAILY]
    )

    # Data leakage prevention
    strict_temporal: bool = True
    gap_periods: int = 0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not (self.time_split or self.market_split or self.frequency_split):
            raise InvalidSplitError("At least one split dimension must be enabled")

        if abs(self.train_ratio + self.valid_ratio + self.test_ratio - 1.0) > 1e-6:
            raise InvalidSplitError("Split ratios must sum to 1")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "time_split": self.time_split,
            "market_split": self.market_split,
            "frequency_split": self.frequency_split,
            "train_ratio": self.train_ratio,
            "valid_ratio": self.valid_ratio,
            "test_ratio": self.test_ratio,
            "rolling_mode": self.rolling_mode,
            "window_size": self.window_size,
            "step_size": self.step_size,
            "min_train_size": self.min_train_size,
            "frequencies": [f.value for f in self.frequencies],
            "strict_temporal": self.strict_temporal,
            "gap_periods": self.gap_periods,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CVSplitConfig":
        """Create from dictionary."""
        if "frequencies" in d and isinstance(d["frequencies"], list):
            d = d.copy()
            d["frequencies"] = [
                TimeFrequency(f) if isinstance(f, str) else f
                for f in d["frequencies"]
            ]
        return cls(**d)


class TimeSplitter:
    """Time-based data splitter."""

    def __init__(
        self,
        train_ratio: float = 0.6,
        valid_ratio: float = 0.2,
        test_ratio: float = 0.2,
        mode: str = "sequential",
        window_size: int = 30,
        step_size: int = 7,
    ) -> None:
        """Initialize time splitter.

        Args:
            train_ratio: Ratio of data for training
            valid_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            mode: "sequential" or "rolling"
            window_size: Window size for rolling mode
            step_size: Step size for rolling mode
        """
        if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
            raise InvalidSplitError("Split ratios must sum to 1")

        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.mode = mode
        self.window_size = window_size
        self.step_size = step_size

    def split(self, data: pd.DataFrame) -> SplitResult:
        """Split data by time.

        Args:
            data: DataFrame with 'datetime' column

        Returns:
            SplitResult with train/valid/test sets
        """
        if "datetime" not in data.columns:
            raise InvalidSplitError("Data must have 'datetime' column")

        # Sort by time
        data = data.sort_values("datetime").reset_index(drop=True)

        n = len(data)
        train_end = int(n * self.train_ratio)
        valid_end = train_end + int(n * self.valid_ratio)

        return SplitResult(
            train=data.iloc[:train_end].copy(),
            valid=data.iloc[train_end:valid_end].copy(),
            test=data.iloc[valid_end:].copy(),
            metadata={"split_type": "time", "mode": self.mode},
        )

    def split_rolling(self, data: pd.DataFrame) -> Iterator[SplitResult]:
        """Generate rolling window splits.

        Args:
            data: DataFrame with 'datetime' column

        Yields:
            SplitResult for each window
        """
        if "datetime" not in data.columns:
            raise InvalidSplitError("Data must have 'datetime' column")

        data = data.sort_values("datetime").reset_index(drop=True)
        n = len(data)

        start = 0
        window_idx = 0

        while start + self.window_size < n:
            train_end = start + int(self.window_size * self.train_ratio)
            test_start = train_end

            yield SplitResult(
                train=data.iloc[start:train_end].copy(),
                test=data.iloc[test_start:start + self.window_size].copy(),
                metadata={
                    "split_type": "time_rolling",
                    "window_idx": window_idx,
                    "start": start,
                },
            )

            start += self.step_size
            window_idx += 1


class MarketSplitter:
    """Market capitalization-based data splitter."""

    DEFAULT_GROUPS = {
        MarketGroup.LARGE_CAP: ["BTC", "ETH"],
        MarketGroup.MID_CAP: ["BNB", "SOL", "XRP", "ADA", "DOGE", "DOT", "MATIC", "LINK"],
        MarketGroup.SMALL_CAP: [],  # Everything else
    }

    def __init__(
        self,
        custom_groups: Optional[dict[MarketGroup, list[str]]] = None,
    ) -> None:
        """Initialize market splitter.

        Args:
            custom_groups: Custom market group definitions
        """
        self.groups = custom_groups or self.DEFAULT_GROUPS.copy()

    def get_market_groups(self) -> list[MarketGroup]:
        """Get available market groups."""
        return list(self.groups.keys())

    def get_symbols_for_group(self, group: MarketGroup) -> list[str]:
        """Get symbols for a market group."""
        return self.groups.get(group, [])

    def _assign_group(self, symbol: str) -> MarketGroup:
        """Assign a symbol to a market group."""
        for group, symbols in self.groups.items():
            if symbol in symbols:
                return group
        return MarketGroup.SMALL_CAP  # Default to small cap

    def split(self, data: pd.DataFrame) -> dict[MarketGroup, pd.DataFrame]:
        """Split data by market capitalization.

        Args:
            data: DataFrame with 'symbol' column

        Returns:
            Dictionary mapping market groups to data
        """
        if "symbol" not in data.columns:
            return {MarketGroup.OTHER: data}

        result: dict[MarketGroup, pd.DataFrame] = {}

        for symbol in data["symbol"].unique():
            group = self._assign_group(symbol)
            symbol_data = data[data["symbol"] == symbol]

            if group not in result:
                result[group] = symbol_data.copy()
            else:
                result[group] = pd.concat([result[group], symbol_data])

        return result


class FrequencySplitter:
    """Frequency-based data splitter with resampling."""

    def __init__(self) -> None:
        """Initialize frequency splitter."""
        pass

    def get_supported_frequencies(self) -> list[TimeFrequency]:
        """Get list of supported frequencies."""
        return list(TimeFrequency)

    def _detect_frequency(self, data: pd.DataFrame) -> int:
        """Detect data frequency in hours."""
        if "datetime" not in data.columns or len(data) < 2:
            return 1  # Assume hourly

        data_sorted = data.sort_values("datetime")
        diffs = data_sorted["datetime"].diff().dropna()
        if len(diffs) == 0:
            return 1

        median_diff = diffs.median()
        hours = median_diff.total_seconds() / 3600
        return max(1, int(round(hours)))

    def resample(
        self,
        data: pd.DataFrame,
        target_freq: TimeFrequency,
    ) -> pd.DataFrame:
        """Resample data to target frequency.

        Args:
            data: DataFrame with 'datetime' and OHLCV columns
            target_freq: Target frequency

        Returns:
            Resampled DataFrame
        """
        current_hours = self._detect_frequency(data)
        target_hours = target_freq.hours

        if target_hours < current_hours:
            raise InvalidSplitError(
                f"Cannot upsample from {current_hours}h to {target_hours}h"
            )

        if target_hours == current_hours:
            return data.copy()

        # Group by symbol if present
        if "symbol" in data.columns:
            result_dfs = []
            for symbol in data["symbol"].unique():
                symbol_data = data[data["symbol"] == symbol].copy()
                resampled = self._resample_single(symbol_data, target_freq)
                resampled["symbol"] = symbol
                result_dfs.append(resampled)
            return pd.concat(result_dfs, ignore_index=True)
        else:
            return self._resample_single(data, target_freq)

    def _resample_single(
        self,
        data: pd.DataFrame,
        target_freq: TimeFrequency,
    ) -> pd.DataFrame:
        """Resample a single symbol's data."""
        data = data.set_index("datetime")

        agg_rules: dict[str, str] = {}
        if "open" in data.columns:
            agg_rules["open"] = "first"
        if "high" in data.columns:
            agg_rules["high"] = "max"
        if "low" in data.columns:
            agg_rules["low"] = "min"
        if "close" in data.columns:
            agg_rules["close"] = "last"
        if "volume" in data.columns:
            agg_rules["volume"] = "sum"

        if not agg_rules:
            agg_rules = {data.columns[0]: "last"}

        resampled = data.resample(target_freq.pandas_freq).agg(agg_rules)
        resampled = resampled.dropna()
        resampled = resampled.reset_index()

        return resampled

    def split(
        self,
        data: pd.DataFrame,
        frequencies: list[TimeFrequency],
    ) -> dict[TimeFrequency, pd.DataFrame]:
        """Split data into multiple frequencies.

        Args:
            data: Input DataFrame
            frequencies: List of target frequencies

        Returns:
            Dictionary mapping frequencies to data
        """
        result: dict[TimeFrequency, pd.DataFrame] = {}

        for freq in frequencies:
            try:
                result[freq] = self.resample(data, freq)
            except InvalidSplitError:
                # Skip frequencies we can't resample to
                pass

        return result


class CryptoCVSplitter:
    """Multi-dimensional cross-validation splitter for crypto data."""

    def __init__(self, config: CVSplitConfig) -> None:
        """Initialize CV splitter.

        Args:
            config: Split configuration
        """
        self.config = config
        self.time_splitter = TimeSplitter(
            train_ratio=config.train_ratio,
            valid_ratio=config.valid_ratio,
            test_ratio=config.test_ratio,
            mode="rolling" if config.rolling_mode else "sequential",
            window_size=config.window_size,
            step_size=config.step_size,
        )
        self.market_splitter = MarketSplitter()
        self.frequency_splitter = FrequencySplitter()

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data."""
        if data is None or len(data) == 0:
            raise InvalidSplitError("Empty data provided")

        if self.config.time_split and "datetime" not in data.columns:
            raise InvalidSplitError("Data must have 'datetime' column for time split")

        if len(data) < self.config.min_train_size:
            raise InvalidSplitError(
                f"Insufficient data: {len(data)} rows, "
                f"minimum required: {self.config.min_train_size}"
            )

    def _apply_gap(self, split: SplitResult) -> SplitResult:
        """Apply gap between train and test sets."""
        if self.config.gap_periods <= 0:
            return split

        # Remove gap_periods from the end of train
        train = split.train.copy()
        if len(train) > self.config.gap_periods:
            train = train.iloc[:-self.config.gap_periods]

        return SplitResult(
            train=train,
            valid=split.valid,
            test=split.test,
            metadata=split.metadata,
        )

    def get_n_splits(self, data: pd.DataFrame) -> int:
        """Get the number of splits that will be generated.

        Args:
            data: Input DataFrame

        Returns:
            Number of splits
        """
        n_splits = 1

        if self.config.market_split and "symbol" in data.columns:
            n_symbols = len(data["symbol"].unique())
            n_splits *= min(n_symbols, 3)  # Max 3 market groups

        if self.config.frequency_split:
            n_splits *= len(self.config.frequencies)

        return n_splits

    def split(self, data: pd.DataFrame) -> Iterator[SplitResult]:
        """Generate cross-validation splits.

        Args:
            data: Input DataFrame

        Yields:
            SplitResult for each split configuration
        """
        self._validate_data(data)

        # Get data variants to process
        data_variants: list[tuple[pd.DataFrame, dict[str, Any]]] = [(data, {})]

        # Apply market split
        if self.config.market_split:
            new_variants = []
            for d, meta in data_variants:
                market_splits = self.market_splitter.split(d)
                for group, group_data in market_splits.items():
                    new_meta = {**meta, "market_group": group.value}
                    new_variants.append((group_data, new_meta))
            data_variants = new_variants if new_variants else data_variants

        # Apply frequency split
        if self.config.frequency_split:
            new_variants = []
            for d, meta in data_variants:
                freq_splits = self.frequency_splitter.split(d, self.config.frequencies)
                for freq, freq_data in freq_splits.items():
                    new_meta = {**meta, "frequency": freq.value}
                    new_variants.append((freq_data, new_meta))
            data_variants = new_variants if new_variants else data_variants

        # Apply time split to each variant
        for d, meta in data_variants:
            if len(d) == 0:
                continue

            if self.config.time_split:
                if self.config.rolling_mode:
                    for split in self.time_splitter.split_rolling(d):
                        split.metadata.update(meta)
                        yield self._apply_gap(split)
                else:
                    split = self.time_splitter.split(d)
                    split.metadata.update(meta)
                    yield self._apply_gap(split)
            else:
                # No time split, just return the data
                yield SplitResult(
                    train=d.iloc[:int(len(d) * 0.8)].copy(),
                    test=d.iloc[int(len(d) * 0.8):].copy(),
                    metadata=meta,
                )
