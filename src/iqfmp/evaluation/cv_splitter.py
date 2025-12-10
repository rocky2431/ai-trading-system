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


class MarketRegime(Enum):
    """Market regime classification."""

    LOW_VOL = "low_vol"
    MEDIUM_VOL = "medium_vol"
    HIGH_VOL = "high_vol"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"


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
    regime_split: bool = False

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

    # Regime split settings
    volatility_window: int = 20
    volatility_bins: list[float] = field(
        default_factory=lambda: [0.0, 0.02, 0.05, float("inf")]
    )
    trend_window: int = 20
    trend_threshold: float = 0.02

    # Data leakage prevention
    strict_temporal: bool = True
    gap_periods: int = 0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not (self.time_split or self.market_split or self.frequency_split or self.regime_split):
            raise InvalidSplitError("At least one split dimension must be enabled")

        if abs(self.train_ratio + self.valid_ratio + self.test_ratio - 1.0) > 1e-6:
            raise InvalidSplitError("Split ratios must sum to 1")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "time_split": self.time_split,
            "market_split": self.market_split,
            "frequency_split": self.frequency_split,
            "regime_split": self.regime_split,
            "train_ratio": self.train_ratio,
            "valid_ratio": self.valid_ratio,
            "test_ratio": self.test_ratio,
            "rolling_mode": self.rolling_mode,
            "window_size": self.window_size,
            "step_size": self.step_size,
            "min_train_size": self.min_train_size,
            "frequencies": [f.value for f in self.frequencies],
            "volatility_window": self.volatility_window,
            "volatility_bins": self.volatility_bins,
            "trend_window": self.trend_window,
            "trend_threshold": self.trend_threshold,
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


class RegimeSplitter:
    """Market regime-based data splitter.

    Supports two types of regime detection:
    1. Volatility-based: Low/Medium/High volatility regimes
    2. Trend-based: Trending Up/Trending Down/Ranging regimes
    """

    def __init__(
        self,
        volatility_window: int = 20,
        volatility_bins: Optional[list[float]] = None,
        trend_window: int = 20,
        trend_threshold: float = 0.02,
    ) -> None:
        """Initialize regime splitter.

        Args:
            volatility_window: Rolling window for volatility calculation
            volatility_bins: Bin edges for volatility classification [low, med, high, inf]
            trend_window: Rolling window for trend detection
            trend_threshold: Threshold for trend classification (abs return)
        """
        self.volatility_window = volatility_window
        self.volatility_bins = volatility_bins or [0.0, 0.02, 0.05, float("inf")]
        self.trend_window = trend_window
        self.trend_threshold = trend_threshold

    def detect_volatility_regime(
        self,
        data: pd.DataFrame,
        price_column: str = "close",
    ) -> pd.Series:
        """Detect volatility regime for each data point.

        Args:
            data: DataFrame with price data
            price_column: Name of price column

        Returns:
            Series with regime labels
        """
        if price_column not in data.columns:
            raise InvalidSplitError(f"Price column '{price_column}' not found")

        # Calculate rolling volatility
        returns = data[price_column].pct_change()
        volatility = returns.rolling(self.volatility_window).std()

        # Classify into regimes
        regime_labels = ["low_vol", "medium_vol", "high_vol"]
        regime = pd.cut(
            volatility,
            bins=self.volatility_bins,
            labels=regime_labels,
            include_lowest=True,
        )

        return regime

    def detect_trend_regime(
        self,
        data: pd.DataFrame,
        price_column: str = "close",
    ) -> pd.Series:
        """Detect trend regime for each data point.

        Args:
            data: DataFrame with price data
            price_column: Name of price column

        Returns:
            Series with regime labels
        """
        if price_column not in data.columns:
            raise InvalidSplitError(f"Price column '{price_column}' not found")

        # Calculate rolling return
        rolling_return = data[price_column].pct_change(self.trend_window)

        # Classify into regimes
        conditions = [
            rolling_return > self.trend_threshold,
            rolling_return < -self.trend_threshold,
        ]
        choices = [MarketRegime.TRENDING_UP.value, MarketRegime.TRENDING_DOWN.value]
        regime = pd.Series(
            np.select(conditions, choices, default=MarketRegime.RANGING.value),
            index=data.index,
        )

        return regime

    def detect_combined_regime(
        self,
        data: pd.DataFrame,
        price_column: str = "close",
    ) -> pd.Series:
        """Detect combined volatility + trend regime.

        Args:
            data: DataFrame with price data
            price_column: Name of price column

        Returns:
            Series with combined regime labels (e.g., "high_vol_trending_up")
        """
        vol_regime = self.detect_volatility_regime(data, price_column)
        trend_regime = self.detect_trend_regime(data, price_column)

        # Combine regimes
        combined = vol_regime.astype(str) + "_" + trend_regime.astype(str)
        return combined

    def split_by_regime(
        self,
        data: pd.DataFrame,
        regime_type: str = "volatility",
        regime_column: Optional[str] = None,
        price_column: str = "close",
    ) -> dict[str, pd.DataFrame]:
        """Split data by market regime.

        Args:
            data: Input DataFrame
            regime_type: "volatility", "trend", or "combined"
            regime_column: Pre-existing regime column name (optional)
            price_column: Price column for auto-detection

        Returns:
            Dictionary mapping regime names to DataFrames
        """
        if regime_column and regime_column in data.columns:
            regime = data[regime_column]
        elif regime_type == "volatility":
            regime = self.detect_volatility_regime(data, price_column)
        elif regime_type == "trend":
            regime = self.detect_trend_regime(data, price_column)
        elif regime_type == "combined":
            regime = self.detect_combined_regime(data, price_column)
        else:
            raise InvalidSplitError(f"Unknown regime type: {regime_type}")

        result: dict[str, pd.DataFrame] = {}

        for regime_name in regime.dropna().unique():
            mask = regime == regime_name
            regime_data = data[mask].copy()
            if len(regime_data) > 0:
                result[str(regime_name)] = regime_data

        return result


class CryptoCVSplitter:
    """Multi-dimensional cross-validation splitter for crypto data.

    Supports multiple split dimensions:
    - Time-based splits (Train/Valid/Test with sequential or rolling windows)
    - Market-based splits (Large/Mid/Small cap)
    - Frequency-based splits (1h/4h/1d)
    - Regime-based splits (High/Low volatility, Trending/Ranging)
    """

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
        self.regime_splitter = RegimeSplitter(
            volatility_window=config.volatility_window,
            volatility_bins=config.volatility_bins,
            trend_window=config.trend_window,
            trend_threshold=config.trend_threshold,
        )

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

        if self.config.regime_split:
            # Estimate 3 volatility regimes (low/medium/high)
            n_splits *= 3

        return n_splits

    def split(
        self,
        data: pd.DataFrame,
        regime_column: Optional[str] = None,
        regime_type: str = "volatility",
    ) -> Iterator[SplitResult]:
        """Generate cross-validation splits.

        Supports multi-dimensional splitting:
        - Time-based (sequential or rolling)
        - Market-based (large/mid/small cap)
        - Frequency-based (1h/4h/1d)
        - Regime-based (volatility or trend)

        Args:
            data: Input DataFrame
            regime_column: Optional pre-existing regime column name
            regime_type: Type of regime detection ("volatility", "trend", "combined")

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

        # Apply regime split
        if self.config.regime_split:
            new_variants = []
            for d, meta in data_variants:
                try:
                    regime_splits = self.regime_splitter.split_by_regime(
                        d,
                        regime_type=regime_type,
                        regime_column=regime_column,
                        price_column="close" if "close" in d.columns else d.columns[0],
                    )
                    for regime_name, regime_data in regime_splits.items():
                        new_meta = {**meta, "regime": regime_name}
                        new_variants.append((regime_data, new_meta))
                except InvalidSplitError:
                    # If regime split fails, keep original data
                    new_variants.append((d, meta))
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

    def split_with_regime_stratification(
        self,
        data: pd.DataFrame,
        regime_type: str = "volatility",
    ) -> Iterator[SplitResult]:
        """Generate regime-stratified cross-validation splits.

        Ensures each regime is proportionally represented in train/test splits.

        Args:
            data: Input DataFrame with price data
            regime_type: Type of regime detection

        Yields:
            SplitResult with stratified sampling across regimes
        """
        # Detect regime
        if regime_type == "volatility":
            regime = self.regime_splitter.detect_volatility_regime(data)
        elif regime_type == "trend":
            regime = self.regime_splitter.detect_trend_regime(data)
        else:
            regime = self.regime_splitter.detect_combined_regime(data)

        # Add regime column to data
        data_with_regime = data.copy()
        data_with_regime["_regime"] = regime

        # Get unique regimes
        unique_regimes = regime.dropna().unique()

        # For each regime, get time-ordered splits
        train_parts = []
        test_parts = []
        valid_parts = []

        for regime_name in unique_regimes:
            regime_data = data_with_regime[data_with_regime["_regime"] == regime_name]

            if len(regime_data) < self.config.min_train_size:
                continue

            # Apply time split within regime
            n = len(regime_data)
            train_end = int(n * self.config.train_ratio)
            valid_end = train_end + int(n * self.config.valid_ratio)

            regime_data_sorted = regime_data.sort_values("datetime")
            train_parts.append(regime_data_sorted.iloc[:train_end])
            if self.config.valid_ratio > 0:
                valid_parts.append(regime_data_sorted.iloc[train_end:valid_end])
            test_parts.append(regime_data_sorted.iloc[valid_end:])

        if not train_parts or not test_parts:
            return

        # Combine and sort by time
        train = pd.concat(train_parts).sort_values("datetime").drop(columns=["_regime"])
        test = pd.concat(test_parts).sort_values("datetime").drop(columns=["_regime"])
        valid = None
        if valid_parts:
            valid = pd.concat(valid_parts).sort_values("datetime").drop(columns=["_regime"])

        yield SplitResult(
            train=train,
            test=test,
            valid=valid,
            metadata={
                "split_type": "regime_stratified",
                "regime_type": regime_type,
                "regimes": list(unique_regimes),
            },
        )
