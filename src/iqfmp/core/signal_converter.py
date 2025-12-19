"""Signal Converter for bridging pandas factors to Qlib backtest.

This module provides the critical conversion layer between:
- pandas DataFrame factors (from LLM generation)
- Qlib Dataset format (for Qlib backtest engine)

The SignalConverter ensures compatibility between LLM-generated factor code
(which outputs pandas DataFrames) and Qlib's backtest infrastructure
(which expects Dataset format with MultiIndex).
"""

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class SignalConfig:
    """Configuration for signal conversion."""

    # Normalization
    normalize_method: str = "zscore"  # zscore, minmax, rank, none
    clip_std: float = 3.0  # Clip outliers beyond N std

    # Signal generation
    signal_threshold: float = 0.0  # Threshold for long/short
    top_k: Optional[int] = None  # Top-k selection

    # Position sizing
    position_scale: float = 1.0
    max_position: float = 0.1  # Max position per asset


class SignalConverter:
    """Convert pandas factors to trading signals and Qlib Dataset.

    This is the bridge between LLM-generated factor code and Qlib backtest.

    Usage:
        converter = SignalConverter(config)

        # From factor values to normalized signal
        signal = converter.to_signal(factor_values)

        # From signal to Qlib-compatible Dataset
        dataset = converter.to_qlib_dataset(signal, instruments)
    """

    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()

    def normalize(self, factor: pd.Series) -> pd.Series:
        """Normalize factor values.

        Args:
            factor: Raw factor values

        Returns:
            Normalized factor values
        """
        if self.config.normalize_method == "zscore":
            mean = factor.mean()
            std = factor.std()
            if std == 0 or pd.isna(std):
                return pd.Series(0, index=factor.index)
            normalized = (factor - mean) / std
            # Clip outliers
            return normalized.clip(-self.config.clip_std, self.config.clip_std)

        elif self.config.normalize_method == "minmax":
            min_val = factor.min()
            max_val = factor.max()
            if max_val == min_val:
                return pd.Series(0.5, index=factor.index)
            return (factor - min_val) / (max_val - min_val)

        elif self.config.normalize_method == "rank":
            return factor.rank(pct=True)

        else:  # none
            return factor

    def to_signal(
        self,
        factor: Union[pd.Series, pd.DataFrame],
        normalize: bool = True,
    ) -> pd.Series:
        """Convert factor values to trading signal.

        Args:
            factor: Factor values (can be DataFrame or Series)
            normalize: Whether to normalize first

        Returns:
            Trading signal (-1 to 1)
        """
        # Handle DataFrame input - extract first column or 'value' column
        if isinstance(factor, pd.DataFrame):
            if "value" in factor.columns:
                factor = factor["value"]
            elif "score" in factor.columns:
                factor = factor["score"]
            elif "factor" in factor.columns:
                factor = factor["factor"]
            else:
                factor = factor.iloc[:, 0]

        # Drop NaN values for calculation
        factor = factor.dropna()

        if normalize:
            factor = self.normalize(factor)

        if self.config.top_k:
            # Top-k selection: long top k, short bottom k
            k = min(self.config.top_k, len(factor) // 2)
            signal = pd.Series(0.0, index=factor.index)

            if k > 0:
                # Top k = long
                top_k_idx = factor.nlargest(k).index
                signal.loc[top_k_idx] = 1.0

                # Bottom k = short
                bottom_k_idx = factor.nsmallest(k).index
                signal.loc[bottom_k_idx] = -1.0

            return signal
        else:
            # Threshold-based: above threshold = long, below = short
            threshold = self.config.signal_threshold
            signal = pd.Series(0.0, index=factor.index)
            signal[factor > threshold] = factor[factor > threshold]
            signal[factor < -threshold] = factor[factor < -threshold]
            return signal.clip(-1, 1)

    def to_position(self, signal: pd.Series) -> pd.Series:
        """Convert signal to position weights.

        Args:
            signal: Trading signal

        Returns:
            Position weights (scaled and bounded)
        """
        position = signal * self.config.position_scale
        return position.clip(-self.config.max_position, self.config.max_position)

    def to_qlib_format(
        self,
        factor_df: pd.DataFrame,
        datetime_col: str = "datetime",
        instrument_col: str = "instrument",
    ) -> pd.DataFrame:
        """Convert pandas DataFrame to Qlib-compatible format.

        Qlib expects MultiIndex: (datetime, instrument)

        Args:
            factor_df: Factor DataFrame with datetime, instrument columns
            datetime_col: Name of datetime column
            instrument_col: Name of instrument column

        Returns:
            DataFrame with Qlib-compatible MultiIndex
        """
        df = factor_df.copy()

        # Handle 'symbol' as alias for 'instrument'
        if instrument_col not in df.columns and "symbol" in df.columns:
            df[instrument_col] = df["symbol"]

        # Ensure datetime column
        if datetime_col in df.columns:
            df[datetime_col] = pd.to_datetime(df[datetime_col])

        # Set MultiIndex
        if datetime_col in df.columns and instrument_col in df.columns:
            df = df.set_index([datetime_col, instrument_col])
            df.index.names = ["datetime", "instrument"]

        return df.sort_index()

    def create_prediction_dataset(
        self,
        signal: Union[pd.Series, pd.DataFrame],
        instruments: list[str],
        start_time: str,
        end_time: str,
    ) -> "QlibPredictionDataset":
        """Create a Qlib-compatible prediction dataset.

        This creates a minimal dataset structure that can be used
        with Qlib's backtest engine.

        Args:
            signal: Trading signal
            instruments: List of instrument codes
            start_time: Start datetime
            end_time: End datetime

        Returns:
            QlibPredictionDataset compatible with Qlib backtest
        """
        return QlibPredictionDataset(
            signal=signal,
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
        )

    def from_factor_output(
        self,
        factor_output: dict[str, Any],
        price_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Convert factor execution output to trading signal.

        Args:
            factor_output: Output from factor execution containing:
                - factor_values: The computed factor values
                - metadata: Optional metadata
            price_data: Optional price data for context

        Returns:
            Trading signal Series
        """
        factor_values = factor_output.get("factor_values")

        if factor_values is None:
            raise ValueError("factor_output must contain 'factor_values'")

        if isinstance(factor_values, dict):
            factor_values = pd.Series(factor_values)
        elif isinstance(factor_values, pd.DataFrame):
            if "value" in factor_values.columns:
                factor_values = factor_values["value"]
            else:
                factor_values = factor_values.iloc[:, 0]

        return self.to_signal(factor_values)


class QlibPredictionDataset:
    """Minimal Qlib-compatible prediction dataset.

    This class provides the minimal interface required by Qlib's
    backtest engine without requiring full Qlib data infrastructure.

    It wraps pandas DataFrame/Series signals and provides the necessary
    methods for Qlib backtest compatibility.
    """

    def __init__(
        self,
        signal: Union[pd.Series, pd.DataFrame],
        instruments: list[str],
        start_time: str,
        end_time: str,
    ):
        self.signal = signal
        self.instruments = instruments
        self.start_time = pd.Timestamp(start_time)
        self.end_time = pd.Timestamp(end_time)
        self._prepared = False

    def prepare(self, *args: Any, **kwargs: Any) -> "QlibPredictionDataset":
        """Prepare dataset (Qlib interface compatibility)."""
        self._prepared = True
        return self

    def get_segments(self) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
        """Get data segments (Qlib interface)."""
        return {
            "train": (self.start_time, self.end_time),
            "test": (self.start_time, self.end_time),
        }

    def __getitem__(self, key: Any) -> Union[pd.Series, pd.DataFrame, float]:
        """Get prediction for date/instrument (Qlib interface)."""
        if isinstance(self.signal, pd.DataFrame):
            if key in self.signal.index:
                return self.signal.loc[key]
        elif isinstance(self.signal, pd.Series):
            if key in self.signal.index:
                return self.signal.loc[key]
        return self.signal

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame (utility method)."""
        if isinstance(self.signal, pd.Series):
            return self.signal.to_frame(name="score")
        return self.signal

    @property
    def is_prepared(self) -> bool:
        """Check if dataset is prepared."""
        return self._prepared


# Factory function
def create_signal_converter(
    normalize: str = "zscore",
    top_k: Optional[int] = None,
    max_position: float = 0.1,
) -> SignalConverter:
    """Create a configured SignalConverter.

    Args:
        normalize: Normalization method
        top_k: Top-k selection (None for threshold-based)
        max_position: Maximum position per asset

    Returns:
        Configured SignalConverter
    """
    config = SignalConfig(
        normalize_method=normalize,
        top_k=top_k,
        max_position=max_position,
    )
    return SignalConverter(config)
