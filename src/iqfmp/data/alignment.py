"""Time Alignment Utilities for Crypto Derivative Data.

This module provides tools for aligning different timeframe data,
specifically for handling the 8-hour funding rate data with hourly
or other frequency OHLCV data.

Funding Rate Schedule (Binance/Bybit):
- Settlement times: 00:00, 08:00, 16:00 UTC
- Rate applies for the 8 hours AFTER settlement
- For prediction: use the rate announced BEFORE settlement

Key Functions:
- align_funding_to_ohlcv: Align 8h funding to any OHLCV frequency
- get_funding_settlement_times: Get settlement timestamps
- merge_derivative_data: Merge multiple derivative data sources
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Literal

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# Funding settlement hours (UTC)
FUNDING_SETTLEMENT_HOURS = [0, 8, 16]  # 00:00, 08:00, 16:00 UTC


def get_funding_settlement_times(
    start_date: datetime,
    end_date: datetime,
) -> list[datetime]:
    """Get all funding settlement timestamps between two dates.

    Funding settles at 00:00, 08:00, 16:00 UTC on Binance/Bybit.

    Args:
        start_date: Start datetime (UTC)
        end_date: End datetime (UTC)

    Returns:
        List of settlement timestamps
    """
    settlements = []

    # Start from beginning of start_date
    current = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)

    # Ensure end_date has timezone
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    while current <= end_date:
        for hour in FUNDING_SETTLEMENT_HOURS:
            settlement = current.replace(hour=hour)
            if start_date <= settlement <= end_date:
                settlements.append(settlement)
        current += timedelta(days=1)

    return sorted(settlements)


def get_previous_settlement_time(timestamp: datetime) -> datetime:
    """Get the most recent funding settlement time before a timestamp.

    Args:
        timestamp: Target datetime

    Returns:
        Most recent settlement datetime
    """
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    hour = timestamp.hour

    # Find the most recent settlement hour
    if hour >= 16:
        settlement_hour = 16
    elif hour >= 8:
        settlement_hour = 8
    else:
        settlement_hour = 0

    settlement = timestamp.replace(
        hour=settlement_hour, minute=0, second=0, microsecond=0
    )

    # If timestamp is exactly at settlement, we consider the previous one
    if settlement >= timestamp:
        if settlement_hour == 0:
            settlement = settlement - timedelta(days=1)
            settlement = settlement.replace(hour=16)
        elif settlement_hour == 8:
            settlement = settlement.replace(hour=0)
        else:  # 16
            settlement = settlement.replace(hour=8)

    return settlement


def get_next_settlement_time(timestamp: datetime) -> datetime:
    """Get the next funding settlement time after a timestamp.

    Args:
        timestamp: Target datetime

    Returns:
        Next settlement datetime
    """
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    hour = timestamp.hour

    # Find the next settlement hour
    if hour < 8:
        settlement_hour = 8
        settlement_date = timestamp.date()
    elif hour < 16:
        settlement_hour = 16
        settlement_date = timestamp.date()
    else:
        settlement_hour = 0
        settlement_date = timestamp.date() + timedelta(days=1)

    return datetime(
        settlement_date.year,
        settlement_date.month,
        settlement_date.day,
        settlement_hour,
        tzinfo=timezone.utc,
    )


def align_funding_to_ohlcv(
    funding_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    funding_col: str = "funding_rate",
    method: Literal["ffill", "last", "next"] = "ffill",
) -> pd.DataFrame:
    """Align 8-hour funding rate data to OHLCV frequency.

    Funding rate applies for the 8 hours AFTER the settlement time.
    For example, funding announced at 08:00 UTC applies to 08:00-16:00.

    Methods:
    - 'ffill': Forward-fill the funding rate (most common)
    - 'last': Use the last known funding rate
    - 'next': Use the next funding rate (for prediction tasks)

    Args:
        funding_df: DataFrame with funding rate data
            Required columns: 'timestamp', funding_col
        ohlcv_df: DataFrame with OHLCV data
            Required columns: 'timestamp'
        funding_col: Name of the funding rate column
        method: Alignment method ('ffill', 'last', 'next')

    Returns:
        OHLCV DataFrame with aligned funding rate column added
    """
    if funding_df.empty or ohlcv_df.empty:
        logger.warning("Empty DataFrame provided for alignment")
        result = ohlcv_df.copy()
        result[funding_col] = 0.0
        return result

    # Ensure timestamp columns exist
    if "timestamp" not in funding_df.columns:
        raise ValueError("funding_df must have 'timestamp' column")
    if "timestamp" not in ohlcv_df.columns:
        raise ValueError("ohlcv_df must have 'timestamp' column")

    # Prepare DataFrames
    funding = funding_df[["timestamp", funding_col]].copy()
    ohlcv = ohlcv_df.copy()

    # Ensure timezone-aware timestamps
    funding["timestamp"] = pd.to_datetime(funding["timestamp"], utc=True)
    ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], utc=True)

    # Set timestamp as index for merging
    funding = funding.set_index("timestamp")
    ohlcv_index = ohlcv.set_index("timestamp").index

    # Reindex funding to OHLCV frequency
    if method == "ffill":
        # Forward-fill: use the most recent funding rate
        aligned_funding = funding.reindex(
            funding.index.union(ohlcv_index)
        ).ffill()
        aligned_funding = aligned_funding.reindex(ohlcv_index)

    elif method == "last":
        # Last: use asof merge (similar to ffill but cleaner)
        aligned_funding = pd.merge_asof(
            ohlcv[["timestamp"]].sort_values("timestamp"),
            funding.reset_index().sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        ).set_index("timestamp")[[funding_col]]

    elif method == "next":
        # Next: use the upcoming funding rate (for prediction)
        aligned_funding = pd.merge_asof(
            ohlcv[["timestamp"]].sort_values("timestamp"),
            funding.reset_index().sort_values("timestamp"),
            on="timestamp",
            direction="forward",
        ).set_index("timestamp")[[funding_col]]

    else:
        raise ValueError(f"Unknown method: {method}. Use 'ffill', 'last', or 'next'")

    # Add aligned funding to OHLCV
    result = ohlcv.copy()
    result = result.set_index("timestamp")
    result[funding_col] = aligned_funding[funding_col]
    result = result.reset_index()

    # Fill any remaining NaN with 0
    result[funding_col] = result[funding_col].fillna(0.0)

    return result


def merge_derivative_data(
    ohlcv_df: pd.DataFrame,
    funding_df: Optional[pd.DataFrame] = None,
    oi_df: Optional[pd.DataFrame] = None,
    ls_ratio_df: Optional[pd.DataFrame] = None,
    liquidation_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge multiple derivative data sources with OHLCV data.

    Handles different timeframes:
    - Funding rate: 8h → forward-fill
    - Open Interest: 5m/15m → forward-fill
    - Long/Short ratio: 5m/15m → forward-fill
    - Liquidations: aggregate to OHLCV frequency

    Args:
        ohlcv_df: Base OHLCV DataFrame with 'timestamp'
        funding_df: Funding rate data with 'timestamp', 'funding_rate'
        oi_df: Open interest data with 'timestamp', 'open_interest'
        ls_ratio_df: Long/short ratio with 'timestamp', 'long_short_ratio'
        liquidation_df: Liquidation data with 'timestamp', 'liquidation_long', 'liquidation_short'

    Returns:
        Merged DataFrame with all columns aligned to OHLCV timestamps
    """
    result = ohlcv_df.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result = result.set_index("timestamp").sort_index()

    # Align funding rate (8h data)
    if funding_df is not None and not funding_df.empty:
        funding = funding_df.copy()
        funding["timestamp"] = pd.to_datetime(funding["timestamp"], utc=True)
        funding = funding.set_index("timestamp")["funding_rate"]

        # Forward-fill to OHLCV frequency
        aligned = funding.reindex(
            funding.index.union(result.index)
        ).ffill()
        result["funding_rate"] = aligned.reindex(result.index)
        logger.info(f"Aligned funding rate: {len(funding)} → {len(result)} rows")

    # Align open interest (5m/15m data, but often available at OHLCV freq)
    if oi_df is not None and not oi_df.empty:
        oi = oi_df.copy()
        oi["timestamp"] = pd.to_datetime(oi["timestamp"], utc=True)
        oi = oi.set_index("timestamp")["open_interest"]

        aligned = oi.reindex(oi.index.union(result.index)).ffill()
        result["open_interest"] = aligned.reindex(result.index)

        # Calculate OI change
        result["open_interest_change"] = result["open_interest"].pct_change()
        logger.info(f"Aligned open interest: {len(oi)} → {len(result)} rows")

    # Align long/short ratio
    if ls_ratio_df is not None and not ls_ratio_df.empty:
        ls = ls_ratio_df.copy()
        ls["timestamp"] = pd.to_datetime(ls["timestamp"], utc=True)
        ls = ls.set_index("timestamp")["long_short_ratio"]

        aligned = ls.reindex(ls.index.union(result.index)).ffill()
        result["long_short_ratio"] = aligned.reindex(result.index)
        logger.info(f"Aligned long/short ratio: {len(ls)} → {len(result)} rows")

    # Aggregate liquidations (sum within each OHLCV period)
    if liquidation_df is not None and not liquidation_df.empty:
        liq = liquidation_df.copy()
        liq["timestamp"] = pd.to_datetime(liq["timestamp"], utc=True)
        liq = liq.set_index("timestamp")

        # Determine OHLCV frequency
        if len(result) >= 2:
            freq = pd.infer_freq(result.index)
            if freq is None:
                # Estimate from first two timestamps
                diff = result.index[1] - result.index[0]
                freq = diff
            else:
                freq = pd.Timedelta(freq)
        else:
            freq = pd.Timedelta("1H")

        # Resample liquidations to OHLCV frequency (sum)
        if "liquidation_long" in liq.columns:
            result["liquidation_long"] = liq["liquidation_long"].resample(freq).sum().reindex(result.index).fillna(0)
        if "liquidation_short" in liq.columns:
            result["liquidation_short"] = liq["liquidation_short"].resample(freq).sum().reindex(result.index).fillna(0)

        # Total liquidation
        if "liquidation_long" in result.columns and "liquidation_short" in result.columns:
            result["liquidation_total"] = result["liquidation_long"] + result["liquidation_short"]

        logger.info(f"Aggregated liquidations: {len(liq)} → {len(result)} rows")

    # Reset index
    result = result.reset_index()

    # Fill remaining NaN
    for col in ["funding_rate", "open_interest", "long_short_ratio",
                "liquidation_long", "liquidation_short", "liquidation_total"]:
        if col in result.columns:
            result[col] = result[col].fillna(0.0)

    return result


def calculate_funding_features(
    df: pd.DataFrame,
    funding_col: str = "funding_rate",
) -> pd.DataFrame:
    """Calculate derived features from funding rate.

    Features:
    - funding_ma_8h: 8-hour moving average (1 funding period)
    - funding_ma_24h: 24-hour moving average (3 funding periods)
    - funding_momentum: Change over last 3 periods
    - funding_zscore: Z-score vs 30-day average
    - funding_extreme: Boolean flag for extreme funding (|z| > 2)
    - funding_annualized: Annualized funding rate (rate * 3 * 365)

    Args:
        df: DataFrame with funding_rate column
        funding_col: Name of funding rate column

    Returns:
        DataFrame with additional funding features
    """
    result = df.copy()

    if funding_col not in result.columns:
        logger.warning(f"Column {funding_col} not found, skipping funding features")
        return result

    funding = result[funding_col]

    # Detect data frequency (hourly vs 8-hourly)
    if "timestamp" in result.columns:
        result["timestamp"] = pd.to_datetime(result["timestamp"])
        if len(result) >= 2:
            freq_hours = (result["timestamp"].iloc[1] - result["timestamp"].iloc[0]).total_seconds() / 3600
        else:
            freq_hours = 1
    else:
        freq_hours = 1

    # Periods per 8 hours / 24 hours / 30 days
    periods_8h = max(1, int(8 / freq_hours))
    periods_24h = max(1, int(24 / freq_hours))
    periods_30d = max(1, int(30 * 24 / freq_hours))

    # Moving averages
    result["funding_ma_8h"] = funding.rolling(periods_8h).mean()
    result["funding_ma_24h"] = funding.rolling(periods_24h).mean()

    # Momentum (3 funding periods = 24 hours)
    result["funding_momentum"] = funding.diff(periods_24h)

    # Z-score vs 30-day average
    mean_30d = funding.rolling(periods_30d).mean()
    std_30d = funding.rolling(periods_30d).std()
    result["funding_zscore"] = (funding - mean_30d) / (std_30d + 1e-10)

    # Extreme funding flag (|z| > 2)
    result["funding_extreme"] = np.abs(result["funding_zscore"]) > 2

    # Annualized funding rate
    # Assuming funding is per 8-hour period, annualize: rate * 3 * 365
    result["funding_annualized"] = funding * 3 * 365

    return result


def validate_time_alignment(
    df: pd.DataFrame,
    expected_freq: str = "1H",
    max_gap_ratio: float = 0.05,
) -> tuple[bool, dict]:
    """Validate that DataFrame has proper time alignment.

    Args:
        df: DataFrame with 'timestamp' column
        expected_freq: Expected frequency string (e.g., '1H', '4H', '1D')
        max_gap_ratio: Maximum allowed ratio of gaps/missing periods

    Returns:
        Tuple of (is_valid, details_dict)
    """
    if "timestamp" not in df.columns:
        return False, {"error": "No timestamp column"}

    timestamps = pd.to_datetime(df["timestamp"]).sort_values()

    if len(timestamps) < 2:
        return True, {"warning": "Insufficient data for validation"}

    # Calculate expected number of periods
    date_range = pd.date_range(
        start=timestamps.min(),
        end=timestamps.max(),
        freq=expected_freq,
    )
    expected_count = len(date_range)
    actual_count = len(timestamps)

    # Check for gaps
    gaps = expected_count - actual_count
    gap_ratio = gaps / expected_count if expected_count > 0 else 0

    # Check for duplicates
    duplicates = len(timestamps) - len(timestamps.unique())

    details = {
        "expected_rows": expected_count,
        "actual_rows": actual_count,
        "gaps": gaps,
        "gap_ratio": gap_ratio,
        "duplicates": duplicates,
        "start": timestamps.min().isoformat(),
        "end": timestamps.max().isoformat(),
    }

    is_valid = gap_ratio <= max_gap_ratio and duplicates == 0

    return is_valid, details
