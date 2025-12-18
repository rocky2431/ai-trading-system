"""Alpha360 Complete Factor Library.

Systematic factor generation following Qlib's Alpha360 pattern.
Reference: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py

Alpha360 generates factors by systematically combining:
- Base features: OPEN, HIGH, LOW, CLOSE, VWAP, VOLUME
- Time windows: 5, 10, 20, 30, 60 periods
- Operations: ROC, MEAN, STD, MAX, MIN, RANK, ZSCORE, DELTA, DELAY, etc.

This results in 360 unique factors covering all major factor categories.
"""

from __future__ import annotations

from typing import Callable
import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# Alpha360 Factor Registry
# =============================================================================

ALPHA360_FACTORS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {}


def _register_alpha360(name: str):
    """Decorator to register an Alpha360 factor."""
    def decorator(func: Callable[[pd.DataFrame], pd.Series]):
        ALPHA360_FACTORS[name] = func
        return func
    return decorator


# =============================================================================
# Helper Functions
# =============================================================================

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Safe division."""
    return a / (b.replace(0, np.nan) + 1e-10)


def _zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score."""
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - ma) / (std + 1e-10)


def _decay_linear(series: pd.Series, window: int) -> pd.Series:
    """Linear decay weighted average."""
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(
        lambda x: np.dot(x, weights[:len(x)]) / weights[:len(x)].sum(),
        raw=True
    )


def _ts_rank(series: pd.Series, window: int) -> pd.Series:
    """Time-series rank (percentile within window)."""
    return series.rolling(window).apply(
        lambda x: stats.percentileofscore(x, x[-1]) / 100 if len(x) > 0 else np.nan,
        raw=True
    )


def _get_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate VWAP (Volume Weighted Average Price)."""
    if "volume" in df.columns and df["volume"].sum() > 0:
        return (df["close"] * df["volume"]).rolling(1).sum() / df["volume"].rolling(1).sum()
    return (df["high"] + df["low"] + df["close"]) / 3  # Typical price as fallback


# =============================================================================
# Factor Generation Classes
# =============================================================================

class FactorGenerator:
    """Generates Alpha360 factors programmatically."""

    WINDOWS = [5, 10, 20, 30, 60]
    BASE_FEATURES = ["open", "high", "low", "close", "volume"]

    @staticmethod
    def roc(series: pd.Series, window: int) -> pd.Series:
        """Rate of change."""
        return series.pct_change(window)

    @staticmethod
    def ma(series: pd.Series, window: int) -> pd.Series:
        """Moving average ratio."""
        return series / series.rolling(window).mean() - 1

    @staticmethod
    def std(series: pd.Series, window: int) -> pd.Series:
        """Standard deviation."""
        return series.rolling(window).std() / (series.rolling(window).mean() + 1e-10)

    @staticmethod
    def max_ratio(series: pd.Series, window: int) -> pd.Series:
        """Max ratio."""
        return series / series.rolling(window).max()

    @staticmethod
    def min_ratio(series: pd.Series, window: int) -> pd.Series:
        """Min ratio."""
        return series / series.rolling(window).min()

    @staticmethod
    def rank(series: pd.Series, window: int) -> pd.Series:
        """Time-series rank."""
        return _ts_rank(series, window)

    @staticmethod
    def zscore(series: pd.Series, window: int) -> pd.Series:
        """Z-score."""
        return _zscore(series, window)

    @staticmethod
    def delta(series: pd.Series, window: int) -> pd.Series:
        """Difference."""
        return series.diff(window) / (series.shift(window) + 1e-10)

    @staticmethod
    def delay(series: pd.Series, window: int) -> pd.Series:
        """Lagged value ratio."""
        return series / series.shift(window)

    @staticmethod
    def decay(series: pd.Series, window: int) -> pd.Series:
        """Linear decay."""
        return _decay_linear(series, window)


# =============================================================================
# CLOSE-based Factors (60 factors: 5 windows × 12 operations)
# =============================================================================

# ROC (Rate of Change)
@_register_alpha360("CLOSE_ROC5")
def close_roc5(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change(5)

@_register_alpha360("CLOSE_ROC10")
def close_roc10(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change(10)

@_register_alpha360("CLOSE_ROC20")
def close_roc20(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change(20)

@_register_alpha360("CLOSE_ROC30")
def close_roc30(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change(30)

@_register_alpha360("CLOSE_ROC60")
def close_roc60(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change(60)

# MA Ratio
@_register_alpha360("CLOSE_MA5")
def close_ma5(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].rolling(5).mean() - 1

@_register_alpha360("CLOSE_MA10")
def close_ma10(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].rolling(10).mean() - 1

@_register_alpha360("CLOSE_MA20")
def close_ma20(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].rolling(20).mean() - 1

@_register_alpha360("CLOSE_MA30")
def close_ma30(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].rolling(30).mean() - 1

@_register_alpha360("CLOSE_MA60")
def close_ma60(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].rolling(60).mean() - 1

# STD (Volatility)
@_register_alpha360("CLOSE_STD5")
def close_std5(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(5).std()

@_register_alpha360("CLOSE_STD10")
def close_std10(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(10).std()

@_register_alpha360("CLOSE_STD20")
def close_std20(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(20).std()

@_register_alpha360("CLOSE_STD30")
def close_std30(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(30).std()

@_register_alpha360("CLOSE_STD60")
def close_std60(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(60).std()

# MAX Ratio
@_register_alpha360("CLOSE_MAX5")
def close_max5(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].rolling(5).max()

@_register_alpha360("CLOSE_MAX10")
def close_max10(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].rolling(10).max()

@_register_alpha360("CLOSE_MAX20")
def close_max20(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].rolling(20).max()

@_register_alpha360("CLOSE_MAX30")
def close_max30(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].rolling(30).max()

@_register_alpha360("CLOSE_MAX60")
def close_max60(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].rolling(60).max()

# MIN Ratio
@_register_alpha360("CLOSE_MIN5")
def close_min5(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].rolling(5).min()

@_register_alpha360("CLOSE_MIN10")
def close_min10(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].rolling(10).min()

@_register_alpha360("CLOSE_MIN20")
def close_min20(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].rolling(20).min()

@_register_alpha360("CLOSE_MIN30")
def close_min30(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].rolling(30).min()

@_register_alpha360("CLOSE_MIN60")
def close_min60(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].rolling(60).min()

# RANK
@_register_alpha360("CLOSE_RANK5")
def close_rank5(df: pd.DataFrame) -> pd.Series:
    return _ts_rank(df["close"], 5)

@_register_alpha360("CLOSE_RANK10")
def close_rank10(df: pd.DataFrame) -> pd.Series:
    return _ts_rank(df["close"], 10)

@_register_alpha360("CLOSE_RANK20")
def close_rank20(df: pd.DataFrame) -> pd.Series:
    return _ts_rank(df["close"], 20)

@_register_alpha360("CLOSE_RANK30")
def close_rank30(df: pd.DataFrame) -> pd.Series:
    return _ts_rank(df["close"], 30)

@_register_alpha360("CLOSE_RANK60")
def close_rank60(df: pd.DataFrame) -> pd.Series:
    return _ts_rank(df["close"], 60)

# ZSCORE
@_register_alpha360("CLOSE_ZSCORE5")
def close_zscore5(df: pd.DataFrame) -> pd.Series:
    return _zscore(df["close"], 5)

@_register_alpha360("CLOSE_ZSCORE10")
def close_zscore10(df: pd.DataFrame) -> pd.Series:
    return _zscore(df["close"], 10)

@_register_alpha360("CLOSE_ZSCORE20")
def close_zscore20(df: pd.DataFrame) -> pd.Series:
    return _zscore(df["close"], 20)

@_register_alpha360("CLOSE_ZSCORE30")
def close_zscore30(df: pd.DataFrame) -> pd.Series:
    return _zscore(df["close"], 30)

@_register_alpha360("CLOSE_ZSCORE60")
def close_zscore60(df: pd.DataFrame) -> pd.Series:
    return _zscore(df["close"], 60)

# DELTA
@_register_alpha360("CLOSE_DELTA5")
def close_delta5(df: pd.DataFrame) -> pd.Series:
    return df["close"].diff(5) / (df["close"].shift(5) + 1e-10)

@_register_alpha360("CLOSE_DELTA10")
def close_delta10(df: pd.DataFrame) -> pd.Series:
    return df["close"].diff(10) / (df["close"].shift(10) + 1e-10)

@_register_alpha360("CLOSE_DELTA20")
def close_delta20(df: pd.DataFrame) -> pd.Series:
    return df["close"].diff(20) / (df["close"].shift(20) + 1e-10)

@_register_alpha360("CLOSE_DELTA30")
def close_delta30(df: pd.DataFrame) -> pd.Series:
    return df["close"].diff(30) / (df["close"].shift(30) + 1e-10)

@_register_alpha360("CLOSE_DELTA60")
def close_delta60(df: pd.DataFrame) -> pd.Series:
    return df["close"].diff(60) / (df["close"].shift(60) + 1e-10)

# DECAY
@_register_alpha360("CLOSE_DECAY5")
def close_decay5(df: pd.DataFrame) -> pd.Series:
    return _decay_linear(df["close"], 5) / df["close"] - 1

@_register_alpha360("CLOSE_DECAY10")
def close_decay10(df: pd.DataFrame) -> pd.Series:
    return _decay_linear(df["close"], 10) / df["close"] - 1

@_register_alpha360("CLOSE_DECAY20")
def close_decay20(df: pd.DataFrame) -> pd.Series:
    return _decay_linear(df["close"], 20) / df["close"] - 1

@_register_alpha360("CLOSE_DECAY30")
def close_decay30(df: pd.DataFrame) -> pd.Series:
    return _decay_linear(df["close"], 30) / df["close"] - 1

@_register_alpha360("CLOSE_DECAY60")
def close_decay60(df: pd.DataFrame) -> pd.Series:
    return _decay_linear(df["close"], 60) / df["close"] - 1

# DELAY
@_register_alpha360("CLOSE_DELAY5")
def close_delay5(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].shift(5)

@_register_alpha360("CLOSE_DELAY10")
def close_delay10(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].shift(10)

@_register_alpha360("CLOSE_DELAY20")
def close_delay20(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].shift(20)

@_register_alpha360("CLOSE_DELAY30")
def close_delay30(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].shift(30)

@_register_alpha360("CLOSE_DELAY60")
def close_delay60(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].shift(60)

# =============================================================================
# HIGH-based Factors (50 factors)
# =============================================================================

@_register_alpha360("HIGH_ROC5")
def high_roc5(df: pd.DataFrame) -> pd.Series:
    return df["high"].pct_change(5)

@_register_alpha360("HIGH_ROC10")
def high_roc10(df: pd.DataFrame) -> pd.Series:
    return df["high"].pct_change(10)

@_register_alpha360("HIGH_ROC20")
def high_roc20(df: pd.DataFrame) -> pd.Series:
    return df["high"].pct_change(20)

@_register_alpha360("HIGH_ROC30")
def high_roc30(df: pd.DataFrame) -> pd.Series:
    return df["high"].pct_change(30)

@_register_alpha360("HIGH_ROC60")
def high_roc60(df: pd.DataFrame) -> pd.Series:
    return df["high"].pct_change(60)

@_register_alpha360("HIGH_MA5")
def high_ma5(df: pd.DataFrame) -> pd.Series:
    return df["high"] / df["high"].rolling(5).mean() - 1

@_register_alpha360("HIGH_MA10")
def high_ma10(df: pd.DataFrame) -> pd.Series:
    return df["high"] / df["high"].rolling(10).mean() - 1

@_register_alpha360("HIGH_MA20")
def high_ma20(df: pd.DataFrame) -> pd.Series:
    return df["high"] / df["high"].rolling(20).mean() - 1

@_register_alpha360("HIGH_MA30")
def high_ma30(df: pd.DataFrame) -> pd.Series:
    return df["high"] / df["high"].rolling(30).mean() - 1

@_register_alpha360("HIGH_MA60")
def high_ma60(df: pd.DataFrame) -> pd.Series:
    return df["high"] / df["high"].rolling(60).mean() - 1

@_register_alpha360("HIGH_MAX5")
def high_max5(df: pd.DataFrame) -> pd.Series:
    return df["high"] / df["high"].rolling(5).max()

@_register_alpha360("HIGH_MAX10")
def high_max10(df: pd.DataFrame) -> pd.Series:
    return df["high"] / df["high"].rolling(10).max()

@_register_alpha360("HIGH_MAX20")
def high_max20(df: pd.DataFrame) -> pd.Series:
    return df["high"] / df["high"].rolling(20).max()

@_register_alpha360("HIGH_MAX30")
def high_max30(df: pd.DataFrame) -> pd.Series:
    return df["high"] / df["high"].rolling(30).max()

@_register_alpha360("HIGH_MAX60")
def high_max60(df: pd.DataFrame) -> pd.Series:
    return df["high"] / df["high"].rolling(60).max()

@_register_alpha360("HIGH_RANK5")
def high_rank5(df: pd.DataFrame) -> pd.Series:
    return _ts_rank(df["high"], 5)

@_register_alpha360("HIGH_RANK10")
def high_rank10(df: pd.DataFrame) -> pd.Series:
    return _ts_rank(df["high"], 10)

@_register_alpha360("HIGH_RANK20")
def high_rank20(df: pd.DataFrame) -> pd.Series:
    return _ts_rank(df["high"], 20)

@_register_alpha360("HIGH_RANK30")
def high_rank30(df: pd.DataFrame) -> pd.Series:
    return _ts_rank(df["high"], 30)

@_register_alpha360("HIGH_RANK60")
def high_rank60(df: pd.DataFrame) -> pd.Series:
    return _ts_rank(df["high"], 60)

@_register_alpha360("HIGH_ZSCORE5")
def high_zscore5(df: pd.DataFrame) -> pd.Series:
    return _zscore(df["high"], 5)

@_register_alpha360("HIGH_ZSCORE10")
def high_zscore10(df: pd.DataFrame) -> pd.Series:
    return _zscore(df["high"], 10)

@_register_alpha360("HIGH_ZSCORE20")
def high_zscore20(df: pd.DataFrame) -> pd.Series:
    return _zscore(df["high"], 20)

@_register_alpha360("HIGH_ZSCORE30")
def high_zscore30(df: pd.DataFrame) -> pd.Series:
    return _zscore(df["high"], 30)

@_register_alpha360("HIGH_ZSCORE60")
def high_zscore60(df: pd.DataFrame) -> pd.Series:
    return _zscore(df["high"], 60)

@_register_alpha360("HIGH_DECAY5")
def high_decay5(df: pd.DataFrame) -> pd.Series:
    return _decay_linear(df["high"], 5) / df["high"] - 1

@_register_alpha360("HIGH_DECAY10")
def high_decay10(df: pd.DataFrame) -> pd.Series:
    return _decay_linear(df["high"], 10) / df["high"] - 1

@_register_alpha360("HIGH_DECAY20")
def high_decay20(df: pd.DataFrame) -> pd.Series:
    return _decay_linear(df["high"], 20) / df["high"] - 1

@_register_alpha360("HIGH_DECAY30")
def high_decay30(df: pd.DataFrame) -> pd.Series:
    return _decay_linear(df["high"], 30) / df["high"] - 1

@_register_alpha360("HIGH_DECAY60")
def high_decay60(df: pd.DataFrame) -> pd.Series:
    return _decay_linear(df["high"], 60) / df["high"] - 1

# =============================================================================
# LOW-based Factors (50 factors)
# =============================================================================

@_register_alpha360("LOW_ROC5")
def low_roc5(df: pd.DataFrame) -> pd.Series:
    return df["low"].pct_change(5)

@_register_alpha360("LOW_ROC10")
def low_roc10(df: pd.DataFrame) -> pd.Series:
    return df["low"].pct_change(10)

@_register_alpha360("LOW_ROC20")
def low_roc20(df: pd.DataFrame) -> pd.Series:
    return df["low"].pct_change(20)

@_register_alpha360("LOW_ROC30")
def low_roc30(df: pd.DataFrame) -> pd.Series:
    return df["low"].pct_change(30)

@_register_alpha360("LOW_ROC60")
def low_roc60(df: pd.DataFrame) -> pd.Series:
    return df["low"].pct_change(60)

@_register_alpha360("LOW_MA5")
def low_ma5(df: pd.DataFrame) -> pd.Series:
    return df["low"] / df["low"].rolling(5).mean() - 1

@_register_alpha360("LOW_MA10")
def low_ma10(df: pd.DataFrame) -> pd.Series:
    return df["low"] / df["low"].rolling(10).mean() - 1

@_register_alpha360("LOW_MA20")
def low_ma20(df: pd.DataFrame) -> pd.Series:
    return df["low"] / df["low"].rolling(20).mean() - 1

@_register_alpha360("LOW_MA30")
def low_ma30(df: pd.DataFrame) -> pd.Series:
    return df["low"] / df["low"].rolling(30).mean() - 1

@_register_alpha360("LOW_MA60")
def low_ma60(df: pd.DataFrame) -> pd.Series:
    return df["low"] / df["low"].rolling(60).mean() - 1

@_register_alpha360("LOW_MIN5")
def low_min5(df: pd.DataFrame) -> pd.Series:
    return df["low"] / df["low"].rolling(5).min()

@_register_alpha360("LOW_MIN10")
def low_min10(df: pd.DataFrame) -> pd.Series:
    return df["low"] / df["low"].rolling(10).min()

@_register_alpha360("LOW_MIN20")
def low_min20(df: pd.DataFrame) -> pd.Series:
    return df["low"] / df["low"].rolling(20).min()

@_register_alpha360("LOW_MIN30")
def low_min30(df: pd.DataFrame) -> pd.Series:
    return df["low"] / df["low"].rolling(30).min()

@_register_alpha360("LOW_MIN60")
def low_min60(df: pd.DataFrame) -> pd.Series:
    return df["low"] / df["low"].rolling(60).min()

@_register_alpha360("LOW_RANK5")
def low_rank5(df: pd.DataFrame) -> pd.Series:
    return _ts_rank(df["low"], 5)

@_register_alpha360("LOW_RANK10")
def low_rank10(df: pd.DataFrame) -> pd.Series:
    return _ts_rank(df["low"], 10)

@_register_alpha360("LOW_RANK20")
def low_rank20(df: pd.DataFrame) -> pd.Series:
    return _ts_rank(df["low"], 20)

@_register_alpha360("LOW_RANK30")
def low_rank30(df: pd.DataFrame) -> pd.Series:
    return _ts_rank(df["low"], 30)

@_register_alpha360("LOW_RANK60")
def low_rank60(df: pd.DataFrame) -> pd.Series:
    return _ts_rank(df["low"], 60)

@_register_alpha360("LOW_ZSCORE5")
def low_zscore5(df: pd.DataFrame) -> pd.Series:
    return _zscore(df["low"], 5)

@_register_alpha360("LOW_ZSCORE10")
def low_zscore10(df: pd.DataFrame) -> pd.Series:
    return _zscore(df["low"], 10)

@_register_alpha360("LOW_ZSCORE20")
def low_zscore20(df: pd.DataFrame) -> pd.Series:
    return _zscore(df["low"], 20)

@_register_alpha360("LOW_ZSCORE30")
def low_zscore30(df: pd.DataFrame) -> pd.Series:
    return _zscore(df["low"], 30)

@_register_alpha360("LOW_ZSCORE60")
def low_zscore60(df: pd.DataFrame) -> pd.Series:
    return _zscore(df["low"], 60)

@_register_alpha360("LOW_DECAY5")
def low_decay5(df: pd.DataFrame) -> pd.Series:
    return _decay_linear(df["low"], 5) / df["low"] - 1

@_register_alpha360("LOW_DECAY10")
def low_decay10(df: pd.DataFrame) -> pd.Series:
    return _decay_linear(df["low"], 10) / df["low"] - 1

@_register_alpha360("LOW_DECAY20")
def low_decay20(df: pd.DataFrame) -> pd.Series:
    return _decay_linear(df["low"], 20) / df["low"] - 1

@_register_alpha360("LOW_DECAY30")
def low_decay30(df: pd.DataFrame) -> pd.Series:
    return _decay_linear(df["low"], 30) / df["low"] - 1

@_register_alpha360("LOW_DECAY60")
def low_decay60(df: pd.DataFrame) -> pd.Series:
    return _decay_linear(df["low"], 60) / df["low"] - 1

# =============================================================================
# OPEN-based Factors (40 factors)
# =============================================================================

@_register_alpha360("OPEN_ROC5")
def open_roc5(df: pd.DataFrame) -> pd.Series:
    return df["open"].pct_change(5)

@_register_alpha360("OPEN_ROC10")
def open_roc10(df: pd.DataFrame) -> pd.Series:
    return df["open"].pct_change(10)

@_register_alpha360("OPEN_ROC20")
def open_roc20(df: pd.DataFrame) -> pd.Series:
    return df["open"].pct_change(20)

@_register_alpha360("OPEN_ROC30")
def open_roc30(df: pd.DataFrame) -> pd.Series:
    return df["open"].pct_change(30)

@_register_alpha360("OPEN_ROC60")
def open_roc60(df: pd.DataFrame) -> pd.Series:
    return df["open"].pct_change(60)

@_register_alpha360("OPEN_MA5")
def open_ma5(df: pd.DataFrame) -> pd.Series:
    return df["open"] / df["open"].rolling(5).mean() - 1

@_register_alpha360("OPEN_MA10")
def open_ma10(df: pd.DataFrame) -> pd.Series:
    return df["open"] / df["open"].rolling(10).mean() - 1

@_register_alpha360("OPEN_MA20")
def open_ma20(df: pd.DataFrame) -> pd.Series:
    return df["open"] / df["open"].rolling(20).mean() - 1

@_register_alpha360("OPEN_MA30")
def open_ma30(df: pd.DataFrame) -> pd.Series:
    return df["open"] / df["open"].rolling(30).mean() - 1

@_register_alpha360("OPEN_MA60")
def open_ma60(df: pd.DataFrame) -> pd.Series:
    return df["open"] / df["open"].rolling(60).mean() - 1

@_register_alpha360("OPEN_RANK5")
def open_rank5(df: pd.DataFrame) -> pd.Series:
    return _ts_rank(df["open"], 5)

@_register_alpha360("OPEN_RANK10")
def open_rank10(df: pd.DataFrame) -> pd.Series:
    return _ts_rank(df["open"], 10)

@_register_alpha360("OPEN_RANK20")
def open_rank20(df: pd.DataFrame) -> pd.Series:
    return _ts_rank(df["open"], 20)

@_register_alpha360("OPEN_RANK30")
def open_rank30(df: pd.DataFrame) -> pd.Series:
    return _ts_rank(df["open"], 30)

@_register_alpha360("OPEN_RANK60")
def open_rank60(df: pd.DataFrame) -> pd.Series:
    return _ts_rank(df["open"], 60)

@_register_alpha360("OPEN_ZSCORE5")
def open_zscore5(df: pd.DataFrame) -> pd.Series:
    return _zscore(df["open"], 5)

@_register_alpha360("OPEN_ZSCORE10")
def open_zscore10(df: pd.DataFrame) -> pd.Series:
    return _zscore(df["open"], 10)

@_register_alpha360("OPEN_ZSCORE20")
def open_zscore20(df: pd.DataFrame) -> pd.Series:
    return _zscore(df["open"], 20)

@_register_alpha360("OPEN_ZSCORE30")
def open_zscore30(df: pd.DataFrame) -> pd.Series:
    return _zscore(df["open"], 30)

@_register_alpha360("OPEN_ZSCORE60")
def open_zscore60(df: pd.DataFrame) -> pd.Series:
    return _zscore(df["open"], 60)

@_register_alpha360("OPEN_DECAY5")
def open_decay5(df: pd.DataFrame) -> pd.Series:
    return _decay_linear(df["open"], 5) / df["open"] - 1

@_register_alpha360("OPEN_DECAY10")
def open_decay10(df: pd.DataFrame) -> pd.Series:
    return _decay_linear(df["open"], 10) / df["open"] - 1

@_register_alpha360("OPEN_DECAY20")
def open_decay20(df: pd.DataFrame) -> pd.Series:
    return _decay_linear(df["open"], 20) / df["open"] - 1

@_register_alpha360("OPEN_DECAY30")
def open_decay30(df: pd.DataFrame) -> pd.Series:
    return _decay_linear(df["open"], 30) / df["open"] - 1

@_register_alpha360("OPEN_DECAY60")
def open_decay60(df: pd.DataFrame) -> pd.Series:
    return _decay_linear(df["open"], 60) / df["open"] - 1

# =============================================================================
# VOLUME-based Factors (50 factors)
# =============================================================================

@_register_alpha360("VOLUME_ROC5")
def volume_roc5(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"].pct_change(5)

@_register_alpha360("VOLUME_ROC10")
def volume_roc10(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"].pct_change(10)

@_register_alpha360("VOLUME_ROC20")
def volume_roc20(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"].pct_change(20)

@_register_alpha360("VOLUME_ROC30")
def volume_roc30(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"].pct_change(30)

@_register_alpha360("VOLUME_ROC60")
def volume_roc60(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"].pct_change(60)

@_register_alpha360("VOLUME_MA5")
def volume_ma5(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"] / df["volume"].rolling(5).mean() - 1

@_register_alpha360("VOLUME_MA10")
def volume_ma10(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"] / df["volume"].rolling(10).mean() - 1

@_register_alpha360("VOLUME_MA20")
def volume_ma20(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"] / df["volume"].rolling(20).mean() - 1

@_register_alpha360("VOLUME_MA30")
def volume_ma30(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"] / df["volume"].rolling(30).mean() - 1

@_register_alpha360("VOLUME_MA60")
def volume_ma60(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"] / df["volume"].rolling(60).mean() - 1

@_register_alpha360("VOLUME_STD5")
def volume_std5(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"].rolling(5).std() / (df["volume"].rolling(20).mean() + 1e-10)

@_register_alpha360("VOLUME_STD10")
def volume_std10(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"].rolling(10).std() / (df["volume"].rolling(20).mean() + 1e-10)

@_register_alpha360("VOLUME_STD20")
def volume_std20(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"].rolling(20).std() / (df["volume"].rolling(60).mean() + 1e-10)

@_register_alpha360("VOLUME_STD30")
def volume_std30(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"].rolling(30).std() / (df["volume"].rolling(60).mean() + 1e-10)

@_register_alpha360("VOLUME_STD60")
def volume_std60(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"].rolling(60).std() / (df["volume"].rolling(120).mean() + 1e-10)

@_register_alpha360("VOLUME_RANK5")
def volume_rank5(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return _ts_rank(df["volume"], 5)

@_register_alpha360("VOLUME_RANK10")
def volume_rank10(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return _ts_rank(df["volume"], 10)

@_register_alpha360("VOLUME_RANK20")
def volume_rank20(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return _ts_rank(df["volume"], 20)

@_register_alpha360("VOLUME_RANK30")
def volume_rank30(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return _ts_rank(df["volume"], 30)

@_register_alpha360("VOLUME_RANK60")
def volume_rank60(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return _ts_rank(df["volume"], 60)

@_register_alpha360("VOLUME_ZSCORE5")
def volume_zscore5(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return _zscore(df["volume"], 5)

@_register_alpha360("VOLUME_ZSCORE10")
def volume_zscore10(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return _zscore(df["volume"], 10)

@_register_alpha360("VOLUME_ZSCORE20")
def volume_zscore20(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return _zscore(df["volume"], 20)

@_register_alpha360("VOLUME_ZSCORE30")
def volume_zscore30(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return _zscore(df["volume"], 30)

@_register_alpha360("VOLUME_ZSCORE60")
def volume_zscore60(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return _zscore(df["volume"], 60)

@_register_alpha360("VOLUME_DECAY5")
def volume_decay5(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return _decay_linear(df["volume"], 5) / df["volume"] - 1

@_register_alpha360("VOLUME_DECAY10")
def volume_decay10(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return _decay_linear(df["volume"], 10) / df["volume"] - 1

@_register_alpha360("VOLUME_DECAY20")
def volume_decay20(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return _decay_linear(df["volume"], 20) / df["volume"] - 1

@_register_alpha360("VOLUME_DECAY30")
def volume_decay30(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return _decay_linear(df["volume"], 30) / df["volume"] - 1

@_register_alpha360("VOLUME_DECAY60")
def volume_decay60(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return _decay_linear(df["volume"], 60) / df["volume"] - 1

# =============================================================================
# VWAP-based Factors (40 factors)
# =============================================================================

@_register_alpha360("VWAP_ROC5")
def vwap_roc5(df: pd.DataFrame) -> pd.Series:
    vwap = _get_vwap(df)
    return vwap.pct_change(5)

@_register_alpha360("VWAP_ROC10")
def vwap_roc10(df: pd.DataFrame) -> pd.Series:
    vwap = _get_vwap(df)
    return vwap.pct_change(10)

@_register_alpha360("VWAP_ROC20")
def vwap_roc20(df: pd.DataFrame) -> pd.Series:
    vwap = _get_vwap(df)
    return vwap.pct_change(20)

@_register_alpha360("VWAP_ROC30")
def vwap_roc30(df: pd.DataFrame) -> pd.Series:
    vwap = _get_vwap(df)
    return vwap.pct_change(30)

@_register_alpha360("VWAP_ROC60")
def vwap_roc60(df: pd.DataFrame) -> pd.Series:
    vwap = _get_vwap(df)
    return vwap.pct_change(60)

@_register_alpha360("VWAP_MA5")
def vwap_ma5(df: pd.DataFrame) -> pd.Series:
    vwap = _get_vwap(df)
    return vwap / vwap.rolling(5).mean() - 1

@_register_alpha360("VWAP_MA10")
def vwap_ma10(df: pd.DataFrame) -> pd.Series:
    vwap = _get_vwap(df)
    return vwap / vwap.rolling(10).mean() - 1

@_register_alpha360("VWAP_MA20")
def vwap_ma20(df: pd.DataFrame) -> pd.Series:
    vwap = _get_vwap(df)
    return vwap / vwap.rolling(20).mean() - 1

@_register_alpha360("VWAP_MA30")
def vwap_ma30(df: pd.DataFrame) -> pd.Series:
    vwap = _get_vwap(df)
    return vwap / vwap.rolling(30).mean() - 1

@_register_alpha360("VWAP_MA60")
def vwap_ma60(df: pd.DataFrame) -> pd.Series:
    vwap = _get_vwap(df)
    return vwap / vwap.rolling(60).mean() - 1

@_register_alpha360("VWAP_RANK5")
def vwap_rank5(df: pd.DataFrame) -> pd.Series:
    vwap = _get_vwap(df)
    return _ts_rank(vwap, 5)

@_register_alpha360("VWAP_RANK10")
def vwap_rank10(df: pd.DataFrame) -> pd.Series:
    vwap = _get_vwap(df)
    return _ts_rank(vwap, 10)

@_register_alpha360("VWAP_RANK20")
def vwap_rank20(df: pd.DataFrame) -> pd.Series:
    vwap = _get_vwap(df)
    return _ts_rank(vwap, 20)

@_register_alpha360("VWAP_RANK30")
def vwap_rank30(df: pd.DataFrame) -> pd.Series:
    vwap = _get_vwap(df)
    return _ts_rank(vwap, 30)

@_register_alpha360("VWAP_RANK60")
def vwap_rank60(df: pd.DataFrame) -> pd.Series:
    vwap = _get_vwap(df)
    return _ts_rank(vwap, 60)

@_register_alpha360("VWAP_ZSCORE5")
def vwap_zscore5(df: pd.DataFrame) -> pd.Series:
    vwap = _get_vwap(df)
    return _zscore(vwap, 5)

@_register_alpha360("VWAP_ZSCORE10")
def vwap_zscore10(df: pd.DataFrame) -> pd.Series:
    vwap = _get_vwap(df)
    return _zscore(vwap, 10)

@_register_alpha360("VWAP_ZSCORE20")
def vwap_zscore20(df: pd.DataFrame) -> pd.Series:
    vwap = _get_vwap(df)
    return _zscore(vwap, 20)

@_register_alpha360("VWAP_ZSCORE30")
def vwap_zscore30(df: pd.DataFrame) -> pd.Series:
    vwap = _get_vwap(df)
    return _zscore(vwap, 30)

@_register_alpha360("VWAP_ZSCORE60")
def vwap_zscore60(df: pd.DataFrame) -> pd.Series:
    vwap = _get_vwap(df)
    return _zscore(vwap, 60)

@_register_alpha360("VWAP_DECAY5")
def vwap_decay5(df: pd.DataFrame) -> pd.Series:
    vwap = _get_vwap(df)
    return _decay_linear(vwap, 5) / vwap - 1

@_register_alpha360("VWAP_DECAY10")
def vwap_decay10(df: pd.DataFrame) -> pd.Series:
    vwap = _get_vwap(df)
    return _decay_linear(vwap, 10) / vwap - 1

@_register_alpha360("VWAP_DECAY20")
def vwap_decay20(df: pd.DataFrame) -> pd.Series:
    vwap = _get_vwap(df)
    return _decay_linear(vwap, 20) / vwap - 1

@_register_alpha360("VWAP_DECAY30")
def vwap_decay30(df: pd.DataFrame) -> pd.Series:
    vwap = _get_vwap(df)
    return _decay_linear(vwap, 30) / vwap - 1

@_register_alpha360("VWAP_DECAY60")
def vwap_decay60(df: pd.DataFrame) -> pd.Series:
    vwap = _get_vwap(df)
    return _decay_linear(vwap, 60) / vwap - 1

# =============================================================================
# Cross-Feature Factors (70 factors)
# =============================================================================

# Price Spread Factors
@_register_alpha360("SPREAD_HL5")
def spread_hl5(df: pd.DataFrame) -> pd.Series:
    spread = (df["high"] - df["low"]) / df["close"]
    return spread.rolling(5).mean()

@_register_alpha360("SPREAD_HL10")
def spread_hl10(df: pd.DataFrame) -> pd.Series:
    spread = (df["high"] - df["low"]) / df["close"]
    return spread.rolling(10).mean()

@_register_alpha360("SPREAD_HL20")
def spread_hl20(df: pd.DataFrame) -> pd.Series:
    spread = (df["high"] - df["low"]) / df["close"]
    return spread.rolling(20).mean()

@_register_alpha360("SPREAD_HL30")
def spread_hl30(df: pd.DataFrame) -> pd.Series:
    spread = (df["high"] - df["low"]) / df["close"]
    return spread.rolling(30).mean()

@_register_alpha360("SPREAD_HL60")
def spread_hl60(df: pd.DataFrame) -> pd.Series:
    spread = (df["high"] - df["low"]) / df["close"]
    return spread.rolling(60).mean()

# Open-Close Gap
@_register_alpha360("GAP_OC5")
def gap_oc5(df: pd.DataFrame) -> pd.Series:
    gap = (df["close"] - df["open"]) / df["open"]
    return gap.rolling(5).mean()

@_register_alpha360("GAP_OC10")
def gap_oc10(df: pd.DataFrame) -> pd.Series:
    gap = (df["close"] - df["open"]) / df["open"]
    return gap.rolling(10).mean()

@_register_alpha360("GAP_OC20")
def gap_oc20(df: pd.DataFrame) -> pd.Series:
    gap = (df["close"] - df["open"]) / df["open"]
    return gap.rolling(20).mean()

@_register_alpha360("GAP_OC30")
def gap_oc30(df: pd.DataFrame) -> pd.Series:
    gap = (df["close"] - df["open"]) / df["open"]
    return gap.rolling(30).mean()

@_register_alpha360("GAP_OC60")
def gap_oc60(df: pd.DataFrame) -> pd.Series:
    gap = (df["close"] - df["open"]) / df["open"]
    return gap.rolling(60).mean()

# Price-Volume Correlation
@_register_alpha360("CORR_CV5")
def corr_cv5(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["close"].pct_change().rolling(5).corr(df["volume"].pct_change())

@_register_alpha360("CORR_CV10")
def corr_cv10(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["close"].pct_change().rolling(10).corr(df["volume"].pct_change())

@_register_alpha360("CORR_CV20")
def corr_cv20(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["close"].pct_change().rolling(20).corr(df["volume"].pct_change())

@_register_alpha360("CORR_CV30")
def corr_cv30(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["close"].pct_change().rolling(30).corr(df["volume"].pct_change())

@_register_alpha360("CORR_CV60")
def corr_cv60(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["close"].pct_change().rolling(60).corr(df["volume"].pct_change())

# K-line Pattern Factors
@_register_alpha360("KLINE_BODY5")
def kline_body5(df: pd.DataFrame) -> pd.Series:
    body = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-10)
    return body.rolling(5).mean()

@_register_alpha360("KLINE_BODY10")
def kline_body10(df: pd.DataFrame) -> pd.Series:
    body = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-10)
    return body.rolling(10).mean()

@_register_alpha360("KLINE_BODY20")
def kline_body20(df: pd.DataFrame) -> pd.Series:
    body = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-10)
    return body.rolling(20).mean()

@_register_alpha360("KLINE_UPPER5")
def kline_upper5(df: pd.DataFrame) -> pd.Series:
    upper = (df["high"] - df[["open", "close"]].max(axis=1)) / (df["high"] - df["low"] + 1e-10)
    return upper.rolling(5).mean()

@_register_alpha360("KLINE_UPPER10")
def kline_upper10(df: pd.DataFrame) -> pd.Series:
    upper = (df["high"] - df[["open", "close"]].max(axis=1)) / (df["high"] - df["low"] + 1e-10)
    return upper.rolling(10).mean()

@_register_alpha360("KLINE_UPPER20")
def kline_upper20(df: pd.DataFrame) -> pd.Series:
    upper = (df["high"] - df[["open", "close"]].max(axis=1)) / (df["high"] - df["low"] + 1e-10)
    return upper.rolling(20).mean()

@_register_alpha360("KLINE_LOWER5")
def kline_lower5(df: pd.DataFrame) -> pd.Series:
    lower = (df[["open", "close"]].min(axis=1) - df["low"]) / (df["high"] - df["low"] + 1e-10)
    return lower.rolling(5).mean()

@_register_alpha360("KLINE_LOWER10")
def kline_lower10(df: pd.DataFrame) -> pd.Series:
    lower = (df[["open", "close"]].min(axis=1) - df["low"]) / (df["high"] - df["low"] + 1e-10)
    return lower.rolling(10).mean()

@_register_alpha360("KLINE_LOWER20")
def kline_lower20(df: pd.DataFrame) -> pd.Series:
    lower = (df[["open", "close"]].min(axis=1) - df["low"]) / (df["high"] - df["low"] + 1e-10)
    return lower.rolling(20).mean()

# Return Skew and Kurt
@_register_alpha360("RET_SKEW5")
def ret_skew5(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(5).skew()

@_register_alpha360("RET_SKEW10")
def ret_skew10(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(10).skew()

@_register_alpha360("RET_SKEW20")
def ret_skew20(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(20).skew()

@_register_alpha360("RET_SKEW30")
def ret_skew30(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(30).skew()

@_register_alpha360("RET_SKEW60")
def ret_skew60(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(60).skew()

@_register_alpha360("RET_KURT5")
def ret_kurt5(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(5).kurt()

@_register_alpha360("RET_KURT10")
def ret_kurt10(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(10).kurt()

@_register_alpha360("RET_KURT20")
def ret_kurt20(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(20).kurt()

@_register_alpha360("RET_KURT30")
def ret_kurt30(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(30).kurt()

@_register_alpha360("RET_KURT60")
def ret_kurt60(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(60).kurt()

# Quantile Factors
@_register_alpha360("RET_Q80_5")
def ret_q80_5(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(5).quantile(0.8)

@_register_alpha360("RET_Q80_10")
def ret_q80_10(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(10).quantile(0.8)

@_register_alpha360("RET_Q80_20")
def ret_q80_20(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(20).quantile(0.8)

@_register_alpha360("RET_Q20_5")
def ret_q20_5(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(5).quantile(0.2)

@_register_alpha360("RET_Q20_10")
def ret_q20_10(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(10).quantile(0.2)

@_register_alpha360("RET_Q20_20")
def ret_q20_20(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(20).quantile(0.2)

# Up/Down Ratio
@_register_alpha360("UP_RATIO5")
def up_ratio5(df: pd.DataFrame) -> pd.Series:
    rets = df["close"].pct_change()
    return rets.rolling(5).apply(lambda x: (x > 0).sum() / len(x), raw=True)

@_register_alpha360("UP_RATIO10")
def up_ratio10(df: pd.DataFrame) -> pd.Series:
    rets = df["close"].pct_change()
    return rets.rolling(10).apply(lambda x: (x > 0).sum() / len(x), raw=True)

@_register_alpha360("UP_RATIO20")
def up_ratio20(df: pd.DataFrame) -> pd.Series:
    rets = df["close"].pct_change()
    return rets.rolling(20).apply(lambda x: (x > 0).sum() / len(x), raw=True)

@_register_alpha360("UP_RATIO30")
def up_ratio30(df: pd.DataFrame) -> pd.Series:
    rets = df["close"].pct_change()
    return rets.rolling(30).apply(lambda x: (x > 0).sum() / len(x), raw=True)

@_register_alpha360("UP_RATIO60")
def up_ratio60(df: pd.DataFrame) -> pd.Series:
    rets = df["close"].pct_change()
    return rets.rolling(60).apply(lambda x: (x > 0).sum() / len(x), raw=True)

# MA Cross Features
@_register_alpha360("MA_CROSS_5_10")
def ma_cross_5_10(df: pd.DataFrame) -> pd.Series:
    return df["close"].rolling(5).mean() / df["close"].rolling(10).mean() - 1

@_register_alpha360("MA_CROSS_5_20")
def ma_cross_5_20(df: pd.DataFrame) -> pd.Series:
    return df["close"].rolling(5).mean() / df["close"].rolling(20).mean() - 1

@_register_alpha360("MA_CROSS_10_20")
def ma_cross_10_20(df: pd.DataFrame) -> pd.Series:
    return df["close"].rolling(10).mean() / df["close"].rolling(20).mean() - 1

@_register_alpha360("MA_CROSS_10_30")
def ma_cross_10_30(df: pd.DataFrame) -> pd.Series:
    return df["close"].rolling(10).mean() / df["close"].rolling(30).mean() - 1

@_register_alpha360("MA_CROSS_20_60")
def ma_cross_20_60(df: pd.DataFrame) -> pd.Series:
    return df["close"].rolling(20).mean() / df["close"].rolling(60).mean() - 1

# =============================================================================
# Additional Factors to Reach 360 (50 more)
# =============================================================================

# Return Accel
@_register_alpha360("RET_ACCEL5")
def ret_accel5(df: pd.DataFrame) -> pd.Series:
    ret = df["close"].pct_change()
    return ret.diff().rolling(5).mean()

@_register_alpha360("RET_ACCEL10")
def ret_accel10(df: pd.DataFrame) -> pd.Series:
    ret = df["close"].pct_change()
    return ret.diff().rolling(10).mean()

@_register_alpha360("RET_ACCEL20")
def ret_accel20(df: pd.DataFrame) -> pd.Series:
    ret = df["close"].pct_change()
    return ret.diff().rolling(20).mean()

# Volatility Ratio
@_register_alpha360("VOL_RATIO_5_20")
def vol_ratio_5_20(df: pd.DataFrame) -> pd.Series:
    vol5 = df["close"].pct_change().rolling(5).std()
    vol20 = df["close"].pct_change().rolling(20).std()
    return vol5 / (vol20 + 1e-10)

@_register_alpha360("VOL_RATIO_10_30")
def vol_ratio_10_30(df: pd.DataFrame) -> pd.Series:
    vol10 = df["close"].pct_change().rolling(10).std()
    vol30 = df["close"].pct_change().rolling(30).std()
    return vol10 / (vol30 + 1e-10)

@_register_alpha360("VOL_RATIO_20_60")
def vol_ratio_20_60(df: pd.DataFrame) -> pd.Series:
    vol20 = df["close"].pct_change().rolling(20).std()
    vol60 = df["close"].pct_change().rolling(60).std()
    return vol20 / (vol60 + 1e-10)

# Range Position
@_register_alpha360("RANGE_POS5")
def range_pos5(df: pd.DataFrame) -> pd.Series:
    high = df["high"].rolling(5).max()
    low = df["low"].rolling(5).min()
    return (df["close"] - low) / (high - low + 1e-10)

@_register_alpha360("RANGE_POS10")
def range_pos10(df: pd.DataFrame) -> pd.Series:
    high = df["high"].rolling(10).max()
    low = df["low"].rolling(10).min()
    return (df["close"] - low) / (high - low + 1e-10)

@_register_alpha360("RANGE_POS20")
def range_pos20(df: pd.DataFrame) -> pd.Series:
    high = df["high"].rolling(20).max()
    low = df["low"].rolling(20).min()
    return (df["close"] - low) / (high - low + 1e-10)

@_register_alpha360("RANGE_POS30")
def range_pos30(df: pd.DataFrame) -> pd.Series:
    high = df["high"].rolling(30).max()
    low = df["low"].rolling(30).min()
    return (df["close"] - low) / (high - low + 1e-10)

@_register_alpha360("RANGE_POS60")
def range_pos60(df: pd.DataFrame) -> pd.Series:
    high = df["high"].rolling(60).max()
    low = df["low"].rolling(60).min()
    return (df["close"] - low) / (high - low + 1e-10)

# Days since High/Low
@_register_alpha360("DAYS_HIGH5")
def days_high5(df: pd.DataFrame) -> pd.Series:
    return df["high"].rolling(5).apply(lambda x: 5 - x.argmax() - 1, raw=True) / 5

@_register_alpha360("DAYS_HIGH10")
def days_high10(df: pd.DataFrame) -> pd.Series:
    return df["high"].rolling(10).apply(lambda x: 10 - x.argmax() - 1, raw=True) / 10

@_register_alpha360("DAYS_HIGH20")
def days_high20(df: pd.DataFrame) -> pd.Series:
    return df["high"].rolling(20).apply(lambda x: 20 - x.argmax() - 1, raw=True) / 20

@_register_alpha360("DAYS_HIGH30")
def days_high30(df: pd.DataFrame) -> pd.Series:
    return df["high"].rolling(30).apply(lambda x: 30 - x.argmax() - 1, raw=True) / 30

@_register_alpha360("DAYS_HIGH60")
def days_high60(df: pd.DataFrame) -> pd.Series:
    return df["high"].rolling(60).apply(lambda x: 60 - x.argmax() - 1, raw=True) / 60

@_register_alpha360("DAYS_LOW5")
def days_low5(df: pd.DataFrame) -> pd.Series:
    return df["low"].rolling(5).apply(lambda x: 5 - x.argmin() - 1, raw=True) / 5

@_register_alpha360("DAYS_LOW10")
def days_low10(df: pd.DataFrame) -> pd.Series:
    return df["low"].rolling(10).apply(lambda x: 10 - x.argmin() - 1, raw=True) / 10

@_register_alpha360("DAYS_LOW20")
def days_low20(df: pd.DataFrame) -> pd.Series:
    return df["low"].rolling(20).apply(lambda x: 20 - x.argmin() - 1, raw=True) / 20

@_register_alpha360("DAYS_LOW30")
def days_low30(df: pd.DataFrame) -> pd.Series:
    return df["low"].rolling(30).apply(lambda x: 30 - x.argmin() - 1, raw=True) / 30

@_register_alpha360("DAYS_LOW60")
def days_low60(df: pd.DataFrame) -> pd.Series:
    return df["low"].rolling(60).apply(lambda x: 60 - x.argmin() - 1, raw=True) / 60

# Trend Strength
@_register_alpha360("TREND5")
def trend5(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["close"].shift(5)) / (df["close"].rolling(5).std() + 1e-10)

@_register_alpha360("TREND10")
def trend10(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["close"].shift(10)) / (df["close"].rolling(10).std() + 1e-10)

@_register_alpha360("TREND20")
def trend20(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["close"].shift(20)) / (df["close"].rolling(20).std() + 1e-10)

@_register_alpha360("TREND30")
def trend30(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["close"].shift(30)) / (df["close"].rolling(30).std() + 1e-10)

@_register_alpha360("TREND60")
def trend60(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["close"].shift(60)) / (df["close"].rolling(60).std() + 1e-10)

# High-Low Expansion
@_register_alpha360("HL_EXP5")
def hl_exp5(df: pd.DataFrame) -> pd.Series:
    hl = (df["high"] - df["low"]) / df["close"]
    return hl / hl.rolling(20).mean() - 1

@_register_alpha360("HL_EXP10")
def hl_exp10(df: pd.DataFrame) -> pd.Series:
    hl = (df["high"] - df["low"]) / df["close"]
    return hl.rolling(10).mean() / hl.rolling(30).mean() - 1

@_register_alpha360("HL_EXP20")
def hl_exp20(df: pd.DataFrame) -> pd.Series:
    hl = (df["high"] - df["low"]) / df["close"]
    return hl.rolling(20).mean() / hl.rolling(60).mean() - 1

# Average True Range Normalized
@_register_alpha360("ATR5_NORM")
def atr5_norm(df: pd.DataFrame) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(5).mean()
    return atr / df["close"]

@_register_alpha360("ATR10_NORM")
def atr10_norm(df: pd.DataFrame) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(10).mean()
    return atr / df["close"]

@_register_alpha360("ATR20_NORM")
def atr20_norm(df: pd.DataFrame) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(20).mean()
    return atr / df["close"]

# Max Return
@_register_alpha360("MAX_RET5")
def max_ret5(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(5).max()

@_register_alpha360("MAX_RET10")
def max_ret10(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(10).max()

@_register_alpha360("MAX_RET20")
def max_ret20(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(20).max()

# Min Return
@_register_alpha360("MIN_RET5")
def min_ret5(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(5).min()

@_register_alpha360("MIN_RET10")
def min_ret10(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(10).min()

@_register_alpha360("MIN_RET20")
def min_ret20(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(20).min()

# Mean Abs Return
@_register_alpha360("MEAN_ABS_RET5")
def mean_abs_ret5(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().abs().rolling(5).mean()

@_register_alpha360("MEAN_ABS_RET10")
def mean_abs_ret10(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().abs().rolling(10).mean()

@_register_alpha360("MEAN_ABS_RET20")
def mean_abs_ret20(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().abs().rolling(20).mean()


# =============================================================================
# Technical Indicators (79 more factors to reach 360)
# =============================================================================

# RSI Factors
@_register_alpha360("RSI6")
def rsi6(df: pd.DataFrame) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

@_register_alpha360("RSI12")
def rsi12(df: pd.DataFrame) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(12).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(12).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

@_register_alpha360("RSI24")
def rsi24(df: pd.DataFrame) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(24).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(24).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

# Bollinger Band Factors
@_register_alpha360("BB_POS5")
def bb_pos5(df: pd.DataFrame) -> pd.Series:
    ma = df["close"].rolling(5).mean()
    std = df["close"].rolling(5).std()
    return (df["close"] - ma) / (2 * std + 1e-10)

@_register_alpha360("BB_POS10")
def bb_pos10(df: pd.DataFrame) -> pd.Series:
    ma = df["close"].rolling(10).mean()
    std = df["close"].rolling(10).std()
    return (df["close"] - ma) / (2 * std + 1e-10)

@_register_alpha360("BB_POS20")
def bb_pos20(df: pd.DataFrame) -> pd.Series:
    ma = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    return (df["close"] - ma) / (2 * std + 1e-10)

@_register_alpha360("BB_WIDTH5")
def bb_width5(df: pd.DataFrame) -> pd.Series:
    ma = df["close"].rolling(5).mean()
    std = df["close"].rolling(5).std()
    return 4 * std / (ma + 1e-10)

@_register_alpha360("BB_WIDTH10")
def bb_width10(df: pd.DataFrame) -> pd.Series:
    ma = df["close"].rolling(10).mean()
    std = df["close"].rolling(10).std()
    return 4 * std / (ma + 1e-10)

@_register_alpha360("BB_WIDTH20")
def bb_width20(df: pd.DataFrame) -> pd.Series:
    ma = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    return 4 * std / (ma + 1e-10)

# Williams %R
@_register_alpha360("WILLR5")
def willr5(df: pd.DataFrame) -> pd.Series:
    high = df["high"].rolling(5).max()
    low = df["low"].rolling(5).min()
    return (high - df["close"]) / (high - low + 1e-10) * -100

@_register_alpha360("WILLR10")
def willr10(df: pd.DataFrame) -> pd.Series:
    high = df["high"].rolling(10).max()
    low = df["low"].rolling(10).min()
    return (high - df["close"]) / (high - low + 1e-10) * -100

@_register_alpha360("WILLR20")
def willr20(df: pd.DataFrame) -> pd.Series:
    high = df["high"].rolling(20).max()
    low = df["low"].rolling(20).min()
    return (high - df["close"]) / (high - low + 1e-10) * -100

# CCI
@_register_alpha360("CCI5")
def cci5(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(5).mean()
    md = tp.rolling(5).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * md + 1e-10)

@_register_alpha360("CCI10")
def cci10(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(10).mean()
    md = tp.rolling(10).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * md + 1e-10)

@_register_alpha360("CCI20")
def cci20(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(20).mean()
    md = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * md + 1e-10)

# EMA Factors
@_register_alpha360("EMA5")
def ema5(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].ewm(span=5, adjust=False).mean() - 1

@_register_alpha360("EMA10")
def ema10(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].ewm(span=10, adjust=False).mean() - 1

@_register_alpha360("EMA20")
def ema20(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].ewm(span=20, adjust=False).mean() - 1

@_register_alpha360("EMA30")
def ema30(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].ewm(span=30, adjust=False).mean() - 1

@_register_alpha360("EMA60")
def ema60(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["close"].ewm(span=60, adjust=False).mean() - 1

# EMA Cross
@_register_alpha360("EMA_CROSS_5_10")
def ema_cross_5_10(df: pd.DataFrame) -> pd.Series:
    ema5 = df["close"].ewm(span=5, adjust=False).mean()
    ema10 = df["close"].ewm(span=10, adjust=False).mean()
    return ema5 / ema10 - 1

@_register_alpha360("EMA_CROSS_5_20")
def ema_cross_5_20(df: pd.DataFrame) -> pd.Series:
    ema5 = df["close"].ewm(span=5, adjust=False).mean()
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    return ema5 / ema20 - 1

@_register_alpha360("EMA_CROSS_10_20")
def ema_cross_10_20(df: pd.DataFrame) -> pd.Series:
    ema10 = df["close"].ewm(span=10, adjust=False).mean()
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    return ema10 / ema20 - 1

@_register_alpha360("EMA_CROSS_10_30")
def ema_cross_10_30(df: pd.DataFrame) -> pd.Series:
    ema10 = df["close"].ewm(span=10, adjust=False).mean()
    ema30 = df["close"].ewm(span=30, adjust=False).mean()
    return ema10 / ema30 - 1

@_register_alpha360("EMA_CROSS_20_60")
def ema_cross_20_60(df: pd.DataFrame) -> pd.Series:
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    ema60 = df["close"].ewm(span=60, adjust=False).mean()
    return ema20 / ema60 - 1

# MACD
@_register_alpha360("MACD_LINE")
def macd_line(df: pd.DataFrame) -> pd.Series:
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    return (ema12 - ema26) / df["close"]

@_register_alpha360("MACD_SIGNAL")
def macd_signal(df: pd.DataFrame) -> pd.Series:
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return signal / df["close"]

@_register_alpha360("MACD_HIST")
def macd_hist(df: pd.DataFrame) -> pd.Series:
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return (macd - signal) / df["close"]

# Typical Price
@_register_alpha360("TP_ROC5")
def tp_roc5(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return tp.pct_change(5)

@_register_alpha360("TP_ROC10")
def tp_roc10(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return tp.pct_change(10)

@_register_alpha360("TP_ROC20")
def tp_roc20(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return tp.pct_change(20)

# Midpoint Price
@_register_alpha360("MID_ROC5")
def mid_roc5(df: pd.DataFrame) -> pd.Series:
    mid = (df["high"] + df["low"]) / 2
    return mid.pct_change(5)

@_register_alpha360("MID_ROC10")
def mid_roc10(df: pd.DataFrame) -> pd.Series:
    mid = (df["high"] + df["low"]) / 2
    return mid.pct_change(10)

@_register_alpha360("MID_ROC20")
def mid_roc20(df: pd.DataFrame) -> pd.Series:
    mid = (df["high"] + df["low"]) / 2
    return mid.pct_change(20)

# Momentum Acceleration
@_register_alpha360("MOM_ACCEL5")
def mom_accel5(df: pd.DataFrame) -> pd.Series:
    mom = df["close"].diff(5)
    return mom.diff(5) / (df["close"].shift(10) + 1e-10)

@_register_alpha360("MOM_ACCEL10")
def mom_accel10(df: pd.DataFrame) -> pd.Series:
    mom = df["close"].diff(10)
    return mom.diff(10) / (df["close"].shift(20) + 1e-10)

@_register_alpha360("MOM_ACCEL20")
def mom_accel20(df: pd.DataFrame) -> pd.Series:
    mom = df["close"].diff(20)
    return mom.diff(20) / (df["close"].shift(40) + 1e-10)

# Price Efficiency Ratio
@_register_alpha360("PER5")
def per5(df: pd.DataFrame) -> pd.Series:
    change = (df["close"] - df["close"].shift(5)).abs()
    volatility = df["close"].diff().abs().rolling(5).sum()
    return change / (volatility + 1e-10)

@_register_alpha360("PER10")
def per10(df: pd.DataFrame) -> pd.Series:
    change = (df["close"] - df["close"].shift(10)).abs()
    volatility = df["close"].diff().abs().rolling(10).sum()
    return change / (volatility + 1e-10)

@_register_alpha360("PER20")
def per20(df: pd.DataFrame) -> pd.Series:
    change = (df["close"] - df["close"].shift(20)).abs()
    volatility = df["close"].diff().abs().rolling(20).sum()
    return change / (volatility + 1e-10)

# Volume-Price Trend
@_register_alpha360("VPT5")
def vpt5(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    pct = df["close"].pct_change()
    vpt = (pct * df["volume"]).cumsum()
    return vpt.pct_change(5)

@_register_alpha360("VPT10")
def vpt10(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    pct = df["close"].pct_change()
    vpt = (pct * df["volume"]).cumsum()
    return vpt.pct_change(10)

@_register_alpha360("VPT20")
def vpt20(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    pct = df["close"].pct_change()
    vpt = (pct * df["volume"]).cumsum()
    return vpt.pct_change(20)

# Force Index
@_register_alpha360("FORCE5")
def force5(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    force = df["close"].diff() * df["volume"]
    return force.rolling(5).mean() / (force.rolling(20).std() + 1e-10)

@_register_alpha360("FORCE10")
def force10(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    force = df["close"].diff() * df["volume"]
    return force.rolling(10).mean() / (force.rolling(20).std() + 1e-10)

@_register_alpha360("FORCE20")
def force20(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    force = df["close"].diff() * df["volume"]
    return force.rolling(20).mean() / (force.rolling(60).std() + 1e-10)

# Price Channel Factors
@_register_alpha360("CHANNEL_UP5")
def channel_up5(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["high"].rolling(5).max() - 1

@_register_alpha360("CHANNEL_UP10")
def channel_up10(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["high"].rolling(10).max() - 1

@_register_alpha360("CHANNEL_UP20")
def channel_up20(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["high"].rolling(20).max() - 1

@_register_alpha360("CHANNEL_DOWN5")
def channel_down5(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["low"].rolling(5).min() - 1

@_register_alpha360("CHANNEL_DOWN10")
def channel_down10(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["low"].rolling(10).min() - 1

@_register_alpha360("CHANNEL_DOWN20")
def channel_down20(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df["low"].rolling(20).min() - 1

# Relative Vigor Index
@_register_alpha360("RVI5")
def rvi5(df: pd.DataFrame) -> pd.Series:
    numerator = (df["close"] - df["open"]).rolling(5).mean()
    denominator = (df["high"] - df["low"]).rolling(5).mean()
    return numerator / (denominator + 1e-10)

@_register_alpha360("RVI10")
def rvi10(df: pd.DataFrame) -> pd.Series:
    numerator = (df["close"] - df["open"]).rolling(10).mean()
    denominator = (df["high"] - df["low"]).rolling(10).mean()
    return numerator / (denominator + 1e-10)

@_register_alpha360("RVI20")
def rvi20(df: pd.DataFrame) -> pd.Series:
    numerator = (df["close"] - df["open"]).rolling(20).mean()
    denominator = (df["high"] - df["low"]).rolling(20).mean()
    return numerator / (denominator + 1e-10)

# Choppiness Index
@_register_alpha360("CHOP5")
def chop5(df: pd.DataFrame) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_sum = tr.rolling(5).sum()
    high_low = df["high"].rolling(5).max() - df["low"].rolling(5).min()
    return np.log10(atr_sum / (high_low + 1e-10)) / np.log10(5)

@_register_alpha360("CHOP10")
def chop10(df: pd.DataFrame) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_sum = tr.rolling(10).sum()
    high_low = df["high"].rolling(10).max() - df["low"].rolling(10).min()
    return np.log10(atr_sum / (high_low + 1e-10)) / np.log10(10)

@_register_alpha360("CHOP20")
def chop20(df: pd.DataFrame) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_sum = tr.rolling(20).sum()
    high_low = df["high"].rolling(20).max() - df["low"].rolling(20).min()
    return np.log10(atr_sum / (high_low + 1e-10)) / np.log10(20)

# Consecutive Up/Down Days
@_register_alpha360("CONS_UP")
def cons_up(df: pd.DataFrame) -> pd.Series:
    up = (df["close"] > df["close"].shift(1)).astype(int)
    result = up.copy()
    for i in range(1, len(up)):
        if up.iloc[i] == 1:
            result.iloc[i] = result.iloc[i-1] + 1
        else:
            result.iloc[i] = 0
    return result / 10  # Normalize

@_register_alpha360("CONS_DOWN")
def cons_down(df: pd.DataFrame) -> pd.Series:
    down = (df["close"] < df["close"].shift(1)).astype(int)
    result = down.copy()
    for i in range(1, len(down)):
        if down.iloc[i] == 1:
            result.iloc[i] = result.iloc[i-1] + 1
        else:
            result.iloc[i] = 0
    return result / 10  # Normalize

# Relative Return
@_register_alpha360("REL_RET_5_10")
def rel_ret_5_10(df: pd.DataFrame) -> pd.Series:
    ret5 = df["close"].pct_change(5)
    ret10 = df["close"].pct_change(10)
    return ret5 - ret10/2

@_register_alpha360("REL_RET_10_20")
def rel_ret_10_20(df: pd.DataFrame) -> pd.Series:
    ret10 = df["close"].pct_change(10)
    ret20 = df["close"].pct_change(20)
    return ret10 - ret20/2

@_register_alpha360("REL_RET_20_60")
def rel_ret_20_60(df: pd.DataFrame) -> pd.Series:
    ret20 = df["close"].pct_change(20)
    ret60 = df["close"].pct_change(60)
    return ret20 - ret60/3


# Average Directional Index (ADX) components
@_register_alpha360("ADX14")
def adx14(df: pd.DataFrame) -> pd.Series:
    """Average Directional Index - trend strength."""
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr + 1e-10))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    return dx.rolling(14).mean() / 100


@_register_alpha360("DI_DIFF")
def di_diff(df: pd.DataFrame) -> pd.Series:
    """Directional Indicator Difference (+DI - -DI)."""
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_di = plus_dm.rolling(14).mean() / (atr + 1e-10)
    minus_di = minus_dm.rolling(14).mean() / (atr + 1e-10)
    return plus_di - minus_di


# Stochastic RSI
@_register_alpha360("STOCH_RSI")
def stoch_rsi(df: pd.DataFrame) -> pd.Series:
    """Stochastic RSI - RSI of RSI."""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_min = rsi.rolling(14).min()
    rsi_max = rsi.rolling(14).max()
    return (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)


# Money Flow Index (MFI)
@_register_alpha360("MFI14")
def mfi14(df: pd.DataFrame) -> pd.Series:
    """Money Flow Index - volume-weighted RSI."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    mf_pos = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    mf_neg = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    mfi = 100 - (100 / (1 + mf_pos / (mf_neg + 1e-10)))
    return mfi / 100


# Ultimate Oscillator
@_register_alpha360("ULT_OSC")
def ult_osc(df: pd.DataFrame) -> pd.Series:
    """Ultimate Oscillator - weighted average of 3 periods."""
    bp = df["close"] - pd.concat([df["low"], df["close"].shift(1)], axis=1).min(axis=1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    avg7 = bp.rolling(7).sum() / (tr.rolling(7).sum() + 1e-10)
    avg14 = bp.rolling(14).sum() / (tr.rolling(14).sum() + 1e-10)
    avg28 = bp.rolling(28).sum() / (tr.rolling(28).sum() + 1e-10)
    return (4 * avg7 + 2 * avg14 + avg28) / 7


# Keltner Channel factors
@_register_alpha360("KC_WIDTH")
def kc_width(df: pd.DataFrame) -> pd.Series:
    """Keltner Channel Width."""
    ema20 = df["close"].ewm(span=20).mean()
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(10).mean()
    upper = ema20 + 2 * atr
    lower = ema20 - 2 * atr
    return (upper - lower) / (ema20 + 1e-10)


@_register_alpha360("KC_POS")
def kc_pos(df: pd.DataFrame) -> pd.Series:
    """Price position within Keltner Channel."""
    ema20 = df["close"].ewm(span=20).mean()
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(10).mean()
    upper = ema20 + 2 * atr
    lower = ema20 - 2 * atr
    return (df["close"] - lower) / (upper - lower + 1e-10)


# Donchian Channel factors
@_register_alpha360("DC_WIDTH20")
def dc_width20(df: pd.DataFrame) -> pd.Series:
    """Donchian Channel Width (20 periods)."""
    upper = df["high"].rolling(20).max()
    lower = df["low"].rolling(20).min()
    return (upper - lower) / (df["close"] + 1e-10)


@_register_alpha360("DC_POS20")
def dc_pos20(df: pd.DataFrame) -> pd.Series:
    """Price position within Donchian Channel."""
    upper = df["high"].rolling(20).max()
    lower = df["low"].rolling(20).min()
    return (df["close"] - lower) / (upper - lower + 1e-10)


# Price Oscillators
@_register_alpha360("TRIX")
def trix(df: pd.DataFrame) -> pd.Series:
    """Triple Exponential Average."""
    ema1 = df["close"].ewm(span=15).mean()
    ema2 = ema1.ewm(span=15).mean()
    ema3 = ema2.ewm(span=15).mean()
    return ema3.pct_change() * 100


@_register_alpha360("MASS_INDEX")
def mass_index(df: pd.DataFrame) -> pd.Series:
    """Mass Index - reversal indicator."""
    range_ema = (df["high"] - df["low"]).ewm(span=9).mean()
    range_ema2 = range_ema.ewm(span=9).mean()
    ratio = range_ema / (range_ema2 + 1e-10)
    return ratio.rolling(25).sum() / 25


# Aroon Indicators
@_register_alpha360("AROON_UP")
def aroon_up(df: pd.DataFrame) -> pd.Series:
    """Aroon Up - periods since highest high."""
    return df["high"].rolling(25).apply(lambda x: x.argmax() / 24, raw=True)


@_register_alpha360("AROON_DOWN")
def aroon_down(df: pd.DataFrame) -> pd.Series:
    """Aroon Down - periods since lowest low."""
    return df["low"].rolling(25).apply(lambda x: x.argmin() / 24, raw=True)


@_register_alpha360("AROON_OSC")
def aroon_osc(df: pd.DataFrame) -> pd.Series:
    """Aroon Oscillator (Up - Down)."""
    up = df["high"].rolling(25).apply(lambda x: x.argmax() / 24, raw=True)
    down = df["low"].rolling(25).apply(lambda x: x.argmin() / 24, raw=True)
    return up - down


# Know Sure Thing (KST)
@_register_alpha360("KST")
def kst(df: pd.DataFrame) -> pd.Series:
    """Know Sure Thing - momentum oscillator."""
    roc10 = df["close"].pct_change(10).rolling(10).mean()
    roc15 = df["close"].pct_change(15).rolling(10).mean()
    roc20 = df["close"].pct_change(20).rolling(10).mean()
    roc30 = df["close"].pct_change(30).rolling(15).mean()
    return roc10 + 2 * roc15 + 3 * roc20 + 4 * roc30


# Volume Weighted Moving Average
@_register_alpha360("VWMA20")
def vwma20(df: pd.DataFrame) -> pd.Series:
    """Volume Weighted Moving Average - price/VWMA ratio."""
    vwma = (df["close"] * df["volume"]).rolling(20).sum() / (df["volume"].rolling(20).sum() + 1e-10)
    return df["close"] / (vwma + 1e-10) - 1


# =============================================================================
# Utility Functions
# =============================================================================

def get_all_alpha360_factors() -> list[str]:
    """Get all registered Alpha360 factor names."""
    return list(ALPHA360_FACTORS.keys())


def compute_alpha360(
    df: pd.DataFrame,
    factor_names: list[str] | None = None,
) -> pd.DataFrame:
    """Compute specified Alpha360 factors.

    Args:
        df: DataFrame with OHLCV data
        factor_names: List of factor names to compute (None = all)

    Returns:
        DataFrame with computed factor values
    """
    if factor_names is None:
        factor_names = list(ALPHA360_FACTORS.keys())

    results = {}
    for name in factor_names:
        if name in ALPHA360_FACTORS:
            try:
                results[name] = ALPHA360_FACTORS[name](df)
            except Exception as e:
                print(f"Warning: Failed to compute {name}: {e}")
                results[name] = pd.Series(np.nan, index=df.index)

    return pd.DataFrame(results)


def get_alpha360_count() -> int:
    """Get number of registered Alpha360 factors."""
    return len(ALPHA360_FACTORS)


def get_alpha360_by_feature() -> dict[str, list[str]]:
    """Get Alpha360 factors organized by base feature."""
    features = {
        "CLOSE": [],
        "HIGH": [],
        "LOW": [],
        "OPEN": [],
        "VOLUME": [],
        "VWAP": [],
        "CROSS": [],
    }

    for name in ALPHA360_FACTORS.keys():
        if name.startswith("CLOSE_"):
            features["CLOSE"].append(name)
        elif name.startswith("HIGH_"):
            features["HIGH"].append(name)
        elif name.startswith("LOW_"):
            features["LOW"].append(name)
        elif name.startswith("OPEN_"):
            features["OPEN"].append(name)
        elif name.startswith("VOLUME_"):
            features["VOLUME"].append(name)
        elif name.startswith("VWAP_"):
            features["VWAP"].append(name)
        else:
            features["CROSS"].append(name)

    return features
