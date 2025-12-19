"""Alpha158 Complete Factor Library.

Complete implementation of Qlib's Alpha158 factor set.
Reference: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py

158 factors organized into categories:
- K-Line (price pattern) factors
- Momentum factors (ROC, returns)
- Moving average factors
- Volatility factors
- Volume factors
- Technical indicators (RSI, MACD, etc.)
- Statistical factors (beta, correlation, etc.)

All factors are adapted for cryptocurrency 24/7 trading.
"""

from __future__ import annotations

from typing import Callable
import numpy as np
import pandas as pd

# Use Qlib-native statistical functions instead of scipy
from iqfmp.evaluation.qlib_stats import rank_percentile

# =============================================================================
# Alpha158 Factor Registry
# =============================================================================

ALPHA158_FACTORS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {}


def _register_alpha158(name: str):
    """Decorator to register an Alpha158 factor."""
    def decorator(func: Callable[[pd.DataFrame], pd.Series]):
        ALPHA158_FACTORS[name] = func
        return func
    return decorator


# =============================================================================
# Helper Functions
# =============================================================================

def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    """Safe division avoiding zero."""
    return a / (b + 1e-10)


def _window_percentile_rank(x: pd.Series) -> float:
    """Calculate percentile rank of last value within window.

    Pure numpy/pandas implementation replacing scipy.stats.percentileofscore.
    Returns value between 0 and 1 (normalized percentile).
    """
    last_val = x.iloc[-1]
    n = len(x)
    # Percentile: proportion of values less than or equal to last value
    return (x <= last_val).sum() / n


def _ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=window, adjust=False).mean()


def _wma(series: pd.Series, window: int) -> pd.Series:
    """Weighted moving average."""
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(
        lambda x: np.dot(x, weights[:len(x)]) / weights[:len(x)].sum(),
        raw=True
    )


def _ts_rank(series: pd.Series, window: int) -> pd.Series:
    """Time-series rank (percentile within window).

    Delegates to Qlib-native implementation for architectural consistency.
    Returns value between 0 and 1 (rank_percentile returns 0-100, so we divide by 100).
    """
    return rank_percentile(series, window) / 100


# =============================================================================
# K-Line Factors (9 factors)
# =============================================================================

@_register_alpha158("KMID")
def kmid(df: pd.DataFrame) -> pd.Series:
    """Close position relative to High-Low range."""
    return _safe_divide(df["close"] - df["low"], df["high"] - df["low"])


@_register_alpha158("KLEN")
def klen(df: pd.DataFrame) -> pd.Series:
    """K-line length normalized by close."""
    return (df["high"] - df["low"]) / df["close"]


@_register_alpha158("KMID2")
def kmid2(df: pd.DataFrame) -> pd.Series:
    """Close-Open relative to High-Low range."""
    return _safe_divide(df["close"] - df["open"], df["high"] - df["low"])


@_register_alpha158("KUP")
def kup(df: pd.DataFrame) -> pd.Series:
    """Upper shadow ratio."""
    body_high = df[["open", "close"]].max(axis=1)
    return _safe_divide(df["high"] - body_high, df["high"] - df["low"])


@_register_alpha158("KUP2")
def kup2(df: pd.DataFrame) -> pd.Series:
    """High-Open relative to range."""
    return _safe_divide(df["high"] - df["open"], df["high"] - df["low"])


@_register_alpha158("KLOW")
def klow(df: pd.DataFrame) -> pd.Series:
    """Lower shadow ratio."""
    body_low = df[["open", "close"]].min(axis=1)
    return _safe_divide(body_low - df["low"], df["high"] - df["low"])


@_register_alpha158("KLOW2")
def klow2(df: pd.DataFrame) -> pd.Series:
    """Open-Low relative to range."""
    return _safe_divide(df["open"] - df["low"], df["high"] - df["low"])


@_register_alpha158("KSFT")
def ksft(df: pd.DataFrame) -> pd.Series:
    """K-line center shift."""
    return _safe_divide(2 * df["close"] - df["high"] - df["low"], df["high"] - df["low"])


@_register_alpha158("KSFT2")
def ksft2(df: pd.DataFrame) -> pd.Series:
    """K-line center shift normalized by close."""
    return (2 * df["close"] - df["high"] - df["low"]) / df["close"]


# =============================================================================
# Return/Momentum Factors (15 factors - 5 lookbacks x 3 types)
# =============================================================================

# Simple Returns (ROC)
@_register_alpha158("ROC5")
def roc5(df: pd.DataFrame) -> pd.Series:
    """5-period rate of change."""
    return df["close"].pct_change(5)


@_register_alpha158("ROC10")
def roc10(df: pd.DataFrame) -> pd.Series:
    """10-period rate of change."""
    return df["close"].pct_change(10)


@_register_alpha158("ROC20")
def roc20(df: pd.DataFrame) -> pd.Series:
    """20-period rate of change."""
    return df["close"].pct_change(20)


@_register_alpha158("ROC30")
def roc30(df: pd.DataFrame) -> pd.Series:
    """30-period rate of change."""
    return df["close"].pct_change(30)


@_register_alpha158("ROC60")
def roc60(df: pd.DataFrame) -> pd.Series:
    """60-period rate of change."""
    return df["close"].pct_change(60)


# High-Low Change
@_register_alpha158("HLROC5")
def hlroc5(df: pd.DataFrame) -> pd.Series:
    """5-period high-low range change."""
    hlr = (df["high"] - df["low"]) / df["close"]
    return hlr.pct_change(5)


@_register_alpha158("HLROC10")
def hlroc10(df: pd.DataFrame) -> pd.Series:
    """10-period high-low range change."""
    hlr = (df["high"] - df["low"]) / df["close"]
    return hlr.pct_change(10)


@_register_alpha158("HLROC20")
def hlroc20(df: pd.DataFrame) -> pd.Series:
    """20-period high-low range change."""
    hlr = (df["high"] - df["low"]) / df["close"]
    return hlr.pct_change(20)


@_register_alpha158("HLROC30")
def hlroc30(df: pd.DataFrame) -> pd.Series:
    """30-period high-low range change."""
    hlr = (df["high"] - df["low"]) / df["close"]
    return hlr.pct_change(30)


@_register_alpha158("HLROC60")
def hlroc60(df: pd.DataFrame) -> pd.Series:
    """60-period high-low range change."""
    hlr = (df["high"] - df["low"]) / df["close"]
    return hlr.pct_change(60)


# Log Returns
@_register_alpha158("LOGRET5")
def logret5(df: pd.DataFrame) -> pd.Series:
    """5-period log return."""
    return np.log(df["close"] / df["close"].shift(5))


@_register_alpha158("LOGRET10")
def logret10(df: pd.DataFrame) -> pd.Series:
    """10-period log return."""
    return np.log(df["close"] / df["close"].shift(10))


@_register_alpha158("LOGRET20")
def logret20(df: pd.DataFrame) -> pd.Series:
    """20-period log return."""
    return np.log(df["close"] / df["close"].shift(20))


@_register_alpha158("LOGRET30")
def logret30(df: pd.DataFrame) -> pd.Series:
    """30-period log return."""
    return np.log(df["close"] / df["close"].shift(30))


@_register_alpha158("LOGRET60")
def logret60(df: pd.DataFrame) -> pd.Series:
    """60-period log return."""
    return np.log(df["close"] / df["close"].shift(60))


# =============================================================================
# Moving Average Factors (20 factors)
# =============================================================================

# MA Ratio (Close / MA - 1)
@_register_alpha158("MA5_RATIO")
def ma5_ratio(df: pd.DataFrame) -> pd.Series:
    """Close / MA5 - 1."""
    return df["close"] / df["close"].rolling(5).mean() - 1


@_register_alpha158("MA10_RATIO")
def ma10_ratio(df: pd.DataFrame) -> pd.Series:
    """Close / MA10 - 1."""
    return df["close"] / df["close"].rolling(10).mean() - 1


@_register_alpha158("MA20_RATIO")
def ma20_ratio(df: pd.DataFrame) -> pd.Series:
    """Close / MA20 - 1."""
    return df["close"] / df["close"].rolling(20).mean() - 1


@_register_alpha158("MA30_RATIO")
def ma30_ratio(df: pd.DataFrame) -> pd.Series:
    """Close / MA30 - 1."""
    return df["close"] / df["close"].rolling(30).mean() - 1


@_register_alpha158("MA60_RATIO")
def ma60_ratio(df: pd.DataFrame) -> pd.Series:
    """Close / MA60 - 1."""
    return df["close"] / df["close"].rolling(60).mean() - 1


# EMA Ratio
@_register_alpha158("EMA5_RATIO")
def ema5_ratio(df: pd.DataFrame) -> pd.Series:
    """Close / EMA5 - 1."""
    return df["close"] / _ema(df["close"], 5) - 1


@_register_alpha158("EMA10_RATIO")
def ema10_ratio(df: pd.DataFrame) -> pd.Series:
    """Close / EMA10 - 1."""
    return df["close"] / _ema(df["close"], 10) - 1


@_register_alpha158("EMA20_RATIO")
def ema20_ratio(df: pd.DataFrame) -> pd.Series:
    """Close / EMA20 - 1."""
    return df["close"] / _ema(df["close"], 20) - 1


@_register_alpha158("EMA30_RATIO")
def ema30_ratio(df: pd.DataFrame) -> pd.Series:
    """Close / EMA30 - 1."""
    return df["close"] / _ema(df["close"], 30) - 1


@_register_alpha158("EMA60_RATIO")
def ema60_ratio(df: pd.DataFrame) -> pd.Series:
    """Close / EMA60 - 1."""
    return df["close"] / _ema(df["close"], 60) - 1


# MA Cross Signals
@_register_alpha158("MA5_10_CROSS")
def ma5_10_cross(df: pd.DataFrame) -> pd.Series:
    """MA5 / MA10 - 1."""
    return df["close"].rolling(5).mean() / df["close"].rolling(10).mean() - 1


@_register_alpha158("MA5_20_CROSS")
def ma5_20_cross(df: pd.DataFrame) -> pd.Series:
    """MA5 / MA20 - 1."""
    return df["close"].rolling(5).mean() / df["close"].rolling(20).mean() - 1


@_register_alpha158("MA10_20_CROSS")
def ma10_20_cross(df: pd.DataFrame) -> pd.Series:
    """MA10 / MA20 - 1."""
    return df["close"].rolling(10).mean() / df["close"].rolling(20).mean() - 1


@_register_alpha158("MA10_30_CROSS")
def ma10_30_cross(df: pd.DataFrame) -> pd.Series:
    """MA10 / MA30 - 1."""
    return df["close"].rolling(10).mean() / df["close"].rolling(30).mean() - 1


@_register_alpha158("MA20_60_CROSS")
def ma20_60_cross(df: pd.DataFrame) -> pd.Series:
    """MA20 / MA60 - 1."""
    return df["close"].rolling(20).mean() / df["close"].rolling(60).mean() - 1


# WMA Ratio
@_register_alpha158("WMA5_RATIO")
def wma5_ratio(df: pd.DataFrame) -> pd.Series:
    """Close / WMA5 - 1."""
    return df["close"] / _wma(df["close"], 5) - 1


@_register_alpha158("WMA10_RATIO")
def wma10_ratio(df: pd.DataFrame) -> pd.Series:
    """Close / WMA10 - 1."""
    return df["close"] / _wma(df["close"], 10) - 1


@_register_alpha158("WMA20_RATIO")
def wma20_ratio(df: pd.DataFrame) -> pd.Series:
    """Close / WMA20 - 1."""
    return df["close"] / _wma(df["close"], 20) - 1


@_register_alpha158("WMA30_RATIO")
def wma30_ratio(df: pd.DataFrame) -> pd.Series:
    """Close / WMA30 - 1."""
    return df["close"] / _wma(df["close"], 30) - 1


@_register_alpha158("WMA60_RATIO")
def wma60_ratio(df: pd.DataFrame) -> pd.Series:
    """Close / WMA60 - 1."""
    return df["close"] / _wma(df["close"], 60) - 1


# =============================================================================
# Volatility Factors (25 factors)
# =============================================================================

# Standard Deviation
@_register_alpha158("STD5")
def std5(df: pd.DataFrame) -> pd.Series:
    """5-period return standard deviation."""
    return df["close"].pct_change().rolling(5).std()


@_register_alpha158("STD10")
def std10(df: pd.DataFrame) -> pd.Series:
    """10-period return standard deviation."""
    return df["close"].pct_change().rolling(10).std()


@_register_alpha158("STD20")
def std20(df: pd.DataFrame) -> pd.Series:
    """20-period return standard deviation."""
    return df["close"].pct_change().rolling(20).std()


@_register_alpha158("STD30")
def std30(df: pd.DataFrame) -> pd.Series:
    """30-period return standard deviation."""
    return df["close"].pct_change().rolling(30).std()


@_register_alpha158("STD60")
def std60(df: pd.DataFrame) -> pd.Series:
    """60-period return standard deviation."""
    return df["close"].pct_change().rolling(60).std()


# True Range
@_register_alpha158("ATR5")
def atr5(df: pd.DataFrame) -> pd.Series:
    """5-period Average True Range."""
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(5).mean() / df["close"]


@_register_alpha158("ATR10")
def atr10(df: pd.DataFrame) -> pd.Series:
    """10-period Average True Range."""
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(10).mean() / df["close"]


@_register_alpha158("ATR20")
def atr20(df: pd.DataFrame) -> pd.Series:
    """20-period Average True Range."""
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(20).mean() / df["close"]


@_register_alpha158("ATR30")
def atr30(df: pd.DataFrame) -> pd.Series:
    """30-period Average True Range."""
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(30).mean() / df["close"]


@_register_alpha158("ATR60")
def atr60(df: pd.DataFrame) -> pd.Series:
    """60-period Average True Range."""
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(60).mean() / df["close"]


# Bollinger Band Width
@_register_alpha158("BBWIDTH5")
def bbwidth5(df: pd.DataFrame) -> pd.Series:
    """5-period Bollinger Band Width."""
    ma = df["close"].rolling(5).mean()
    std = df["close"].rolling(5).std()
    return (2 * std * 2) / ma


@_register_alpha158("BBWIDTH10")
def bbwidth10(df: pd.DataFrame) -> pd.Series:
    """10-period Bollinger Band Width."""
    ma = df["close"].rolling(10).mean()
    std = df["close"].rolling(10).std()
    return (2 * std * 2) / ma


@_register_alpha158("BBWIDTH20")
def bbwidth20(df: pd.DataFrame) -> pd.Series:
    """20-period Bollinger Band Width."""
    ma = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    return (2 * std * 2) / ma


@_register_alpha158("BBWIDTH30")
def bbwidth30(df: pd.DataFrame) -> pd.Series:
    """30-period Bollinger Band Width."""
    ma = df["close"].rolling(30).mean()
    std = df["close"].rolling(30).std()
    return (2 * std * 2) / ma


@_register_alpha158("BBWIDTH60")
def bbwidth60(df: pd.DataFrame) -> pd.Series:
    """60-period Bollinger Band Width."""
    ma = df["close"].rolling(60).mean()
    std = df["close"].rolling(60).std()
    return (2 * std * 2) / ma


# Bollinger Band Position
@_register_alpha158("BBPOS5")
def bbpos5(df: pd.DataFrame) -> pd.Series:
    """5-period Bollinger Band Position."""
    ma = df["close"].rolling(5).mean()
    std = df["close"].rolling(5).std()
    return (df["close"] - ma) / (2 * std + 1e-10)


@_register_alpha158("BBPOS10")
def bbpos10(df: pd.DataFrame) -> pd.Series:
    """10-period Bollinger Band Position."""
    ma = df["close"].rolling(10).mean()
    std = df["close"].rolling(10).std()
    return (df["close"] - ma) / (2 * std + 1e-10)


@_register_alpha158("BBPOS20")
def bbpos20(df: pd.DataFrame) -> pd.Series:
    """20-period Bollinger Band Position."""
    ma = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    return (df["close"] - ma) / (2 * std + 1e-10)


@_register_alpha158("BBPOS30")
def bbpos30(df: pd.DataFrame) -> pd.Series:
    """30-period Bollinger Band Position."""
    ma = df["close"].rolling(30).mean()
    std = df["close"].rolling(30).std()
    return (df["close"] - ma) / (2 * std + 1e-10)


@_register_alpha158("BBPOS60")
def bbpos60(df: pd.DataFrame) -> pd.Series:
    """60-period Bollinger Band Position."""
    ma = df["close"].rolling(60).mean()
    std = df["close"].rolling(60).std()
    return (df["close"] - ma) / (2 * std + 1e-10)


# Volatility Ratio
@_register_alpha158("VOLRATIO5_20")
def volratio5_20(df: pd.DataFrame) -> pd.Series:
    """5-period / 20-period volatility ratio."""
    vol5 = df["close"].pct_change().rolling(5).std()
    vol20 = df["close"].pct_change().rolling(20).std()
    return vol5 / (vol20 + 1e-10)


@_register_alpha158("VOLRATIO10_30")
def volratio10_30(df: pd.DataFrame) -> pd.Series:
    """10-period / 30-period volatility ratio."""
    vol10 = df["close"].pct_change().rolling(10).std()
    vol30 = df["close"].pct_change().rolling(30).std()
    return vol10 / (vol30 + 1e-10)


@_register_alpha158("VOLRATIO20_60")
def volratio20_60(df: pd.DataFrame) -> pd.Series:
    """20-period / 60-period volatility ratio."""
    vol20 = df["close"].pct_change().rolling(20).std()
    vol60 = df["close"].pct_change().rolling(60).std()
    return vol20 / (vol60 + 1e-10)


# =============================================================================
# Volume Factors (25 factors)
# =============================================================================

# Volume MA Ratio
@_register_alpha158("VMA5")
def vma5(df: pd.DataFrame) -> pd.Series:
    """Volume MA5 ratio."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"] / df["volume"].rolling(5).mean() - 1


@_register_alpha158("VMA10")
def vma10(df: pd.DataFrame) -> pd.Series:
    """Volume MA10 ratio."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"] / df["volume"].rolling(10).mean() - 1


@_register_alpha158("VMA20")
def vma20(df: pd.DataFrame) -> pd.Series:
    """Volume MA20 ratio."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"] / df["volume"].rolling(20).mean() - 1


@_register_alpha158("VMA30")
def vma30(df: pd.DataFrame) -> pd.Series:
    """Volume MA30 ratio."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"] / df["volume"].rolling(30).mean() - 1


@_register_alpha158("VMA60")
def vma60(df: pd.DataFrame) -> pd.Series:
    """Volume MA60 ratio."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"] / df["volume"].rolling(60).mean() - 1


# Volume Standard Deviation
@_register_alpha158("VSTD5")
def vstd5(df: pd.DataFrame) -> pd.Series:
    """5-period volume standard deviation ratio."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"].rolling(5).std() / (df["volume"].rolling(20).mean() + 1e-10)


@_register_alpha158("VSTD10")
def vstd10(df: pd.DataFrame) -> pd.Series:
    """10-period volume standard deviation ratio."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"].rolling(10).std() / (df["volume"].rolling(20).mean() + 1e-10)


@_register_alpha158("VSTD20")
def vstd20(df: pd.DataFrame) -> pd.Series:
    """20-period volume standard deviation ratio."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"].rolling(20).std() / (df["volume"].rolling(60).mean() + 1e-10)


@_register_alpha158("VSTD30")
def vstd30(df: pd.DataFrame) -> pd.Series:
    """30-period volume standard deviation ratio."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"].rolling(30).std() / (df["volume"].rolling(60).mean() + 1e-10)


@_register_alpha158("VSTD60")
def vstd60(df: pd.DataFrame) -> pd.Series:
    """60-period volume standard deviation ratio."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["volume"].rolling(60).std() / (df["volume"].rolling(120).mean() + 1e-10)


# Weighted Volume MA
@_register_alpha158("WVMA5")
def wvma5(df: pd.DataFrame) -> pd.Series:
    """5-period weighted volume MA (by absolute return)."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    weighted = df["volume"] * df["close"].pct_change().abs()
    return weighted.rolling(5).mean() / (weighted.rolling(20).mean() + 1e-10)


@_register_alpha158("WVMA10")
def wvma10(df: pd.DataFrame) -> pd.Series:
    """10-period weighted volume MA."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    weighted = df["volume"] * df["close"].pct_change().abs()
    return weighted.rolling(10).mean() / (weighted.rolling(20).mean() + 1e-10)


@_register_alpha158("WVMA20")
def wvma20(df: pd.DataFrame) -> pd.Series:
    """20-period weighted volume MA."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    weighted = df["volume"] * df["close"].pct_change().abs()
    return weighted.rolling(20).mean() / (weighted.rolling(60).mean() + 1e-10)


# Turnover
@_register_alpha158("TURN5")
def turn5(df: pd.DataFrame) -> pd.Series:
    """5-period turnover ratio."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    turn = df["volume"] / (df["volume"].rolling(20).mean() + 1e-10)
    return turn.rolling(5).mean() - 1


@_register_alpha158("TURN10")
def turn10(df: pd.DataFrame) -> pd.Series:
    """10-period turnover ratio."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    turn = df["volume"] / (df["volume"].rolling(20).mean() + 1e-10)
    return turn.rolling(10).mean() - 1


@_register_alpha158("TURN20")
def turn20(df: pd.DataFrame) -> pd.Series:
    """20-period turnover ratio."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    turn = df["volume"] / (df["volume"].rolling(60).mean() + 1e-10)
    return turn.rolling(20).mean() - 1


# Price-Volume Correlation
@_register_alpha158("CORR5")
def corr5(df: pd.DataFrame) -> pd.Series:
    """5-period return-volume correlation."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    returns = df["close"].pct_change()
    vol_change = df["volume"].pct_change()
    return returns.rolling(5).corr(vol_change)


@_register_alpha158("CORR10")
def corr10(df: pd.DataFrame) -> pd.Series:
    """10-period return-volume correlation."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    returns = df["close"].pct_change()
    vol_change = df["volume"].pct_change()
    return returns.rolling(10).corr(vol_change)


@_register_alpha158("CORR20")
def corr20(df: pd.DataFrame) -> pd.Series:
    """20-period return-volume correlation."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    returns = df["close"].pct_change()
    vol_change = df["volume"].pct_change()
    return returns.rolling(20).corr(vol_change)


@_register_alpha158("CORR30")
def corr30(df: pd.DataFrame) -> pd.Series:
    """30-period return-volume correlation."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    returns = df["close"].pct_change()
    vol_change = df["volume"].pct_change()
    return returns.rolling(30).corr(vol_change)


@_register_alpha158("CORR60")
def corr60(df: pd.DataFrame) -> pd.Series:
    """60-period return-volume correlation."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    returns = df["close"].pct_change()
    vol_change = df["volume"].pct_change()
    return returns.rolling(60).corr(vol_change)


# OBV (On-Balance Volume) based
@_register_alpha158("OBV5")
def obv5(df: pd.DataFrame) -> pd.Series:
    """5-period OBV change ratio."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    direction = np.sign(df["close"].diff())
    obv = (direction * df["volume"]).cumsum()
    return obv.pct_change(5)


@_register_alpha158("OBV10")
def obv10(df: pd.DataFrame) -> pd.Series:
    """10-period OBV change ratio."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    direction = np.sign(df["close"].diff())
    obv = (direction * df["volume"]).cumsum()
    return obv.pct_change(10)


@_register_alpha158("OBV20")
def obv20(df: pd.DataFrame) -> pd.Series:
    """20-period OBV change ratio."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    direction = np.sign(df["close"].diff())
    obv = (direction * df["volume"]).cumsum()
    return obv.pct_change(20)


# VWAP factors
@_register_alpha158("VWAP5_RATIO")
def vwap5_ratio(df: pd.DataFrame) -> pd.Series:
    """5-period VWAP ratio."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    vwap = (df["close"] * df["volume"]).rolling(5).sum() / df["volume"].rolling(5).sum()
    return df["close"] / vwap - 1


@_register_alpha158("VWAP20_RATIO")
def vwap20_ratio(df: pd.DataFrame) -> pd.Series:
    """20-period VWAP ratio."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    vwap = (df["close"] * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
    return df["close"] / vwap - 1


# =============================================================================
# Technical Indicators (30 factors)
# =============================================================================

# RSI
@_register_alpha158("RSI6")
def rsi6(df: pd.DataFrame) -> pd.Series:
    """6-period RSI."""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


@_register_alpha158("RSI14")
def rsi14(df: pd.DataFrame) -> pd.Series:
    """14-period RSI."""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


@_register_alpha158("RSI24")
def rsi24(df: pd.DataFrame) -> pd.Series:
    """24-period RSI."""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(24).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(24).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


# Stochastic
@_register_alpha158("RSV5")
def rsv5(df: pd.DataFrame) -> pd.Series:
    """5-period Raw Stochastic Value."""
    lowest = df["low"].rolling(5).min()
    highest = df["high"].rolling(5).max()
    return _safe_divide(df["close"] - lowest, highest - lowest)


@_register_alpha158("RSV10")
def rsv10(df: pd.DataFrame) -> pd.Series:
    """10-period Raw Stochastic Value."""
    lowest = df["low"].rolling(10).min()
    highest = df["high"].rolling(10).max()
    return _safe_divide(df["close"] - lowest, highest - lowest)


@_register_alpha158("RSV20")
def rsv20(df: pd.DataFrame) -> pd.Series:
    """20-period Raw Stochastic Value."""
    lowest = df["low"].rolling(20).min()
    highest = df["high"].rolling(20).max()
    return _safe_divide(df["close"] - lowest, highest - lowest)


# Williams %R
@_register_alpha158("WILLR5")
def willr5(df: pd.DataFrame) -> pd.Series:
    """5-period Williams %R."""
    highest = df["high"].rolling(5).max()
    lowest = df["low"].rolling(5).min()
    return _safe_divide(highest - df["close"], highest - lowest) * -100


@_register_alpha158("WILLR10")
def willr10(df: pd.DataFrame) -> pd.Series:
    """10-period Williams %R."""
    highest = df["high"].rolling(10).max()
    lowest = df["low"].rolling(10).min()
    return _safe_divide(highest - df["close"], highest - lowest) * -100


@_register_alpha158("WILLR20")
def willr20(df: pd.DataFrame) -> pd.Series:
    """20-period Williams %R."""
    highest = df["high"].rolling(20).max()
    lowest = df["low"].rolling(20).min()
    return _safe_divide(highest - df["close"], highest - lowest) * -100


# CCI (Commodity Channel Index)
@_register_alpha158("CCI5")
def cci5(df: pd.DataFrame) -> pd.Series:
    """5-period CCI."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(5).mean()
    md = tp.rolling(5).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * md + 1e-10)


@_register_alpha158("CCI10")
def cci10(df: pd.DataFrame) -> pd.Series:
    """10-period CCI."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(10).mean()
    md = tp.rolling(10).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * md + 1e-10)


@_register_alpha158("CCI20")
def cci20(df: pd.DataFrame) -> pd.Series:
    """20-period CCI."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(20).mean()
    md = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * md + 1e-10)


# ADX (Average Directional Index) components
@_register_alpha158("PLUS_DI14")
def plus_di14(df: pd.DataFrame) -> pd.Series:
    """14-period +DI."""
    high_diff = df["high"].diff()
    low_diff = -df["low"].diff()
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    return 100 * plus_dm.rolling(14).mean() / (atr + 1e-10)


@_register_alpha158("MINUS_DI14")
def minus_di14(df: pd.DataFrame) -> pd.Series:
    """14-period -DI."""
    high_diff = df["high"].diff()
    low_diff = -df["low"].diff()
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    return 100 * minus_dm.rolling(14).mean() / (atr + 1e-10)


@_register_alpha158("ADX14")
def adx14(df: pd.DataFrame) -> pd.Series:
    """14-period ADX."""
    plus_di = plus_di14(df)
    minus_di = minus_di14(df)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.rolling(14).mean()


# MACD
@_register_alpha158("MACD")
def macd(df: pd.DataFrame) -> pd.Series:
    """MACD (12-26)."""
    ema12 = _ema(df["close"], 12)
    ema26 = _ema(df["close"], 26)
    return (ema12 - ema26) / df["close"]


@_register_alpha158("MACD_SIGNAL")
def macd_signal(df: pd.DataFrame) -> pd.Series:
    """MACD Signal (9-period EMA of MACD)."""
    macd_line = macd(df) * df["close"]
    return _ema(macd_line, 9) / df["close"]


@_register_alpha158("MACD_HIST")
def macd_hist(df: pd.DataFrame) -> pd.Series:
    """MACD Histogram."""
    return macd(df) - macd_signal(df)


# Money Flow Index
@_register_alpha158("MFI14")
def mfi14(df: pd.DataFrame) -> pd.Series:
    """14-period Money Flow Index."""
    if "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    tp_diff = tp.diff()
    pos_mf = mf.where(tp_diff > 0, 0).rolling(14).sum()
    neg_mf = mf.where(tp_diff < 0, 0).rolling(14).sum()
    mfi = 100 - 100 / (1 + pos_mf / (neg_mf + 1e-10))
    return mfi


# ROC-based indicators
@_register_alpha158("ROC_ACCEL5")
def roc_accel5(df: pd.DataFrame) -> pd.Series:
    """5-period ROC acceleration (2nd derivative)."""
    roc = df["close"].pct_change()
    return roc.diff().rolling(5).mean()


@_register_alpha158("ROC_ACCEL10")
def roc_accel10(df: pd.DataFrame) -> pd.Series:
    """10-period ROC acceleration."""
    roc = df["close"].pct_change()
    return roc.diff().rolling(10).mean()


# Aroon
@_register_alpha158("AROON_UP25")
def aroon_up25(df: pd.DataFrame) -> pd.Series:
    """25-period Aroon Up."""
    return df["high"].rolling(25).apply(
        lambda x: (25 - (25 - x.argmax() - 1)) / 25 * 100,
        raw=True
    )


@_register_alpha158("AROON_DOWN25")
def aroon_down25(df: pd.DataFrame) -> pd.Series:
    """25-period Aroon Down."""
    return df["low"].rolling(25).apply(
        lambda x: (25 - (25 - x.argmin() - 1)) / 25 * 100,
        raw=True
    )


@_register_alpha158("AROON_OSC25")
def aroon_osc25(df: pd.DataFrame) -> pd.Series:
    """25-period Aroon Oscillator."""
    return aroon_up25(df) - aroon_down25(df)


# Keltner Channel Position
@_register_alpha158("KELTNER_POS20")
def keltner_pos20(df: pd.DataFrame) -> pd.Series:
    """20-period Keltner Channel Position."""
    ema20 = _ema(df["close"], 20)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = _ema(tr, 20)
    upper = ema20 + 2 * atr
    lower = ema20 - 2 * atr
    return _safe_divide(df["close"] - lower, upper - lower)


# Donchian Channel Position
@_register_alpha158("DONCHIAN_POS20")
def donchian_pos20(df: pd.DataFrame) -> pd.Series:
    """20-period Donchian Channel Position."""
    upper = df["high"].rolling(20).max()
    lower = df["low"].rolling(20).min()
    return _safe_divide(df["close"] - lower, upper - lower)


# =============================================================================
# Statistical/Regression Factors (20 factors)
# =============================================================================

# Beta approximations
@_register_alpha158("BETA5")
def beta5(df: pd.DataFrame) -> pd.Series:
    """5-period beta (vs rolling mean proxy)."""
    returns = df["close"].pct_change()
    market_proxy = returns.rolling(20).mean()
    cov = returns.rolling(5).cov(market_proxy)
    var = market_proxy.rolling(5).var()
    return cov / (var + 1e-10)


@_register_alpha158("BETA10")
def beta10(df: pd.DataFrame) -> pd.Series:
    """10-period beta."""
    returns = df["close"].pct_change()
    market_proxy = returns.rolling(20).mean()
    cov = returns.rolling(10).cov(market_proxy)
    var = market_proxy.rolling(10).var()
    return cov / (var + 1e-10)


@_register_alpha158("BETA20")
def beta20(df: pd.DataFrame) -> pd.Series:
    """20-period beta."""
    returns = df["close"].pct_change()
    market_proxy = returns.rolling(60).mean()
    cov = returns.rolling(20).cov(market_proxy)
    var = market_proxy.rolling(20).var()
    return cov / (var + 1e-10)


# R-Squared
@_register_alpha158("RSQR5")
def rsqr5(df: pd.DataFrame) -> pd.Series:
    """5-period R-squared of returns vs trend."""
    returns = df["close"].pct_change()
    ma = returns.rolling(5).mean()
    return returns.rolling(5).corr(ma) ** 2


@_register_alpha158("RSQR10")
def rsqr10(df: pd.DataFrame) -> pd.Series:
    """10-period R-squared."""
    returns = df["close"].pct_change()
    ma = returns.rolling(10).mean()
    return returns.rolling(10).corr(ma) ** 2


@_register_alpha158("RSQR20")
def rsqr20(df: pd.DataFrame) -> pd.Series:
    """20-period R-squared."""
    returns = df["close"].pct_change()
    ma = returns.rolling(20).mean()
    return returns.rolling(20).corr(ma) ** 2


# Residual Volatility
@_register_alpha158("RESI5")
def resi5(df: pd.DataFrame) -> pd.Series:
    """5-period residual volatility."""
    returns = df["close"].pct_change()
    ma = returns.rolling(5).mean()
    residual = returns - ma
    return residual.rolling(5).std()


@_register_alpha158("RESI10")
def resi10(df: pd.DataFrame) -> pd.Series:
    """10-period residual volatility."""
    returns = df["close"].pct_change()
    ma = returns.rolling(10).mean()
    residual = returns - ma
    return residual.rolling(10).std()


@_register_alpha158("RESI20")
def resi20(df: pd.DataFrame) -> pd.Series:
    """20-period residual volatility."""
    returns = df["close"].pct_change()
    ma = returns.rolling(20).mean()
    residual = returns - ma
    return residual.rolling(20).std()


# Max/Min
@_register_alpha158("MAX5")
def max5(df: pd.DataFrame) -> pd.Series:
    """5-period max return."""
    return df["close"].pct_change().rolling(5).max()


@_register_alpha158("MAX10")
def max10(df: pd.DataFrame) -> pd.Series:
    """10-period max return."""
    return df["close"].pct_change().rolling(10).max()


@_register_alpha158("MAX20")
def max20(df: pd.DataFrame) -> pd.Series:
    """20-period max return."""
    return df["close"].pct_change().rolling(20).max()


@_register_alpha158("MIN5")
def min5(df: pd.DataFrame) -> pd.Series:
    """5-period min return."""
    return df["close"].pct_change().rolling(5).min()


@_register_alpha158("MIN10")
def min10(df: pd.DataFrame) -> pd.Series:
    """10-period min return."""
    return df["close"].pct_change().rolling(10).min()


@_register_alpha158("MIN20")
def min20(df: pd.DataFrame) -> pd.Series:
    """20-period min return."""
    return df["close"].pct_change().rolling(20).min()


# Quantiles
@_register_alpha158("QTLU5")
def qtlu5(df: pd.DataFrame) -> pd.Series:
    """5-period 80th percentile return."""
    return df["close"].pct_change().rolling(5).quantile(0.8)


@_register_alpha158("QTLD5")
def qtld5(df: pd.DataFrame) -> pd.Series:
    """5-period 20th percentile return."""
    return df["close"].pct_change().rolling(5).quantile(0.2)


@_register_alpha158("QTLU10")
def qtlu10(df: pd.DataFrame) -> pd.Series:
    """10-period 80th percentile return."""
    return df["close"].pct_change().rolling(10).quantile(0.8)


@_register_alpha158("QTLD10")
def qtld10(df: pd.DataFrame) -> pd.Series:
    """10-period 20th percentile return."""
    return df["close"].pct_change().rolling(10).quantile(0.2)


@_register_alpha158("QTLU20")
def qtlu20(df: pd.DataFrame) -> pd.Series:
    """20-period 80th percentile return."""
    return df["close"].pct_change().rolling(20).quantile(0.8)


# =============================================================================
# Rank/Position Factors (14 factors)
# =============================================================================

@_register_alpha158("RANK5")
def rank5(df: pd.DataFrame) -> pd.Series:
    """Rank of close within 5-period window."""
    return df["close"].rolling(5).apply(
        _window_percentile_rank,
        raw=False,
    )


@_register_alpha158("RANK10")
def rank10(df: pd.DataFrame) -> pd.Series:
    """Rank of close within 10-period window."""
    return df["close"].rolling(10).apply(
        _window_percentile_rank,
        raw=False,
    )


@_register_alpha158("RANK20")
def rank20(df: pd.DataFrame) -> pd.Series:
    """Rank of close within 20-period window."""
    return df["close"].rolling(20).apply(
        _window_percentile_rank,
        raw=False,
    )


# Days since high/low
@_register_alpha158("IMAX5")
def imax5(df: pd.DataFrame) -> pd.Series:
    """Days since 5-period high."""
    return df["high"].rolling(5).apply(lambda x: 5 - x.argmax() - 1, raw=True) / 5


@_register_alpha158("IMAX10")
def imax10(df: pd.DataFrame) -> pd.Series:
    """Days since 10-period high."""
    return df["high"].rolling(10).apply(lambda x: 10 - x.argmax() - 1, raw=True) / 10


@_register_alpha158("IMAX20")
def imax20(df: pd.DataFrame) -> pd.Series:
    """Days since 20-period high."""
    return df["high"].rolling(20).apply(lambda x: 20 - x.argmax() - 1, raw=True) / 20


@_register_alpha158("IMIN5")
def imin5(df: pd.DataFrame) -> pd.Series:
    """Days since 5-period low."""
    return df["low"].rolling(5).apply(lambda x: 5 - x.argmin() - 1, raw=True) / 5


@_register_alpha158("IMIN10")
def imin10(df: pd.DataFrame) -> pd.Series:
    """Days since 10-period low."""
    return df["low"].rolling(10).apply(lambda x: 10 - x.argmin() - 1, raw=True) / 10


@_register_alpha158("IMIN20")
def imin20(df: pd.DataFrame) -> pd.Series:
    """Days since 20-period low."""
    return df["low"].rolling(20).apply(lambda x: 20 - x.argmin() - 1, raw=True) / 20


# High-Low Index Diff
@_register_alpha158("IMXD5")
def imxd5(df: pd.DataFrame) -> pd.Series:
    """5-period max-min index difference."""
    return imax5(df) - imin5(df)


@_register_alpha158("IMXD10")
def imxd10(df: pd.DataFrame) -> pd.Series:
    """10-period max-min index difference."""
    return imax10(df) - imin10(df)


@_register_alpha158("IMXD20")
def imxd20(df: pd.DataFrame) -> pd.Series:
    """20-period max-min index difference."""
    return imax20(df) - imin20(df)


# Skewness and Kurtosis
@_register_alpha158("SKEW5")
def skew5(df: pd.DataFrame) -> pd.Series:
    """5-period return skewness."""
    return df["close"].pct_change().rolling(5).skew()


@_register_alpha158("SKEW20")
def skew20(df: pd.DataFrame) -> pd.Series:
    """20-period return skewness."""
    return df["close"].pct_change().rolling(20).skew()


@_register_alpha158("KURT5")
def kurt5(df: pd.DataFrame) -> pd.Series:
    """5-period return kurtosis."""
    return df["close"].pct_change().rolling(5).kurt()


@_register_alpha158("KURT20")
def kurt20(df: pd.DataFrame) -> pd.Series:
    """20-period return kurtosis."""
    return df["close"].pct_change().rolling(20).kurt()


@_register_alpha158("QTLD20")
def qtld20(df: pd.DataFrame) -> pd.Series:
    """20-period 20th percentile return."""
    return df["close"].pct_change().rolling(20).quantile(0.2)


@_register_alpha158("CNTP5")
def cntp5(df: pd.DataFrame) -> pd.Series:
    """5-period positive return count ratio."""
    returns = df["close"].pct_change()
    return returns.rolling(5).apply(lambda x: (x > 0).sum() / len(x), raw=True)


@_register_alpha158("CNTN5")
def cntn5(df: pd.DataFrame) -> pd.Series:
    """5-period negative return count ratio."""
    returns = df["close"].pct_change()
    return returns.rolling(5).apply(lambda x: (x < 0).sum() / len(x), raw=True)


# =============================================================================
# Utility Functions
# =============================================================================

def get_all_alpha158_factors() -> list[str]:
    """Get all registered Alpha158 factor names."""
    return list(ALPHA158_FACTORS.keys())


def compute_alpha158(
    df: pd.DataFrame,
    factor_names: list[str] | None = None,
) -> pd.DataFrame:
    """Compute specified Alpha158 factors.

    Args:
        df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
        factor_names: List of factor names to compute (None = all)

    Returns:
        DataFrame with computed factor values
    """
    if factor_names is None:
        factor_names = list(ALPHA158_FACTORS.keys())

    results = {}
    for name in factor_names:
        if name in ALPHA158_FACTORS:
            try:
                results[name] = ALPHA158_FACTORS[name](df)
            except Exception as e:
                print(f"Warning: Failed to compute {name}: {e}")
                results[name] = pd.Series(np.nan, index=df.index)

    return pd.DataFrame(results)


def get_alpha158_count() -> int:
    """Get number of registered Alpha158 factors."""
    return len(ALPHA158_FACTORS)


def get_alpha158_by_category() -> dict[str, list[str]]:
    """Get Alpha158 factors organized by category."""
    categories = {
        "kline": [],
        "momentum": [],
        "ma": [],
        "volatility": [],
        "volume": [],
        "technical": [],
        "statistical": [],
        "rank": [],
    }

    for name in ALPHA158_FACTORS.keys():
        if name.startswith(("KMID", "KLEN", "KUP", "KLOW", "KSFT")):
            categories["kline"].append(name)
        elif name.startswith(("ROC", "LOGRET", "HLROC")):
            categories["momentum"].append(name)
        elif name.startswith(("MA", "EMA", "WMA")):
            categories["ma"].append(name)
        elif name.startswith(("STD", "ATR", "BB", "VOL")):
            categories["volatility"].append(name)
        elif name.startswith(("VMA", "VSTD", "WVMA", "TURN", "CORR", "OBV", "VWAP")):
            categories["volume"].append(name)
        elif name.startswith(("RSI", "RSV", "WILL", "CCI", "ADX", "MACD", "MFI", "AROON", "KELTNER", "DONCHIAN", "PLUS", "MINUS")):
            categories["technical"].append(name)
        elif name.startswith(("BETA", "RSQR", "RESI", "MAX", "MIN", "QTL")):
            categories["statistical"].append(name)
        elif name.startswith(("RANK", "IMAX", "IMIN", "IMXD", "SKEW")):
            categories["rank"].append(name)

    return categories
