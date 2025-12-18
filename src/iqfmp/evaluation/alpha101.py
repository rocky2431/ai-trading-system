"""Alpha101 Factor Library.

Complete implementation of WorldQuant's 101 Formulaic Alphas.
Reference: "101 Formulaic Alphas" by Zura Kakushadze

These factors are adapted for cryptocurrency markets with 24/7 trading.

Factor Categories:
- Momentum factors (trend following)
- Mean reversion factors (contrarian)
- Volatility factors
- Volume factors
- Correlation factors
- Technical indicators

Note: Some factors require additional data (industry, market cap) which may
not be available for crypto. These are marked with [REQUIRES_EXTENDED_DATA].
"""

from __future__ import annotations

from typing import Callable
import numpy as np
import pandas as pd
from scipy.stats import rankdata


# =============================================================================
# Helper Functions for Alpha Calculation
# =============================================================================

def ts_rank(series: pd.Series, window: int) -> pd.Series:
    """Time-series rank over window."""
    def rank_pct(x):
        if len(x) == 0 or np.all(np.isnan(x)):
            return np.nan
        ranked = rankdata(x, nan_policy='omit')
        return ranked[-1] / len(x[~np.isnan(x)])
    return series.rolling(window).apply(rank_pct, raw=True)


def ts_argmax(series: pd.Series, window: int) -> pd.Series:
    """Index of max value over window (0-indexed from end)."""
    def argmax_func(x):
        if len(x) == 0 or np.all(np.isnan(x)):
            return np.nan
        return np.nanargmax(x)
    return series.rolling(window).apply(argmax_func, raw=True)


def ts_argmin(series: pd.Series, window: int) -> pd.Series:
    """Index of min value over window (0-indexed from end)."""
    def argmin_func(x):
        if len(x) == 0 or np.all(np.isnan(x)):
            return np.nan
        return np.nanargmin(x)
    return series.rolling(window).apply(argmin_func, raw=True)


def ts_decay_linear(series: pd.Series, window: int) -> pd.Series:
    """Linear decay weighted average."""
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(
        lambda x: np.sum(weights[:len(x)] * x) / np.sum(weights[:len(x)]),
        raw=True
    )


def ts_sum(series: pd.Series, window: int) -> pd.Series:
    """Time-series sum."""
    return series.rolling(window).sum()


def ts_product(series: pd.Series, window: int) -> pd.Series:
    """Time-series product."""
    return series.rolling(window).apply(np.prod, raw=True)


def ts_min(series: pd.Series, window: int) -> pd.Series:
    """Time-series minimum."""
    return series.rolling(window).min()


def ts_max(series: pd.Series, window: int) -> pd.Series:
    """Time-series maximum."""
    return series.rolling(window).max()


def ts_delta(series: pd.Series, period: int) -> pd.Series:
    """Difference between current and n-period ago value."""
    return series.diff(period)


def ts_delay(series: pd.Series, period: int) -> pd.Series:
    """Value from n periods ago."""
    return series.shift(period)


def ts_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Rolling correlation."""
    return x.rolling(window).corr(y)


def ts_cov(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Rolling covariance."""
    return x.rolling(window).cov(y)


def ts_stddev(series: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation."""
    return series.rolling(window).std()


def cross_rank(series: pd.Series) -> pd.Series:
    """Cross-sectional rank (for single asset, returns normalized value)."""
    # For single asset, normalize to 0-1
    return (series - series.min()) / (series.max() - series.min() + 1e-10)


def sign(series: pd.Series) -> pd.Series:
    """Sign function."""
    return np.sign(series)


def log(series: pd.Series) -> pd.Series:
    """Natural logarithm (with protection)."""
    return np.log(np.maximum(series, 1e-10))


def abs_val(series: pd.Series) -> pd.Series:
    """Absolute value."""
    return np.abs(series)


def power(series: pd.Series, exp: float) -> pd.Series:
    """Power function."""
    return np.power(np.maximum(series, 0), exp)


# =============================================================================
# Alpha101 Factor Registry
# =============================================================================

ALPHA101_FACTORS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {}


def _register_alpha(name: str):
    """Decorator to register an Alpha101 factor."""
    def decorator(func: Callable[[pd.DataFrame], pd.Series]):
        ALPHA101_FACTORS[name] = func
        return func
    return decorator


# =============================================================================
# Alpha001 - Alpha025
# =============================================================================

@_register_alpha("alpha001")
def alpha001(df: pd.DataFrame) -> pd.Series:
    """Rank of returns sign weighted by volatility."""
    returns = df["close"].pct_change()
    cond = (returns < 0)
    result = ts_stddev(returns, 20)
    result[cond] = ts_stddev(returns, 20)[cond] ** 2
    return cross_rank(ts_argmax(power(result, 2), 5)) - 0.5


@_register_alpha("alpha002")
def alpha002(df: pd.DataFrame) -> pd.Series:
    """Volume-weighted log price correlation."""
    result = -1 * ts_corr(
        cross_rank(ts_delta(log(df["volume"]), 2)),
        cross_rank((df["close"] - df["open"]) / df["open"]),
        6
    )
    return result.fillna(0)


@_register_alpha("alpha003")
def alpha003(df: pd.DataFrame) -> pd.Series:
    """Open-close correlation factor."""
    result = -1 * ts_corr(
        cross_rank(df["open"]),
        cross_rank(df["volume"]),
        10
    )
    return result.fillna(0)


@_register_alpha("alpha004")
def alpha004(df: pd.DataFrame) -> pd.Series:
    """Low price rank momentum."""
    return -1 * ts_rank(cross_rank(df["low"]), 9)


@_register_alpha("alpha005")
def alpha005(df: pd.DataFrame) -> pd.Series:
    """Open-vwap rank difference."""
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    return (cross_rank(df["open"] - ts_sum(vwap, 10) / 10) *
            (-1 * abs_val(cross_rank(df["close"] - vwap))))


@_register_alpha("alpha006")
def alpha006(df: pd.DataFrame) -> pd.Series:
    """Open-volume correlation."""
    return -1 * ts_corr(df["open"], df["volume"], 10)


@_register_alpha("alpha007")
def alpha007(df: pd.DataFrame) -> pd.Series:
    """Volume-price divergence."""
    delta_close = ts_delta(df["close"], 7)
    adv20 = df["volume"].rolling(20).mean()
    condition = (adv20 < df["volume"])
    result = -1 * ts_rank(abs_val(delta_close), 60) * sign(delta_close)
    result[condition] = -1
    return result


@_register_alpha("alpha008")
def alpha008(df: pd.DataFrame) -> pd.Series:
    """Decay linear rank."""
    result = -1 * cross_rank(
        ts_sum(df["open"], 5) * ts_sum(df["close"].pct_change(), 5) -
        ts_delay(ts_sum(df["open"], 5) * ts_sum(df["close"].pct_change(), 5), 10)
    )
    return result


@_register_alpha("alpha009")
def alpha009(df: pd.DataFrame) -> pd.Series:
    """Close price momentum with delay."""
    delta = ts_delta(df["close"], 1)
    cond = (0 < ts_min(delta, 5))
    result = delta
    cond2 = (ts_max(delta, 5) < 0)
    result[cond] = delta[cond]
    result[cond2] = delta[cond2]
    result[~(cond | cond2)] = -1 * delta[~(cond | cond2)]
    return result


@_register_alpha("alpha010")
def alpha010(df: pd.DataFrame) -> pd.Series:
    """Delta close rank with direction."""
    delta = ts_delta(df["close"], 1)
    cond = (0 < ts_min(delta, 4))
    result = delta
    cond2 = (ts_max(delta, 4) < 0)
    result[cond] = delta[cond]
    result[cond2] = delta[cond2]
    result[~(cond | cond2)] = -1 * delta[~(cond | cond2)]
    return cross_rank(result)


@_register_alpha("alpha011")
def alpha011(df: pd.DataFrame) -> pd.Series:
    """VWAP-close rank correlation."""
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    return (cross_rank(ts_max(vwap - df["close"], 3)) +
            cross_rank(ts_min(vwap - df["close"], 3))) * cross_rank(ts_delta(df["volume"], 3))


@_register_alpha("alpha012")
def alpha012(df: pd.DataFrame) -> pd.Series:
    """Volume-weighted close change."""
    return sign(ts_delta(df["volume"], 1)) * (-1 * ts_delta(df["close"], 1))


@_register_alpha("alpha013")
def alpha013(df: pd.DataFrame) -> pd.Series:
    """Volume-close covariance rank."""
    return -1 * cross_rank(ts_cov(cross_rank(df["close"]), cross_rank(df["volume"]), 5))


@_register_alpha("alpha014")
def alpha014(df: pd.DataFrame) -> pd.Series:
    """Returns-volume correlation with delay."""
    returns = df["close"].pct_change()
    return -1 * cross_rank(ts_delta(returns, 3)) * ts_corr(df["open"], df["volume"], 10)


@_register_alpha("alpha015")
def alpha015(df: pd.DataFrame) -> pd.Series:
    """High-volume correlation sum."""
    return -1 * ts_sum(
        cross_rank(ts_corr(cross_rank(df["high"]), cross_rank(df["volume"]), 3)), 3
    )


@_register_alpha("alpha016")
def alpha016(df: pd.DataFrame) -> pd.Series:
    """High-volume covariance rank."""
    return -1 * cross_rank(ts_cov(cross_rank(df["high"]), cross_rank(df["volume"]), 5))


@_register_alpha("alpha017")
def alpha017(df: pd.DataFrame) -> pd.Series:
    """Close price rank with delay."""
    adv20 = df["volume"].rolling(20).mean()
    return (
        -1 * cross_rank(ts_rank(df["close"], 10)) *
        cross_rank(ts_delta(ts_delta(df["close"], 1), 1)) *
        cross_rank(ts_rank(df["volume"] / adv20, 5))
    )


@_register_alpha("alpha018")
def alpha018(df: pd.DataFrame) -> pd.Series:
    """Close-open correlation rank."""
    return -1 * cross_rank(
        ts_stddev(abs_val(df["close"] - df["open"]), 5) +
        (df["close"] - df["open"]) +
        ts_corr(df["close"], df["open"], 10)
    )


@_register_alpha("alpha019")
def alpha019(df: pd.DataFrame) -> pd.Series:
    """Returns sign with delay."""
    returns_7d = df["close"].pct_change(7)
    return -1 * sign(returns_7d + ts_delta(df["close"], 7)) * (1 + cross_rank(1 + ts_sum(returns_7d, 250)))


@_register_alpha("alpha020")
def alpha020(df: pd.DataFrame) -> pd.Series:
    """Open-high-low-close rank."""
    return (
        -1 * cross_rank(df["open"] - ts_delay(df["high"], 1)) *
        cross_rank(df["open"] - ts_delay(df["close"], 1)) *
        cross_rank(df["open"] - ts_delay(df["low"], 1))
    )


@_register_alpha("alpha021")
def alpha021(df: pd.DataFrame) -> pd.Series:
    """Volume and close mean deviation."""
    sma8 = df["close"].rolling(8).mean()
    std8 = df["close"].rolling(8).std()
    sma2 = df["volume"].rolling(2).mean()
    cond1 = (sma8 + std8 < df["close"].rolling(2).mean())
    cond2 = (df["close"].rolling(2).mean() < sma8 - std8)
    cond3 = (df["volume"] / df["volume"].rolling(20).mean() >= 1)
    result = pd.Series(-1, index=df.index)
    result[cond1 | cond2] = 1
    result[cond3] = 1
    return result


@_register_alpha("alpha022")
def alpha022(df: pd.DataFrame) -> pd.Series:
    """High correlation with volume decay."""
    return -1 * ts_delta(
        ts_corr(df["high"], df["volume"], 5), 5
    ) * cross_rank(ts_stddev(df["close"], 20))


@_register_alpha("alpha023")
def alpha023(df: pd.DataFrame) -> pd.Series:
    """High price delta conditional."""
    sma20 = df["high"].rolling(20).mean()
    cond = (sma20 < df["high"])
    result = -1 * ts_delta(df["high"], 2)
    result[~cond] = 0
    return result


@_register_alpha("alpha024")
def alpha024(df: pd.DataFrame) -> pd.Series:
    """Close delta with SMA condition."""
    sma100 = df["close"].rolling(100).mean()
    delta_sma100 = ts_delta(sma100, 100) / 100
    cond = (delta_sma100 < df["close"] - sma100)
    result = -1 * ts_delta(df["close"], 3)
    result[cond] = df["close"][cond] - ts_min(df["close"], 100)[cond]
    return result


@_register_alpha("alpha025")
def alpha025(df: pd.DataFrame) -> pd.Series:
    """Returns rank with volume and ADV."""
    returns = df["close"].pct_change()
    adv20 = df["volume"].rolling(20).mean()
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    return cross_rank(
        -1 * returns * adv20 * vwap * (df["high"] - df["close"])
    )


# =============================================================================
# Alpha026 - Alpha050
# =============================================================================

@_register_alpha("alpha026")
def alpha026(df: pd.DataFrame) -> pd.Series:
    """High-volume time series max correlation."""
    return -1 * ts_max(ts_corr(ts_rank(df["volume"], 5), ts_rank(df["high"], 5), 5), 3)


@_register_alpha("alpha027")
def alpha027(df: pd.DataFrame) -> pd.Series:
    """Volume rank with correlation decay."""
    cond = (0.5 < cross_rank(
        ts_sum(ts_corr(cross_rank(df["volume"]), cross_rank(df["close"]), 6), 2) / 2
    ))
    result = -1 * pd.Series(1, index=df.index)
    result[cond] = 1
    return result


@_register_alpha("alpha028")
def alpha028(df: pd.DataFrame) -> pd.Series:
    """ADV20 and correlation product."""
    adv20 = df["volume"].rolling(20).mean()
    return cross_rank(
        ts_corr(adv20, df["low"], 5) +
        ((df["high"] + df["low"]) / 2 - df["close"])
    )


@_register_alpha("alpha029")
def alpha029(df: pd.DataFrame) -> pd.Series:
    """Returns rank product."""
    returns = df["close"].pct_change()
    ts_returns_sum = ts_sum(returns, 5)
    ts_rank_vol = ts_rank(ts_delay(-1 * returns, 6), 5)
    return (
        ts_min(cross_rank(cross_rank(ts_rank(ts_returns_sum, 2))), 5) +
        ts_rank(ts_delay(-1 * returns, 6), 5)
    )


@_register_alpha("alpha030")
def alpha030(df: pd.DataFrame) -> pd.Series:
    """Sign and rank product."""
    close_diff = df["close"] - ts_delay(df["close"], 1)
    sign_diff = sign(close_diff)
    sign_delay = sign(ts_delay(close_diff, 1))
    sign_delay2 = sign(ts_delay(close_diff, 2))
    return (
        (1 - cross_rank(sign_diff + sign_delay + sign_delay2)) *
        ts_sum(df["volume"], 5)
    ) / ts_sum(df["volume"], 20)


@_register_alpha("alpha031")
def alpha031(df: pd.DataFrame) -> pd.Series:
    """Decay linear rank product."""
    adv20 = df["volume"].rolling(20).mean()
    return (
        cross_rank(ts_decay_linear(ts_rank(ts_rank(df["close"], 10), 10), 10)) +
        cross_rank(ts_delta(ts_delta(df["close"], 3), 3)) +
        sign(ts_delta(adv20, 5))
    )


@_register_alpha("alpha032")
def alpha032(df: pd.DataFrame) -> pd.Series:
    """Close-volume correlation scaled."""
    return (
        cross_rank(ts_sum(df["close"], 7) / 7) -
        cross_rank(df["volume"]) +
        cross_rank(ts_corr(df["close"], df["volume"], 6))
    ) * cross_rank(ts_sum(df["close"], 20) / 20)


@_register_alpha("alpha033")
def alpha033(df: pd.DataFrame) -> pd.Series:
    """Open-close ratio rank."""
    return cross_rank(-1 + (df["open"] / df["close"]))


@_register_alpha("alpha034")
def alpha034(df: pd.DataFrame) -> pd.Series:
    """Returns stddev and close ratio."""
    returns = df["close"].pct_change()
    return cross_rank(
        (1 - cross_rank(ts_stddev(returns, 2) / ts_stddev(returns, 5))) +
        (1 - cross_rank(ts_delta(df["close"], 1)))
    )


@_register_alpha("alpha035")
def alpha035(df: pd.DataFrame) -> pd.Series:
    """Volume-returns rank product."""
    returns = df["close"].pct_change()
    return (
        ts_rank(df["volume"], 32) *
        (1 - ts_rank(df["close"] + df["high"] - df["low"], 16)) *
        (1 - ts_rank(returns, 32))
    )


@_register_alpha("alpha036")
def alpha036(df: pd.DataFrame) -> pd.Series:
    """Correlation sum with decay."""
    adv20 = df["volume"].rolling(20).mean()
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    return (
        2.21 * cross_rank(ts_corr(df["close"] - df["open"], ts_delay(df["volume"], 1), 15)) +
        0.7 * cross_rank(df["open"] - df["close"]) +
        0.73 * cross_rank(ts_rank(ts_delay(-1 * (df["close"].pct_change()), 6), 5)) +
        cross_rank(abs_val(ts_corr(vwap, adv20, 6))) +
        0.6 * cross_rank((ts_sum(df["close"], 200) / 200 - df["open"]) * (df["close"] - df["open"]))
    )


@_register_alpha("alpha037")
def alpha037(df: pd.DataFrame) -> pd.Series:
    """Open-close correlation rank."""
    return cross_rank(ts_corr(ts_delay(df["open"] - df["close"], 1), df["close"], 200)) + cross_rank(df["open"] - df["close"])


@_register_alpha("alpha038")
def alpha038(df: pd.DataFrame) -> pd.Series:
    """High time series rank."""
    return -1 * cross_rank(ts_rank(df["close"], 10)) * cross_rank(df["close"] / df["open"])


@_register_alpha("alpha039")
def alpha039(df: pd.DataFrame) -> pd.Series:
    """Volume-price decay rank."""
    adv20 = df["volume"].rolling(20).mean()
    returns = df["close"].pct_change()
    return (
        -1 * cross_rank(ts_delta(df["close"], 7) * (1 - cross_rank(ts_decay_linear(df["volume"] / adv20, 9)))) *
        (1 + cross_rank(ts_sum(returns, 250)))
    )


@_register_alpha("alpha040")
def alpha040(df: pd.DataFrame) -> pd.Series:
    """High-volume stddev product."""
    return -1 * cross_rank(ts_stddev(df["high"], 10)) * ts_corr(df["high"], df["volume"], 10)


@_register_alpha("alpha041")
def alpha041(df: pd.DataFrame) -> pd.Series:
    """High-low power factor."""
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    return power(df["high"] * df["low"], 0.5) - vwap


@_register_alpha("alpha042")
def alpha042(df: pd.DataFrame) -> pd.Series:
    """Close-vwap rank difference."""
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    return cross_rank(vwap - df["close"]) / cross_rank(vwap + df["close"])


@_register_alpha("alpha043")
def alpha043(df: pd.DataFrame) -> pd.Series:
    """Volume delta rank."""
    adv20 = df["volume"].rolling(20).mean()
    return ts_rank(df["volume"] / adv20, 20) * ts_rank(-1 * ts_delta(df["close"], 7), 8)


@_register_alpha("alpha044")
def alpha044(df: pd.DataFrame) -> pd.Series:
    """High-volume rank correlation."""
    return -1 * ts_corr(df["high"], cross_rank(df["volume"]), 5)


@_register_alpha("alpha045")
def alpha045(df: pd.DataFrame) -> pd.Series:
    """Close-volume delay correlation."""
    adv20 = df["volume"].rolling(20).mean()
    return -1 * (
        cross_rank(ts_sum(ts_delay(df["close"], 5), 20) / 20) *
        ts_corr(df["close"], df["volume"], 2) *
        cross_rank(ts_corr(ts_sum(df["close"], 5), ts_sum(df["close"], 20), 2))
    )


@_register_alpha("alpha046")
def alpha046(df: pd.DataFrame) -> pd.Series:
    """Close delta condition factor."""
    cond = (0.25 < ts_delay(ts_delta(df["close"], 10), 10) / 10 - ts_delta(df["close"], 10) / 10)
    result = -1 * ts_delta(df["close"], 1)
    result[cond] = 1
    cond2 = (ts_delay(ts_delta(df["close"], 10), 10) / 10 - ts_delta(df["close"], 10) / 10 < 0)
    result[cond2] = -1 * ts_delta(df["close"], 1)[cond2]
    return result


@_register_alpha("alpha047")
def alpha047(df: pd.DataFrame) -> pd.Series:
    """High rank volume product."""
    adv20 = df["volume"].rolling(20).mean()
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    return (
        cross_rank((1 / df["close"]) * df["volume"] / adv20) *
        df["high"] *
        cross_rank(df["high"] - df["close"]) /
        (ts_sum(df["high"], 5) / 5) -
        cross_rank(vwap - ts_delay(vwap, 5))
    )


@_register_alpha("alpha048")
def alpha048(df: pd.DataFrame) -> pd.Series:
    """[REQUIRES_EXTENDED_DATA] Industry neutralized correlation."""
    # Simplified version without industry data
    return -1 * ts_corr(df["close"], df["volume"], 20)


@_register_alpha("alpha049")
def alpha049(df: pd.DataFrame) -> pd.Series:
    """Close delta condition v2."""
    delta_1 = ts_delta(ts_delay(df["close"], 10), 10) / 10
    delta_2 = ts_delta(df["close"], 10) / 10
    cond = (delta_1 - delta_2 < -0.1 * df["close"])
    result = 1 * pd.Series(1, index=df.index)
    result[cond] = -1 * ts_delta(df["close"], 1)[cond]
    return result


@_register_alpha("alpha050")
def alpha050(df: pd.DataFrame) -> pd.Series:
    """Volume-close max correlation."""
    return -1 * ts_max(cross_rank(ts_corr(cross_rank(df["volume"]), cross_rank(df["close"]), 5)), 5)


# =============================================================================
# Alpha051 - Alpha075
# =============================================================================

@_register_alpha("alpha051")
def alpha051(df: pd.DataFrame) -> pd.Series:
    """Close delta condition v3."""
    delta = ts_delta(ts_delay(df["close"], 10), 10) / 10 - ts_delta(df["close"], 10) / 10
    cond = (delta < -0.05 * df["close"])
    result = 1 * pd.Series(1, index=df.index)
    result[cond] = -1 * ts_delta(df["close"], 1)[cond]
    return result


@_register_alpha("alpha052")
def alpha052(df: pd.DataFrame) -> pd.Series:
    """Low delta and returns sum."""
    returns = df["close"].pct_change()
    return (
        ts_delta(ts_min(df["low"], 5), 5) *
        cross_rank(ts_sum(returns, 240) - ts_sum(returns, 20)) / 220 *
        ts_rank(df["volume"], 5)
    )


@_register_alpha("alpha053")
def alpha053(df: pd.DataFrame) -> pd.Series:
    """High-low difference delta."""
    return -1 * ts_delta((df["close"] - df["low"]) - (df["high"] - df["close"]) / (df["close"] - df["low"] + 1e-10), 9)


@_register_alpha("alpha054")
def alpha054(df: pd.DataFrame) -> pd.Series:
    """Low-close power rank."""
    return -1 * (df["low"] - df["close"]) * power(df["open"], 5) / ((df["low"] - df["high"] + 1e-10) * power(df["close"], 5))


@_register_alpha("alpha055")
def alpha055(df: pd.DataFrame) -> pd.Series:
    """High-low-close correlation."""
    hl_ratio = (df["close"] - ts_min(df["low"], 12)) / (ts_max(df["high"], 12) - ts_min(df["low"], 12) + 1e-10)
    return -1 * ts_corr(cross_rank(hl_ratio), cross_rank(df["volume"]), 6)


@_register_alpha("alpha056")
def alpha056(df: pd.DataFrame) -> pd.Series:
    """Returns-cap rank product."""
    # [REQUIRES_EXTENDED_DATA] - using volume as proxy for cap
    returns = df["close"].pct_change()
    return -1 * cross_rank(ts_sum(returns, 10) / ts_sum(ts_sum(returns, 2), 3)) * cross_rank(returns * df["volume"])


@_register_alpha("alpha057")
def alpha057(df: pd.DataFrame) -> pd.Series:
    """Close-VWAP ratio SMA."""
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    return -1 * (df["close"] - vwap) / ts_decay_linear(cross_rank(ts_argmax(df["close"], 30)), 2)


@_register_alpha("alpha058")
def alpha058(df: pd.DataFrame) -> pd.Series:
    """[REQUIRES_EXTENDED_DATA] Industry-adjusted correlation."""
    # Simplified version
    return -1 * ts_corr(df["close"], df["volume"], 20) * ts_rank(df["volume"], 10)


@_register_alpha("alpha059")
def alpha059(df: pd.DataFrame) -> pd.Series:
    """[REQUIRES_EXTENDED_DATA] Industry-adjusted factor."""
    # Simplified version
    return -1 * ts_corr(df["close"], df["volume"], 10)


@_register_alpha("alpha060")
def alpha060(df: pd.DataFrame) -> pd.Series:
    """Close-low-high rank scale."""
    return -1 * (2 * df["close"] - df["low"] - df["high"]) / (df["high"] - df["low"] + 1e-10) * df["volume"]


@_register_alpha("alpha061")
def alpha061(df: pd.DataFrame) -> pd.Series:
    """VWAP delta rank product."""
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    adv180 = df["volume"].rolling(180).mean()
    return cross_rank(vwap - ts_min(vwap, 16)) < cross_rank(ts_corr(vwap, adv180, 18))


@_register_alpha("alpha062")
def alpha062(df: pd.DataFrame) -> pd.Series:
    """Volume-VWAP correlation rank."""
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    adv20 = df["volume"].rolling(20).mean()
    cond = (cross_rank(ts_corr(vwap, ts_sum(adv20, 22), 10)) < cross_rank(vwap - df["close"]))
    result = -1 * pd.Series(1, index=df.index)
    result[cond] = 0
    return result


@_register_alpha("alpha063")
def alpha063(df: pd.DataFrame) -> pd.Series:
    """[REQUIRES_EXTENDED_DATA] Industry neutralized decay."""
    # Simplified version
    adv180 = df["volume"].rolling(180).mean()
    return (
        cross_rank(ts_decay_linear(ts_delta(df["close"], 1), 2)) -
        cross_rank(ts_decay_linear(ts_corr(df["close"], df["volume"], 2), 3)) +
        cross_rank(ts_decay_linear(ts_corr(adv180, df["close"], 13), 4))
    )


@_register_alpha("alpha064")
def alpha064(df: pd.DataFrame) -> pd.Series:
    """ADV120 weighted rank."""
    adv120 = df["volume"].rolling(120).mean()
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    return (
        cross_rank(ts_corr(ts_sum(df["open"] * 0.178404 + df["low"] * (1 - 0.178404), 13),
                           ts_sum(adv120, 13), 17)) <
        cross_rank(ts_delta(vwap - df["close"], 3))
    )


@_register_alpha("alpha065")
def alpha065(df: pd.DataFrame) -> pd.Series:
    """Open-VWAP correlation rank."""
    adv60 = df["volume"].rolling(60).mean()
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    return (
        cross_rank(ts_corr((df["open"] * 0.00817205 + vwap * (1 - 0.00817205)), ts_sum(adv60, 9), 6)) <
        cross_rank(df["open"] - ts_min(df["open"], 14))
    )


@_register_alpha("alpha066")
def alpha066(df: pd.DataFrame) -> pd.Series:
    """VWAP decay linear rank."""
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    return -1 * cross_rank(ts_decay_linear(ts_delta(vwap, 4), 7))


@_register_alpha("alpha067")
def alpha067(df: pd.DataFrame) -> pd.Series:
    """[REQUIRES_EXTENDED_DATA] Industry rank factor."""
    # Simplified version
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    adv20 = df["volume"].rolling(20).mean()
    return (
        power(cross_rank(df["high"] - ts_min(df["high"], 2)), cross_rank(ts_corr(vwap, adv20, 6))) *
        -1
    )


@_register_alpha("alpha068")
def alpha068(df: pd.DataFrame) -> pd.Series:
    """High-low volume correlation."""
    adv15 = df["volume"].rolling(15).mean()
    return (
        ts_rank(ts_corr(cross_rank(df["high"]), cross_rank(adv15), 9), 14) <
        cross_rank(ts_delta(df["close"] * 0.518371 + df["low"] * (1 - 0.518371), 1))
    )


@_register_alpha("alpha069")
def alpha069(df: pd.DataFrame) -> pd.Series:
    """[REQUIRES_EXTENDED_DATA] Industry adjusted VWAP."""
    # Simplified version
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    adv20 = df["volume"].rolling(20).mean()
    return (
        power(cross_rank(ts_max(ts_delta(vwap, 3), 5)), ts_rank(ts_corr(df["close"], adv20, 4), 19)) *
        -1
    )


@_register_alpha("alpha070")
def alpha070(df: pd.DataFrame) -> pd.Series:
    """[REQUIRES_EXTENDED_DATA] Industry neutralized factor."""
    # Simplified version
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    return power(cross_rank(ts_delta(vwap, 2)), ts_rank(ts_corr(df["close"], df["volume"], 18), 8))


@_register_alpha("alpha071")
def alpha071(df: pd.DataFrame) -> pd.Series:
    """Decay linear max factor."""
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    adv180 = df["volume"].rolling(180).mean()
    p1 = ts_rank(ts_decay_linear(ts_corr(ts_rank(df["close"], 4), ts_rank(adv180, 8), 8), 16), 8)
    p2 = ts_rank(ts_decay_linear(ts_corr(vwap, ts_rank(df["low"], 20), 7), 3), 7)
    cond = (p1 > p2)
    result = ts_rank(ts_decay_linear(cross_rank(ts_decay_linear(ts_corr(df["low"], df["volume"], 8), 11)), 18), 13)
    result[cond] = p1[cond]
    return result


@_register_alpha("alpha072")
def alpha072(df: pd.DataFrame) -> pd.Series:
    """ADV40 VWAP correlation."""
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    adv40 = df["volume"].rolling(40).mean()
    return (
        cross_rank(ts_decay_linear(ts_corr(df["high"] * 0.335 + df["low"] * 0.665, adv40, 9), 11)) /
        (cross_rank(ts_decay_linear(ts_corr(ts_rank(vwap, 4), ts_rank(df["volume"], 19), 7), 3)) + 1e-10)
    )


@_register_alpha("alpha073")
def alpha073(df: pd.DataFrame) -> pd.Series:
    """VWAP decay max factor."""
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    p1 = cross_rank(ts_decay_linear(ts_delta(vwap, 5), 3))
    p2 = ts_rank(ts_decay_linear((ts_delta(df["open"] * 0.147155 + df["low"] * (1 - 0.147155), 2) /
                                   (df["open"] * 0.147155 + df["low"] * (1 - 0.147155) + 1e-10) * -1), 3), 17)
    cond = (p1 > p2)
    result = p2
    result[cond] = p1[cond]
    return -1 * result


@_register_alpha("alpha074")
def alpha074(df: pd.DataFrame) -> pd.Series:
    """High-volume correlation rank product."""
    adv30 = df["volume"].rolling(30).mean()
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    return (
        cross_rank(ts_corr(df["close"], ts_sum(adv30, 37), 15)) <
        cross_rank(ts_corr(cross_rank(df["high"] * 0.0261661 + vwap * (1 - 0.0261661)), cross_rank(df["volume"]), 11))
    ) * -1


@_register_alpha("alpha075")
def alpha075(df: pd.DataFrame) -> pd.Series:
    """Low-VWAP correlation rank."""
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    adv50 = df["volume"].rolling(50).mean()
    return cross_rank(ts_corr(vwap, df["volume"], 4)) < cross_rank(ts_corr(cross_rank(df["low"]), cross_rank(adv50), 12))


# =============================================================================
# Alpha076 - Alpha101
# =============================================================================

@_register_alpha("alpha076")
def alpha076(df: pd.DataFrame) -> pd.Series:
    """[REQUIRES_EXTENDED_DATA] Industry neutralized max."""
    # Simplified version
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    adv81 = df["volume"].rolling(81).mean()
    return (
        cross_rank(ts_decay_linear(ts_delta(vwap, 2), 12)) <
        ts_rank(ts_decay_linear(ts_rank(ts_corr(df["low"], adv81, 8), 20), 17), 19)
    ) * -1


@_register_alpha("alpha077")
def alpha077(df: pd.DataFrame) -> pd.Series:
    """High-low VWAP decay."""
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    adv40 = df["volume"].rolling(40).mean()
    p1 = cross_rank(ts_decay_linear((df["high"] + df["low"]) / 2 + df["high"] - vwap - df["high"], 20))
    p2 = cross_rank(ts_decay_linear(ts_corr(df["high"] + df["low"]) / 2, adv40, 3), 6)
    cond = (p1 < p2)
    result = p1
    result[~cond] = p2[~cond]
    return result


@_register_alpha("alpha078")
def alpha078(df: pd.DataFrame) -> pd.Series:
    """Low-volume correlation power."""
    adv40 = df["volume"].rolling(40).mean()
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    return power(
        cross_rank(ts_corr(ts_sum((df["low"] * 0.352233 + vwap * (1 - 0.352233)), 20), ts_sum(adv40, 20), 7)),
        cross_rank(ts_corr(cross_rank(vwap), cross_rank(df["volume"]), 6))
    )


@_register_alpha("alpha079")
def alpha079(df: pd.DataFrame) -> pd.Series:
    """[REQUIRES_EXTENDED_DATA] Industry delta rank."""
    # Simplified version
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    adv150 = df["volume"].rolling(150).mean()
    return (
        cross_rank(ts_delta(vwap, 5)) <
        ts_rank(ts_corr(df["open"], adv150, 15), 21)
    ) * -1


@_register_alpha("alpha080")
def alpha080(df: pd.DataFrame) -> pd.Series:
    """[REQUIRES_EXTENDED_DATA] Industry correlation."""
    # Simplified version
    adv10 = df["volume"].rolling(10).mean()
    return (
        power(cross_rank(sign(ts_delta(df["close"], 1)) + sign(ts_delay(ts_delta(df["close"], 1), 1)) +
              sign(ts_delay(ts_delta(df["close"], 1), 2))),
              ts_rank(ts_corr(df["high"], adv10, 5), 49)) *
        -1
    )


@_register_alpha("alpha081")
def alpha081(df: pd.DataFrame) -> pd.Series:
    """ADV10 VWAP decay."""
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    adv10 = df["volume"].rolling(10).mean()
    cond = (cross_rank(log(ts_product(cross_rank(cross_rank(ts_corr(vwap, ts_sum(adv10, 50), 8))), 15))) <
            cross_rank(ts_corr(cross_rank(vwap), cross_rank(df["volume"]), 5)))
    result = 1 * pd.Series(1, index=df.index)
    result[cond] = -1
    return result


@_register_alpha("alpha082")
def alpha082(df: pd.DataFrame) -> pd.Series:
    """[REQUIRES_EXTENDED_DATA] Sector neutralized."""
    # Simplified version
    return -1 * ts_rank(ts_delta(df["open"], 1), 10)


@_register_alpha("alpha083")
def alpha083(df: pd.DataFrame) -> pd.Series:
    """High-low-close shadow factor."""
    return (df["high"] - df["low"]) / (df["close"].rolling(5).mean() + 1e-10) * ts_rank(df["volume"], 5)


@_register_alpha("alpha084")
def alpha084(df: pd.DataFrame) -> pd.Series:
    """Close-VWAP sign rank."""
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    return power(ts_rank(vwap - ts_max(vwap, 15), 21), ts_delta(df["close"], 5))


@_register_alpha("alpha085")
def alpha085(df: pd.DataFrame) -> pd.Series:
    """High-low ADV correlation."""
    adv30 = df["volume"].rolling(30).mean()
    return power(
        cross_rank(ts_corr(df["high"] * 0.876703 + df["close"] * (1 - 0.876703), adv30, 10)),
        cross_rank(ts_corr(ts_rank(df["high"] * 0.329 + df["low"] * 0.671, 4), ts_rank(df["volume"], 5), 7))
    )


@_register_alpha("alpha086")
def alpha086(df: pd.DataFrame) -> pd.Series:
    """ADV20 VWAP correlation condition."""
    adv20 = df["volume"].rolling(20).mean()
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    cond = (
        ts_rank(ts_corr(df["close"], ts_sum(adv20, 15), 6), 20) <
        cross_rank(df["open"] + df["close"] - vwap - df["open"])
    )
    result = -1 * pd.Series(1, index=df.index)
    result[cond] = 1
    return result


@_register_alpha("alpha087")
def alpha087(df: pd.DataFrame) -> pd.Series:
    """ADV81 decay correlation."""
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    adv81 = df["volume"].rolling(81).mean()
    p1 = cross_rank(ts_decay_linear(ts_delta(vwap, 4), 7))
    p2 = ts_rank(ts_decay_linear((df["low"] * 0.388 + vwap * (1 - 0.388) - df["open"]) / (df["open"] - (df["high"] + df["low"]) / 2 + 1e-10), 11), 7)
    cond = (p1 > p2)
    result = p2
    result[cond] = p1[cond]
    return -1 * result


@_register_alpha("alpha088")
def alpha088(df: pd.DataFrame) -> pd.Series:
    """Open-high-low-close rank decay."""
    return cross_rank(
        ts_decay_linear(
            (cross_rank(df["open"]) + cross_rank(df["low"])) -
            (cross_rank(df["high"]) + cross_rank(df["close"])),
            8
        )
    )


@_register_alpha("alpha089")
def alpha089(df: pd.DataFrame) -> pd.Series:
    """ADV10 low correlation decay."""
    adv10 = df["volume"].rolling(10).mean()
    p1 = ts_rank(ts_decay_linear(ts_corr(df["low"] * 0.967285 + df["low"] * (1 - 0.967285), adv10, 7), 6), 4)
    p2 = ts_rank(ts_decay_linear(ts_delta(df["close"], 3), 10), 15)
    cond = (p1 < p2)
    result = p1
    result[~cond] = p2[~cond]
    return result


@_register_alpha("alpha090")
def alpha090(df: pd.DataFrame) -> pd.Series:
    """[REQUIRES_EXTENDED_DATA] Industry delta."""
    # Simplified version
    adv40 = df["volume"].rolling(40).mean()
    return (cross_rank(ts_corr(df["close"], df["volume"], 5)) < cross_rank((df["high"] + df["low"]) / 2 - df["close"])) * -1


@_register_alpha("alpha091")
def alpha091(df: pd.DataFrame) -> pd.Series:
    """ADV30 close correlation decay."""
    adv30 = df["volume"].rolling(30).mean()
    p1 = ts_rank(ts_decay_linear(ts_decay_linear(ts_corr(df["close"], df["volume"], 10), 16), 4), 5)
    p2 = cross_rank(ts_decay_linear(ts_corr(df["close"], adv30, 4), 3))
    cond = (p1 < p2)
    result = p1
    result[~cond] = p2[~cond]
    return -1 * result


@_register_alpha("alpha092")
def alpha092(df: pd.DataFrame) -> pd.Series:
    """ADV30 high-low correlation."""
    adv30 = df["volume"].rolling(30).mean()
    p1 = ts_rank(ts_decay_linear((df["high"] + df["low"]) / 2 + df["close"] - df["low"] - df["high"], 15), 19)
    p2 = ts_rank(ts_decay_linear(ts_corr(cross_rank(df["low"]), cross_rank(adv30), 8), 7), 7)
    cond = (p1 > p2)
    result = p2
    result[cond] = p1[cond]
    return result


@_register_alpha("alpha093")
def alpha093(df: pd.DataFrame) -> pd.Series:
    """ADV81 open correlation decay."""
    adv81 = df["volume"].rolling(81).mean()
    return ts_rank(ts_decay_linear(ts_corr(df["close"], adv81, 17), 20), 7) / (
        cross_rank(ts_decay_linear(ts_delta((df["close"] * 0.524 + df["low"] * 0.476), 3), 16)) + 1e-10
    )


@_register_alpha("alpha094")
def alpha094(df: pd.DataFrame) -> pd.Series:
    """ADV60 VWAP rank."""
    adv60 = df["volume"].rolling(60).mean()
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    return (
        cross_rank(vwap - ts_min(vwap, 12)) <
        ts_rank(ts_corr(ts_rank(vwap, 20), ts_rank(adv60, 4), 18), 3)
    ) * -1


@_register_alpha("alpha095")
def alpha095(df: pd.DataFrame) -> pd.Series:
    """ADV40 high correlation."""
    adv40 = df["volume"].rolling(40).mean()
    return cross_rank(df["open"] - ts_min(df["open"], 12)) < ts_rank(cross_rank(ts_corr(ts_sum((df["high"] + df["low"]) / 2, 19), ts_sum(adv40, 19), 13)), 12)


@_register_alpha("alpha096")
def alpha096(df: pd.DataFrame) -> pd.Series:
    """ADV60 VWAP decay max."""
    adv60 = df["volume"].rolling(60).mean()
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    p1 = ts_rank(ts_decay_linear(ts_corr(cross_rank(vwap), cross_rank(df["volume"]), 4), 4), 8)
    p2 = ts_rank(ts_decay_linear(ts_argmax(ts_corr(ts_rank(df["close"], 7), ts_rank(adv60, 4), 4), 13), 14), 13)
    cond = (p1 > p2)
    result = p2
    result[cond] = p1[cond]
    return -1 * result


@_register_alpha("alpha097")
def alpha097(df: pd.DataFrame) -> pd.Series:
    """[REQUIRES_EXTENDED_DATA] Industry volume correlation."""
    # Simplified version
    adv60 = df["volume"].rolling(60).mean()
    return (
        cross_rank(ts_decay_linear(ts_delta(df["close"], 1), 4)) -
        ts_rank(ts_decay_linear(ts_corr(df["close"], adv60, 5), 16), 8)
    )


@_register_alpha("alpha098")
def alpha098(df: pd.DataFrame) -> pd.Series:
    """ADV5 VWAP sum decay."""
    adv5 = df["volume"].rolling(5).mean()
    adv15 = df["volume"].rolling(15).mean()
    vwap = (df["close"] * df["volume"]).rolling(10).sum() / df["volume"].rolling(10).sum()
    return (
        cross_rank(ts_decay_linear(ts_corr(vwap, ts_sum(adv5, 26), 5), 7)) -
        cross_rank(ts_decay_linear(ts_rank(ts_argmin(ts_corr(cross_rank(df["open"]), cross_rank(adv15), 21), 9), 7), 8))
    )


@_register_alpha("alpha099")
def alpha099(df: pd.DataFrame) -> pd.Series:
    """ADV60 volume correlation cond."""
    adv60 = df["volume"].rolling(60).mean()
    cond = (
        cross_rank(ts_corr(ts_sum((df["high"] + df["low"]) / 2, 20), ts_sum(adv60, 20), 9)) <
        cross_rank(ts_corr(df["low"], df["volume"], 6))
    )
    result = -1 * pd.Series(1, index=df.index)
    result[cond] = 1
    return result


@_register_alpha("alpha100")
def alpha100(df: pd.DataFrame) -> pd.Series:
    """[REQUIRES_EXTENDED_DATA] Industry neutralized factor."""
    # Simplified version
    adv20 = df["volume"].rolling(20).mean()
    return (
        -1 * (1 - cross_rank(100 * (1 - (df["close"] - ts_max(df["close"], 20)) / ts_max(df["close"], 20) + 1e-10))) *
        cross_rank(ts_corr(df["close"], adv20, 5))
    )


@_register_alpha("alpha101")
def alpha101(df: pd.DataFrame) -> pd.Series:
    """Final alpha - open-high-low-close composite."""
    return (df["close"] - df["open"]) / ((df["high"] - df["low"]) + 0.001)


# =============================================================================
# Utility Functions
# =============================================================================

def get_all_alpha101_factors() -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    """Get all registered Alpha101 factors."""
    return ALPHA101_FACTORS.copy()


def compute_alpha101(df: pd.DataFrame, alpha_names: list[str] | None = None) -> pd.DataFrame:
    """Compute specified Alpha101 factors.

    Args:
        df: DataFrame with OHLCV data
        alpha_names: List of alpha names to compute (None = all)

    Returns:
        DataFrame with computed alpha values
    """
    if alpha_names is None:
        alpha_names = list(ALPHA101_FACTORS.keys())

    results = {}
    for name in alpha_names:
        if name in ALPHA101_FACTORS:
            try:
                results[name] = ALPHA101_FACTORS[name](df)
            except Exception as e:
                print(f"Warning: Failed to compute {name}: {e}")
                results[name] = pd.Series(np.nan, index=df.index)

    return pd.DataFrame(results)


def get_alpha101_count() -> int:
    """Get number of registered Alpha101 factors."""
    return len(ALPHA101_FACTORS)
