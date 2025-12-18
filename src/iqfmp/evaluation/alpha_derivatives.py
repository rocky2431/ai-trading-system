"""Crypto Derivative Alpha Factors.

This module provides alpha factors specific to crypto derivatives markets,
utilizing funding rates, open interest, liquidations, and other derivative data.

These factors are designed for perpetual futures trading strategies.
"""

from typing import Callable

import numpy as np
import pandas as pd


# Registry for derivative alpha factors
DERIVATIVE_FACTORS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {}


def _register_derivative(name: str):
    """Decorator to register a derivative factor."""
    def decorator(func: Callable[[pd.DataFrame], pd.Series]):
        DERIVATIVE_FACTORS[name] = func
        return func
    return decorator


# =============================================================================
# Funding Rate Factors
# =============================================================================


@_register_derivative("FUNDING_RATE")
def funding_rate(df: pd.DataFrame) -> pd.Series:
    """Raw funding rate."""
    return df.get("funding_rate", pd.Series(0, index=df.index))


@_register_derivative("FUNDING_RATE_MA8")
def funding_rate_ma8(df: pd.DataFrame) -> pd.Series:
    """8-period moving average of funding rate (typical funding cycle)."""
    fr = df.get("funding_rate", pd.Series(0, index=df.index))
    return fr.rolling(8).mean()


@_register_derivative("FUNDING_RATE_ZSCORE")
def funding_rate_zscore(df: pd.DataFrame) -> pd.Series:
    """Z-score of funding rate (30-period window)."""
    fr = df.get("funding_rate", pd.Series(0, index=df.index))
    mean = fr.rolling(30).mean()
    std = fr.rolling(30).std()
    return (fr - mean) / (std + 1e-10)


@_register_derivative("FUNDING_CUMSUM_8H")
def funding_cumsum_8h(df: pd.DataFrame) -> pd.Series:
    """Cumulative funding over 8 hours (1 funding cycle)."""
    fr = df.get("funding_rate", pd.Series(0, index=df.index))
    return fr.rolling(8).sum()


@_register_derivative("FUNDING_CUMSUM_24H")
def funding_cumsum_24h(df: pd.DataFrame) -> pd.Series:
    """Cumulative funding over 24 hours (3 funding cycles)."""
    fr = df.get("funding_rate", pd.Series(0, index=df.index))
    return fr.rolling(24).sum()


@_register_derivative("FUNDING_MOMENTUM")
def funding_momentum(df: pd.DataFrame) -> pd.Series:
    """Funding rate momentum (current vs 8-period MA)."""
    fr = df.get("funding_rate", pd.Series(0, index=df.index))
    ma = fr.rolling(8).mean()
    return fr - ma


@_register_derivative("FUNDING_REVERSAL")
def funding_reversal(df: pd.DataFrame) -> pd.Series:
    """Funding rate reversal signal (contrarian)."""
    fr = df.get("funding_rate", pd.Series(0, index=df.index))
    zscore = (fr - fr.rolling(30).mean()) / (fr.rolling(30).std() + 1e-10)
    # Signal is negative when funding is extremely high (crowded longs)
    return -zscore.clip(-3, 3) / 3


@_register_derivative("FUNDING_EXTREME")
def funding_extreme(df: pd.DataFrame) -> pd.Series:
    """Binary flag for extreme funding (|zscore| > 2)."""
    fr = df.get("funding_rate", pd.Series(0, index=df.index))
    zscore = (fr - fr.rolling(30).mean()) / (fr.rolling(30).std() + 1e-10)
    return (zscore.abs() > 2).astype(float)


@_register_derivative("FUNDING_POSITIVE_STREAK")
def funding_positive_streak(df: pd.DataFrame) -> pd.Series:
    """Consecutive periods of positive funding."""
    fr = df.get("funding_rate", pd.Series(0, index=df.index))
    positive = (fr > 0).astype(int)
    result = positive.copy()
    for i in range(1, len(positive)):
        if positive.iloc[i] == 1:
            result.iloc[i] = result.iloc[i-1] + 1
        else:
            result.iloc[i] = 0
    return result / 10  # Normalize


@_register_derivative("FUNDING_APR")
def funding_apr(df: pd.DataFrame) -> pd.Series:
    """Annualized funding rate (assuming 3x daily funding)."""
    fr = df.get("funding_rate", pd.Series(0, index=df.index))
    # 3 fundings per day * 365 days
    return fr * 3 * 365


# =============================================================================
# Open Interest Factors
# =============================================================================


@_register_derivative("OI_RAW")
def oi_raw(df: pd.DataFrame) -> pd.Series:
    """Raw open interest (normalized by MA)."""
    oi = df.get("open_interest", pd.Series(0, index=df.index))
    ma = oi.rolling(20).mean()
    return oi / (ma + 1e-10) - 1


@_register_derivative("OI_CHANGE_1H")
def oi_change_1h(df: pd.DataFrame) -> pd.Series:
    """1-hour open interest change."""
    oi = df.get("open_interest", pd.Series(0, index=df.index))
    return oi.pct_change(1)


@_register_derivative("OI_CHANGE_4H")
def oi_change_4h(df: pd.DataFrame) -> pd.Series:
    """4-hour open interest change."""
    oi = df.get("open_interest", pd.Series(0, index=df.index))
    return oi.pct_change(4)


@_register_derivative("OI_CHANGE_24H")
def oi_change_24h(df: pd.DataFrame) -> pd.Series:
    """24-hour open interest change."""
    oi = df.get("open_interest", pd.Series(0, index=df.index))
    return oi.pct_change(24)


@_register_derivative("OI_ZSCORE")
def oi_zscore(df: pd.DataFrame) -> pd.Series:
    """Z-score of open interest (30-period)."""
    oi = df.get("open_interest", pd.Series(0, index=df.index))
    mean = oi.rolling(30).mean()
    std = oi.rolling(30).std()
    return (oi - mean) / (std + 1e-10)


@_register_derivative("OI_PRICE_DIVERGENCE")
def oi_price_divergence(df: pd.DataFrame) -> pd.Series:
    """OI vs price divergence (OI up + price down = bearish, vice versa)."""
    oi = df.get("open_interest", pd.Series(0, index=df.index))
    close = df.get("close", pd.Series(0, index=df.index))
    oi_change = oi.pct_change(4)
    price_change = close.pct_change(4)
    # Positive when OI and price move together, negative when diverging
    return oi_change * price_change * 100


@_register_derivative("OI_ACCELERATION")
def oi_acceleration(df: pd.DataFrame) -> pd.Series:
    """OI momentum acceleration."""
    oi = df.get("open_interest", pd.Series(0, index=df.index))
    change = oi.pct_change(1)
    return change - change.shift(1)


@_register_derivative("OI_VALUE_RATIO")
def oi_value_ratio(df: pd.DataFrame) -> pd.Series:
    """OI value vs contracts ratio (indicates average position size)."""
    oi = df.get("open_interest", pd.Series(0, index=df.index))
    oi_value = df.get("open_interest_value", pd.Series(0, index=df.index))
    return oi_value / (oi + 1e-10)


@_register_derivative("OI_VOLUME_RATIO")
def oi_volume_ratio(df: pd.DataFrame) -> pd.Series:
    """OI to volume ratio (turnover indicator)."""
    oi = df.get("open_interest", pd.Series(0, index=df.index))
    volume = df.get("volume", pd.Series(1, index=df.index))
    return oi / (volume + 1e-10)


# =============================================================================
# Long/Short Ratio Factors
# =============================================================================


@_register_derivative("LS_RATIO")
def ls_ratio(df: pd.DataFrame) -> pd.Series:
    """Raw long/short ratio."""
    return df.get("long_short_ratio", pd.Series(1, index=df.index))


@_register_derivative("LS_RATIO_ZSCORE")
def ls_ratio_zscore(df: pd.DataFrame) -> pd.Series:
    """Z-score of long/short ratio."""
    ls = df.get("long_short_ratio", pd.Series(1, index=df.index))
    mean = ls.rolling(30).mean()
    std = ls.rolling(30).std()
    return (ls - mean) / (std + 1e-10)


@_register_derivative("LS_CONTRARIAN")
def ls_contrarian(df: pd.DataFrame) -> pd.Series:
    """Contrarian signal based on L/S ratio extremes."""
    ls = df.get("long_short_ratio", pd.Series(1, index=df.index))
    zscore = (ls - ls.rolling(30).mean()) / (ls.rolling(30).std() + 1e-10)
    # Go short when too many longs, go long when too many shorts
    return -zscore.clip(-2, 2) / 2


@_register_derivative("LS_MOMENTUM")
def ls_momentum(df: pd.DataFrame) -> pd.Series:
    """L/S ratio momentum (change in positioning)."""
    ls = df.get("long_short_ratio", pd.Series(1, index=df.index))
    return ls.pct_change(4)


@_register_derivative("LS_LONG_PCT")
def ls_long_pct(df: pd.DataFrame) -> pd.Series:
    """Long percentage (0-1)."""
    return df.get("long_ratio", pd.Series(0.5, index=df.index))


@_register_derivative("LS_SHORT_PCT")
def ls_short_pct(df: pd.DataFrame) -> pd.Series:
    """Short percentage (0-1)."""
    return df.get("short_ratio", pd.Series(0.5, index=df.index))


@_register_derivative("LS_IMBALANCE")
def ls_imbalance(df: pd.DataFrame) -> pd.Series:
    """L/S imbalance (long - short)."""
    long_r = df.get("long_ratio", pd.Series(0.5, index=df.index))
    short_r = df.get("short_ratio", pd.Series(0.5, index=df.index))
    return long_r - short_r


# =============================================================================
# Liquidation Factors
# =============================================================================


@_register_derivative("LIQ_LONG_VOLUME")
def liq_long_volume(df: pd.DataFrame) -> pd.Series:
    """Long liquidation volume (normalized)."""
    liq = df.get("liquidation_long", pd.Series(0, index=df.index))
    ma = liq.rolling(24).mean()
    return liq / (ma + 1e-10) - 1


@_register_derivative("LIQ_SHORT_VOLUME")
def liq_short_volume(df: pd.DataFrame) -> pd.Series:
    """Short liquidation volume (normalized)."""
    liq = df.get("liquidation_short", pd.Series(0, index=df.index))
    ma = liq.rolling(24).mean()
    return liq / (ma + 1e-10) - 1


@_register_derivative("LIQ_RATIO")
def liq_ratio(df: pd.DataFrame) -> pd.Series:
    """Long/Short liquidation ratio."""
    long_liq = df.get("liquidation_long", pd.Series(0, index=df.index))
    short_liq = df.get("liquidation_short", pd.Series(0, index=df.index))
    return long_liq / (short_liq + 1e-10)


@_register_derivative("LIQ_TOTAL_ZSCORE")
def liq_total_zscore(df: pd.DataFrame) -> pd.Series:
    """Total liquidation z-score (extreme liquidations = reversals)."""
    long_liq = df.get("liquidation_long", pd.Series(0, index=df.index))
    short_liq = df.get("liquidation_short", pd.Series(0, index=df.index))
    total = long_liq + short_liq
    mean = total.rolling(24).mean()
    std = total.rolling(24).std()
    return (total - mean) / (std + 1e-10)


@_register_derivative("LIQ_CASCADE_SIGNAL")
def liq_cascade_signal(df: pd.DataFrame) -> pd.Series:
    """Liquidation cascade signal (extreme liqs often lead to reversals)."""
    long_liq = df.get("liquidation_long", pd.Series(0, index=df.index))
    short_liq = df.get("liquidation_short", pd.Series(0, index=df.index))
    total = long_liq + short_liq
    zscore = (total - total.rolling(24).mean()) / (total.rolling(24).std() + 1e-10)
    liq_ratio = (long_liq - short_liq) / (total + 1e-10)
    # Positive when shorts liquidated (go long), negative when longs liquidated
    return (zscore > 2).astype(float) * liq_ratio


# =============================================================================
# Basis/Premium Factors
# =============================================================================


@_register_derivative("BASIS")
def basis(df: pd.DataFrame) -> pd.Series:
    """Mark price - Index price basis."""
    return df.get("basis", pd.Series(0, index=df.index))


@_register_derivative("BASIS_RATE")
def basis_rate(df: pd.DataFrame) -> pd.Series:
    """Basis rate (basis / index price)."""
    return df.get("basis_rate", pd.Series(0, index=df.index))


@_register_derivative("BASIS_ZSCORE")
def basis_zscore(df: pd.DataFrame) -> pd.Series:
    """Z-score of basis rate."""
    br = df.get("basis_rate", pd.Series(0, index=df.index))
    mean = br.rolling(30).mean()
    std = br.rolling(30).std()
    return (br - mean) / (std + 1e-10)


@_register_derivative("BASIS_MOMENTUM")
def basis_momentum(df: pd.DataFrame) -> pd.Series:
    """Basis rate momentum (change over 4 periods)."""
    br = df.get("basis_rate", pd.Series(0, index=df.index))
    return br - br.shift(4)


@_register_derivative("BASIS_CONTANGO")
def basis_contango(df: pd.DataFrame) -> pd.Series:
    """Contango indicator (1 if positive basis, -1 if backwardation)."""
    br = df.get("basis_rate", pd.Series(0, index=df.index))
    return np.sign(br)


# =============================================================================
# Taker Buy/Sell Factors
# =============================================================================


@_register_derivative("TAKER_BUY_RATIO")
def taker_buy_ratio(df: pd.DataFrame) -> pd.Series:
    """Taker buy ratio (buy / total)."""
    return df.get("buy_sell_ratio", pd.Series(0.5, index=df.index))


@_register_derivative("TAKER_NET_FLOW")
def taker_net_flow(df: pd.DataFrame) -> pd.Series:
    """Net taker flow (normalized)."""
    buy = df.get("buy_volume", pd.Series(0, index=df.index))
    sell = df.get("sell_volume", pd.Series(0, index=df.index))
    total = buy + sell
    return (buy - sell) / (total + 1e-10)


@_register_derivative("TAKER_NET_FLOW_MA")
def taker_net_flow_ma(df: pd.DataFrame) -> pd.Series:
    """Smoothed net taker flow (4-period MA)."""
    buy = df.get("buy_volume", pd.Series(0, index=df.index))
    sell = df.get("sell_volume", pd.Series(0, index=df.index))
    total = buy + sell
    net = (buy - sell) / (total + 1e-10)
    return net.rolling(4).mean()


@_register_derivative("TAKER_MOMENTUM")
def taker_momentum(df: pd.DataFrame) -> pd.Series:
    """Taker flow momentum."""
    ratio = df.get("buy_sell_ratio", pd.Series(0.5, index=df.index))
    return ratio - ratio.shift(4)


@_register_derivative("TAKER_PRESSURE")
def taker_pressure(df: pd.DataFrame) -> pd.Series:
    """Cumulative taker pressure (12-period)."""
    buy = df.get("buy_volume", pd.Series(0, index=df.index))
    sell = df.get("sell_volume", pd.Series(0, index=df.index))
    total = buy + sell
    net = (buy - sell) / (total + 1e-10)
    return net.rolling(12).sum()


# =============================================================================
# Composite Factors
# =============================================================================


@_register_derivative("CRYPTO_SENTIMENT")
def crypto_sentiment(df: pd.DataFrame) -> pd.Series:
    """Composite crypto sentiment indicator."""
    # Combine multiple signals
    funding_signal = -df.get("funding_rate", pd.Series(0, index=df.index)).rolling(8).mean() * 100
    ls_signal = -(df.get("long_short_ratio", pd.Series(1, index=df.index)) - 1)
    taker_signal = df.get("buy_sell_ratio", pd.Series(0.5, index=df.index)) - 0.5

    # Equal weight combination
    return (funding_signal + ls_signal + taker_signal) / 3


@_register_derivative("LEVERAGE_PRESSURE")
def leverage_pressure(df: pd.DataFrame) -> pd.Series:
    """Leverage pressure indicator (high OI + high funding = crowded)."""
    oi_z = df.get("open_interest", pd.Series(0, index=df.index))
    oi_z = (oi_z - oi_z.rolling(30).mean()) / (oi_z.rolling(30).std() + 1e-10)

    fr = df.get("funding_rate", pd.Series(0, index=df.index))
    fr_z = (fr - fr.rolling(30).mean()) / (fr.rolling(30).std() + 1e-10)

    # High leverage pressure when both are extreme in same direction
    return oi_z * fr_z


@_register_derivative("SMART_MONEY_FLOW")
def smart_money_flow(df: pd.DataFrame) -> pd.Series:
    """Smart money flow indicator."""
    # Combines taker flow with OI changes
    taker = df.get("buy_sell_ratio", pd.Series(0.5, index=df.index)) - 0.5
    oi = df.get("open_interest", pd.Series(0, index=df.index))
    oi_change = oi.pct_change(1)

    # Smart money: aggressive buying + increasing OI = bullish
    return taker * (1 + oi_change * 10)


@_register_derivative("LIQUIDATION_RISK")
def liquidation_risk(df: pd.DataFrame) -> pd.Series:
    """Liquidation risk indicator."""
    # High leverage + extreme positioning = high liquidation risk
    oi = df.get("open_interest", pd.Series(0, index=df.index))
    oi_z = (oi - oi.rolling(30).mean()) / (oi.rolling(30).std() + 1e-10)

    ls = df.get("long_short_ratio", pd.Series(1, index=df.index))
    ls_extreme = (ls - 1).abs()

    return oi_z.abs() * ls_extreme


# =============================================================================
# Utility Functions
# =============================================================================


def get_all_derivative_factors() -> list[str]:
    """Get all registered derivative factor names."""
    return list(DERIVATIVE_FACTORS.keys())


def compute_derivative_factors(
    df: pd.DataFrame,
    factor_names: list[str] | None = None,
) -> pd.DataFrame:
    """Compute specified derivative factors.

    Args:
        df: DataFrame with derivative data columns
        factor_names: List of factor names to compute (None = all)

    Returns:
        DataFrame with computed factor values
    """
    if factor_names is None:
        factor_names = list(DERIVATIVE_FACTORS.keys())

    results = {}
    for name in factor_names:
        if name in DERIVATIVE_FACTORS:
            try:
                results[name] = DERIVATIVE_FACTORS[name](df)
            except Exception as e:
                print(f"Warning: Failed to compute {name}: {e}")
                results[name] = pd.Series(np.nan, index=df.index)

    return pd.DataFrame(results)


def get_derivative_factor_count() -> int:
    """Get number of registered derivative factors."""
    return len(DERIVATIVE_FACTORS)


def get_derivative_factors_by_category() -> dict[str, list[str]]:
    """Get derivative factors organized by category."""
    categories = {
        "funding": [],
        "open_interest": [],
        "long_short": [],
        "liquidation": [],
        "basis": [],
        "taker": [],
        "composite": [],
    }

    for name in DERIVATIVE_FACTORS.keys():
        if name.startswith("FUNDING"):
            categories["funding"].append(name)
        elif name.startswith("OI_"):
            categories["open_interest"].append(name)
        elif name.startswith("LS_"):
            categories["long_short"].append(name)
        elif name.startswith("LIQ_"):
            categories["liquidation"].append(name)
        elif name.startswith("BASIS"):
            categories["basis"].append(name)
        elif name.startswith("TAKER"):
            categories["taker"].append(name)
        else:
            categories["composite"].append(name)

    return categories
