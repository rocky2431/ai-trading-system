"""Smoke tests for alpha factor libraries.

These tests execute all registered factors to ensure the libraries are callable
and to drive coverage over the factor implementations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from iqfmp.evaluation.alpha101 import ALPHA101_FACTORS
from iqfmp.evaluation.alpha158 import ALPHA158_FACTORS
from iqfmp.evaluation.alpha360 import ALPHA360_FACTORS
from iqfmp.evaluation.alpha_derivatives import DERIVATIVE_FACTORS


def _make_sample_df(rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed=7)
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = pd.Series(100 + rng.normal(0, 1, size=rows)).cumsum()
    open_ = close + rng.normal(0, 0.5, size=rows)
    high = np.maximum(open_, close) + rng.random(rows)
    low = np.minimum(open_, close) - rng.random(rows)
    volume = rng.uniform(100, 1000, size=rows)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "funding_rate": rng.normal(0, 0.001, size=rows),
            "open_interest": rng.uniform(1000, 2000, size=rows),
            "liquidation_volume": rng.uniform(0, 100, size=rows),
            "long_short_ratio": rng.uniform(0.5, 1.5, size=rows),
            "basis_pct": rng.normal(0, 0.001, size=rows),
            "bid_ask_spread": rng.uniform(0.0001, 0.001, size=rows),
            "orderbook_imbalance": rng.uniform(-1, 1, size=rows),
            "premium_index": rng.normal(0, 0.001, size=rows),
            "taker_buy_volume": rng.uniform(0, 1000, size=rows),
            "taker_sell_volume": rng.uniform(0, 1000, size=rows),
        },
        index=dates,
    )
    return df


def _assert_series_like(result: object, expected_len: int) -> None:
    if isinstance(result, pd.Series):
        assert len(result) == expected_len
        return
    if isinstance(result, np.ndarray):
        assert result.shape[0] == expected_len
        return
    assert result is not None


def test_alpha101_all_factors_execute() -> None:
    df = _make_sample_df()
    assert len(ALPHA101_FACTORS) >= 101

    for name, func in ALPHA101_FACTORS.items():
        result = func(df)
        _assert_series_like(result, len(df))


def test_alpha158_all_factors_execute() -> None:
    df = _make_sample_df()
    assert len(ALPHA158_FACTORS) > 0

    for name, func in ALPHA158_FACTORS.items():
        result = func(df)
        _assert_series_like(result, len(df))


def test_alpha360_all_factors_execute() -> None:
    df = _make_sample_df()
    assert len(ALPHA360_FACTORS) > 0

    for name, func in ALPHA360_FACTORS.items():
        result = func(df)
        _assert_series_like(result, len(df))


def test_alpha_derivative_factors_execute() -> None:
    df = _make_sample_df()
    assert len(DERIVATIVE_FACTORS) > 0

    for name, func in DERIVATIVE_FACTORS.items():
        result = func(df)
        _assert_series_like(result, len(df))
