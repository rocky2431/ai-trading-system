"""Additional tests for qlib_stats utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iqfmp.evaluation import qlib_stats


def _returns_series() -> pd.Series:
    rng = np.random.default_rng(seed=9)
    return pd.Series(rng.normal(0, 0.01, size=100))


def test_basic_statistics() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    engine = qlib_stats.QlibStatisticalEngine()

    assert engine.calculate_mean(values) == 2.5
    assert engine.calculate_std(values) > 0
    assert engine.calculate_correlation(values, values) == 1.0


def test_sharpe_ratio_uses_risk_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qlib_stats, "QLIB_AVAILABLE", True)
    monkeypatch.setattr(
        qlib_stats,
        "risk_analysis",
        lambda returns, N=252: {"information_ratio": 1.23},
    )

    returns = _returns_series()
    result = qlib_stats.QlibStatisticalEngine.calculate_sharpe_ratio(returns)
    assert result == 1.23


def test_deflated_sharpe_calculator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qlib_stats, "QLIB_AVAILABLE", True)
    monkeypatch.setattr(
        qlib_stats,
        "risk_analysis",
        lambda returns, N=252: {"information_ratio": 1.5},
    )

    calc = qlib_stats.DeflatedSharpeCalculator()
    returns = _returns_series()
    dsr = calc.calculate_dsr(returns, n_trials=10)
    assert dsr.raw_sharpe >= 0.0
    assert 0.0 <= dsr.p_value <= 1.0


def test_tail_risk_helpers() -> None:
    returns = _returns_series()

    var_95 = qlib_stats.calculate_var(returns, 0.95)
    es_95 = qlib_stats.calculate_expected_shortfall(returns, 0.95)
    sortino = qlib_stats.calculate_sortino_ratio(returns)
    calmar = qlib_stats.calculate_calmar_ratio(returns)
    info = qlib_stats.calculate_information_ratio(returns, returns * 0.5)

    assert isinstance(var_95, float)
    assert isinstance(es_95, float)
    assert isinstance(sortino, float)
    assert isinstance(calmar, float)
    assert isinstance(info, float)
