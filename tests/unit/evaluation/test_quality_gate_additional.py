"""Additional tests for quality gate utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from iqfmp.evaluation.quality_gate import (
    CryptoRegimeDetector,
    RealisticBacktestEngine,
    BacktestCostConfig,
    AntiPHackingGate,
    CryptoRegime,
    DeflatedSharpeCalculator,
    AntiPHackingConfig,
    TrialBudget,
    RegimeConfig,
)


def _sample_market_data(rows: int = 48) -> pd.DataFrame:
    rng = np.random.default_rng(seed=6)
    dates = pd.date_range("2023-01-01", periods=rows, freq="H")
    close = pd.Series(100 + rng.normal(0, 1, size=rows)).cumsum()
    return pd.DataFrame(
        {
            "timestamp": dates,
            "close": close,
            "funding_rate": rng.normal(0, 0.0005, size=rows),
            "liquidation_total": rng.uniform(0, 10, size=rows),
        }
    )


def test_regime_detector_funding_missing() -> None:
    data = _sample_market_data()[["timestamp", "close"]]
    detector = CryptoRegimeDetector()
    funding_regime = detector.detect_funding_regime(data)
    assert set(funding_regime.unique()) == {CryptoRegime.FUNDING_NORMAL.value}


def test_regime_detector_missing_price_column() -> None:
    detector = CryptoRegimeDetector()
    with pytest.raises(ValueError):
        detector.detect_volatility_regime(pd.DataFrame({"open": [1, 2, 3]}), price_column="close")


def test_regime_detector_all_regimes() -> None:
    data = _sample_market_data()
    detector = CryptoRegimeDetector()
    result = detector.detect_all_regimes(data)
    assert {
        "regime_volatility",
        "regime_funding",
        "regime_liquidation",
        "regime_combined",
    }.issubset(result.columns)


def test_regime_detector_funding_extreme() -> None:
    data = pd.DataFrame(
        {
            "funding_rate": [0.0, 0.0, 0.02, 0.02],
        }
    )
    detector = CryptoRegimeDetector(
        RegimeConfig(funding_window=3, funding_zscore_threshold=0.5)
    )
    regime = detector.detect_funding_regime(data)
    assert regime.iloc[-1] == CryptoRegime.FUNDING_EXTREME_POSITIVE.value


def test_regime_detector_liquidation_extreme() -> None:
    data = pd.DataFrame(
        {
            "liquidation_total": [1.0, 1.2, 5.0, 10.0],
        }
    )
    detector = CryptoRegimeDetector(
        RegimeConfig(liquidation_window=3, liquidation_percentile=0.5)
    )
    regime = detector.detect_liquidation_regime(data)
    assert regime.iloc[-1] == CryptoRegime.HIGH_LIQUIDATION.value


def test_realistic_backtest_net_returns() -> None:
    data = _sample_market_data()
    signals = pd.Series(np.linspace(-1, 1, len(data)), index=data.index)
    engine = RealisticBacktestEngine(BacktestCostConfig(include_funding=True))

    results = engine.calculate_net_returns(data, signals)
    assert {"gross_return", "total_cost", "net_return"}.issubset(results.columns)
    assert np.allclose(
        results["net_return"],
        results["gross_return"] + results["total_cost"],
        equal_nan=True,
    )


def test_realistic_backtest_cost_components() -> None:
    config = BacktestCostConfig(slippage_dynamic=False, funding_settlement_hours=[0])
    engine = RealisticBacktestEngine(config)
    timestamps = pd.date_range("2023-01-01", periods=2, freq="h")
    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": [100.0, 101.0],
            "funding_rate": [0.001, 0.001],
        }
    )
    positions = pd.Series([1.0, -1.0], index=data.index)
    trades = positions.diff().fillna(positions)

    funding_cost = engine.calculate_funding_cost(data, positions)
    trading_cost = engine.calculate_trading_cost(trades, data["close"], is_maker=True)
    slippage_cost = engine.calculate_slippage(trades, data["close"].pct_change().fillna(0))

    assert funding_cost.iloc[0] != 0.0
    assert funding_cost.iloc[1] == 0.0
    assert (trading_cost <= 0).all()
    assert (slippage_cost <= 0).all()


def test_anti_p_hacking_gate_evaluation() -> None:
    gate = AntiPHackingGate()
    outcome = gate.evaluate_factor_quality(
        sharpe_ratio=1.2,
        ic=0.04,
        ir=0.7,
        expression="Mean($close, 5)",
        hypothesis="momentum persists",
        family="momentum",
    )
    assert "grade" in outcome
    assert "recommendations" in outcome


def test_anti_p_hacking_gate_low_scores() -> None:
    gate = AntiPHackingGate()
    outcome = gate.evaluate_factor_quality(
        sharpe_ratio=0.1,
        ic=0.0,
        ir=0.0,
        expression="Mean(Mean(Mean(Mean(Mean(Mean(Mean(Mean(Mean($close, 2),2),2),2),2),2),2),2),2)",
        hypothesis="random noise",
        family="volatility",
    )
    assert outcome["grade"] == "F"
    assert outcome["recommendations"]


def test_complexity_penalty_and_deflated_sharpe() -> None:
    gate = AntiPHackingGate()
    penalty, details = gate.calculate_complexity_penalty("Mean(Ref($close, 1), 5) + Std($close, 10)")
    assert penalty >= 0.0
    assert details["n_operators"] >= 2

    calc = DeflatedSharpeCalculator(min_trials=5)
    result = calc.calculate(sharpe_ratio=1.0, n_trials=2)
    assert result.deflated_sharpe == 1.0


def test_deflated_sharpe_full_calculation() -> None:
    calc = DeflatedSharpeCalculator(threshold=0.5, min_trials=1)
    result = calc.calculate(sharpe_ratio=1.5, n_trials=20, skewness=0.1, kurtosis=3.2, t_observations=252)
    assert result.deflated_sharpe <= result.raw_sharpe
    assert result.haircut_pct >= 0.0
    assert result.confidence in {"Low", "Medium", "High"}


def test_trial_budget_limits() -> None:
    budget = TrialBudget()
    config = AntiPHackingConfig(max_total_trials=1, max_trials_per_hypothesis=1)

    allowed, _ = budget.can_try("hyp1", config)
    assert allowed
    budget.record_trial("hyp1")

    allowed_after, _ = budget.can_try("hyp1", config)
    assert not allowed_after


def test_trial_budget_cooldown() -> None:
    budget = TrialBudget()
    config = AntiPHackingConfig(max_total_trials=10, max_trials_per_hypothesis=10, hypothesis_cooldown_minutes=10)
    budget.last_trial_time["hyp2"] = datetime.now() - timedelta(minutes=1)

    allowed, reason = budget.can_try("hyp2", config)
    assert not allowed
    assert "Cooldown" in reason
