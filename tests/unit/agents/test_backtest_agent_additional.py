"""Additional tests for backtest agent data models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iqfmp.agents import backtest_agent
from iqfmp.agents.backtest_agent import (
    ParameterSpace,
    BacktestConfig,
    BacktestMetrics,
    TrialResult,
    OptimizationResult,
    BacktestEngine,
    QlibBacktestEngine,
    BacktestOptimizationAgent,
)


def test_parameter_space_sampling() -> None:
    continuous = ParameterSpace(name="alpha", min_value=0.1, max_value=1.0)
    samples = continuous.sample(n=3)
    assert len(samples) == 3

    discrete = ParameterSpace(name="steps", min_value=1, max_value=5, step=1, discrete=True)
    samples = discrete.sample(n=4)
    assert all(1 <= s <= 5 for s in samples)


def test_backtest_metrics_constraints() -> None:
    config = BacktestConfig(min_sharpe=1.0, min_win_rate=0.4, max_drawdown=0.2)
    metrics = BacktestMetrics(sharpe_ratio=1.2, win_rate=0.5, max_drawdown=0.1)
    assert metrics.passes_constraints(config)


def test_optimization_result_to_dict() -> None:
    metrics = BacktestMetrics(sharpe_ratio=1.1, win_rate=0.45, max_drawdown=0.1)
    trial = TrialResult(trial_id=1, parameters={"alpha": 0.5}, metrics=metrics)
    result = OptimizationResult(
        best_parameters={"alpha": 0.5},
        best_metrics=metrics,
        all_trials=[trial],
        n_trials=1,
        optimization_time=0.1,
        converged=True,
    )

    as_dict = result.to_dict()
    assert as_dict["best_parameters"]["alpha"] == 0.5


def test_legacy_backtest_engine_run() -> None:
    engine = BacktestEngine()
    idx = pd.RangeIndex(start=0, stop=15)
    signals = pd.Series(np.where(np.arange(15) % 2 == 0, 0.1, -0.1), index=idx)
    returns = pd.Series(np.linspace(-0.01, 0.02, 15), index=idx)

    metrics = engine.run(signals, returns)
    assert isinstance(metrics, BacktestMetrics)
    assert metrics.n_trades >= 0


def test_qlib_manual_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backtest_agent, "QLIB_AVAILABLE", True)
    engine = QlibBacktestEngine()

    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.0])
    position_changes = pd.Series([0, 1, 0, 1, 0])
    metrics = engine._calculate_metrics_manually(returns, position_changes)

    assert metrics.max_drawdown >= 0.0


def test_qlib_extract_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backtest_agent, "QLIB_AVAILABLE", True)
    monkeypatch.setattr(
        backtest_agent,
        "risk_analysis",
        lambda returns: {
            "total_return": 0.1,
            "annualized_return": 0.2,
            "sharpe": 1.5,
            "sortino": 1.1,
            "max_drawdown": 0.05,
            "win_rate": 0.6,
        },
    )

    engine = QlibBacktestEngine()
    portfolio_df = pd.DataFrame({"return": [0.01, -0.02, 0.03]})
    portfolio_dict = {"1day": (portfolio_df, {"total_trades": 3})}

    metrics = engine._extract_qlib_metrics(portfolio_dict, {})
    assert metrics.n_trades == 3
    assert metrics.sharpe_ratio == 1.5


def test_qlib_simulate_components(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backtest_agent, "QLIB_AVAILABLE", True)
    monkeypatch.setattr(
        backtest_agent,
        "risk_analysis",
        lambda returns: {"information_ratio": 1.0, "max_drawdown": 0.1, "total_return": 0.05},
    )

    engine = QlibBacktestEngine()
    idx = pd.RangeIndex(start=0, stop=12)
    signals = pd.Series(np.where(np.arange(12) % 2 == 0, 0.2, -0.2), index=idx)
    returns = pd.Series(np.linspace(-0.01, 0.02, 12), index=idx)
    prices = pd.Series(100 + np.arange(12), index=idx)

    metrics = engine._simulate_with_qlib_components(signals, returns, prices)
    assert metrics.sharpe_ratio == 1.0


def _make_agent(monkeypatch: pytest.MonkeyPatch) -> BacktestOptimizationAgent:
    monkeypatch.setattr(backtest_agent, "QLIB_AVAILABLE", True)
    config = BacktestConfig(n_trials=3, timeout=5.0, train_ratio=0.5, use_realistic_costs=False)
    agent = BacktestOptimizationAgent(config=config)

    class _StubEngine:
        def run(self, signals: pd.Series, returns: pd.Series) -> BacktestMetrics:
            sharpe = float(signals.mean()) if len(signals) else 0.0
            return BacktestMetrics(
                total_return=0.1,
                annualized_return=0.2,
                sharpe_ratio=sharpe,
                sortino_ratio=0.5,
                max_drawdown=0.1,
                win_rate=0.6,
                profit_factor=1.2,
                n_trades=5,
                avg_trade_return=0.01,
            )

    agent.engine = _StubEngine()
    return agent


def test_apply_parameters_adjusts_signals(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _make_agent(monkeypatch)
    signals = pd.Series([0.1, 0.6, -0.8, 0.2])
    returns = pd.Series([0.01, -0.02, 0.03, -0.04])
    params = {"position_scale": 2.0, "signal_threshold": 0.5, "stop_loss": 0.02}

    adjusted = agent._apply_parameters(signals, params, returns)
    assert adjusted.abs().max() <= 1.0
    assert adjusted.iloc[0] == 0


def test_run_optimization_records_trials(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _make_agent(monkeypatch)
    param_space = [
        ParameterSpace(name="signal_threshold", min_value=0.5, max_value=0.5, step=0.5),
        ParameterSpace(name="position_scale", min_value=1.0, max_value=1.0, step=1.0),
        ParameterSpace(name="stop_loss", min_value=0.05, max_value=0.05, step=0.01),
    ]
    price_data = pd.DataFrame({"close": np.linspace(1.0, 2.0, 30)})
    signals = [{"combined_signal": 0.1} for _ in range(30)]

    result = agent._run_optimization(param_space, price_data, signals)
    assert result.n_trials > 0
    assert result.best_parameters


def test_validate_out_of_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _make_agent(monkeypatch)
    price_data = pd.DataFrame({"close": np.linspace(1.0, 2.0, 120)})
    signals = [{"combined_signal": 0.2} for _ in range(120)]

    metrics = agent._validate_out_of_sample(
        params={"signal_threshold": 0.1},
        price_data=price_data,
        strategy_signals=signals,
    )
    assert metrics is not None


def test_check_overfitting() -> None:
    train = BacktestMetrics(sharpe_ratio=2.0)
    test = BacktestMetrics(sharpe_ratio=1.0)
    agent = BacktestOptimizationAgent.__new__(BacktestOptimizationAgent)
    ratio = agent._check_overfitting(train, test)
    assert ratio == 2.0


def test_backtest_strategy_realistic_costs(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _make_agent(monkeypatch)

    class _StubRealisticEngine:
        def calculate_net_returns(self, data, signals, price_column, funding_column):
            series = pd.Series([0.0] * len(data), index=data.index)
            return pd.DataFrame({"net_returns": series, "total_costs": series})

    agent.realistic_engine = _StubRealisticEngine()

    signals = pd.Series([0.1] * 10)
    returns = pd.Series([0.01] * 10)
    price_data = pd.DataFrame({"close": np.linspace(1.0, 2.0, 10)})

    metrics = agent.backtest_strategy(signals, returns, price_data=price_data)
    assert isinstance(metrics, BacktestMetrics)


def test_walk_forward_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _make_agent(monkeypatch)
    signals = pd.Series(np.linspace(-1, 1, 20))
    returns = pd.Series(np.linspace(-0.01, 0.02, 20))

    results = agent.walk_forward_validation(signals, returns, n_folds=4)
    assert len(results) == 4
