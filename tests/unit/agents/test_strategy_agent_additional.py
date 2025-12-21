"""Additional unit tests for StrategyAssemblyAgent."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iqfmp.agents.strategy_agent import (
    StrategyAssemblyAgent,
    StrategyConfig,
    CombinationMethod,
    SignalTransform,
    PortfolioConstruction,
    FactorWeight,
)
from iqfmp.agents.orchestrator import AgentState


def _evaluation_results() -> list[dict[str, object]]:
    return [
        {"factor_name": "alpha", "metrics": {"ic": 0.05, "ir": 1.2}},
        {"factor_name": "beta", "metrics": {"ic": 0.03, "ir": 0.8}},
        {"factor_name": "gamma", "metrics": {"ic": 0.08, "ir": 1.5}},
    ]


def _factor_data() -> pd.DataFrame:
    rng = np.random.default_rng(seed=21)
    df = pd.DataFrame(
        {
            "factor_alpha": rng.normal(0, 1, size=50),
            "factor_beta": rng.normal(0, 1, size=50),
            "factor_gamma": rng.normal(0, 1, size=50),
        }
    )
    return df


def _signal_series() -> pd.Series:
    return pd.Series([1.5, -2.0, 0.5, 3.0], index=["a", "b", "c", "d"])


def test_select_factors_by_ir() -> None:
    config = StrategyConfig(combination_method=CombinationMethod.IR_WEIGHTED, max_factors=2)
    agent = StrategyAssemblyAgent(config=config)

    selected = agent._select_factors(["alpha", "beta", "gamma"], _evaluation_results())
    assert selected[0] == "gamma"
    assert len(selected) == 2


def test_calculate_weights_ranked() -> None:
    config = StrategyConfig(combination_method=CombinationMethod.RANK_WEIGHTED)
    agent = StrategyAssemblyAgent(config=config)

    weights = agent._calculate_weights(["alpha", "beta"], _evaluation_results())
    total = sum(w.weight for w in weights)
    assert abs(total - 1.0) < 1e-6


def test_generate_signals_and_positions() -> None:
    config = StrategyConfig(
        combination_method=CombinationMethod.EQUAL_WEIGHT,
        signal_transform=SignalTransform.ZSCORE,
        construction_method=PortfolioConstruction.LONG_SHORT,
    )
    agent = StrategyAssemblyAgent(config=config)

    weights = agent._calculate_weights(["alpha", "beta"], _evaluation_results())
    signals = agent._generate_signals(weights, _factor_data())

    assert "combined_signal" in signals.columns
    assert "position" in signals.columns
    assert "rank" in signals.columns


def test_generate_strategy_name() -> None:
    agent = StrategyAssemblyAgent(config=StrategyConfig())
    name = agent._generate_strategy_name(["alpha", "beta"])
    assert "multi_factor" in name.lower()
    assert "2" in name


def test_parse_llm_recommendations() -> None:
    agent = StrategyAssemblyAgent(config=StrategyConfig())
    response = '{"assessment": "ok", "risks": ["r1"], "recommendations": ["r2"]}'
    parsed = agent._parse_llm_recommendations(response)
    assert parsed["assessment"] == "ok"
    assert parsed["risks"] == ["r1"]


def test_apply_weight_constraints() -> None:
    config = StrategyConfig(min_factor_weight=0.2, max_factor_weight=0.6)
    agent = StrategyAssemblyAgent(config=config)
    raw = {"alpha": 0.9, "beta": 0.1}

    constrained = agent._apply_weight_constraints(raw)
    assert abs(sum(constrained.values()) - 1.0) < 1e-6
    assert all(w > 0 for w in constrained.values())
    assert constrained["alpha"] > constrained["beta"]


def test_validate_weights_negative() -> None:
    agent = StrategyAssemblyAgent(config=StrategyConfig())
    bad_weights = [
        FactorWeight(factor_name="alpha", weight=-0.1, ic=0.0, ir=0.0),
        FactorWeight(factor_name="beta", weight=1.1, ic=0.0, ir=0.0),
    ]
    with pytest.raises(Exception):
        agent._validate_weights(bad_weights)


def test_transform_signal_variants() -> None:
    signal = _signal_series()
    agent = StrategyAssemblyAgent(config=StrategyConfig(signal_transform=SignalTransform.RAW))
    assert agent._transform_signal(signal).equals(signal)

    agent.config.signal_transform = SignalTransform.RANK
    ranked = agent._transform_signal(signal)
    assert ranked.between(-0.5, 0.5).all()

    agent.config.signal_transform = SignalTransform.PERCENTILE
    percentile = agent._transform_signal(signal)
    assert percentile.between(0, 1).all()

    agent.config.signal_transform = SignalTransform.WINSORIZE
    winsorized = agent._transform_signal(signal)
    assert winsorized.max() <= signal.max()
    assert winsorized.min() >= signal.min()


def test_generate_positions_variants() -> None:
    signal = _signal_series()

    agent = StrategyAssemblyAgent(
        config=StrategyConfig(construction_method=PortfolioConstruction.LONG_ONLY)
    )
    long_only = agent._generate_positions(signal)
    assert (long_only >= 0).all()
    assert abs(long_only.sum() - 1.0) < 1e-6

    agent.config = StrategyConfig(
        construction_method=PortfolioConstruction.TOP_N,
        top_n=2,
    )
    top_n = agent._generate_positions(signal)
    assert (top_n > 0).sum() == 2
    assert np.isclose(top_n[top_n > 0].sum(), 1.0)

    agent.config = StrategyConfig(
        construction_method=PortfolioConstruction.THRESHOLD,
        signal_threshold=1.0,
    )
    thresholded = agent._generate_positions(signal)
    assert (thresholded[signal > 1.0] > 0).all()
    assert (thresholded[signal < -1.0] < 0).all()
    assert (thresholded[(signal <= 1.0) & (signal >= -1.0)] == 0).all()


def test_template_recommendations_low_factor_count() -> None:
    agent = StrategyAssemblyAgent(config=StrategyConfig())
    weights = [
        FactorWeight(factor_name="alpha", weight=0.6, ic=0.03, ir=1.0),
        FactorWeight(factor_name="beta", weight=0.4, ic=0.02, ir=0.8),
    ]
    recommendations = agent._generate_template_recommendations(weights)
    assert any("Low number of factors" in risk for risk in recommendations["risks"])


@pytest.mark.asyncio
async def test_generate_llm_recommendations_without_llm() -> None:
    agent = StrategyAssemblyAgent(config=StrategyConfig())
    weights = [
        FactorWeight(factor_name="alpha", weight=0.5, ic=0.03, ir=1.1),
        FactorWeight(factor_name="beta", weight=0.5, ic=0.02, ir=0.9),
    ]
    recs = await agent.generate_llm_recommendations(weights, [])
    assert "assessment" in recs


@pytest.mark.asyncio
async def test_generate_llm_recommendations_with_llm() -> None:
    class _DummyLLM:
        async def complete(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            temperature: float | None = None,
            max_tokens: int | None = None,
        ) -> object:
            return type("Resp", (), {"content": '{"assessment": "ok", "risks": [], "recommendations": []}'})()

    agent = StrategyAssemblyAgent(config=StrategyConfig(), llm_provider=_DummyLLM())
    weights = [
        FactorWeight(factor_name="alpha", weight=0.5, ic=0.03, ir=1.1),
        FactorWeight(factor_name="beta", weight=0.5, ic=0.02, ir=0.9),
    ]
    recs = await agent.generate_llm_recommendations(weights, [])
    assert recs["assessment"] == "ok"


@pytest.mark.asyncio
async def test_assemble_no_factors_returns_error() -> None:
    agent = StrategyAssemblyAgent(config=StrategyConfig())
    state = AgentState(context={"factors_passed": []})
    updated = await agent.assemble(state)
    assert updated.context["strategy_result"] is None
    assert updated.context["strategy_error"]


@pytest.mark.asyncio
async def test_assemble_with_factors_and_data() -> None:
    agent = StrategyAssemblyAgent(config=StrategyConfig())
    state = AgentState(
        context={
            "factors_passed": ["alpha", "beta", "gamma"],
            "evaluation_results": _evaluation_results(),
            "factor_data": _factor_data(),
        }
    )
    updated = await agent.assemble(state)
    assert updated.context["strategy_result"]
    assert updated.context["factor_weights"]
    assert updated.context["strategy_signals"]
