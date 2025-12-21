"""Tests for PipelineBuilder routing helpers."""

from __future__ import annotations

from iqfmp.agents.orchestrator import AgentState
from iqfmp.agents.pipeline_builder import PipelineBuilder, PipelineConfig


def test_route_after_evaluation() -> None:
    builder = PipelineBuilder(PipelineConfig(enable_strategy=True))
    state_with_pass = AgentState(context={"factors_passed": ["alpha"]})
    state_without_pass = AgentState(context={"factors_passed": []})

    assert builder._route_after_evaluation(state_with_pass) == "strategy"
    assert builder._route_after_evaluation(state_without_pass) == "finish"


def test_route_after_risk() -> None:
    builder = PipelineBuilder(PipelineConfig(enable_risk_check=True))
    approved = AgentState(context={"strategy_approved": True})
    rejected = AgentState(context={"strategy_approved": False})

    assert builder._route_after_risk(approved) == "finish"
    assert builder._route_after_risk(rejected) == "finish"


def test_generate_summary() -> None:
    builder = PipelineBuilder(PipelineConfig())
    summary = builder._generate_summary(
        {
            "generated_factors": [{"name": "alpha"}],
            "factors_passed": ["alpha"],
            "strategy_result": {"strategy_name": "alpha_strategy"},
            "backtest_metrics": {"sharpe_ratio": 1.2},
            "risk_level": "low",
            "strategy_approved": True,
        }
    )

    assert summary["generated_factors"] == 1
    assert summary["factors_passed"] == 1
    assert summary["approved"] is True
