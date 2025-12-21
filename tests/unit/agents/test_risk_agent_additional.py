"""Additional unit tests for RiskCheckAgent."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from iqfmp.agents.orchestrator import AgentState
from iqfmp.agents.risk_agent import (
    RiskCheckAgent,
    RiskConfig,
    RiskMetrics,
    ApprovalStatus,
    RiskLevel,
)


def _sample_returns() -> pd.Series:
    rng = np.random.default_rng(seed=11)
    return pd.Series(rng.normal(0, 0.01, size=120))


@pytest.mark.asyncio
async def test_risk_check_end_to_end() -> None:
    agent = RiskCheckAgent(config=RiskConfig())

    returns = _sample_returns()
    state = AgentState(
        context={
            "backtest_metrics": {"max_drawdown": 0.1, "sharpe_ratio": 1.2},
            "strategy_signals": [
                {"position": 0.1, "combined_signal": 0.2},
                {"position": -0.05, "combined_signal": -0.1},
            ],
            "factor_weights": [
                {"factor_name": "factor_a", "weight": 0.6},
                {"factor_name": "factor_b", "weight": 0.4},
            ],
            "price_data": pd.DataFrame({"returns": returns}),
        }
    )

    result = await agent.check(state)
    assert result.context.get("risk_assessment") is not None
    assert "strategy_approved" in result.context


def test_risk_scoring_and_warnings() -> None:
    agent = RiskCheckAgent(config=RiskConfig())

    metrics = RiskMetrics(
        max_drawdown=0.12,
        portfolio_volatility=0.2,
        var_95=0.03,
        concentration_ratio=0.2,
        avg_turnover=0.1,
    )
    limits = agent._create_limits()
    breached = agent._check_limits(limits, metrics)
    score = agent._calculate_risk_score(metrics, breached)
    approval = agent._determine_approval(score, breached)
    warnings = agent._generate_warnings(metrics, breached)
    recommendations = agent._generate_recommendations(metrics, breached)

    assert 0.0 <= score <= 1.0
    assert approval in {ApprovalStatus.APPROVED, ApprovalStatus.CONDITIONAL, ApprovalStatus.MANUAL_REVIEW, ApprovalStatus.REJECTED}
    assert isinstance(warnings, list)
    assert isinstance(recommendations, list)


def test_parse_llm_risk_response() -> None:
    agent = RiskCheckAgent()
    payload = {
        "analysis": "ok",
        "warnings": ["w1"],
        "recommendations": ["r1"],
        "risk_level": "medium",
    }
    response = json.dumps(payload)

    parsed = agent._parse_llm_risk_response(response)
    assert parsed["analysis"] == "ok"
    assert "warnings" in parsed


def test_template_risk_analysis() -> None:
    agent = RiskCheckAgent()
    metrics = RiskMetrics(max_drawdown=0.2, portfolio_volatility=0.3, var_95=0.05)
    report = agent._generate_template_risk_analysis(metrics, ["max_drawdown"])
    assert "summary" in report
    assert "mitigation_actions" in report


def test_calculate_metrics_from_inputs() -> None:
    agent = RiskCheckAgent()
    returns = _sample_returns()
    metrics = agent._calculate_metrics(
        backtest_metrics={"max_drawdown": 0.05, "sharpe_ratio": 1.0},
        strategy_signals=[
            {"position": 0.2, "combined_signal": 0.1},
            {"position": -0.1, "combined_signal": -0.2},
        ],
        price_data=pd.DataFrame({"returns": returns}),
    )

    assert metrics.max_drawdown >= 0.05
    assert metrics.max_position >= 0.1


def test_determine_risk_level() -> None:
    agent = RiskCheckAgent()
    assert agent._determine_risk_level(0.1) == RiskLevel.LOW
    assert agent._determine_risk_level(0.5) in {RiskLevel.MEDIUM, RiskLevel.HIGH}
