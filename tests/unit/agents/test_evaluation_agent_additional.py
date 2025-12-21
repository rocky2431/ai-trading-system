"""Additional tests for FactorEvaluationAgent helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from iqfmp.agents.evaluation_agent import (
    FactorEvaluationAgent,
    EvaluationAgentConfig,
    EvaluationAgentResult,
    DataValidationError,
)
from iqfmp.evaluation.factor_evaluator import EvaluationResult, FactorMetrics


def _sample_data(rows: int = 120) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "close": [1 + i * 0.01 for i in range(rows)],
            "forward_return": [0.001] * rows,
        }
    )
    return df


def test_validate_data_success() -> None:
    agent = FactorEvaluationAgent(config=EvaluationAgentConfig(min_data_points=10))
    agent._validate_data(_sample_data())


def test_validate_data_missing_columns() -> None:
    agent = FactorEvaluationAgent(config=EvaluationAgentConfig(min_data_points=10))
    with pytest.raises(Exception):
        agent._validate_data(pd.DataFrame({"close": [1.0] * 20}))


def test_prepare_factor_data_python_code() -> None:
    config = EvaluationAgentConfig(min_data_points=10, allow_python_factors=True)
    agent = FactorEvaluationAgent(config=config)
    df = _sample_data()

    code = "def factor(df):\n    return df['close']\n"
    prepared = agent._prepare_factor_data(df, code, "factor")

    assert "factor_value" in prepared.columns
    assert len(prepared["factor_value"]) == len(df)


def test_prepare_factor_data_python_not_allowed() -> None:
    config = EvaluationAgentConfig(min_data_points=10, allow_python_factors=False)
    agent = FactorEvaluationAgent(config=config)
    df = _sample_data()
    code = "def factor(df):\n    return df['close']\n"

    with pytest.raises(DataValidationError):
        agent._prepare_factor_data(df, code, "factor")


def test_prepare_factor_data_qlib_expression(monkeypatch: pytest.MonkeyPatch) -> None:
    config = EvaluationAgentConfig(min_data_points=10)
    agent = FactorEvaluationAgent(config=config)
    df = _sample_data()

    monkeypatch.setattr(
        agent._expression_engine,
        "compute_expression",
        lambda expression, df, result_name: df["close"],
    )

    prepared = agent._prepare_factor_data(df, "$close", "close_factor")
    assert "factor_value" in prepared.columns


def test_calculate_summary() -> None:
    agent = FactorEvaluationAgent(config=EvaluationAgentConfig(min_data_points=10))

    metrics = FactorMetrics(ic=0.05, ir=1.2, sharpe_ratio=1.1)
    result = EvaluationResult(
        factor_name="alpha",
        factor_family="momentum",
        metrics=metrics,
        passes_threshold=True,
        threshold_used=0.03,
        trial_id="trial_1",
    )
    results = [
        EvaluationAgentResult(factor_name="alpha", success=True, result=result),
        EvaluationAgentResult(factor_name="beta", success=False, error="failed"),
    ]

    summary = agent._calculate_summary(results)
    assert summary["total_evaluated"] == 2
    assert summary["passed_count"] == 1


def test_parse_llm_insights() -> None:
    agent = FactorEvaluationAgent(config=EvaluationAgentConfig(min_data_points=10))
    response = '{"analysis": "ok", "insights": ["a"], "recommendations": ["b"]}'
    parsed = agent._parse_llm_insights(response)
    assert parsed["analysis"] == "ok"
    assert parsed["insights"] == ["a"]


def test_generate_template_insights_pass_fail() -> None:
    agent = FactorEvaluationAgent(config=EvaluationAgentConfig(min_data_points=10))

    passed = agent._generate_template_insights(
        "alpha", {"ic": 0.05, "ir": 1.2}, passes_threshold=True
    )
    failed = agent._generate_template_insights(
        "beta", {"ic": 0.01, "ir": 0.1}, passes_threshold=False
    )

    assert "passed evaluation" in passed["analysis"].lower()
    assert "failed evaluation" in failed["analysis"].lower()


def test_evaluate_single_and_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = FactorEvaluationAgent(config=EvaluationAgentConfig(min_data_points=10, max_factors_per_batch=1))
    df = _sample_data()

    class _StubEvaluator:
        def evaluate(self, factor_name: str, factor_family: str, data: pd.DataFrame) -> EvaluationResult:
            metrics = FactorMetrics(ic=0.04, ir=1.1, sharpe_ratio=1.0)
            return EvaluationResult(
                factor_name=factor_name,
                factor_family=factor_family,
                metrics=metrics,
                passes_threshold=True,
                threshold_used=0.03,
                trial_id="trial_stub",
            )

    agent.evaluator = _StubEvaluator()

    single = agent.evaluate_single("alpha", "momentum", df)
    assert single.success

    batch = agent.evaluate_batch(
        [{"name": "alpha", "family": "momentum"}, {"name": "beta", "family": "value"}],
        df,
    )
    assert len(batch) == 1
