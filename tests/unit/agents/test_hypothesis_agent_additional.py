"""Additional tests for hypothesis generation utilities."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from iqfmp.agents.hypothesis_agent import (
    HypothesisGenerator,
    HypothesisFamily,
    HypothesisToCode,
    Hypothesis,
    FeedbackAnalyzer,
    HypothesisAgent,
    HypothesisStatus,
)


def _sample_market_df(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed=13)
    close = pd.Series(100 + rng.normal(0, 1, size=rows)).cumsum()
    df = pd.DataFrame(
        {
            "close": close,
            "volume": rng.uniform(100, 1000, size=rows),
            "funding_rate": rng.normal(0, 0.001, size=rows),
        }
    )
    return df


def test_generate_from_analysis() -> None:
    generator = HypothesisGenerator()
    hypotheses = generator.generate_from_analysis(_sample_market_df())
    assert len(hypotheses) > 0
    assert all(h.family in HypothesisFamily for h in hypotheses)


def test_parse_llm_response() -> None:
    generator = HypothesisGenerator()
    payload = [
        {
            "name": "Test Hypothesis",
            "description": "Momentum should persist",
            "family": "momentum",
            "rationale": "Trend continuation",
            "expected_ic": 0.05,
            "expected_direction": "long",
        }
    ]
    response = json.dumps(payload)
    parsed = generator._parse_llm_response(response)
    assert len(parsed) == 1
    assert parsed[0].family == HypothesisFamily.MOMENTUM


def test_market_summary_generation() -> None:
    generator = HypothesisGenerator()
    df = _sample_market_df()
    conditions = generator._analyze_market_conditions(df)
    summary = generator._create_market_summary(df, conditions)
    assert "return" in summary.lower()


def test_hypothesis_to_code_template() -> None:
    converter = HypothesisToCode()
    hypothesis = Hypothesis(
        name="Price Momentum",
        description="Momentum effect",
        family=HypothesisFamily.MOMENTUM,
        rationale="Trend continuation",
    )
    expression = converter.convert(hypothesis)
    assert "$close" in expression
    assert "Ref" in expression


def test_extract_code_from_response() -> None:
    converter = HypothesisToCode()
    response = "```qlib\nMean($close, 5) - Mean($close, 20)\n```"
    expression = converter._extract_code(response, "factor_price_momentum")
    assert expression.startswith("Mean(")


def test_generate_from_template() -> None:
    generator = HypothesisGenerator()
    hypothesis = generator.generate_from_template(HypothesisFamily.MEAN_REVERSION)
    assert hypothesis.family == HypothesisFamily.MEAN_REVERSION


def test_feedback_analyzer_templates() -> None:
    analyzer = FeedbackAnalyzer()
    hypothesis = Hypothesis(
        name="Test",
        description="desc",
        family=HypothesisFamily.MOMENTUM,
        rationale="rationale",
    )
    feedback = analyzer._generate_rejection_feedback(hypothesis, {"ic": 0.0, "ir": 0.0})
    suggestions = analyzer._generate_suggestions(hypothesis, {"ic": 0.0, "ir": 0.0})

    assert "ic" in feedback.lower()
    assert isinstance(suggestions, list)


def test_hypothesis_agent_flow() -> None:
    agent = HypothesisAgent()
    market = _sample_market_df()

    hypotheses = agent.generate_hypotheses(market, n_hypotheses=3)
    assert len(hypotheses) == 3

    converted = agent.convert_to_factors(hypotheses)
    assert all("$" in code for _, code in converted)

    result = agent.process_results(
        hypotheses[0],
        {"metrics": {"ic": 0.05, "ir": 1.2}},
    )
    assert result.status == HypothesisStatus.VALIDATED

    stats = agent.get_statistics()
    assert stats["total_hypotheses_tested"] >= 1
