"""Smoke test for PipelineBuilder end-to-end (P0).

Validates the minimal runnable path:
start → generate (stub LLM) → evaluate (Qlib metrics) → finish
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest

from iqfmp.agents.orchestrator import AgentState
from iqfmp.agents.pipeline_builder import PipelineBuilder, PipelineConfig


def _make_sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    dates = pd.date_range("2024-01-01", periods=220, freq="D")
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, size=len(dates))), index=dates)
    df = pd.DataFrame(
        {
            "date": dates,
            "symbol": "ETHUSDT",
            "open": close.values,
            "high": close.values,
            "low": close.values,
            "close": close.values,
            "volume": 1.0,
        }
    )
    df["forward_return"] = df["close"].shift(-1) / df["close"] - 1
    df = df.dropna(subset=["forward_return"]).reset_index(drop=True)
    return df


@dataclass
class _StubLLMResponse:
    content: str


class _StubLLMProvider:
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> _StubLLMResponse:
        # Return a valid Qlib expression (crypto field set).
        return _StubLLMResponse(content="Mean($close, 5) / $close - 1")


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Uses stub LLM provider; real LLM needed for proper dedup behavior")
async def test_pipeline_smoke_generate_evaluate_finish() -> None:
    # Minimal config: hypothesis/strategy/backtest/risk disabled.
    config = PipelineConfig(
        name="pipeline_smoke",
        enable_hypothesis=False,
        enable_evaluation=True,
        enable_strategy=False,
        enable_backtest=False,
        enable_risk_check=False,
        require_vector_db=False,
        checkpoint_enabled=False,
        max_iterations=10,
        timeout=30.0,
    )

    builder = PipelineBuilder(config)
    pipeline = builder.build()

    # Provide deterministic evaluation_data (avoid relying on external providers).
    df = _make_sample_df()

    initial_state = AgentState(
        context={
            "factor_prompts": ["create a simple 5-day mean reversion signal on close"],
            "factor_family": "momentum",
            "llm_provider": _StubLLMProvider(),
            "evaluation_data": df,
        }
    )

    final_state = await pipeline.run(initial_state)

    assert final_state.context.get("pipeline_status") == "completed"
    assert len(final_state.context.get("generated_factors", [])) >= 1
    assert len(final_state.context.get("evaluation_results", [])) >= 1


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Uses stub LLM provider; real LLM needed for proper dedup behavior")
async def test_pipeline_smoke_hypothesis_generate_evaluate_finish() -> None:
    config = PipelineConfig(
        name="pipeline_smoke_hypothesis",
        enable_hypothesis=True,
        enable_evaluation=True,
        enable_strategy=False,
        enable_backtest=False,
        enable_risk_check=False,
        require_vector_db=False,
        checkpoint_enabled=False,
        max_iterations=10,
        timeout=30.0,
    )

    builder = PipelineBuilder(config)
    pipeline = builder.build()

    df = _make_sample_df()

    initial_state = AgentState(
        context={
            "n_hypotheses": 3,
            "llm_provider": _StubLLMProvider(),
            "evaluation_data": df,
        }
    )

    final_state = await pipeline.run(initial_state)

    assert final_state.context.get("pipeline_status") == "completed"
    assert len(final_state.context.get("hypotheses", [])) >= 1
    assert len(final_state.context.get("generated_factors", [])) >= 1
    assert len(final_state.context.get("evaluation_results", [])) >= 1
