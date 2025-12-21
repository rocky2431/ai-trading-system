"""Additional tests for alpha_benchmark utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from iqfmp.evaluation.alpha_benchmark import (
    AlphaBenchmarker,
    BenchmarkResult,
    create_alpha_benchmarker,
)


def _sample_df(rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(seed=4)
    close = pd.Series(100 + rng.normal(0, 1, size=rows)).cumsum()
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.5, size=rows),
            "high": close + rng.normal(1, 0.5, size=rows),
            "low": close - rng.normal(1, 0.5, size=rows),
            "close": close,
            "volume": rng.uniform(100, 1000, size=rows),
        }
    )


def test_create_alpha_benchmarker_filters_volume() -> None:
    benchmarker = create_alpha_benchmarker(include_volume_factors=False)
    assert all("$volume" not in expr for expr in benchmarker.expressions.values())


def test_alpha_benchmark_flow() -> None:
    df = _sample_df()
    forward_returns = df["close"].pct_change().shift(-1).fillna(0)

    benchmarker = AlphaBenchmarker(
        expressions={
            "ROC5": "Ref($close, 5) / $close - 1",
            "MA5_RATIO": "$close / Mean($close, 5) - 1",
        }
    )

    # Stub expression engine to avoid Qlib dependency
    benchmarker._engine.compute_expression = lambda expr, df_in: df_in["close"].pct_change().fillna(0)

    factors = benchmarker.compute_benchmark_factors(df)
    assert set(factors.columns) == {"ROC5", "MA5_RATIO"}

    metrics = benchmarker.evaluate_benchmark_factors(df, forward_returns)
    assert "ROC5" in metrics

    result = benchmarker.benchmark(
        "NEW_FACTOR",
        df["close"].pct_change().fillna(0),
        df,
        forward_returns,
    )
    assert isinstance(result, BenchmarkResult)
    assert "Alpha158 Benchmark Result" in result.get_summary()

    top = benchmarker.get_top_factors(n=1)
    assert len(top) <= 1
