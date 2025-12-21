"""Additional tests for redundancy detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from iqfmp.evaluation.redundancy_detector import (
    RedundancyDetector,
    RedundancyConfig,
    IncrementalRedundancyChecker,
)


def _factor_data(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed=12)
    base = rng.normal(0, 1, size=rows)
    return pd.DataFrame(
        {
            "factor_a": base + rng.normal(0, 0.01, size=rows),
            "factor_b": base + rng.normal(0, 0.01, size=rows),
            "factor_c": rng.normal(0, 1, size=rows),
        }
    )


def test_redundancy_detector_selects_best() -> None:
    config = RedundancyConfig(correlation_threshold=0.8, min_samples=30)
    detector = RedundancyDetector(config=config)
    data = _factor_data()
    metrics = {
        "factor_a": {"sharpe": 1.0},
        "factor_b": {"sharpe": 0.5},
        "factor_c": {"sharpe": 0.8},
    }

    result = detector.detect(data, metrics)
    assert result.total_factors == 3
    assert result.retained_count >= 2
    assert "factor_a" in result.retained_factors
    assert "factor_b" in result.removed_factors


def test_pairwise_correlations() -> None:
    detector = RedundancyDetector(RedundancyConfig(correlation_threshold=0.7, min_samples=30))
    pairs = detector.get_pairwise_correlations(_factor_data())
    assert pairs
    assert pairs[0][2] >= 0.7


def test_incremental_redundancy_checker() -> None:
    data = _factor_data()
    checker = IncrementalRedundancyChecker(data, correlation_threshold=0.8)

    new_factor = data["factor_a"] + 0.001
    redundant, most_corr, corr = checker.is_redundant(new_factor, "factor_new")
    assert redundant
    assert most_corr in {"factor_a", "factor_b"}
    assert corr >= 0.8
