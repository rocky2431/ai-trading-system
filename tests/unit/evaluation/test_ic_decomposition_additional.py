"""Additional tests for IC decomposition analyzer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iqfmp.evaluation import ic_decomposition


class _DummyCalculator:
    def calculate_rank_ic(self, x: pd.Series, y: pd.Series) -> float:
        return float(pd.Series(x).corr(pd.Series(y), method="spearman") or 0.0)


def _sample_data(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed=8)
    dates = pd.date_range("2023-01-01", periods=rows, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "factor_value": rng.normal(0, 1, size=rows),
            "forward_return": rng.normal(0, 0.01, size=rows),
            "market_cap": rng.uniform(1e8, 1e10, size=rows),
            "volatility": rng.uniform(0.01, 0.05, size=rows),
        }
    )


def test_ic_decomposition_analyze(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ic_decomposition, "MetricsCalculator", _DummyCalculator)

    analyzer = ic_decomposition.ICDecompositionAnalyzer()
    result = analyzer.analyze(_sample_data())

    assert result.n_periods > 0
    assert result.ic_by_month
    assert isinstance(result.to_dict(), dict)


def test_ic_decomposition_empty_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ic_decomposition, "MetricsCalculator", _DummyCalculator)
    analyzer = ic_decomposition.ICDecompositionAnalyzer()

    with pytest.raises(ic_decomposition.InsufficientDataError):
        analyzer.analyze(pd.DataFrame())
