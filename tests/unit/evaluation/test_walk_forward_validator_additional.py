"""Additional tests for walk-forward validator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iqfmp.evaluation import walk_forward_validator


class _DummyCalculator:
    def calculate_rank_ic(self, x: pd.Series, y: pd.Series) -> float:
        return float(pd.Series(x).corr(pd.Series(y), method="spearman") or 0.0)

    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        std = returns.std()
        return float(returns.mean() / std * np.sqrt(252)) if std > 0 else 0.0


def _sample_data(rows: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed=5)
    dates = pd.date_range("2023-01-01", periods=rows, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "factor_value": rng.normal(0, 1, size=rows),
            "forward_return": rng.normal(0, 0.01, size=rows),
        }
    )


def test_walk_forward_validate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(walk_forward_validator, "MetricsCalculator", _DummyCalculator)
    config = walk_forward_validator.WalkForwardConfig(
        window_size=20,
        step_size=10,
        min_train_samples=10,
        min_test_samples=5,
        use_deflated_sharpe=True,
        detect_ic_decay=True,
    )
    validator = walk_forward_validator.WalkForwardValidator(config=config)
    result = validator.validate(_sample_data())

    assert result.n_windows > 0
    assert isinstance(result.to_dict(), dict)
    assert result.get_diagnosis()


def test_walk_forward_insufficient_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(walk_forward_validator, "MetricsCalculator", _DummyCalculator)
    validator = walk_forward_validator.WalkForwardValidator(
        config=walk_forward_validator.WalkForwardConfig(window_size=30, min_test_samples=10)
    )

    with pytest.raises(walk_forward_validator.InsufficientDataError):
        validator.validate(_sample_data(rows=20))
