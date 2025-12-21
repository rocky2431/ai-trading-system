"""Tests for QlibFactorLibrary strict (Qlib-only) behavior."""

from __future__ import annotations

import pandas as pd
import pytest

from iqfmp.evaluation import qlib_factor_library as qfl


def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.05, 1.2],
            "high": [1.1, 1.2, 1.15, 1.25],
            "low": [0.9, 1.0, 1.0, 1.1],
            "close": [1.0, 1.15, 1.1, 1.22],
            "volume": [100, 110, 120, 130],
        }
    )


def test_list_factors_and_categories() -> None:
    factors = qfl.list_factors()
    assert len(factors) > 0

    categories = qfl.list_factors_by_category()
    assert isinstance(categories, dict)
    assert len(categories) > 0
    assert any("alpha101" in key for key in categories.keys())


def test_get_factor_expression_missing() -> None:
    with pytest.raises(KeyError):
        qfl.get_factor_expression("NOT_A_FACTOR")


def test_compute_expression_requires_qlib(monkeypatch: pytest.MonkeyPatch) -> None:
    library = qfl.QlibFactorLibrary()
    monkeypatch.setattr(library, "_qlib_available", False)

    with pytest.raises(qfl.QlibUnavailableError):
        library.compute_expression("$close", _make_df(), "factor")


def test_compute_expression_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    library = qfl.QlibFactorLibrary()
    df = _make_df()

    monkeypatch.setattr(library, "_qlib_available", True)
    monkeypatch.setattr(library, "_prepare_data", lambda data: {"$close": data["close"]})
    monkeypatch.setattr(library, "_evaluate", lambda expr, data: data["$close"])

    result = library.compute_expression("$close", df, "factor")
    assert isinstance(result, pd.Series)
    assert result.name == "factor"
    assert len(result) == len(df)


def test_compute_batch_handles_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    library = qfl.QlibFactorLibrary()
    df = _make_df()

    monkeypatch.setattr(library, "_qlib_available", True)

    def _fake_compute(name: str, data: pd.DataFrame) -> pd.Series:
        if name == "BAD":
            raise ValueError("boom")
        return data["close"]

    monkeypatch.setattr(library, "compute", _fake_compute)

    result = library.compute_batch(["OK", "BAD"], df)
    assert "OK" in result.columns
    assert "BAD" in result.columns
    assert result["BAD"].isna().all()


def test_prepare_data_maps_numeric_fields_and_vwap() -> None:
    library = qfl.QlibFactorLibrary()
    df = pd.DataFrame(
        {
            "open": [1.0, 1.1],
            "high": [1.2, 1.3],
            "low": [0.9, 1.0],
            "close": [1.05, 1.15],
            "volume": [100, 120],
            "funding_rate": [0.001, 0.002],
            "symbol": ["BTC", "ETH"],
        }
    )
    data = library._prepare_data(df)
    assert "$open" in data
    assert "$funding_rate" in data
    assert "funding_rate" in data
    assert "$vwap" in data
    assert "symbol" not in data


def test_evaluate_expression_with_ops() -> None:
    library = qfl.QlibFactorLibrary()
    df = _make_df()
    library._ops_cache = {"Mean": lambda s, n: s.rolling(n, min_periods=1).mean()}
    data = library._prepare_data(df)
    data["close"] = df["close"]
    result = library._evaluate("Mean(close, 2)", data)
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)


def test_compute_factor_convenience(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _make_df()

    monkeypatch.setattr(
        qfl.QlibFactorLibrary,
        "compute",
        lambda self, name, data: data["close"],
    )
    result = qfl.compute_factor("KMID", df)
    assert isinstance(result, pd.Series)


def test_compute_factors_convenience(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _make_df()

    monkeypatch.setattr(
        qfl.QlibFactorLibrary,
        "compute_batch",
        lambda self, names, data: pd.DataFrame({name: data["close"] for name in names}),
    )
    result = qfl.compute_factors(["KMID", "ROC5"], df)
    assert list(result.columns) == ["KMID", "ROC5"]
