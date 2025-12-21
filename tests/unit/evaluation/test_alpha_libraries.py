"""Smoke tests for Qlib-based alpha factor libraries.

These tests execute all registered factors through Qlib expression engine
to ensure the factor expressions are valid and computable.

P0-2 Migration: All factors now use Qlib expressions (no more Pandas functions).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iqfmp.evaluation.qlib_factor_library import (
    QLIB_FACTOR_EXPRESSIONS,
    ALPHA158_EXPRESSIONS,
    ADDITIONAL_EXPRESSIONS,
    list_factors,
    list_factors_by_category,
)
from iqfmp.evaluation.alpha_derivatives import DERIVATIVE_FACTORS
from iqfmp.core.qlib_crypto import QlibExpressionEngine, QLIB_AVAILABLE


def _make_sample_df(rows: int = 200) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    rng = np.random.default_rng(seed=7)
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = pd.Series(100 + rng.normal(0, 1, size=rows)).cumsum()
    open_ = close + rng.normal(0, 0.5, size=rows)
    high = np.maximum(open_, close) + rng.random(rows)
    low = np.minimum(open_, close) - rng.random(rows)
    volume = rng.uniform(100, 1000, size=rows)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "funding_rate": rng.normal(0, 0.001, size=rows),
            "open_interest": rng.uniform(1000, 2000, size=rows),
            "liquidation_volume": rng.uniform(0, 100, size=rows),
            "long_short_ratio": rng.uniform(0.5, 1.5, size=rows),
            "basis_pct": rng.normal(0, 0.001, size=rows),
            "bid_ask_spread": rng.uniform(0.0001, 0.001, size=rows),
            "orderbook_imbalance": rng.uniform(-1, 1, size=rows),
            "premium_index": rng.normal(0, 0.001, size=rows),
            "taker_buy_volume": rng.uniform(0, 1000, size=rows),
            "taker_sell_volume": rng.uniform(0, 1000, size=rows),
        },
        index=dates,
    )
    return df


def _assert_series_like(result: object, expected_len: int) -> None:
    """Assert result is a valid series-like object."""
    if isinstance(result, pd.Series):
        assert len(result) == expected_len
        return
    if isinstance(result, np.ndarray):
        assert result.shape[0] == expected_len
        return
    assert result is not None


class TestQlibFactorExpressions:
    """Tests for Qlib-based factor expressions."""

    def test_alpha158_expressions_available(self) -> None:
        """Verify Alpha158 expressions are loaded."""
        assert len(ALPHA158_EXPRESSIONS) >= 40, f"Expected 40+ Alpha158 expressions, got {len(ALPHA158_EXPRESSIONS)}"

    def test_additional_expressions_available(self) -> None:
        """Verify additional expressions are loaded."""
        assert len(ADDITIONAL_EXPRESSIONS) >= 20, f"Expected 20+ additional expressions, got {len(ADDITIONAL_EXPRESSIONS)}"

    def test_total_expressions_available(self) -> None:
        """Verify total expression count."""
        total = len(QLIB_FACTOR_EXPRESSIONS)
        assert total >= 60, f"Expected 60+ total expressions, got {total}"

    def test_list_factors_function(self) -> None:
        """Test list_factors utility function."""
        factors = list_factors()
        assert len(factors) == len(QLIB_FACTOR_EXPRESSIONS)

    def test_list_factors_by_category(self) -> None:
        """Test list_factors_by_category utility function."""
        categories = list_factors_by_category()
        assert isinstance(categories, dict)
        assert len(categories) > 0

    @pytest.mark.skipif(not QLIB_AVAILABLE, reason="Qlib not available")
    def test_alpha158_expressions_valid_syntax(self) -> None:
        """Verify Alpha158 expressions have valid Qlib syntax."""
        df = _make_sample_df()
        engine = QlibExpressionEngine()

        failed = []
        for name, expression in ALPHA158_EXPRESSIONS.items():
            try:
                result = engine.compute_expression(expression, df, name)
                _assert_series_like(result, len(df))
            except Exception as e:
                failed.append(f"{name}: {e}")

        assert len(failed) == 0, f"Failed expressions:\n" + "\n".join(failed)

    @pytest.mark.skipif(not QLIB_AVAILABLE, reason="Qlib not available")
    def test_sample_alpha158_factors(self) -> None:
        """Test a sample of Alpha158 factors."""
        df = _make_sample_df()
        engine = QlibExpressionEngine()

        sample_factors = ["KMID", "ROC5", "STD5", "MA5"]
        for name in sample_factors:
            if name in ALPHA158_EXPRESSIONS:
                result = engine.compute_expression(
                    ALPHA158_EXPRESSIONS[name], df, name
                )
                assert isinstance(result, pd.Series)
                assert len(result) == len(df)


class TestDerivativeFactors:
    """Tests for derivative-specific factors."""

    def test_derivative_factors_available(self) -> None:
        """Verify derivative factors are registered."""
        assert len(DERIVATIVE_FACTORS) > 0

    def test_derivative_factors_execute(self) -> None:
        """Verify derivative factors can execute."""
        df = _make_sample_df()

        for name, func in DERIVATIVE_FACTORS.items():
            result = func(df)
            _assert_series_like(result, len(df))


class TestFactorPerformance:
    """Performance tests for factor computation."""

    @pytest.mark.skipif(not QLIB_AVAILABLE, reason="Qlib not available")
    def test_batch_computation_performance(self) -> None:
        """Test batch computation is reasonably fast."""
        import time

        df = _make_sample_df(500)
        engine = QlibExpressionEngine()

        start = time.time()
        computed = 0
        for name, expression in list(ALPHA158_EXPRESSIONS.items())[:20]:
            try:
                engine.compute_expression(expression, df, name)
                computed += 1
            except Exception:
                pass
        elapsed = time.time() - start

        # Should compute 20 factors in under 2 seconds
        assert elapsed < 2.0, f"Computed {computed} factors in {elapsed:.2f}s (too slow)"
        assert computed >= 15, f"Only computed {computed}/20 factors"
