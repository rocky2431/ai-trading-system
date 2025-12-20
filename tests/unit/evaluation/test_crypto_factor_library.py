"""Tests for crypto factor expressions in QlibFactorLibrary (Task 15).

These tests ensure the baseline crypto factor library is:
- Present (names are registered)
- Syntactically valid under ExpressionGate(FieldSet.CRYPTO)
"""

from __future__ import annotations

from iqfmp.evaluation.qlib_factor_library import get_factor_expression
from iqfmp.llm.validation import ExpressionGate, FieldSet


def test_crypto_factor_expressions_are_valid() -> None:
    gate = ExpressionGate(field_set=FieldSet.CRYPTO)

    factor_names = [
        "CRYPTO_FUNDING_ZSCORE",
        "CRYPTO_FUNDING_MOMENTUM",
        "CRYPTO_PREMIUM_ZSCORE",
        "CRYPTO_OI_CHANGE_1",
        "CRYPTO_OI_ZSCORE",
        "CRYPTO_OI_PRICE_DIVERGENCE",
        "CRYPTO_LIQUIDATION_SPIKE",
        "CRYPTO_BASIS_PCT",
        "CRYPTO_ORDERBOOK_IMBALANCE",
        "CRYPTO_SPREAD_NORM",
        "CRYPTO_LONG_SHORT_SKEW",
    ]

    for name in factor_names:
        expr = get_factor_expression(name)
        result = gate.validate(expr)
        assert result.is_valid, f"{name} invalid: {result.error_message}"

