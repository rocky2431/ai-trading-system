"""Qlib Deep Fork for Cryptocurrency Trading.

This module extends Qlib to support cryptocurrency-specific data fields
and trading features including:
- Funding rates
- Open interest
- Basis (futures vs spot spread)
- Multi-exchange support
"""

from iqfmp.qlib_crypto.data.handler import (
    QLIB_AVAILABLE,
    CryptoDataConfig,
    CryptoDataHandler,
    CryptoField,
    Exchange,
    TimeFrame,
)
from iqfmp.qlib_crypto.data.validator import (
    DataValidator,
    ValidationError,
    ValidationResult,
)

__all__ = [
    # Handler
    "QLIB_AVAILABLE",
    "CryptoDataHandler",
    "CryptoDataConfig",
    "CryptoField",
    "Exchange",
    "TimeFrame",
    # Validator
    "DataValidator",
    "ValidationResult",
    "ValidationError",
]
