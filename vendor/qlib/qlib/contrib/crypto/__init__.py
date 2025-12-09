"""Qlib Crypto Extension for Cryptocurrency Trading.

This module extends Qlib to support cryptocurrency-specific data fields
and trading features including:
- Funding rates (perpetual futures)
- Open interest
- Basis (futures vs spot spread)
- Multi-exchange support (Binance, OKX, Bybit, etc.)
"""

from qlib.contrib.crypto.data.handler import (
    QLIB_AVAILABLE,
    CryptoDataConfig,
    CryptoDataHandler,
    CryptoField,
    Exchange,
    TimeFrame,
)
from qlib.contrib.crypto.data.validator import (
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
