"""Qlib Deep Fork for Cryptocurrency Trading.

This module provides the public interface to the Qlib crypto extension,
which extends Qlib to support cryptocurrency-specific data fields
and trading features including:
- Funding rates (perpetual futures)
- Open interest
- Basis (futures vs spot spread)
- Multi-exchange support (Binance, OKX, Bybit)

Deep Fork Strategy:
The actual implementation lives in vendor/qlib/qlib/contrib/crypto/,
which is a direct extension of the forked Qlib repository.
This module re-exports those components for convenient access.
"""

# Re-export from forked Qlib's crypto extension
from qlib.contrib.crypto import (
    QLIB_AVAILABLE,
    CryptoDataConfig,
    CryptoDataHandler,
    CryptoField,
    DataValidator,
    Exchange,
    TimeFrame,
    ValidationError,
    ValidationResult,
)

# Flag indicating Qlib deep fork is active
QLIB_DEEP_FORK = True

__all__ = [
    # Flags
    "QLIB_DEEP_FORK",
    "QLIB_AVAILABLE",
    # Handler
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
