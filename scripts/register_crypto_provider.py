"""Register IQFMP crypto data handler as a Qlib provider.

Usage:
    python scripts/register_crypto_provider.py --provider-uri ~/.qlib/qlib_data/crypto
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional


def register_crypto_provider(provider_uri: Optional[str]) -> bool:
    """Register CryptoDataHandler with Qlib Provider registry."""
    try:
        from qlib.data.dataset import provider
        from iqfmp.core.qlib_init import init_qlib
        from iqfmp.core.qlib_crypto import CryptoDataHandler
    except Exception as exc:  # pragma: no cover - utility script
        logging.error("Qlib not available: %s", exc)
        return False

    init_qlib(provider_uri=provider_uri, region="cn")

    try:
        provider.Provider.register("crypto", CryptoDataHandler)
        logging.info("Registered CryptoDataHandler as 'crypto' provider")
        return True
    except Exception as exc:  # pragma: no cover - utility script
        logging.error("Failed to register provider: %s", exc)
        return False


if __name__ == "__main__":  # pragma: no cover - script entry
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", type=str, default=None, help="Qlib data directory")
    args = parser.parse_args()
    success = register_crypto_provider(args.provider_uri)
    raise SystemExit(0 if success else 1)
