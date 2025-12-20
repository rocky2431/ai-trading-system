"""Qlib initialization module for IQFMP.

Provides proper Qlib initialization with crypto data support.
Handles initialization on API startup with fallback options.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Global initialization state
_qlib_initialized = False
_qlib_provider_uri: Optional[str] = None


def get_qlib_data_dir() -> Path:
    """Get Qlib data directory from environment or default."""
    env_dir = os.environ.get("QLIB_DATA_DIR")
    if env_dir:
        return Path(env_dir)

    # Default to ~/.qlib/qlib_data
    return Path.home() / ".qlib" / "qlib_data"


def init_qlib(
    provider_uri: Optional[str] = None,
    region: str = "cn",
    kernels: int = 1,
    redis_host: Optional[str] = None,
    redis_port: int = 6379,
    expression_cache: Optional[str] = None,
    **kwargs,
) -> bool:
    """Initialize Qlib with configuration.

    This function should be called once at API startup.
    Subsequent calls will be ignored if already initialized.

    Args:
        provider_uri: Data provider URI (default: ~/.qlib/qlib_data)
        region: Market region ("cn" for China A-share, "us" for US stock)
        kernels: Number of kernels for parallel computation
        redis_host: Redis host for expression caching (optional)
        redis_port: Redis port
        expression_cache: Expression cache type ("redis" or None)
        **kwargs: Additional Qlib config options

    Returns:
        True if initialization successful, False otherwise
    """
    global _qlib_initialized, _qlib_provider_uri

    if _qlib_initialized:
        logger.debug("Qlib already initialized, skipping")
        return True

    try:
        import qlib
        from qlib.config import C

        # Determine provider URI
        uri = provider_uri or str(get_qlib_data_dir())
        _qlib_provider_uri = uri

        # Build config
        config = {
            "region": region,
            "kernels": kernels,
            **kwargs,
        }

        # Add provider URI if data directory exists
        if Path(uri).exists():
            config["provider_uri"] = uri
            logger.info(f"Using Qlib data from: {uri}")
        else:
            logger.warning(f"Qlib data directory not found: {uri}")
            logger.info("Qlib will use expression engine only (no D.features())")

        # Add Redis caching if configured
        if redis_host and expression_cache == "redis":
            config["expression_cache"] = {
                "class": "RedisCache",
                "module_path": "qlib.data.cache",
                "host": redis_host,
                "port": redis_port,
                "db": 0,
            }

        # Initialize Qlib
        qlib.init(**config)

        _qlib_initialized = True
        logger.info(f"Qlib initialized: region={region}, uri={uri}")
        return True

    except ImportError as e:
        logger.error(f"Qlib not installed: {e}")
        return False

    except Exception as e:
        logger.warning(f"Qlib initialization failed: {e}")
        logger.info("Falling back to local expression engine")
        return False


def is_qlib_initialized() -> bool:
    """Check if Qlib is initialized."""
    return _qlib_initialized


def get_provider_uri() -> Optional[str]:
    """Get current Qlib provider URI."""
    return _qlib_provider_uri


def ensure_qlib_initialized() -> bool:
    """Ensure Qlib is initialized, initialize if not.

    Returns:
        True if Qlib is (now) initialized
    """
    if _qlib_initialized:
        return True
    return init_qlib()


def init_qlib_for_crypto(
    provider_uri: Optional[str] = None,
    instruments: Optional[list[str]] = None,
) -> bool:
    """Initialize Qlib specifically for crypto data.

    Args:
        provider_uri: Data provider URI (default: ~/.qlib/qlib_data/crypto)
        instruments: List of crypto instruments (e.g., ["BTCUSDT", "ETHUSDT"])

    Returns:
        True if initialization successful
    """
    # Default to crypto data directory
    uri = provider_uri
    if uri is None:
        crypto_dir = get_qlib_data_dir() / "crypto"
        if crypto_dir.exists():
            uri = str(crypto_dir)
        else:
            uri = str(get_qlib_data_dir())

    return init_qlib(
        provider_uri=uri,
        region="crypto",
    )


# Auto-initialize on import if environment variable set
if os.environ.get("QLIB_AUTO_INIT", "false").lower() == "true":
    init_qlib()
