"""IQFMP Data Module.

This module provides data downloading and management for:
- OHLCV price data
- Derivative data (funding rates, open interest, liquidations, etc.)
"""

from .downloader import (
    CCXTDownloader,
    execute_download_task,
    TIMEFRAME_MAPPING,
    TIMEFRAME_MS,
)
from .derivatives import (
    DerivativeDataType,
    DerivativeDownloader,
    get_derivative_downloader,
    download_all_derivative_data,
)

__all__ = [
    # OHLCV downloader
    "CCXTDownloader",
    "execute_download_task",
    "TIMEFRAME_MAPPING",
    "TIMEFRAME_MS",
    # Derivative downloader
    "DerivativeDataType",
    "DerivativeDownloader",
    "get_derivative_downloader",
    "download_all_derivative_data",
]
