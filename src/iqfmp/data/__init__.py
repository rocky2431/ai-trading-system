"""IQFMP Data Module.

This module provides data downloading and management for:
- OHLCV price data
- Derivative data (funding rates, open interest, liquidations, etc.)
- Time alignment utilities for multi-timeframe data
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
from .alignment import (
    FUNDING_SETTLEMENT_HOURS,
    get_funding_settlement_times,
    get_previous_settlement_time,
    get_next_settlement_time,
    align_funding_to_ohlcv,
    merge_derivative_data,
    calculate_funding_features,
    validate_time_alignment,
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
    # Time alignment
    "FUNDING_SETTLEMENT_HOURS",
    "get_funding_settlement_times",
    "get_previous_settlement_time",
    "get_next_settlement_time",
    "align_funding_to_ohlcv",
    "merge_derivative_data",
    "calculate_funding_features",
    "validate_time_alignment",
]
