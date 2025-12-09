"""Evaluation module for IQFMP.

Provides tools for factor evaluation and validation:
- Cross-validation splitters
- Performance metrics
- Stability analysis
"""

from iqfmp.evaluation.cv_splitter import (
    CryptoCVSplitter,
    CVSplitConfig,
    DataLeakageError,
    FrequencySplitter,
    InvalidSplitError,
    MarketGroup,
    MarketSplitter,
    SplitResult,
    TimeFrequency,
    TimeSplitter,
)

__all__ = [
    "CryptoCVSplitter",
    "CVSplitConfig",
    "DataLeakageError",
    "FrequencySplitter",
    "InvalidSplitError",
    "MarketGroup",
    "MarketSplitter",
    "SplitResult",
    "TimeFrequency",
    "TimeSplitter",
]
