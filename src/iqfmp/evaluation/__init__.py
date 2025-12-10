"""Evaluation module for IQFMP.

Provides tools for factor evaluation and validation:
- Cross-validation splitters
- Performance metrics
- Stability analysis
- Research ledger for trial tracking
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
from iqfmp.evaluation.research_ledger import (
    DynamicThreshold,
    FileStorage,
    InvalidTrialError,
    LedgerStatistics,
    LedgerStorage,
    MemoryStorage,
    ResearchLedger,
    ThresholdConfig,
    ThresholdExceededWarning,
    TrialRecord,
)
from iqfmp.evaluation.stability_analyzer import (
    InsufficientDataError,
    InvalidDataError,
    MarketRegime,
    MarketStabilityAnalyzer,
    MarketStabilityResult,
    RegimeStabilityAnalyzer,
    RegimeStabilityResult,
    StabilityAnalyzer,
    StabilityConfig,
    StabilityReport,
    StabilityScore,
    TimeStabilityAnalyzer,
    TimeStabilityResult,
)
from iqfmp.evaluation.factor_evaluator import (
    CVResult,
    EvaluationConfig,
    EvaluationFailedError,
    EvaluationPipeline,
    EvaluationResult,
    FactorEvaluator,
    FactorMetrics,
    FactorReport,
    InvalidFactorError,
    MetricsCalculator,
)

__all__ = [
    # CV Splitter
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
    # Research Ledger
    "DynamicThreshold",
    "FileStorage",
    "InvalidTrialError",
    "LedgerStatistics",
    "LedgerStorage",
    "MemoryStorage",
    "ResearchLedger",
    "ThresholdConfig",
    "ThresholdExceededWarning",
    "TrialRecord",
    # Stability Analyzer
    "InsufficientDataError",
    "InvalidDataError",
    "MarketRegime",
    "MarketStabilityAnalyzer",
    "MarketStabilityResult",
    "RegimeStabilityAnalyzer",
    "RegimeStabilityResult",
    "StabilityAnalyzer",
    "StabilityConfig",
    "StabilityReport",
    "StabilityScore",
    "TimeStabilityAnalyzer",
    "TimeStabilityResult",
    # Factor Evaluator
    "CVResult",
    "EvaluationConfig",
    "EvaluationFailedError",
    "EvaluationPipeline",
    "EvaluationResult",
    "FactorEvaluator",
    "FactorMetrics",
    "FactorReport",
    "InvalidFactorError",
    "MetricsCalculator",
]
