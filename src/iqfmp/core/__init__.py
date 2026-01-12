"""Core engine modules for IQFMP."""

from iqfmp.core.review import (
    HumanReviewGate,
    NotifierBase,
    RedisReviewQueue,
    ReviewConfig,
    ReviewDecision,
    ReviewQueue,
    ReviewQueueError,
    ReviewRequest,
    ReviewStatus,
    get_review_queue,
)
from iqfmp.core.sandbox import (
    ExecutionResult,
    ExecutionStatus,
    SandboxConfig,
    SandboxExecutor,
)
from iqfmp.core.security import (
    ASTSecurityChecker,
    SecurityCheckResult,
    SecurityViolation,
    ViolationType,
)
from iqfmp.core.factor_engine import (
    FactorEngine,
    FactorEvaluator,
    BUILTIN_FACTORS,
    create_engine_with_sample_data,
    get_default_data_path,
)
from iqfmp.core.backtest_engine import (
    BacktestEngine,
    BacktestResult,
    Trade,
    TradingCosts,
    run_strategy_backtest,
)
from iqfmp.core.rd_loop import (
    IterationResult,
    LoopConfig,
    LoopPhase,
    LoopState,
    RDLoop,
    create_rd_loop,
)
from iqfmp.core.crypto_backtest import (
    CryptoQlibBacktest,
    CryptoBacktestConfig,
    CryptoBacktestResult,
    CryptoExchange,
    InsufficientDataError,
    run_crypto_backtest,
)

__all__ = [
    # Security
    "ASTSecurityChecker",
    "SecurityCheckResult",
    "SecurityViolation",
    "ViolationType",
    # Sandbox
    "ExecutionResult",
    "ExecutionStatus",
    "SandboxConfig",
    "SandboxExecutor",
    # Review
    "HumanReviewGate",
    "NotifierBase",
    "RedisReviewQueue",
    "ReviewConfig",
    "ReviewDecision",
    "ReviewQueue",
    "ReviewQueueError",
    "ReviewRequest",
    "ReviewStatus",
    "get_review_queue",
    # Factor Engine
    "FactorEngine",
    "FactorEvaluator",
    "BUILTIN_FACTORS",
    "create_engine_with_sample_data",
    "get_default_data_path",
    # Backtest Engine
    "BacktestEngine",
    "BacktestResult",
    "Trade",
    "TradingCosts",
    "run_strategy_backtest",
    # RD Loop
    "IterationResult",
    "LoopConfig",
    "LoopPhase",
    "LoopState",
    "RDLoop",
    "create_rd_loop",
    # Crypto Backtest (Unified Engine)
    "CryptoQlibBacktest",
    "CryptoBacktestConfig",
    "CryptoBacktestResult",
    "CryptoExchange",
    "InsufficientDataError",
    "run_crypto_backtest",
]
