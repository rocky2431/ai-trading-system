"""Factor API schemas."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class FactorGenerateRequest(BaseModel):
    """Request to generate a new factor using LLM."""

    description: str = Field(..., min_length=1, max_length=5000)
    family: list[str] = Field(default_factory=list)
    target_task: str = Field(default="price_prediction")


class FactorCreateRequest(BaseModel):
    """Request to create a factor with provided code."""

    name: str = Field(..., min_length=1, max_length=100)
    family: list[str] = Field(default_factory=list)
    code: str = Field(..., min_length=1, max_length=50000)
    target_task: str = Field(default="price_prediction")


class MetricsResponse(BaseModel):
    """Factor metrics response."""

    ic_mean: float
    ic_std: float
    ir: float
    sharpe: float
    max_drawdown: float
    turnover: float
    ic_by_split: dict[str, float] = Field(default_factory=dict)
    sharpe_by_split: dict[str, float] = Field(default_factory=dict)


class StabilityResponse(BaseModel):
    """Factor stability response."""

    time_stability: dict[str, float] = Field(default_factory=dict)
    market_stability: dict[str, float] = Field(default_factory=dict)
    regime_stability: dict[str, float] = Field(default_factory=dict)


class FactorResponse(BaseModel):
    """Factor response schema."""

    id: str
    name: str
    family: list[str]
    code: str
    code_hash: str
    target_task: str
    status: str
    metrics: Optional[MetricsResponse] = None
    stability: Optional[StabilityResponse] = None
    cluster_id: Optional[str] = None
    experiment_number: int = 0
    created_at: datetime


class FactorListResponse(BaseModel):
    """Factor list response schema."""

    factors: list[FactorResponse]
    total: int
    page: int
    page_size: int


class FactorEvaluateRequest(BaseModel):
    """Request to evaluate a factor."""

    splits: list[str] = Field(default=["train", "valid", "test"])
    market_splits: list[str] = Field(default_factory=list)
    frequency_splits: list[str] = Field(default_factory=list)


class FactorEvaluateResponse(BaseModel):
    """Factor evaluation response."""

    factor_id: str
    metrics: MetricsResponse
    stability: Optional[StabilityResponse] = None
    passed_threshold: bool
    experiment_number: int


FactorStatusType = Literal["candidate", "rejected", "core", "redundant"]


class FactorStatusUpdateRequest(BaseModel):
    """Request to update factor status."""

    status: FactorStatusType


class FactorStatsResponse(BaseModel):
    """Factor statistics response.

    Provides comprehensive statistics for monitoring dashboard.
    """

    # Basic counts
    total_factors: int
    by_status: dict[str, int]
    total_trials: int
    current_threshold: float

    # Extended fields for monitoring dashboard
    evaluated_count: int = 0  # Factors that have been evaluated (non-candidate)
    pass_rate: float = 0.0  # Percentage of factors that passed threshold (core)
    avg_ic: float = 0.0  # Average IC across all evaluated factors
    avg_sharpe: float = 0.0  # Average Sharpe across all evaluated factors
    pending_count: int = 0  # Factors pending evaluation (candidate status)


# ============== Mining Task Config Schemas ==============


class MiningDataConfig(BaseModel):
    """Data configuration for mining task."""

    start_date: str = Field(default="2022-01-01", description="Start date (YYYY-MM-DD)")
    end_date: str = Field(default="2024-12-01", description="End date (YYYY-MM-DD)")
    symbols: list[str] = Field(
        default=["BTC", "ETH"],
        description="Trading pair symbols"
    )
    timeframes: list[str] = Field(
        default=["4h", "1d"],
        description="Time intervals: 1h, 4h, 1d, 1w"
    )
    train_ratio: float = Field(default=0.6, ge=0.1, le=0.9, description="Training set ratio")
    valid_ratio: float = Field(default=0.2, ge=0.0, le=0.4, description="Validation set ratio")
    test_ratio: float = Field(default=0.2, ge=0.1, le=0.4, description="Test set ratio")


class MiningBenchmarkConfig(BaseModel):
    """Benchmark configuration for mining task."""

    benchmark_set: str = Field(
        default="alpha158",
        description="Benchmark factor set: alpha158, alpha101, alpha360, custom"
    )
    correlation_threshold: float = Field(
        default=0.70,
        ge=0.3,
        le=0.95,
        description="Redundancy correlation threshold"
    )
    custom_factors: list[str] = Field(
        default_factory=list,
        description="Custom factor IDs for comparison"
    )


class MiningModelConfig(BaseModel):
    """Model configuration for mining task."""

    models: list[str] = Field(
        default=["lightgbm"],
        description="Prediction models: lightgbm, xgboost, linear, catboost, ensemble"
    )
    optimization_method: str = Field(
        default="bayesian",
        description="Optimization algorithm: bayesian, genetic, grid, random, none"
    )
    max_trials: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Max optimization trials"
    )
    early_stopping_rounds: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Early stopping rounds"
    )


class MiningRobustnessConfig(BaseModel):
    """Robustness/anti-overfitting configuration for mining task."""

    # Walk-Forward validation
    use_walk_forward: bool = Field(default=True, description="Enable Walk-Forward validation")
    wf_window_size: int = Field(default=252, ge=63, le=756, description="Training window (days)")
    wf_step_size: int = Field(default=63, ge=21, le=126, description="Rolling step size (days)")
    wf_min_train_samples: int = Field(default=126, ge=30, description="Minimum training samples")

    # Dynamic threshold
    use_dynamic_threshold: bool = Field(default=True, description="Enable dynamic threshold")
    min_sharpe: float = Field(default=1.5, ge=0.5, le=3.0, description="Minimum Sharpe threshold")
    confidence_level: float = Field(default=0.95, ge=0.8, le=0.99, description="Confidence level")

    # IC decay detection
    use_ic_decay_detection: bool = Field(default=True, description="Enable IC decay detection")
    max_half_life: int = Field(default=60, ge=10, le=180, description="Max IC half-life (days)")

    # Redundancy filter
    use_redundancy_filter: bool = Field(default=True, description="Enable redundancy filter")
    cluster_threshold: float = Field(default=0.85, ge=0.5, le=0.95, description="Clustering threshold")


# ============== Mining Task Schemas ==============

class MiningTaskCreateRequest(BaseModel):
    """Request to create a factor mining task.

    Supports both basic and advanced configurations.
    Basic fields are kept for backward compatibility.
    """

    # Basic configuration (backward compatible)
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="")
    factor_families: list[str] = Field(default_factory=list)
    target_count: int = Field(default=10, ge=1, le=100)
    auto_evaluate: bool = Field(default=True)

    # Advanced configuration (optional, new fields)
    data_config: Optional[MiningDataConfig] = Field(
        default=None,
        description="Data configuration"
    )
    benchmark_config: Optional[MiningBenchmarkConfig] = Field(
        default=None,
        description="Benchmark configuration"
    )
    ml_config: Optional[MiningModelConfig] = Field(
        default=None,
        description="ML model configuration (renamed from model_config to avoid Pydantic reserved name)"
    )
    robustness_config: Optional[MiningRobustnessConfig] = Field(
        default=None,
        description="Robustness/anti-overfitting configuration"
    )


class MiningTaskStatus(BaseModel):
    """Mining task status response."""

    id: str
    name: str
    description: str
    factor_families: list[str]
    target_count: int
    generated_count: int
    passed_count: int
    failed_count: int
    status: str  # pending, running, completed, failed, cancelled
    progress: float  # 0-100
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class MiningTaskListResponse(BaseModel):
    """Mining task list response."""

    tasks: list[MiningTaskStatus]
    total: int


class MiningTaskCreateResponse(BaseModel):
    """Mining task creation response."""

    success: bool
    message: str
    task_id: Optional[str] = None


class MiningTaskCancelResponse(BaseModel):
    """Mining task cancel response."""

    success: bool
    message: str


# ============== Factor Library Schemas ==============

class FactorLibraryStats(BaseModel):
    """Factor library statistics."""

    total_factors: int
    core_factors: int
    candidate_factors: int
    rejected_factors: int
    redundant_factors: int
    by_family: dict[str, int]
    avg_sharpe: float
    avg_ic: float
    best_factor_id: Optional[str] = None
    best_sharpe: float = 0.0


class FactorCompareRequest(BaseModel):
    """Request to compare multiple factors."""

    factor_ids: list[str] = Field(..., min_items=2, max_items=10)


class FactorCompareResponse(BaseModel):
    """Factor comparison response."""

    factors: list[FactorResponse]
    correlation_matrix: dict[str, dict[str, float]]
    ranking: list[str]  # Factor IDs ranked by performance


# ============== Walk-Forward Validation Schemas ==============


class WalkForwardWindowResult(BaseModel):
    """Single Walk-Forward window result."""

    window: int
    train_ic: float
    oos_ic: float
    degradation: float
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class WalkForwardResult(BaseModel):
    """Walk-Forward validation result."""

    # Core metrics
    avg_train_ic: float
    avg_oos_ic: float  # Key: out-of-sample IC
    ic_degradation: float  # OOS degradation ratio
    oos_ir: float  # Out-of-sample IR

    # Distribution stats
    min_oos_ic: float
    max_oos_ic: float
    oos_ic_std: float

    # Robustness verdict
    ic_consistency: float  # IC stability score (0-1)
    passes_robustness: bool  # Passes robustness test?

    # Detailed results
    window_results: list[WalkForwardWindowResult] = Field(default_factory=list)


class ICDecompositionResult(BaseModel):
    """IC decomposition analysis result."""

    total_ic: float

    # Time decomposition
    ic_by_month: dict[str, float] = Field(default_factory=dict)
    ic_by_quarter: dict[str, float] = Field(default_factory=dict)

    # Market cap decomposition
    large_cap_ic: float = 0.0
    mid_cap_ic: float = 0.0
    small_cap_ic: float = 0.0

    # Volatility regime decomposition
    high_vol_ic: float = 0.0
    low_vol_ic: float = 0.0

    # Diagnostics
    regime_shift_detected: bool = False
    ic_decay_rate: float = 0.0
    predicted_half_life: int = 999

    # Recommendations
    diagnosis: str = ""
    recommendations: list[str] = Field(default_factory=list)


class RedundancyGroup(BaseModel):
    """Redundant factor group."""

    cluster_factors: list[str]
    best_factor: str
    avg_correlation: float


class RedundancyReport(BaseModel):
    """Redundancy detection report."""

    total_factors: int
    retained_factors: list[str]
    removed_factors: list[str]
    redundant_groups: list[RedundancyGroup] = Field(default_factory=list)
    factor_reduction_ratio: float


class ExtendedFactorEvaluateResponse(BaseModel):
    """Extended factor evaluation response with robustness metrics."""

    # Basic evaluation (same as FactorEvaluateResponse)
    factor_id: str
    metrics: MetricsResponse
    stability: Optional[StabilityResponse] = None
    passed_threshold: bool
    experiment_number: int

    # Walk-Forward validation (new)
    walk_forward: Optional[WalkForwardResult] = None

    # IC decomposition (new)
    ic_decomposition: Optional[ICDecompositionResult] = None

    # Benchmark comparison (new)
    benchmark_rank: Optional[float] = None  # Percentile rank in benchmark
    is_novel: Optional[bool] = None  # Novel vs existing factors
    correlation_with_best: Optional[float] = None

    # Redundancy check (new)
    redundancy_report: Optional[RedundancyReport] = None

    # Overall verdict
    overall_score: float = 0.0  # Composite score (0-100)
    verdict: str = "pending"  # pass, fail, needs_review
