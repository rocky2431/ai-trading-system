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
    """Factor statistics response."""

    total_factors: int
    by_status: dict[str, int]
    total_trials: int
    current_threshold: float


# ============== Mining Task Schemas ==============

class MiningTaskCreateRequest(BaseModel):
    """Request to create a factor mining task."""

    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="")
    factor_families: list[str] = Field(default_factory=list)
    target_count: int = Field(default=10, ge=1, le=100)
    auto_evaluate: bool = Field(default=True)


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
