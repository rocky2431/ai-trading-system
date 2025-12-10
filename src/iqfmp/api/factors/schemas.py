"""Factor API schemas."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class FactorGenerateRequest(BaseModel):
    """Request to generate a new factor."""

    description: str = Field(..., min_length=1, max_length=5000)
    family: list[str] = Field(default_factory=list)
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
