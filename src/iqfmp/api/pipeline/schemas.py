"""Pipeline API schemas."""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


PipelineType = Literal["factor_evaluation", "strategy_backtest", "full_pipeline"]
PipelineStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


class PipelineConfig(BaseModel):
    """Pipeline configuration schema."""

    factor_id: Optional[str] = None
    strategy_id: Optional[str] = None
    date_range: Optional[list[str]] = None
    symbols: list[str] = Field(default_factory=list)
    extra_params: dict[str, Any] = Field(default_factory=dict)


class PipelineRunRequest(BaseModel):
    """Request to run a pipeline."""

    pipeline_type: PipelineType
    config: dict[str, Any] = Field(default_factory=dict)


class PipelineRunResponse(BaseModel):
    """Pipeline run response schema."""

    run_id: str
    status: PipelineStatus
    created_at: datetime


class PipelineStatusResponse(BaseModel):
    """Pipeline status response schema."""

    run_id: str
    status: PipelineStatus
    progress: float = 0.0
    current_step: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class PipelineListResponse(BaseModel):
    """Pipeline list response schema."""

    runs: list[PipelineStatusResponse]
    total: int


class WebSocketMessage(BaseModel):
    """WebSocket message schema."""

    type: Literal["status", "progress", "error", "result"]
    data: dict[str, Any]
