"""Pipeline API schemas."""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


PipelineType = Literal["factor_evaluation", "strategy_backtest", "full_pipeline"]
PipelineStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
RDLoopPhase = Literal[
    "initializing",
    "hypothesis_generation",
    "factor_coding",
    "factor_evaluation",
    "benchmark_comparison",
    "feedback_analysis",
    "factor_combination",
    "factor_selection",
    "completed",
]


# =============================================================================
# RD Loop Schemas
# =============================================================================


class RDLoopConfigRequest(BaseModel):
    """RD Loop configuration request."""

    max_iterations: int = Field(default=100, ge=1, le=1000, description="Maximum iterations")
    max_hypotheses_per_iteration: int = Field(default=5, ge=1, le=20)
    target_core_factors: int = Field(default=10, ge=1, le=100)
    ic_threshold: float = Field(default=0.03, ge=0.0, le=1.0)
    ir_threshold: float = Field(default=1.0, ge=0.0)
    novelty_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    run_benchmark: bool = True
    enable_combination: bool = True
    focus_families: Optional[list[str]] = None


class RDLoopRunRequest(BaseModel):
    """Request to start RD Loop."""

    config: RDLoopConfigRequest = Field(default_factory=RDLoopConfigRequest)
    data_source: Optional[str] = Field(default=None, description="Path to OHLCV CSV or 'default'")


class RDLoopRunResponse(BaseModel):
    """RD Loop run response."""

    run_id: str
    status: str
    message: str
    created_at: datetime


class RDLoopStateResponse(BaseModel):
    """RD Loop state response."""

    run_id: str
    phase: RDLoopPhase
    iteration: int
    total_hypotheses_tested: int
    core_factors_count: int
    core_factors: list[str]
    is_running: bool
    stop_requested: bool


class RDLoopIterationResult(BaseModel):
    """Single iteration result."""

    iteration: int
    hypotheses_tested: int
    factors_validated: int
    best_ic: float
    best_factor_name: str
    benchmark_rank: Optional[int] = None
    phase_durations: dict[str, float] = Field(default_factory=dict)
    timestamp: datetime


class RDLoopStatisticsResponse(BaseModel):
    """RD Loop statistics response."""

    run_id: str
    state: RDLoopStateResponse
    hypothesis_stats: dict[str, Any] = Field(default_factory=dict)
    ledger_stats: dict[str, Any] = Field(default_factory=dict)
    benchmark_top_factors: list[dict[str, Any]] = Field(default_factory=list)
    iteration_results: list[RDLoopIterationResult] = Field(default_factory=list)


class RDLoopCoreFactorResponse(BaseModel):
    """Core factor response."""

    name: str
    family: str
    code: str
    metrics: dict[str, Any]
    hypothesis: dict[str, Any]


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
