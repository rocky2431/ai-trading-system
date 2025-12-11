"""System API schemas."""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


AgentType = Literal["factor_generation", "evaluation", "strategy", "backtest"]
AgentStatusType = Literal["idle", "running", "paused", "error"]
TaskStatusType = Literal["pending", "running", "completed", "failed"]
TaskPriority = Literal["high", "normal", "low"]
SystemHealthType = Literal["healthy", "degraded", "unhealthy"]


class AgentResponse(BaseModel):
    """Agent response schema."""

    id: str
    name: str
    type: AgentType
    status: AgentStatusType
    current_task: Optional[str] = None
    progress: float = 0.0
    started_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None


class TaskQueueItemResponse(BaseModel):
    """Task queue item response schema."""

    id: str
    type: AgentType
    status: TaskStatusType
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    agent_id: Optional[str] = None
    priority: TaskPriority = "normal"


class LLMMetricsResponse(BaseModel):
    """LLM metrics response schema."""

    provider: str
    model: str
    total_requests: int = 0
    success_rate: float = 100.0
    avg_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    tokens_used: int = 0
    cost_estimate: float = 0.0
    last_hour_requests: list[int] = Field(default_factory=list)


class CPUMetrics(BaseModel):
    """CPU metrics schema."""

    usage: float
    cores: int


class MemoryMetrics(BaseModel):
    """Memory metrics schema."""

    used: float
    total: float
    percentage: float


class DiskMetrics(BaseModel):
    """Disk metrics schema."""

    used: float
    total: float
    percentage: float


class ResourceMetricsResponse(BaseModel):
    """Resource metrics response schema."""

    cpu: CPUMetrics
    memory: MemoryMetrics
    disk: DiskMetrics


class DatabaseStatsResponse(BaseModel):
    """Database statistics response schema."""

    total_factors: int = 0
    total_backtests: int = 0
    total_research_trials: int = 0
    total_pipeline_runs: int = 0
    total_ohlcv_records: int = 0
    ohlcv_symbols: list[str] = Field(default_factory=list)
    ohlcv_date_range: Optional[str] = None


class SystemStatusResponse(BaseModel):
    """System status response schema."""

    agents: list[AgentResponse]
    task_queue: list[TaskQueueItemResponse]
    llm_metrics: LLMMetricsResponse
    resources: ResourceMetricsResponse
    database_stats: Optional[DatabaseStatsResponse] = None
    system_health: SystemHealthType
    uptime: int  # seconds


# ============== Agent Config Schemas ==============


class AgentConfigResponse(BaseModel):
    """Agent configuration response schema."""

    id: str
    agent_type: AgentType
    name: str
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    user_prompt_template: Optional[str] = None
    examples: Optional[str] = None
    config: Optional[dict[str, Any]] = None
    is_enabled: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class AgentConfigListResponse(BaseModel):
    """Agent configuration list response schema."""

    configs: list[AgentConfigResponse]
    total: int


class AgentConfigUpdateRequest(BaseModel):
    """Request to update agent configuration."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    system_prompt: Optional[str] = Field(None, max_length=10000)
    user_prompt_template: Optional[str] = Field(None, max_length=10000)
    examples: Optional[str] = Field(None, max_length=20000)
    config: Optional[dict[str, Any]] = None
    is_enabled: Optional[bool] = None


class AgentConfigCreateRequest(BaseModel):
    """Request to create agent configuration (for initialization)."""

    agent_type: AgentType
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    system_prompt: Optional[str] = Field(None, max_length=10000)
    user_prompt_template: Optional[str] = Field(None, max_length=10000)
    examples: Optional[str] = Field(None, max_length=20000)
    config: Optional[dict[str, Any]] = None
    is_enabled: bool = True


class AgentConfigOperationResponse(BaseModel):
    """Response for agent config operations."""

    success: bool
    message: str
    config: Optional[AgentConfigResponse] = None
