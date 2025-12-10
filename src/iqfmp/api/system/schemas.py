"""System API schemas."""

from datetime import datetime
from typing import Literal, Optional

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


class SystemStatusResponse(BaseModel):
    """System status response schema."""

    agents: list[AgentResponse]
    task_queue: list[TaskQueueItemResponse]
    llm_metrics: LLMMetricsResponse
    resources: ResourceMetricsResponse
    system_health: SystemHealthType
    uptime: int  # seconds
