"""RL Training API Schemas.

Pydantic models for RL training request/response validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RLTaskStatus(str, Enum):
    """RL task status enumeration."""

    PENDING = "pending"
    STARTED = "started"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    REVOKED = "revoked"


class RLTrainingConfig(BaseModel):
    """RL training configuration."""

    total_timesteps: int = Field(default=10000, ge=1000, le=1000000)
    learning_rate: float = Field(default=3e-4, ge=1e-6, le=1e-2)
    order_amount: float = Field(default=1.0, ge=0.01, le=1000)
    time_per_step: int = Field(default=60, ge=1, le=3600)
    save_model: bool = True
    model_path: str | None = None


class RLTrainingRequest(BaseModel):
    """RL training request schema."""

    train_data_path: str = Field(..., description="Path to training data (parquet/csv)")
    test_data_path: str = Field(..., description="Path to test data (parquet/csv)")
    config: RLTrainingConfig = Field(default_factory=RLTrainingConfig)
    name: str | None = Field(default=None, description="Optional training job name")


class RLBacktestRequest(BaseModel):
    """RL backtest request schema."""

    model_path: str = Field(..., description="Path to trained policy model")
    data_path: str = Field(..., description="Path to backtest data")
    config: dict[str, Any] = Field(default_factory=dict)
    name: str | None = Field(default=None, description="Optional backtest job name")


class RLTaskResponse(BaseModel):
    """RL task response schema."""

    task_id: str
    celery_task_id: str
    status: RLTaskStatus
    task_type: str  # "training" or "backtest"
    name: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class RLTaskListResponse(BaseModel):
    """Paginated RL task list response."""

    items: list[RLTaskResponse]
    total: int
    page: int
    page_size: int
    has_next: bool


class RLModelInfo(BaseModel):
    """RL model information."""

    model_id: str
    path: str
    task_id: str
    created_at: datetime
    metrics: dict[str, Any] | None = None
    config: dict[str, Any] | None = None


class RLModelListResponse(BaseModel):
    """List of RL models."""

    models: list[RLModelInfo]
    total: int


class RLStatsResponse(BaseModel):
    """RL training statistics."""

    total_training_jobs: int
    successful_jobs: int
    failed_jobs: int
    running_jobs: int
    total_models: int
    average_training_time_seconds: float | None = None
