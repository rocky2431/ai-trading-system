"""Prompts API Schemas.

Pydantic models for prompt template management.
"""

from datetime import datetime
from pydantic import BaseModel, Field


class PromptTemplate(BaseModel):
    """Prompt template information."""

    agent_id: str
    agent_name: str
    prompt_id: str
    version: str
    system_prompt: str
    description: str | None = None
    is_custom: bool = False  # True if custom override is set


class PromptTemplateList(BaseModel):
    """List of prompt templates."""

    templates: list[PromptTemplate]
    total: int


class PromptHistoryEntry(BaseModel):
    """Prompt change history entry."""

    id: str
    agent_id: str
    old_prompt: str | None = None
    new_prompt: str | None = None
    changed_by: str = "system"
    changed_at: datetime
    change_type: str  # "created", "updated", "reset"


class PromptHistoryList(BaseModel):
    """Prompt history list."""

    entries: list[PromptHistoryEntry]
    total: int


class SystemModeConfig(BaseModel):
    """System mode configuration."""

    # Strict mode settings
    strict_mode_enabled: bool = Field(
        default=True,
        description="Require PostgreSQL for all storage (no MemoryStorage fallback)",
    )
    vector_strict_mode: bool = Field(
        default=True,
        description="Require Qdrant for vector storage (no mock fallback)",
    )

    # Sandbox settings
    sandbox_enabled: bool = Field(
        default=True,
        description="Enable sandbox execution for LLM-generated code",
    )
    sandbox_timeout_seconds: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Maximum execution time for sandbox code",
    )
    sandbox_memory_limit_mb: int = Field(
        default=512,
        ge=128,
        le=4096,
        description="Maximum memory for sandbox execution",
    )
    sandbox_network_allowed: bool = Field(
        default=False,
        description="Allow network access in sandbox",
    )

    # Human review settings
    human_review_enabled: bool = Field(
        default=True,
        description="Require human review for generated code",
    )
    auto_reject_timeout_seconds: int = Field(
        default=3600,
        ge=300,
        le=86400,
        description="Auto-reject pending reviews after this time",
    )

    # Feature flags
    ml_signal_enabled: bool = Field(
        default=False,
        description="Enable ML-based signal generation in SignalConverter",
    )
    tool_context_enabled: bool = Field(
        default=False,
        description="Enable tool context in agent calls",
    )
    checkpoint_enabled: bool = Field(
        default=True,
        description="Enable pipeline checkpoint and recovery",
    )


class SystemModeConfigResponse(BaseModel):
    """System mode configuration response."""

    config: SystemModeConfig
    defaults: SystemModeConfig


class SetSystemModeRequest(BaseModel):
    """Request to update system mode configuration."""

    strict_mode_enabled: bool | None = None
    vector_strict_mode: bool | None = None
    sandbox_enabled: bool | None = None
    sandbox_timeout_seconds: int | None = None
    sandbox_memory_limit_mb: int | None = None
    sandbox_network_allowed: bool | None = None
    human_review_enabled: bool | None = None
    auto_reject_timeout_seconds: int | None = None
    ml_signal_enabled: bool | None = None
    tool_context_enabled: bool | None = None
    checkpoint_enabled: bool | None = None
