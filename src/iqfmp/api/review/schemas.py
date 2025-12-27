"""Pydantic schemas for Review API."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ReviewStatusEnum(str, Enum):
    """Review status enum."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


class ReviewRequestCreate(BaseModel):
    """Schema for creating a review request."""

    code: str = Field(..., description="Code to be reviewed")
    code_summary: str = Field(..., description="Summary of the code")
    factor_name: str = Field(..., description="Name of the factor")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    priority: int = Field(default=0, ge=0, le=10, description="Priority (0-10)")


class ReviewRequestResponse(BaseModel):
    """Schema for review request response."""

    request_id: str
    code: str
    code_summary: str
    factor_name: str
    metadata: dict[str, Any]
    priority: int
    status: ReviewStatusEnum
    created_at: datetime

    class Config:
        from_attributes = True


class ReviewDecisionRequest(BaseModel):
    """Schema for making a review decision."""

    approved: bool = Field(..., description="Whether to approve the code")
    reason: Optional[str] = Field(None, description="Reason for decision")


class ReviewDecisionResponse(BaseModel):
    """Schema for review decision response."""

    request_id: str
    status: ReviewStatusEnum
    reviewer: Optional[str]
    reason: Optional[str]
    decided_at: datetime

    class Config:
        from_attributes = True


class ReviewQueueStats(BaseModel):
    """Schema for review queue statistics."""

    pending_count: int
    approved_count: int
    rejected_count: int
    timeout_count: int
    average_review_time_seconds: Optional[float]
    oldest_pending_age_seconds: Optional[float]


class ReviewConfigResponse(BaseModel):
    """Schema for review configuration."""

    timeout_seconds: int
    max_queue_size: int
    auto_reject_on_timeout: bool


class ReviewConfigUpdate(BaseModel):
    """Schema for updating review configuration."""

    timeout_seconds: Optional[int] = Field(None, ge=60, le=86400)
    max_queue_size: Optional[int] = Field(None, ge=10, le=1000)
    auto_reject_on_timeout: Optional[bool] = None


class PaginatedReviewResponse(BaseModel):
    """Paginated response for review requests."""

    items: list[ReviewRequestResponse]
    total: int
    page: int
    page_size: int
    has_next: bool


class PaginatedDecisionResponse(BaseModel):
    """Paginated response for review decisions."""

    items: list[ReviewDecisionResponse]
    total: int
    page: int
    page_size: int
    has_next: bool
