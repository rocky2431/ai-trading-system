"""Review API router for HumanReviewGate management.

This router provides REST endpoints for:
- Submitting code for human review
- Viewing pending review requests
- Approving/rejecting code
- Viewing decision history
- Managing review configuration
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from iqfmp.api.auth.dependencies import get_current_user
from iqfmp.api.review.schemas import (
    PaginatedDecisionResponse,
    PaginatedReviewResponse,
    ReviewConfigResponse,
    ReviewConfigUpdate,
    ReviewDecisionRequest,
    ReviewDecisionResponse,
    ReviewQueueStats,
    ReviewRequestCreate,
    ReviewRequestResponse,
    ReviewStatusEnum,
)
from iqfmp.api.review.service import ReviewService
from iqfmp.core.review import ReviewConfig, ReviewStatus
from iqfmp.db.database import get_db

router = APIRouter(tags=["review"])


async def get_review_service(
    session: AsyncSession = Depends(get_db),
) -> ReviewService:
    """Dependency injection for ReviewService."""
    return ReviewService(session)


@router.post(
    "/requests",
    response_model=ReviewRequestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit code for review",
    description="Submit LLM-generated code for human review before execution.",
)
async def submit_review_request(
    request: ReviewRequestCreate,
    service: ReviewService = Depends(get_review_service),
    current_user: dict = Depends(get_current_user),
) -> ReviewRequestResponse:
    """Submit a new review request."""
    db_request = await service.submit_request(
        code=request.code,
        code_summary=request.code_summary,
        factor_name=request.factor_name,
        metadata=request.metadata,
        priority=request.priority,
    )

    return ReviewRequestResponse(
        request_id=db_request.request_id,
        code=db_request.code,
        code_summary=db_request.code_summary,
        factor_name=db_request.factor_name,
        metadata=db_request.metadata,
        priority=db_request.priority,
        status=ReviewStatusEnum(db_request.status.value),
        created_at=db_request.created_at,
    )


@router.get(
    "/requests/pending",
    response_model=PaginatedReviewResponse,
    summary="Get pending review requests",
    description="Get all pending review requests awaiting human approval.",
)
async def get_pending_requests(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    service: ReviewService = Depends(get_review_service),
    current_user: dict = Depends(get_current_user),
) -> PaginatedReviewResponse:
    """Get pending review requests with pagination."""
    requests, total = await service.get_pending_requests(page=page, page_size=page_size)

    items = [
        ReviewRequestResponse(
            request_id=r.request_id,
            code=r.code,
            code_summary=r.code_summary,
            factor_name=r.factor_name,
            metadata=r.metadata,
            priority=r.priority,
            status=ReviewStatusEnum(r.status.value),
            created_at=r.created_at,
        )
        for r in requests
    ]

    return PaginatedReviewResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        has_next=(page * page_size) < total,
    )


@router.get(
    "/requests/{request_id}",
    response_model=ReviewRequestResponse,
    summary="Get review request details",
    description="Get details of a specific review request.",
)
async def get_review_request(
    request_id: str,
    service: ReviewService = Depends(get_review_service),
    current_user: dict = Depends(get_current_user),
) -> ReviewRequestResponse:
    """Get a specific review request."""
    request = await service.get_request(request_id)
    if request is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review request {request_id} not found",
        )

    return ReviewRequestResponse(
        request_id=request.request_id,
        code=request.code,
        code_summary=request.code_summary,
        factor_name=request.factor_name,
        metadata=request.metadata,
        priority=request.priority,
        status=ReviewStatusEnum(request.status.value),
        created_at=request.created_at,
    )


@router.post(
    "/requests/{request_id}/decide",
    response_model=ReviewDecisionResponse,
    summary="Make review decision",
    description="Approve or reject a pending review request.",
)
async def make_decision(
    request_id: str,
    decision: ReviewDecisionRequest,
    service: ReviewService = Depends(get_review_service),
    current_user: dict = Depends(get_current_user),
) -> ReviewDecisionResponse:
    """Make a decision on a review request."""
    reviewer = current_user.get("email", current_user.get("sub", "unknown"))

    try:
        db_decision = await service.decide(
            request_id=request_id,
            approved=decision.approved,
            reviewer=reviewer,
            reason=decision.reason,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return ReviewDecisionResponse(
        request_id=db_decision.request_id,
        status=ReviewStatusEnum(db_decision.status.value),
        reviewer=db_decision.reviewer,
        reason=db_decision.reason,
        decided_at=db_decision.decided_at,
    )


@router.get(
    "/decisions",
    response_model=PaginatedDecisionResponse,
    summary="Get decision history",
    description="Get history of all review decisions.",
)
async def get_decision_history(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    service: ReviewService = Depends(get_review_service),
    current_user: dict = Depends(get_current_user),
) -> PaginatedDecisionResponse:
    """Get decision history with pagination."""
    decisions, total = await service.get_decision_history(page=page, page_size=page_size)

    items = [
        ReviewDecisionResponse(
            request_id=d.request_id,
            status=ReviewStatusEnum(d.status.value),
            reviewer=d.reviewer,
            reason=d.reason,
            decided_at=d.decided_at,
        )
        for d in decisions
    ]

    return PaginatedDecisionResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        has_next=(page * page_size) < total,
    )


@router.get(
    "/stats",
    response_model=ReviewQueueStats,
    summary="Get review queue statistics",
    description="Get statistics about the review queue.",
)
async def get_queue_stats(
    service: ReviewService = Depends(get_review_service),
    current_user: dict = Depends(get_current_user),
) -> ReviewQueueStats:
    """Get review queue statistics."""
    stats = await service.get_stats()
    return ReviewQueueStats(**stats)


@router.get(
    "/config",
    response_model=ReviewConfigResponse,
    summary="Get review configuration",
    description="Get current review gate configuration.",
)
async def get_review_config(
    service: ReviewService = Depends(get_review_service),
    current_user: dict = Depends(get_current_user),
) -> ReviewConfigResponse:
    """Get current review configuration."""
    config = await service.get_config()
    return ReviewConfigResponse(
        timeout_seconds=config.timeout_seconds,
        max_queue_size=config.max_queue_size,
        auto_reject_on_timeout=config.auto_reject_on_timeout,
    )


@router.patch(
    "/config",
    response_model=ReviewConfigResponse,
    summary="Update review configuration",
    description="Update review gate configuration settings.",
)
async def update_review_config(
    update: ReviewConfigUpdate,
    service: ReviewService = Depends(get_review_service),
    current_user: dict = Depends(get_current_user),
) -> ReviewConfigResponse:
    """Update review configuration."""
    config = await service.update_config(
        timeout_seconds=update.timeout_seconds,
        max_queue_size=update.max_queue_size,
        auto_reject_on_timeout=update.auto_reject_on_timeout,
    )
    return ReviewConfigResponse(
        timeout_seconds=config.timeout_seconds,
        max_queue_size=config.max_queue_size,
        auto_reject_on_timeout=config.auto_reject_on_timeout,
    )


@router.post(
    "/process-timeouts",
    summary="Process timed out requests",
    description="Manually trigger timeout processing for pending requests.",
)
async def process_timeouts(
    service: ReviewService = Depends(get_review_service),
    current_user: dict = Depends(get_current_user),
) -> dict:
    """Process timed out review requests."""
    count = await service.process_timeouts()
    return {
        "message": f"Processed {count} timed out requests",
        "timed_out_count": count,
    }
