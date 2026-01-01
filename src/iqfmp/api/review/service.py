"""Review service with PostgreSQL persistence and WebSocket broadcasting.

This service provides persistent storage for HumanReviewGate,
ensuring review requests and decisions survive server restarts.
Also broadcasts real-time updates via WebSocket for UI updates.
"""

import asyncio
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from sqlalchemy import Column, DateTime, Enum, Integer, String, Text, JSON, Float, select, func
from sqlalchemy.ext.asyncio import AsyncSession

from iqfmp.core.review import (
    HumanReviewGate,
    ReviewConfig,
    ReviewDecision,
    ReviewRequest,
    ReviewStatus,
)
from iqfmp.db import Base
from iqfmp.api.system.websocket import (
    broadcast_review_request_created,
    broadcast_review_decision,
    broadcast_review_timeout,
    broadcast_review_stats_update,
)


class ReviewRequestModel(Base):
    """Database model for review requests."""

    __tablename__ = "review_requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(36), unique=True, nullable=False, index=True)
    code = Column(Text, nullable=False)
    code_summary = Column(Text, nullable=False)
    factor_name = Column(String(255), nullable=False, index=True)
    extra_data = Column(JSON, default=dict)  # renamed from 'metadata' (SQLAlchemy reserved)
    priority = Column(Integer, default=0)
    status = Column(
        Enum(ReviewStatus, name="review_status_enum"),
        default=ReviewStatus.PENDING,
        nullable=False,
        index=True,
    )
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ReviewDecisionModel(Base):
    """Database model for review decisions."""

    __tablename__ = "review_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(36), nullable=False, index=True)
    status = Column(
        Enum(ReviewStatus, name="review_status_enum"),
        nullable=False,
    )
    reviewer = Column(String(255), nullable=True)
    reason = Column(Text, nullable=True)
    decided_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    review_duration_seconds = Column(Float, nullable=True)


class ReviewService:
    """Service for managing review requests with database persistence."""

    def __init__(
        self,
        session: AsyncSession,
        config: Optional[ReviewConfig] = None,
    ) -> None:
        """Initialize review service.

        Args:
            session: Async database session.
            config: Review configuration.
        """
        self._session = session
        self._config = config or ReviewConfig()
        self._gate = HumanReviewGate(config=self._config)
        self._decision_events: dict[str, asyncio.Event] = {}

    async def submit_request(
        self,
        code: str,
        code_summary: str,
        factor_name: str,
        metadata: Optional[dict[str, Any]] = None,
        priority: int = 0,
    ) -> ReviewRequestModel:
        """Submit a new review request.

        Args:
            code: Code to be reviewed.
            code_summary: Summary of the code.
            factor_name: Name of the factor.
            metadata: Additional metadata.
            priority: Request priority (0-10).

        Returns:
            The created review request model.
        """
        request_id = str(uuid4())

        # Create database record
        db_request = ReviewRequestModel(
            request_id=request_id,
            code=code,
            code_summary=code_summary,
            factor_name=factor_name,
            extra_data=metadata or {},
            priority=priority,
            status=ReviewStatus.PENDING,
            created_at=datetime.utcnow(),
        )

        self._session.add(db_request)
        await self._session.commit()
        await self._session.refresh(db_request)

        # Also submit to in-memory gate for real-time waiting
        in_memory_request = ReviewRequest(
            request_id=request_id,
            code=code,
            code_summary=code_summary,
            factor_name=factor_name,
            metadata=metadata or {},
            priority=priority,
        )
        try:
            await self._gate.submit(in_memory_request)
        except RuntimeError:
            pass  # Queue full, but DB record exists

        # Broadcast WebSocket event for real-time UI updates
        try:
            await broadcast_review_request_created(
                request_id=request_id,
                factor_name=factor_name,
                code_summary=code_summary,
                priority=priority,
            )
        except Exception:
            pass  # Don't fail if broadcast fails

        return db_request

    async def get_pending_requests(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[ReviewRequestModel], int]:
        """Get pending review requests with pagination.

        Args:
            page: Page number (1-indexed).
            page_size: Items per page.

        Returns:
            Tuple of (requests, total_count).
        """
        # Count total
        count_stmt = select(func.count(ReviewRequestModel.id)).where(
            ReviewRequestModel.status == ReviewStatus.PENDING
        )
        total = (await self._session.execute(count_stmt)).scalar() or 0

        # Get paginated results
        offset = (page - 1) * page_size
        stmt = (
            select(ReviewRequestModel)
            .where(ReviewRequestModel.status == ReviewStatus.PENDING)
            .order_by(ReviewRequestModel.priority.desc(), ReviewRequestModel.created_at.asc())
            .offset(offset)
            .limit(page_size)
        )
        result = await self._session.execute(stmt)
        requests = list(result.scalars().all())

        return requests, total

    async def get_request(self, request_id: str) -> Optional[ReviewRequestModel]:
        """Get a specific review request.

        Args:
            request_id: The request ID.

        Returns:
            The request or None if not found.
        """
        stmt = select(ReviewRequestModel).where(
            ReviewRequestModel.request_id == request_id
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def decide(
        self,
        request_id: str,
        approved: bool,
        reviewer: str,
        reason: Optional[str] = None,
    ) -> ReviewDecisionModel:
        """Record a review decision.

        Args:
            request_id: The request ID.
            approved: Whether to approve.
            reviewer: Who made the decision.
            reason: Reason for decision.

        Returns:
            The decision model.

        Raises:
            ValueError: If request not found or already decided.
        """
        # Get the request
        request = await self.get_request(request_id)
        if request is None:
            raise ValueError(f"Request {request_id} not found")
        if request.status != ReviewStatus.PENDING:
            raise ValueError(f"Request {request_id} already decided: {request.status}")

        # Calculate review duration
        review_duration = (datetime.utcnow() - request.created_at).total_seconds()

        # Update request status
        status = ReviewStatus.APPROVED if approved else ReviewStatus.REJECTED
        request.status = status
        await self._session.commit()

        # Create decision record
        decision = ReviewDecisionModel(
            request_id=request_id,
            status=status,
            reviewer=reviewer,
            reason=reason,
            decided_at=datetime.utcnow(),
            review_duration_seconds=review_duration,
        )

        self._session.add(decision)
        await self._session.commit()
        await self._session.refresh(decision)

        # Also update in-memory gate
        try:
            await self._gate.decide(request_id, approved, reviewer, reason)
        except ValueError:
            pass  # Not in in-memory queue

        # Broadcast WebSocket event for real-time UI updates
        try:
            await broadcast_review_decision(
                request_id=request_id,
                approved=approved,
                reviewer=reviewer,
                factor_name=request.factor_name,
                reason=reason,
            )
        except Exception:
            pass  # Don't fail if broadcast fails

        return decision

    async def get_decision_history(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[ReviewDecisionModel], int]:
        """Get decision history with pagination.

        Args:
            page: Page number (1-indexed).
            page_size: Items per page.

        Returns:
            Tuple of (decisions, total_count).
        """
        # Count total
        count_stmt = select(func.count(ReviewDecisionModel.id))
        total = (await self._session.execute(count_stmt)).scalar() or 0

        # Get paginated results
        offset = (page - 1) * page_size
        stmt = (
            select(ReviewDecisionModel)
            .order_by(ReviewDecisionModel.decided_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        result = await self._session.execute(stmt)
        decisions = list(result.scalars().all())

        return decisions, total

    async def get_stats(self) -> dict[str, Any]:
        """Get review queue statistics.

        Returns:
            Dictionary with queue statistics.
        """
        # Count by status
        status_counts = {}
        for status in ReviewStatus:
            stmt = select(func.count(ReviewRequestModel.id)).where(
                ReviewRequestModel.status == status
            )
            count = (await self._session.execute(stmt)).scalar() or 0
            status_counts[status.value] = count

        # Average review time
        avg_stmt = select(func.avg(ReviewDecisionModel.review_duration_seconds))
        avg_duration = (await self._session.execute(avg_stmt)).scalar()

        # Oldest pending request age
        oldest_stmt = (
            select(ReviewRequestModel.created_at)
            .where(ReviewRequestModel.status == ReviewStatus.PENDING)
            .order_by(ReviewRequestModel.created_at.asc())
            .limit(1)
        )
        oldest_result = (await self._session.execute(oldest_stmt)).scalar()
        oldest_age = None
        if oldest_result:
            oldest_age = (datetime.utcnow() - oldest_result).total_seconds()

        return {
            "pending_count": status_counts.get("pending", 0),
            "approved_count": status_counts.get("approved", 0),
            "rejected_count": status_counts.get("rejected", 0),
            "timeout_count": status_counts.get("timeout", 0),
            "average_review_time_seconds": avg_duration,
            "oldest_pending_age_seconds": oldest_age,
        }

    async def get_config(self) -> ReviewConfig:
        """Get current review configuration."""
        return self._config

    async def update_config(
        self,
        timeout_seconds: Optional[int] = None,
        max_queue_size: Optional[int] = None,
        auto_reject_on_timeout: Optional[bool] = None,
    ) -> ReviewConfig:
        """Update review configuration.

        Args:
            timeout_seconds: New timeout value.
            max_queue_size: New max queue size.
            auto_reject_on_timeout: New auto-reject setting.

        Returns:
            Updated configuration.
        """
        if timeout_seconds is not None:
            self._config.timeout_seconds = timeout_seconds
        if max_queue_size is not None:
            self._config.max_queue_size = max_queue_size
        if auto_reject_on_timeout is not None:
            self._config.auto_reject_on_timeout = auto_reject_on_timeout

        # Recreate gate with new config
        self._gate = HumanReviewGate(config=self._config)

        return self._config

    async def process_timeouts(self) -> int:
        """Process timed out requests.

        Returns:
            Number of requests timed out.
        """
        if not self._config.auto_reject_on_timeout:
            return 0

        # Find timed out requests
        cutoff = datetime.utcnow()
        stmt = (
            select(ReviewRequestModel)
            .where(ReviewRequestModel.status == ReviewStatus.PENDING)
        )
        result = await self._session.execute(stmt)
        pending_requests = list(result.scalars().all())

        timed_out_count = 0
        timed_out_requests = []
        for request in pending_requests:
            age = (cutoff - request.created_at).total_seconds()
            if age > self._config.timeout_seconds:
                request.status = ReviewStatus.TIMEOUT

                # Create timeout decision
                decision = ReviewDecisionModel(
                    request_id=request.request_id,
                    status=ReviewStatus.TIMEOUT,
                    reviewer=None,
                    reason="Automatic timeout",
                    decided_at=datetime.utcnow(),
                    review_duration_seconds=age,
                )
                self._session.add(decision)
                timed_out_count += 1
                timed_out_requests.append(request)

        if timed_out_count > 0:
            await self._session.commit()

            # Broadcast timeout events for each request
            for request in timed_out_requests:
                try:
                    await broadcast_review_timeout(
                        request_id=request.request_id,
                        factor_name=request.factor_name,
                    )
                except Exception:
                    pass  # Don't fail if broadcast fails

        return timed_out_count
