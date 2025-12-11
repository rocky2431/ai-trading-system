"""Pipeline service for business logic.

This module provides PostgreSQL-backed pipeline management,
ensuring pipeline state persists across server restarts.
"""

import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from iqfmp.api.pipeline.schemas import (
    PipelineRunResponse,
    PipelineStatus,
    PipelineStatusResponse,
    PipelineType,
)
from iqfmp.db.models import PipelineRunORM


class PipelineNotFoundError(Exception):
    """Raised when pipeline run is not found."""

    pass


class PipelineRun:
    """Internal pipeline run representation."""

    def __init__(
        self,
        run_id: str,
        pipeline_type: PipelineType,
        config: dict[str, Any],
    ) -> None:
        """Initialize pipeline run."""
        self.run_id = run_id
        self.pipeline_type = pipeline_type
        self.config = config
        self.status: PipelineStatus = "pending"
        self.progress: float = 0.0
        self.current_step: Optional[str] = None
        self.created_at: datetime = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Optional[dict[str, Any]] = None
        self.error: Optional[str] = None

    @classmethod
    def from_orm(cls, orm: PipelineRunORM) -> "PipelineRun":
        """Create from ORM model."""
        run = cls(
            run_id=orm.id,
            pipeline_type=orm.pipeline_type,
            config=orm.config or {},
        )
        run.status = orm.status
        run.progress = orm.progress
        run.current_step = orm.current_step
        run.created_at = orm.created_at or datetime.now()
        run.started_at = orm.started_at
        run.completed_at = orm.completed_at
        run.result = orm.result
        run.error = orm.error_message
        return run


class PipelineService:
    """Service for pipeline management with PostgreSQL persistence."""

    def __init__(self) -> None:
        """Initialize pipeline service."""
        # Memory cache for fast access (also serves as fallback)
        self._runs: dict[str, PipelineRun] = {}

    def create_run(
        self,
        pipeline_type: PipelineType,
        config: dict[str, Any],
    ) -> PipelineRunResponse:
        """Create a new pipeline run (sync - memory only).

        Args:
            pipeline_type: Type of pipeline to run
            config: Pipeline configuration

        Returns:
            Pipeline run response
        """
        run_id = str(uuid.uuid4())
        run = PipelineRun(
            run_id=run_id,
            pipeline_type=pipeline_type,
            config=config,
        )
        self._runs[run_id] = run

        return PipelineRunResponse(
            run_id=run_id,
            status=run.status,
            created_at=run.created_at,
        )

    async def create_run_async(
        self,
        pipeline_type: PipelineType,
        config: dict[str, Any],
        session: AsyncSession,
        celery_task_id: Optional[str] = None,
    ) -> PipelineRunResponse:
        """Create a new pipeline run with PostgreSQL persistence.

        Args:
            pipeline_type: Type of pipeline to run
            config: Pipeline configuration
            session: Database session
            celery_task_id: Optional Celery task ID

        Returns:
            Pipeline run response
        """
        run_id = str(uuid.uuid4())

        # Create ORM object
        orm_run = PipelineRunORM(
            id=run_id,
            pipeline_type=pipeline_type,
            config=config,
            status="pending",
            progress=0.0,
            celery_task_id=celery_task_id,
        )
        session.add(orm_run)
        await session.commit()

        # Also cache in memory
        run = PipelineRun.from_orm(orm_run)
        self._runs[run_id] = run

        return PipelineRunResponse(
            run_id=run_id,
            status=run.status,
            created_at=run.created_at,
        )

    async def get_run_status_async(
        self, run_id: str, session: AsyncSession
    ) -> Optional[PipelineStatusResponse]:
        """Get pipeline run status from PostgreSQL.

        Args:
            run_id: Pipeline run ID
            session: Database session

        Returns:
            Pipeline status response or None if not found
        """
        result = await session.execute(
            select(PipelineRunORM).where(PipelineRunORM.id == run_id)
        )
        orm_run = result.scalar_one_or_none()

        if not orm_run:
            return None

        return PipelineStatusResponse(
            run_id=orm_run.id,
            status=orm_run.status,
            progress=orm_run.progress,
            current_step=orm_run.current_step,
            started_at=orm_run.started_at,
            completed_at=orm_run.completed_at,
            result=orm_run.result,
            error=orm_run.error_message,
        )

    async def update_run_progress_async(
        self,
        run_id: str,
        progress: float,
        session: AsyncSession,
        current_step: Optional[str] = None,
    ) -> None:
        """Update pipeline run progress in PostgreSQL.

        Args:
            run_id: Pipeline run ID
            progress: Progress value (0.0 to 1.0)
            session: Database session
            current_step: Current step name
        """
        update_values = {"progress": max(0.0, min(1.0, progress))}
        if current_step:
            update_values["current_step"] = current_step

        await session.execute(
            update(PipelineRunORM)
            .where(PipelineRunORM.id == run_id)
            .values(**update_values)
        )
        await session.commit()

        # Update cache
        if run_id in self._runs:
            self._runs[run_id].progress = update_values["progress"]
            if current_step:
                self._runs[run_id].current_step = current_step

    async def complete_run_async(
        self,
        run_id: str,
        session: AsyncSession,
        result: Optional[dict[str, Any]] = None,
    ) -> None:
        """Mark pipeline run as completed in PostgreSQL.

        Args:
            run_id: Pipeline run ID
            session: Database session
            result: Pipeline result data
        """
        await session.execute(
            update(PipelineRunORM)
            .where(PipelineRunORM.id == run_id)
            .values(
                status="completed",
                progress=1.0,
                completed_at=datetime.now(),
                result=result,
            )
        )
        await session.commit()

        # Update cache
        if run_id in self._runs:
            self._runs[run_id].status = "completed"
            self._runs[run_id].progress = 1.0
            self._runs[run_id].completed_at = datetime.now()
            self._runs[run_id].result = result

    async def fail_run_async(
        self, run_id: str, error: str, session: AsyncSession
    ) -> None:
        """Mark pipeline run as failed in PostgreSQL.

        Args:
            run_id: Pipeline run ID
            error: Error message
            session: Database session
        """
        await session.execute(
            update(PipelineRunORM)
            .where(PipelineRunORM.id == run_id)
            .values(
                status="failed",
                completed_at=datetime.now(),
                error_message=error,
            )
        )
        await session.commit()

        # Update cache
        if run_id in self._runs:
            self._runs[run_id].status = "failed"
            self._runs[run_id].completed_at = datetime.now()
            self._runs[run_id].error = error

    async def start_run_async(self, run_id: str, session: AsyncSession) -> None:
        """Start a pipeline run in PostgreSQL.

        Args:
            run_id: Pipeline run ID
            session: Database session
        """
        await session.execute(
            update(PipelineRunORM)
            .where(PipelineRunORM.id == run_id)
            .values(status="running", started_at=datetime.now())
        )
        await session.commit()

        # Update cache
        if run_id in self._runs:
            self._runs[run_id].status = "running"
            self._runs[run_id].started_at = datetime.now()

    def get_run_status(self, run_id: str) -> Optional[PipelineStatusResponse]:
        """Get pipeline run status.

        Args:
            run_id: Pipeline run ID

        Returns:
            Pipeline status response or None if not found
        """
        run = self._runs.get(run_id)
        if not run:
            return None

        return PipelineStatusResponse(
            run_id=run.run_id,
            status=run.status,
            progress=run.progress,
            current_step=run.current_step,
            started_at=run.started_at,
            completed_at=run.completed_at,
            result=run.result,
            error=run.error,
        )

    def start_run(self, run_id: str) -> None:
        """Start a pipeline run.

        Args:
            run_id: Pipeline run ID

        Raises:
            PipelineNotFoundError: If run not found
        """
        run = self._runs.get(run_id)
        if not run:
            raise PipelineNotFoundError(f"Pipeline run {run_id} not found")

        run.status = "running"
        run.started_at = datetime.now()

    def update_run_progress(
        self,
        run_id: str,
        progress: float,
        current_step: Optional[str] = None,
    ) -> None:
        """Update pipeline run progress.

        Args:
            run_id: Pipeline run ID
            progress: Progress value (0.0 to 1.0)
            current_step: Current step name

        Raises:
            PipelineNotFoundError: If run not found
        """
        run = self._runs.get(run_id)
        if not run:
            raise PipelineNotFoundError(f"Pipeline run {run_id} not found")

        run.progress = max(0.0, min(1.0, progress))
        if current_step:
            run.current_step = current_step

    def complete_run(
        self,
        run_id: str,
        result: Optional[dict[str, Any]] = None,
    ) -> None:
        """Mark pipeline run as completed.

        Args:
            run_id: Pipeline run ID
            result: Pipeline result data

        Raises:
            PipelineNotFoundError: If run not found
        """
        run = self._runs.get(run_id)
        if not run:
            raise PipelineNotFoundError(f"Pipeline run {run_id} not found")

        run.status = "completed"
        run.progress = 1.0
        run.completed_at = datetime.now()
        run.result = result

    def fail_run(self, run_id: str, error: str) -> None:
        """Mark pipeline run as failed.

        Args:
            run_id: Pipeline run ID
            error: Error message

        Raises:
            PipelineNotFoundError: If run not found
        """
        run = self._runs.get(run_id)
        if not run:
            raise PipelineNotFoundError(f"Pipeline run {run_id} not found")

        run.status = "failed"
        run.completed_at = datetime.now()
        run.error = error

    def cancel_run(self, run_id: str) -> bool:
        """Cancel a pipeline run.

        Args:
            run_id: Pipeline run ID

        Returns:
            True if cancelled, False if cannot be cancelled

        Raises:
            PipelineNotFoundError: If run not found
        """
        run = self._runs.get(run_id)
        if not run:
            raise PipelineNotFoundError(f"Pipeline run {run_id} not found")

        if run.status in ["completed", "failed", "cancelled"]:
            return False

        run.status = "cancelled"
        run.completed_at = datetime.now()
        return True

    def list_runs(
        self,
        status: Optional[PipelineStatus] = None,
    ) -> list[PipelineStatusResponse]:
        """List pipeline runs (memory cache).

        Args:
            status: Filter by status

        Returns:
            List of pipeline status responses
        """
        runs = list(self._runs.values())

        if status:
            runs = [r for r in runs if r.status == status]

        return [
            PipelineStatusResponse(
                run_id=r.run_id,
                status=r.status,
                progress=r.progress,
                current_step=r.current_step,
                started_at=r.started_at,
                completed_at=r.completed_at,
                result=r.result,
                error=r.error,
            )
            for r in runs
        ]

    async def list_runs_async(
        self,
        status: Optional[PipelineStatus],
        session: AsyncSession,
    ) -> list[PipelineStatusResponse]:
        """List pipeline runs from DB (with memory cache merge)."""
        result = await session.execute(
            select(PipelineRunORM).order_by(PipelineRunORM.created_at.desc())
        )
        orm_runs = result.scalars().all()

        runs = [
            PipelineStatusResponse(
                run_id=orm.id,
                status=orm.status,
                progress=orm.progress,
                current_step=orm.current_step,
                started_at=orm.started_at,
                completed_at=orm.completed_at,
                result=orm.result,
                error=orm.error_message,
            )
            for orm in orm_runs
            if (not status or orm.status == status)
        ]

        # Merge in-memory runs not yet persisted
        mem_runs = self.list_runs(status=status)
        mem_ids = {r.run_id for r in runs}
        runs.extend([r for r in mem_runs if r.run_id not in mem_ids])

        return runs


# Singleton instance
_pipeline_service = PipelineService()


def get_pipeline_service() -> PipelineService:
    """Get pipeline service instance."""
    return _pipeline_service
