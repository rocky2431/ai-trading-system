"""Pipeline service for business logic."""

import uuid
from datetime import datetime
from typing import Any, Optional

from iqfmp.api.pipeline.schemas import (
    PipelineRunResponse,
    PipelineStatus,
    PipelineStatusResponse,
    PipelineType,
)


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


class PipelineService:
    """Service for pipeline management."""

    def __init__(self) -> None:
        """Initialize pipeline service."""
        self._runs: dict[str, PipelineRun] = {}

    def create_run(
        self,
        pipeline_type: PipelineType,
        config: dict[str, Any],
    ) -> PipelineRunResponse:
        """Create a new pipeline run.

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
        """List pipeline runs.

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


# Singleton instance
_pipeline_service = PipelineService()


def get_pipeline_service() -> PipelineService:
    """Get pipeline service instance."""
    return _pipeline_service
