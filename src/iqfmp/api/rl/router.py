"""RL Training API Router.

REST API endpoints for Qlib RL training and backtesting.
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from iqfmp.api.rl.schemas import (
    RLBacktestRequest,
    RLModelListResponse,
    RLStatsResponse,
    RLTaskListResponse,
    RLTaskResponse,
    RLTaskStatus,
    RLTrainingRequest,
)
from iqfmp.api.rl.service import rl_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rl"])


@router.post("/training", response_model=RLTaskResponse)
async def submit_training(request: RLTrainingRequest) -> RLTaskResponse:
    """Submit a new RL training task.

    Starts a Qlib RL training job using PPO algorithm for order execution
    optimization.

    Args:
        request: Training configuration with data paths

    Returns:
        Task response with task ID for tracking
    """
    try:
        return rl_service.submit_training(request)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to submit training: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/backtest", response_model=RLTaskResponse)
async def submit_backtest(request: RLBacktestRequest) -> RLTaskResponse:
    """Submit a new RL backtest task.

    Evaluates a trained RL policy on historical data.

    Args:
        request: Backtest configuration with model and data paths

    Returns:
        Task response with task ID for tracking
    """
    try:
        return rl_service.submit_backtest(request)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to submit backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/tasks", response_model=RLTaskListResponse)
async def list_tasks(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    task_type: str | None = Query(None, pattern="^(training|backtest)$"),
    status: RLTaskStatus | None = None,
) -> RLTaskListResponse:
    """List RL tasks with pagination and filtering.

    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page
        task_type: Filter by type (training or backtest)
        status: Filter by status

    Returns:
        Paginated list of tasks
    """
    tasks, total = rl_service.get_tasks(
        page=page,
        page_size=page_size,
        task_type=task_type,
        status=status,
    )

    return RLTaskListResponse(
        items=tasks,
        total=total,
        page=page,
        page_size=page_size,
        has_next=(page * page_size) < total,
    )


@router.get("/tasks/{task_id}", response_model=RLTaskResponse)
async def get_task(task_id: str) -> RLTaskResponse:
    """Get details of a specific RL task.

    Args:
        task_id: Task identifier

    Returns:
        Task details including status and results
    """
    task = rl_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return task


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str) -> dict:
    """Cancel a running RL task.

    Args:
        task_id: Task identifier

    Returns:
        Cancellation result
    """
    success = rl_service.cancel_task(task_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel task: {task_id} (not found or not cancellable)",
        )
    return {"message": f"Task {task_id} cancelled", "success": True}


@router.get("/models", response_model=RLModelListResponse)
async def list_models() -> RLModelListResponse:
    """List all trained RL models.

    Returns models from successful training jobs.
    """
    return rl_service.get_models()


@router.get("/stats", response_model=RLStatsResponse)
async def get_stats() -> RLStatsResponse:
    """Get RL training statistics.

    Returns aggregate statistics about training jobs.
    """
    return rl_service.get_stats()
