"""Pipeline API router."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, status

from iqfmp.api.pipeline.schemas import (
    PipelineListResponse,
    PipelineRunRequest,
    PipelineRunResponse,
    PipelineStatus,
    PipelineStatusResponse,
)
from iqfmp.api.pipeline.service import PipelineNotFoundError, get_pipeline_service

router = APIRouter(tags=["pipeline"])


@router.post("/run", response_model=PipelineRunResponse, status_code=status.HTTP_201_CREATED)
async def run_pipeline(request: PipelineRunRequest) -> PipelineRunResponse:
    """Start a new pipeline run.

    Args:
        request: Pipeline run request

    Returns:
        Pipeline run response with run_id
    """
    service = get_pipeline_service()
    return service.create_run(
        pipeline_type=request.pipeline_type,
        config=request.config,
    )


@router.get("/{run_id}/status", response_model=PipelineStatusResponse)
async def get_pipeline_status(run_id: str) -> PipelineStatusResponse:
    """Get pipeline run status.

    Args:
        run_id: Pipeline run ID

    Returns:
        Pipeline status response

    Raises:
        HTTPException: If run not found
    """
    service = get_pipeline_service()
    status_resp = service.get_run_status(run_id)

    if not status_resp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline run {run_id} not found",
        )

    return status_resp


@router.get("/runs", response_model=PipelineListResponse)
async def list_pipeline_runs(
    status_filter: Optional[PipelineStatus] = Query(
        default=None, alias="status", description="Filter by status"
    ),
) -> PipelineListResponse:
    """List pipeline runs.

    Args:
        status_filter: Filter by status

    Returns:
        List of pipeline runs
    """
    service = get_pipeline_service()
    runs = service.list_runs(status=status_filter)

    return PipelineListResponse(
        runs=runs,
        total=len(runs),
    )


@router.post("/{run_id}/cancel")
async def cancel_pipeline(run_id: str) -> dict:
    """Cancel a pipeline run.

    Args:
        run_id: Pipeline run ID

    Returns:
        Cancellation result

    Raises:
        HTTPException: If run not found or cannot be cancelled
    """
    service = get_pipeline_service()

    try:
        cancelled = service.cancel_run(run_id)
    except PipelineNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline run {run_id} not found",
        )

    if not cancelled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pipeline run cannot be cancelled (already completed/failed)",
        )

    return {"message": "Pipeline run cancelled", "run_id": run_id}


@router.websocket("/{run_id}/ws")
async def pipeline_websocket(websocket: WebSocket, run_id: str) -> None:
    """WebSocket endpoint for pipeline progress updates.

    Args:
        websocket: WebSocket connection
        run_id: Pipeline run ID
    """
    service = get_pipeline_service()

    # Check if run exists
    status_resp = service.get_run_status(run_id)
    if not status_resp:
        await websocket.close(code=4004, reason="Pipeline run not found")
        return

    await websocket.accept()

    try:
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "data": {
                "run_id": status_resp.run_id,
                "status": status_resp.status,
                "progress": status_resp.progress,
                "current_step": status_resp.current_step,
            },
        })

        # Keep connection alive and send updates
        while True:
            # Wait for messages from client (ping/pong or commands)
            try:
                data = await websocket.receive_json()
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif data.get("type") == "get_status":
                    current_status = service.get_run_status(run_id)
                    if current_status:
                        await websocket.send_json({
                            "type": "status",
                            "data": {
                                "run_id": current_status.run_id,
                                "status": current_status.status,
                                "progress": current_status.progress,
                                "current_step": current_status.current_step,
                            },
                        })
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        pass
