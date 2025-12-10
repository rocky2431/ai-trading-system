"""System API router."""

from fastapi import APIRouter, Depends, WebSocket
from sqlalchemy.ext.asyncio import AsyncSession

from iqfmp.api.system.schemas import (
    AgentResponse,
    LLMMetricsResponse,
    ResourceMetricsResponse,
    SystemStatusResponse,
    TaskQueueItemResponse,
)
from iqfmp.api.system.service import SystemService
from iqfmp.api.system.websocket import websocket_endpoint
from iqfmp.db.database import get_db

router = APIRouter(tags=["system"])


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(
    session: AsyncSession = Depends(get_db),
) -> SystemStatusResponse:
    """Get complete system status including agents, tasks, and metrics.

    Returns:
        Complete system status with real agent/task data from database
    """
    service = SystemService(session=session)
    return await service.get_status()


@router.get("/agents", response_model=list[AgentResponse])
async def get_agents(
    session: AsyncSession = Depends(get_db),
) -> list[AgentResponse]:
    """Get list of agents with their current status.

    Returns:
        List of agent status from database
    """
    service = SystemService(session=session)
    return await service.get_agents()


@router.get("/tasks", response_model=list[TaskQueueItemResponse])
async def get_task_queue(
    session: AsyncSession = Depends(get_db),
) -> list[TaskQueueItemResponse]:
    """Get current task queue.

    Returns:
        List of pending and running tasks from database
    """
    service = SystemService(session=session)
    return await service.get_task_queue()


@router.get("/resources", response_model=ResourceMetricsResponse)
async def get_resources() -> ResourceMetricsResponse:
    """Get system resource metrics (CPU, memory, disk).

    Returns:
        Resource metrics (real-time from psutil)
    """
    service = SystemService()
    return service.get_resource_metrics()


@router.get("/llm", response_model=LLMMetricsResponse)
async def get_llm_metrics() -> LLMMetricsResponse:
    """Get LLM usage metrics.

    Returns:
        LLM metrics
    """
    service = SystemService()
    return service.get_llm_metrics()


@router.websocket("/ws")
async def system_websocket(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time system status updates.

    Supports the following message types:
    - ping: Send a ping to keep connection alive
    - get_status: Request current system status
    - subscribe: Subscribe to specific event types

    Broadcasts the following event types:
    - connected: Initial connection confirmation with system status
    - resource_update: Periodic resource metrics updates (every 2s)
    - agent_update: Agent status changes
    - task_update: Task progress updates
    - factor_created: New factor creation events
    - evaluation_complete: Factor evaluation completion events
    """
    await websocket_endpoint(websocket)
