"""System API router."""

from fastapi import APIRouter

from iqfmp.api.system.schemas import (
    LLMMetricsResponse,
    ResourceMetricsResponse,
    SystemStatusResponse,
)
from iqfmp.api.system.service import get_system_service

router = APIRouter(tags=["system"])


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status() -> SystemStatusResponse:
    """Get complete system status including agents, tasks, and metrics.

    Returns:
        Complete system status
    """
    service = get_system_service()
    return service.get_status()


@router.get("/resources", response_model=ResourceMetricsResponse)
async def get_resources() -> ResourceMetricsResponse:
    """Get system resource metrics (CPU, memory, disk).

    Returns:
        Resource metrics
    """
    service = get_system_service()
    return service.get_resource_metrics()


@router.get("/llm", response_model=LLMMetricsResponse)
async def get_llm_metrics() -> LLMMetricsResponse:
    """Get LLM usage metrics.

    Returns:
        LLM metrics
    """
    service = get_system_service()
    return service.get_llm_metrics()
