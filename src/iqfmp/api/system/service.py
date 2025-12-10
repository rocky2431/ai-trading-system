"""System service for real system metrics."""

import os
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

import psutil

from iqfmp.api.system.schemas import (
    AgentResponse,
    CPUMetrics,
    DiskMetrics,
    LLMMetricsResponse,
    MemoryMetrics,
    ResourceMetricsResponse,
    SystemStatusResponse,
    TaskQueueItemResponse,
)
from iqfmp.llm.provider import LLMConfig


class SystemService:
    """Service for system status and metrics."""

    def __init__(self) -> None:
        """Initialize system service."""
        self._start_time = time.time()
        self._llm_requests: list[dict] = []
        self._llm_config: Optional[LLMConfig] = None

    def _get_llm_config(self) -> LLMConfig:
        """Get or create LLM config."""
        if self._llm_config is None:
            self._llm_config = LLMConfig.from_env()
        return self._llm_config

    def get_agents(self) -> list[AgentResponse]:
        """Get list of agents with their current status."""
        # In a real system, this would query actual agent processes
        # For now, return static agent definitions
        return [
            AgentResponse(
                id="agent-factor-gen",
                name="Factor Generator",
                type="factor_generation",
                status="idle",
                current_task=None,
                progress=0,
                started_at=None,
                last_activity=datetime.now(timezone.utc),
            ),
            AgentResponse(
                id="agent-evaluator",
                name="Factor Evaluator",
                type="evaluation",
                status="idle",
                current_task=None,
                progress=0,
                started_at=None,
                last_activity=datetime.now(timezone.utc),
            ),
            AgentResponse(
                id="agent-strategy",
                name="Strategy Builder",
                type="strategy",
                status="idle",
                current_task=None,
                progress=0,
                started_at=None,
                last_activity=datetime.now(timezone.utc),
            ),
            AgentResponse(
                id="agent-backtest",
                name="Backtester",
                type="backtest",
                status="idle",
                current_task=None,
                progress=0,
                started_at=None,
                last_activity=datetime.now(timezone.utc),
            ),
        ]

    def get_task_queue(self) -> list[TaskQueueItemResponse]:
        """Get current task queue."""
        # In a real system, this would query a task queue (Redis, etc.)
        return []

    def get_llm_metrics(self) -> LLMMetricsResponse:
        """Get LLM usage metrics."""
        config = self._get_llm_config()

        # Extract provider from base_url
        provider = "OpenRouter" if "openrouter" in config.base_url else "Unknown"

        return LLMMetricsResponse(
            provider=provider,
            model=config.default_model.value,
            total_requests=0,
            success_rate=100.0,
            avg_latency=0.0,
            p95_latency=0.0,
            p99_latency=0.0,
            tokens_used=0,
            cost_estimate=0.0,
            last_hour_requests=[],
        )

    def get_resource_metrics(self) -> ResourceMetricsResponse:
        """Get system resource metrics using psutil."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count() or 1

        # Memory metrics
        memory = psutil.virtual_memory()
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)

        # Disk metrics
        disk = psutil.disk_usage("/")
        disk_used_gb = disk.used / (1024**3)
        disk_total_gb = disk.total / (1024**3)

        return ResourceMetricsResponse(
            cpu=CPUMetrics(
                usage=cpu_percent,
                cores=cpu_count,
            ),
            memory=MemoryMetrics(
                used=round(memory_used_gb, 2),
                total=round(memory_total_gb, 2),
                percentage=memory.percent,
            ),
            disk=DiskMetrics(
                used=round(disk_used_gb, 2),
                total=round(disk_total_gb, 2),
                percentage=disk.percent,
            ),
        )

    def get_system_health(self) -> str:
        """Determine overall system health."""
        resources = self.get_resource_metrics()

        # Check for unhealthy conditions
        if resources.cpu.usage > 90 or resources.memory.percentage > 95:
            return "unhealthy"
        elif resources.cpu.usage > 70 or resources.memory.percentage > 80:
            return "degraded"
        return "healthy"

    def get_uptime(self) -> int:
        """Get system uptime in seconds."""
        return int(time.time() - self._start_time)

    def get_status(self) -> SystemStatusResponse:
        """Get complete system status."""
        return SystemStatusResponse(
            agents=self.get_agents(),
            task_queue=self.get_task_queue(),
            llm_metrics=self.get_llm_metrics(),
            resources=self.get_resource_metrics(),
            system_health=self.get_system_health(),  # type: ignore
            uptime=self.get_uptime(),
        )


# Singleton instance
_system_service: Optional[SystemService] = None


def get_system_service() -> SystemService:
    """Get or create system service singleton."""
    global _system_service
    if _system_service is None:
        _system_service = SystemService()
    return _system_service
