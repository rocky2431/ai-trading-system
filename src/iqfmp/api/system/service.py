"""System service for real system metrics.

This module provides real-time system status from:
- Database (mining tasks, pipeline runs, RD Loop runs)
- LLM usage tracking (via global tracker)
- System resources (CPU, memory, disk)
"""

import os
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from typing import Optional

import psutil
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

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
from iqfmp.db.models import MiningTaskORM, PipelineRunORM, RDLoopRunORM
from iqfmp.llm.provider import LLMConfig


# =============================================================================
# Global LLM Usage Tracker
# =============================================================================

class LLMUsageTracker:
    """Tracks LLM API usage across the application."""

    def __init__(self, max_history: int = 60):
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._latencies: deque[float] = deque(maxlen=1000)
        self._hourly_requests: deque[int] = deque(maxlen=max_history)
        self._last_hour_slot = -1

    def record_request(
        self,
        success: bool,
        latency_ms: float,
        tokens_used: int = 0,
        cost: float = 0.0,
    ) -> None:
        """Record an LLM API request."""
        self._total_requests += 1
        if success:
            self._successful_requests += 1
        else:
            self._failed_requests += 1

        self._latencies.append(latency_ms)
        self._total_tokens += tokens_used
        self._total_cost += cost

        # Track hourly requests
        current_hour = datetime.now(timezone.utc).hour
        if current_hour != self._last_hour_slot:
            self._hourly_requests.append(1)
            self._last_hour_slot = current_hour
        elif self._hourly_requests:
            self._hourly_requests[-1] += 1

    @property
    def total_requests(self) -> int:
        return self._total_requests

    @property
    def success_rate(self) -> float:
        if self._total_requests == 0:
            return 100.0
        return (self._successful_requests / self._total_requests) * 100

    @property
    def avg_latency(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    @property
    def p95_latency(self) -> float:
        if len(self._latencies) < 2:
            return self.avg_latency
        sorted_lat = sorted(self._latencies)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def p99_latency(self) -> float:
        if len(self._latencies) < 2:
            return self.avg_latency
        sorted_lat = sorted(self._latencies)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def tokens_used(self) -> int:
        return self._total_tokens

    @property
    def cost_estimate(self) -> float:
        return self._total_cost

    @property
    def hourly_requests(self) -> list[int]:
        return list(self._hourly_requests)


# Global instance
_llm_tracker = LLMUsageTracker()


def get_llm_tracker() -> LLMUsageTracker:
    """Get global LLM usage tracker."""
    return _llm_tracker


class SystemService:
    """Service for system status and metrics."""

    def __init__(self, session: Optional[AsyncSession] = None) -> None:
        """Initialize system service.

        Args:
            session: Optional database session for real status queries
        """
        self._start_time = time.time()
        self._llm_requests: list[dict] = []
        self._llm_config: Optional[LLMConfig] = None
        self._session = session

    def _get_llm_config(self) -> LLMConfig:
        """Get or create LLM config."""
        if self._llm_config is None:
            self._llm_config = LLMConfig.from_env()
        return self._llm_config

    async def get_agents(self) -> list[AgentResponse]:
        """Get list of agents with their current status from database.

        Queries:
        - MiningTaskORM for factor generation tasks
        - PipelineRunORM for pipeline runs
        - RDLoopRunORM for RD Loop runs (hypothesis agent)
        """
        agents = []

        # Base agent definitions
        agent_defs = [
            ("agent-factor-gen", "Factor Generator", "factor_generation"),
            ("agent-evaluator", "Factor Evaluator", "evaluation"),
            ("agent-strategy", "Strategy Builder", "strategy"),
            ("agent-backtest", "Backtester", "backtest"),
        ]

        # Query active mining tasks if we have a session
        active_tasks = {}
        if self._session:
            try:
                result = await self._session.execute(
                    select(MiningTaskORM)
                    .where(MiningTaskORM.status.in_(["running", "pending"]))
                    .order_by(MiningTaskORM.created_at.desc())
                    .limit(10)
                )
                mining_tasks = result.scalars().all()

                for task in mining_tasks:
                    if task.status == "running":
                        # Factor Generator is running
                        active_tasks["agent-factor-gen"] = {
                            "status": "running",
                            "current_task": f"Mining: {task.name}",
                            "progress": task.progress,
                            "started_at": task.started_at,
                        }
            except Exception as e:
                print(f"Warning: Failed to query mining tasks: {e}")

            # Query active pipeline runs
            try:
                result = await self._session.execute(
                    select(PipelineRunORM)
                    .where(PipelineRunORM.status.in_(["running", "pending"]))
                    .order_by(PipelineRunORM.created_at.desc())
                    .limit(10)
                )
                pipeline_runs = result.scalars().all()

                for run in pipeline_runs:
                    if run.pipeline_type == "factor_mining" and run.status == "running":
                        active_tasks["agent-factor-gen"] = {
                            "status": "running",
                            "current_task": f"Pipeline: {run.current_step or 'processing'}",
                            "progress": run.progress * 100,
                            "started_at": run.started_at,
                        }
                    elif run.pipeline_type == "strategy_backtest" and run.status == "running":
                        active_tasks["agent-backtest"] = {
                            "status": "running",
                            "current_task": f"Backtest: {run.current_step or 'running'}",
                            "progress": run.progress * 100,
                            "started_at": run.started_at,
                        }
            except Exception as e:
                print(f"Warning: Failed to query pipeline runs: {e}")

            # Query active RD Loop runs (evaluator agent status)
            try:
                result = await self._session.execute(
                    select(RDLoopRunORM)
                    .where(RDLoopRunORM.status == "running")
                    .order_by(RDLoopRunORM.created_at.desc())
                    .limit(5)
                )
                rd_loop_runs = result.scalars().all()

                for run in rd_loop_runs:
                    # Calculate progress based on iterations
                    max_iter = run.config.get("max_iterations", 10) if run.config else 10
                    progress = (run.iteration / max_iter) * 100 if max_iter > 0 else 0

                    active_tasks["agent-evaluator"] = {
                        "status": "running",
                        "current_task": f"RD Loop: {run.phase} (iter {run.iteration})",
                        "progress": min(progress, 100),
                        "started_at": run.started_at,
                    }
                    break  # Use the most recent one
            except Exception as e:
                print(f"Warning: Failed to query RD Loop runs: {e}")

        # Build agent responses
        for agent_id, name, agent_type in agent_defs:
            if agent_id in active_tasks:
                task_info = active_tasks[agent_id]
                agents.append(AgentResponse(
                    id=agent_id,
                    name=name,
                    type=agent_type,
                    status=task_info["status"],
                    current_task=task_info["current_task"],
                    progress=task_info["progress"],
                    started_at=task_info["started_at"],
                    last_activity=datetime.now(timezone.utc),
                ))
            else:
                agents.append(AgentResponse(
                    id=agent_id,
                    name=name,
                    type=agent_type,
                    status="idle",
                    current_task=None,
                    progress=0,
                    started_at=None,
                    last_activity=datetime.now(timezone.utc),
                ))

        return agents

    async def get_task_queue(self) -> list[TaskQueueItemResponse]:
        """Get current task queue from database."""
        tasks = []

        if not self._session:
            return tasks

        try:
            # Query pending mining tasks
            result = await self._session.execute(
                select(MiningTaskORM)
                .where(MiningTaskORM.status.in_(["pending", "running"]))
                .order_by(MiningTaskORM.created_at.asc())
                .limit(20)
            )
            mining_tasks = result.scalars().all()

            for task in mining_tasks:
                task_status = "running" if task.status == "running" else "pending"
                tasks.append(TaskQueueItemResponse(
                    id=task.id,
                    type="factor_generation",
                    status=task_status,
                    created_at=task.created_at or datetime.now(timezone.utc),
                    started_at=task.started_at,
                    completed_at=task.completed_at,
                    agent_id="agent-factor-gen" if task.status == "running" else None,
                    priority="normal",
                ))

            # Query pending pipeline runs
            result = await self._session.execute(
                select(PipelineRunORM)
                .where(PipelineRunORM.status.in_(["pending", "running"]))
                .order_by(PipelineRunORM.created_at.asc())
                .limit(20)
            )
            pipeline_runs = result.scalars().all()

            for run in pipeline_runs:
                agent_type = "strategy" if run.pipeline_type == "strategy_backtest" else "factor_generation"
                agent_id = "agent-backtest" if run.pipeline_type == "strategy_backtest" else "agent-factor-gen"
                task_status = "running" if run.status == "running" else "pending"

                tasks.append(TaskQueueItemResponse(
                    id=run.id,
                    type=agent_type,
                    status=task_status,
                    created_at=run.created_at or datetime.now(timezone.utc),
                    started_at=run.started_at,
                    completed_at=run.completed_at,
                    agent_id=agent_id if run.status == "running" else None,
                    priority="normal",
                ))

        except Exception as e:
            print(f"Warning: Failed to query task queue: {e}")

        return tasks

    def get_llm_metrics(self) -> LLMMetricsResponse:
        """Get LLM usage metrics from global tracker."""
        config = self._get_llm_config()

        # Extract provider from base_url
        provider = "OpenRouter" if "openrouter" in config.base_url else "Unknown"

        # Get real metrics from global tracker
        tracker = get_llm_tracker()

        return LLMMetricsResponse(
            provider=provider,
            model=config.default_model.value,
            total_requests=tracker.total_requests,
            success_rate=round(tracker.success_rate, 2),
            avg_latency=round(tracker.avg_latency, 2),
            p95_latency=round(tracker.p95_latency, 2),
            p99_latency=round(tracker.p99_latency, 2),
            tokens_used=tracker.tokens_used,
            cost_estimate=round(tracker.cost_estimate, 4),
            last_hour_requests=tracker.hourly_requests,
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

    async def get_status(self) -> SystemStatusResponse:
        """Get complete system status."""
        agents = await self.get_agents()
        task_queue = await self.get_task_queue()

        return SystemStatusResponse(
            agents=agents,
            task_queue=task_queue,
            llm_metrics=self.get_llm_metrics(),
            resources=self.get_resource_metrics(),
            system_health=self.get_system_health(),  # type: ignore
            uptime=self.get_uptime(),
        )


