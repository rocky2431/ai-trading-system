"""WebSocket manager for real-time system status updates."""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from iqfmp.api.system.service import SystemService


class WebSocketMessage(BaseModel):
    """WebSocket message format."""

    type: str  # "status", "agent_update", "task_update", "resource_update", "ping", "pong"
    data: Optional[dict[str, Any]] = None
    timestamp: datetime = datetime.now(timezone.utc)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self) -> None:
        """Initialize connection manager."""
        self._active_connections: list[WebSocket] = []
        self._broadcast_task: Optional[asyncio.Task] = None
        self._broadcast_interval: float = 2.0  # seconds
        self._running: bool = False

    @property
    def connection_count(self) -> int:
        """Get number of active connections."""
        return len(self._active_connections)

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection.

        Args:
            websocket: WebSocket connection to register
        """
        await websocket.accept()
        self._active_connections.append(websocket)
        print(f"WebSocket connected. Total connections: {self.connection_count}")

        # Start broadcast task if not running
        if not self._running and self._broadcast_task is None:
            self._running = True
            self._broadcast_task = asyncio.create_task(self._broadcast_loop())

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection.

        Args:
            websocket: WebSocket connection to remove
        """
        if websocket in self._active_connections:
            self._active_connections.remove(websocket)
            print(f"WebSocket disconnected. Total connections: {self.connection_count}")

        # Stop broadcast task if no connections
        if self.connection_count == 0 and self._broadcast_task is not None:
            self._running = False

    async def send_personal_message(
        self, websocket: WebSocket, message: WebSocketMessage
    ) -> None:
        """Send a message to a specific connection.

        Args:
            websocket: Target WebSocket connection
            message: Message to send
        """
        try:
            await websocket.send_json(message.model_dump(mode="json"))
        except Exception as e:
            print(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: WebSocketMessage) -> None:
        """Broadcast a message to all connected clients.

        Args:
            message: Message to broadcast
        """
        disconnected = []
        for connection in self._active_connections:
            try:
                await connection.send_json(message.model_dump(mode="json"))
            except Exception:
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def _broadcast_loop(self) -> None:
        """Background task to periodically broadcast system status."""
        while self._running:
            try:
                if self.connection_count > 0:
                    # Get system status
                    service = SystemService()
                    resources = service.get_resource_metrics()
                    health = service.get_system_health()

                    # Broadcast resource update
                    message = WebSocketMessage(
                        type="resource_update",
                        data={
                            "cpu": {"usage": resources.cpu.usage, "cores": resources.cpu.cores},
                            "memory": {
                                "used": resources.memory.used,
                                "total": resources.memory.total,
                                "percentage": resources.memory.percentage,
                            },
                            "disk": {
                                "used": resources.disk.used,
                                "total": resources.disk.total,
                                "percentage": resources.disk.percentage,
                            },
                            "system_health": health,
                        },
                    )
                    await self.broadcast(message)

                await asyncio.sleep(self._broadcast_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in broadcast loop: {e}")
                await asyncio.sleep(self._broadcast_interval)

        self._broadcast_task = None


# Global connection manager instance
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket) -> None:
    """Main WebSocket endpoint for system status updates.

    Args:
        websocket: WebSocket connection
    """
    await manager.connect(websocket)

    try:
        # Send initial status
        service = SystemService()
        resources = service.get_resource_metrics()
        health = service.get_system_health()
        llm_metrics = service.get_llm_metrics()

        welcome_message = WebSocketMessage(
            type="connected",
            data={
                "message": "Connected to IQFMP WebSocket",
                "resources": {
                    "cpu": {"usage": resources.cpu.usage, "cores": resources.cpu.cores},
                    "memory": {
                        "used": resources.memory.used,
                        "total": resources.memory.total,
                        "percentage": resources.memory.percentage,
                    },
                    "disk": {
                        "used": resources.disk.used,
                        "total": resources.disk.total,
                        "percentage": resources.disk.percentage,
                    },
                },
                "system_health": health,
                "llm": {
                    "provider": llm_metrics.provider,
                    "model": llm_metrics.model,
                },
            },
        )
        await manager.send_personal_message(websocket, welcome_message)

        # Handle incoming messages
        while True:
            try:
                data = await websocket.receive_json()
                msg_type = data.get("type", "")

                if msg_type == "ping":
                    await manager.send_personal_message(
                        websocket,
                        WebSocketMessage(type="pong"),
                    )

                elif msg_type == "get_status":
                    # Send full status on request
                    resources = service.get_resource_metrics()
                    health = service.get_system_health()

                    status_message = WebSocketMessage(
                        type="status",
                        data={
                            "resources": {
                                "cpu": {"usage": resources.cpu.usage, "cores": resources.cpu.cores},
                                "memory": {
                                    "used": resources.memory.used,
                                    "total": resources.memory.total,
                                    "percentage": resources.memory.percentage,
                                },
                                "disk": {
                                    "used": resources.disk.used,
                                    "total": resources.disk.total,
                                    "percentage": resources.disk.percentage,
                                },
                            },
                            "system_health": health,
                        },
                    )
                    await manager.send_personal_message(websocket, status_message)

                elif msg_type == "subscribe":
                    # Client can subscribe to specific event types
                    topics = data.get("topics", [])
                    await manager.send_personal_message(
                        websocket,
                        WebSocketMessage(
                            type="subscribed",
                            data={"topics": topics},
                        ),
                    )

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    websocket,
                    WebSocketMessage(
                        type="error",
                        data={"message": "Invalid JSON"},
                    ),
                )

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)


# Event broadcaster functions for other services to use
async def broadcast_agent_update(
    agent_id: str,
    status: str,
    current_task: Optional[str] = None,
    progress: float = 0.0,
) -> None:
    """Broadcast an agent status update.

    Args:
        agent_id: Agent identifier
        status: Agent status (idle, running, error)
        current_task: Current task description
        progress: Task progress (0-100)
    """
    message = WebSocketMessage(
        type="agent_update",
        data={
            "agent_id": agent_id,
            "status": status,
            "current_task": current_task,
            "progress": progress,
        },
    )
    await manager.broadcast(message)


async def broadcast_task_update(
    task_id: str,
    task_type: str,
    status: str,
    progress: Optional[float] = None,
    result: Optional[dict] = None,
) -> None:
    """Broadcast a task status update.

    Args:
        task_id: Task identifier
        task_type: Type of task (mining, evaluation, backtest)
        status: Task status
        progress: Task progress
        result: Task result (if completed)
    """
    message = WebSocketMessage(
        type="task_update",
        data={
            "task_id": task_id,
            "task_type": task_type,
            "status": status,
            "progress": progress,
            "result": result,
        },
    )
    await manager.broadcast(message)


async def broadcast_factor_created(
    factor_id: str,
    name: str,
    family: list[str],
) -> None:
    """Broadcast a new factor creation event.

    Args:
        factor_id: Factor identifier
        name: Factor name
        family: Factor families
    """
    message = WebSocketMessage(
        type="factor_created",
        data={
            "factor_id": factor_id,
            "name": name,
            "family": family,
        },
    )
    await manager.broadcast(message)


async def broadcast_evaluation_complete(
    factor_id: str,
    passed: bool,
    metrics: dict,
) -> None:
    """Broadcast a factor evaluation completion event.

    Args:
        factor_id: Factor identifier
        passed: Whether factor passed evaluation
        metrics: Evaluation metrics
    """
    message = WebSocketMessage(
        type="evaluation_complete",
        data={
            "factor_id": factor_id,
            "passed": passed,
            "metrics": metrics,
        },
    )
    await manager.broadcast(message)
