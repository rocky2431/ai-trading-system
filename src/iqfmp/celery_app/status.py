"""
任务状态追踪模块
提供任务状态查询、进度监控和结果获取功能
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from celery.result import AsyncResult

from .app import celery_app

logger = logging.getLogger(__name__)


class TaskState(str, Enum):
    """任务状态枚举"""
    PENDING = "PENDING"  # 等待执行
    STARTED = "STARTED"  # 已开始
    PROGRESS = "PROGRESS"  # 执行中
    SUCCESS = "SUCCESS"  # 成功
    FAILURE = "FAILURE"  # 失败
    RETRY = "RETRY"  # 重试中
    REVOKED = "REVOKED"  # 已撤销


@dataclass
class TaskProgress:
    """任务进度信息"""
    current: int
    total: int
    status: str
    percentage: float

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> "TaskProgress":
        """从元数据创建进度对象"""
        current = meta.get("current", 0)
        total = meta.get("total", 100)
        status = meta.get("status", "Processing...")
        percentage = (current / total * 100) if total > 0 else 0
        return cls(current=current, total=total, status=status, percentage=percentage)


@dataclass
class TaskStatus:
    """任务状态信息"""
    task_id: str
    state: TaskState
    progress: Optional[TaskProgress]
    result: Optional[dict[str, Any]]
    error: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    retries: int


class TaskStatusTracker:
    """任务状态追踪器"""

    def __init__(self):
        """初始化追踪器"""
        self.app = celery_app

    def get_task_status(self, task_id: str) -> TaskStatus:
        """
        获取任务状态

        Args:
            task_id: 任务 ID

        Returns:
            任务状态信息
        """
        result = AsyncResult(task_id, app=self.app)

        state = TaskState(result.state)
        progress = None
        task_result = None
        error = None
        started_at = None
        completed_at = None
        retries = 0

        if state == TaskState.PROGRESS:
            meta = result.info or {}
            progress = TaskProgress.from_meta(meta)
            started_at = self._parse_datetime(meta.get("started_at"))

        elif state == TaskState.SUCCESS:
            task_result = result.result
            if isinstance(task_result, dict):
                completed_at = self._parse_datetime(task_result.get("completed_at"))

        elif state == TaskState.FAILURE:
            error = str(result.result) if result.result else "Unknown error"

        elif state == TaskState.RETRY:
            retries = getattr(result, "retries", 0)

        return TaskStatus(
            task_id=task_id,
            state=state,
            progress=progress,
            result=task_result,
            error=error,
            started_at=started_at,
            completed_at=completed_at,
            retries=retries,
        )

    def get_task_result(self, task_id: str, timeout: float = 10.0) -> dict[str, Any]:
        """
        获取任务结果（阻塞等待）

        Args:
            task_id: 任务 ID
            timeout: 超时时间（秒）

        Returns:
            任务结果

        Raises:
            TimeoutError: 等待超时
            Exception: 任务执行失败
        """
        result = AsyncResult(task_id, app=self.app)

        try:
            return result.get(timeout=timeout)
        except Exception as e:
            logger.error(f"Failed to get result for task {task_id}: {e}")
            raise

    def revoke_task(self, task_id: str, terminate: bool = False) -> bool:
        """
        撤销任务

        Args:
            task_id: 任务 ID
            terminate: 是否终止正在执行的任务

        Returns:
            是否成功撤销
        """
        try:
            self.app.control.revoke(task_id, terminate=terminate)
            logger.info(f"Task {task_id} revoked (terminate={terminate})")
            return True
        except Exception as e:
            logger.error(f"Failed to revoke task {task_id}: {e}")
            return False

    def get_active_tasks(self) -> list[dict[str, Any]]:
        """
        获取所有活跃任务

        Returns:
            活跃任务列表
        """
        inspect = self.app.control.inspect()
        active = inspect.active() or {}

        tasks = []
        for worker, worker_tasks in active.items():
            for task in worker_tasks:
                tasks.append({
                    "task_id": task["id"],
                    "name": task["name"],
                    "worker": worker,
                    "args": task.get("args", []),
                    "kwargs": task.get("kwargs", {}),
                    "started_at": task.get("time_start"),
                })

        return tasks

    def get_scheduled_tasks(self) -> list[dict[str, Any]]:
        """
        获取所有计划任务

        Returns:
            计划任务列表
        """
        inspect = self.app.control.inspect()
        scheduled = inspect.scheduled() or {}

        tasks = []
        for worker, worker_tasks in scheduled.items():
            for task in worker_tasks:
                tasks.append({
                    "task_id": task["request"]["id"],
                    "name": task["request"]["name"],
                    "worker": worker,
                    "eta": task.get("eta"),
                    "priority": task.get("priority"),
                })

        return tasks

    def get_reserved_tasks(self) -> list[dict[str, Any]]:
        """
        获取所有预留任务

        Returns:
            预留任务列表
        """
        inspect = self.app.control.inspect()
        reserved = inspect.reserved() or {}

        tasks = []
        for worker, worker_tasks in reserved.items():
            for task in worker_tasks:
                tasks.append({
                    "task_id": task["id"],
                    "name": task["name"],
                    "worker": worker,
                })

        return tasks

    def get_queue_lengths(self) -> dict[str, int]:
        """
        获取队列长度

        Returns:
            各队列的任务数量
        """
        try:
            with self.app.connection_or_acquire() as conn:
                queues = ["high", "default", "low"]
                lengths = {}

                for queue_name in queues:
                    try:
                        queue = conn.default_channel.queue_declare(
                            queue=queue_name,
                            passive=True,
                        )
                        lengths[queue_name] = queue.message_count
                    except Exception:
                        lengths[queue_name] = 0

                return lengths
        except Exception as e:
            logger.error(f"Failed to get queue lengths: {e}")
            return {"high": 0, "default": 0, "low": 0}

    def get_worker_stats(self) -> dict[str, Any]:
        """
        获取 worker 统计信息

        Returns:
            Worker 统计信息
        """
        inspect = self.app.control.inspect()

        stats = inspect.stats() or {}
        ping = inspect.ping() or {}

        workers = []
        for worker_name, worker_stats in stats.items():
            workers.append({
                "name": worker_name,
                "online": worker_name in ping,
                "concurrency": worker_stats.get("pool", {}).get("max-concurrency", 0),
                "processed": worker_stats.get("total", {}),
                "prefetch_count": worker_stats.get("prefetch_count", 0),
            })

        return {
            "total_workers": len(workers),
            "online_workers": sum(1 for w in workers if w["online"]),
            "workers": workers,
        }

    def _parse_datetime(self, value: Optional[str]) -> Optional[datetime]:
        """解析日期时间字符串"""
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None


# 便捷函数

def get_task_status(task_id: str) -> TaskStatus:
    """获取任务状态的便捷函数"""
    tracker = TaskStatusTracker()
    return tracker.get_task_status(task_id)


def wait_for_task(task_id: str, timeout: float = 10.0) -> dict[str, Any]:
    """等待任务完成的便捷函数"""
    tracker = TaskStatusTracker()
    return tracker.get_task_result(task_id, timeout)


def revoke_task(task_id: str, terminate: bool = False) -> bool:
    """撤销任务的便捷函数"""
    tracker = TaskStatusTracker()
    return tracker.revoke_task(task_id, terminate)
