"""RL Training Service.

Service layer for managing RL training tasks with PostgreSQL persistence.
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Column, DateTime, Enum, String, Text, desc
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session

from iqfmp.api.database import Base, get_db_session
from iqfmp.api.rl.schemas import (
    RLBacktestRequest,
    RLModelInfo,
    RLModelListResponse,
    RLStatsResponse,
    RLTaskResponse,
    RLTaskStatus,
    RLTrainingRequest,
)

logger = logging.getLogger(__name__)


class RLTaskModel(Base):
    """SQLAlchemy model for RL tasks."""

    __tablename__ = "rl_tasks"

    task_id = Column(String(64), primary_key=True)
    celery_task_id = Column(String(64), nullable=True, index=True)
    task_type = Column(String(20), nullable=False)  # training, backtest
    name = Column(String(256), nullable=True)
    status = Column(
        Enum(RLTaskStatus, name="rl_task_status"),
        default=RLTaskStatus.PENDING,
        nullable=False,
        index=True,
    )
    config = Column(JSONB, default=dict)
    result = Column(JSONB, nullable=True)
    error = Column(Text, nullable=True)
    model_path = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)


class RLService:
    """Service for managing RL training tasks."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        logger.info("RLService initialized with PostgreSQL persistence")

    def _get_session(self) -> Session:
        """Get database session."""
        return next(get_db_session())

    def submit_training(self, request: RLTrainingRequest) -> RLTaskResponse:
        """Submit a new RL training task.

        Args:
            request: Training request with data paths and config

        Returns:
            Task response with task ID and initial status
        """
        task_id = f"rl_train_{uuid.uuid4().hex[:12]}"

        # Create task record
        session = self._get_session()
        try:
            task = RLTaskModel(
                task_id=task_id,
                task_type="training",
                name=request.name,
                status=RLTaskStatus.PENDING,
                config={
                    "train_data_path": request.train_data_path,
                    "test_data_path": request.test_data_path,
                    "config": request.config.model_dump(),
                },
            )
            session.add(task)
            session.commit()

            # Submit Celery task
            from iqfmp.celery_app.tasks import run_qlib_rl_training

            celery_result = run_qlib_rl_training.delay(
                task_id=task_id,
                train_data_path=request.train_data_path,
                test_data_path=request.test_data_path,
                config=request.config.model_dump(),
            )

            # Update with Celery task ID
            task.celery_task_id = celery_result.id
            task.status = RLTaskStatus.STARTED
            task.started_at = datetime.utcnow()
            session.commit()

            logger.info(f"Submitted RL training task: {task_id}")

            return RLTaskResponse(
                task_id=task_id,
                celery_task_id=celery_result.id,
                status=RLTaskStatus.STARTED,
                task_type="training",
                name=request.name,
                created_at=task.created_at,
                started_at=task.started_at,
            )
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to submit RL training: {e}")
            raise
        finally:
            session.close()

    def submit_backtest(self, request: RLBacktestRequest) -> RLTaskResponse:
        """Submit a new RL backtest task.

        Args:
            request: Backtest request with model path and data

        Returns:
            Task response with task ID and initial status
        """
        task_id = f"rl_backtest_{uuid.uuid4().hex[:12]}"

        session = self._get_session()
        try:
            task = RLTaskModel(
                task_id=task_id,
                task_type="backtest",
                name=request.name,
                status=RLTaskStatus.PENDING,
                config={
                    "model_path": request.model_path,
                    "data_path": request.data_path,
                    "config": request.config,
                },
                model_path=request.model_path,
            )
            session.add(task)
            session.commit()

            # Submit Celery task
            from iqfmp.celery_app.tasks import run_qlib_rl_backtest

            celery_result = run_qlib_rl_backtest.delay(
                task_id=task_id,
                model_path=request.model_path,
                data_path=request.data_path,
                config=request.config,
            )

            task.celery_task_id = celery_result.id
            task.status = RLTaskStatus.STARTED
            task.started_at = datetime.utcnow()
            session.commit()

            logger.info(f"Submitted RL backtest task: {task_id}")

            return RLTaskResponse(
                task_id=task_id,
                celery_task_id=celery_result.id,
                status=RLTaskStatus.STARTED,
                task_type="backtest",
                name=request.name,
                created_at=task.created_at,
                started_at=task.started_at,
            )
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to submit RL backtest: {e}")
            raise
        finally:
            session.close()

    def get_task(self, task_id: str) -> RLTaskResponse | None:
        """Get task by ID."""
        session = self._get_session()
        try:
            task = session.query(RLTaskModel).filter_by(task_id=task_id).first()
            if not task:
                return None

            # Check Celery status if task is running
            if task.status in (RLTaskStatus.STARTED, RLTaskStatus.RUNNING):
                self._sync_celery_status(session, task)

            return self._task_to_response(task)
        finally:
            session.close()

    def get_tasks(
        self,
        page: int = 1,
        page_size: int = 20,
        task_type: str | None = None,
        status: RLTaskStatus | None = None,
    ) -> tuple[list[RLTaskResponse], int]:
        """Get paginated list of tasks."""
        session = self._get_session()
        try:
            query = session.query(RLTaskModel)

            if task_type:
                query = query.filter_by(task_type=task_type)
            if status:
                query = query.filter_by(status=status)

            total = query.count()
            tasks = (
                query.order_by(desc(RLTaskModel.created_at))
                .offset((page - 1) * page_size)
                .limit(page_size)
                .all()
            )

            # Sync Celery status for running tasks
            for task in tasks:
                if task.status in (RLTaskStatus.STARTED, RLTaskStatus.RUNNING):
                    self._sync_celery_status(session, task)

            return [self._task_to_response(t) for t in tasks], total
        finally:
            session.close()

    def _sync_celery_status(self, session: Session, task: RLTaskModel) -> None:
        """Sync task status with Celery."""
        if not task.celery_task_id:
            return

        try:
            from celery.result import AsyncResult

            from iqfmp.celery_app.celery import celery_app

            result = AsyncResult(task.celery_task_id, app=celery_app)

            if result.ready():
                if result.successful():
                    task.status = RLTaskStatus.SUCCESS
                    task.result = result.result
                    if result.result and "model_path" in result.result:
                        task.model_path = result.result["model_path"]
                else:
                    task.status = RLTaskStatus.FAILED
                    task.error = str(result.result)
                task.completed_at = datetime.utcnow()
                session.commit()
            elif result.state == "STARTED":
                task.status = RLTaskStatus.RUNNING
                session.commit()
        except Exception as e:
            logger.warning(f"Failed to sync Celery status for {task.task_id}: {e}")

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        session = self._get_session()
        try:
            task = session.query(RLTaskModel).filter_by(task_id=task_id).first()
            if not task or not task.celery_task_id:
                return False

            if task.status not in (RLTaskStatus.PENDING, RLTaskStatus.STARTED, RLTaskStatus.RUNNING):
                return False

            from celery.result import AsyncResult

            from iqfmp.celery_app.celery import celery_app

            result = AsyncResult(task.celery_task_id, app=celery_app)
            result.revoke(terminate=True)

            task.status = RLTaskStatus.REVOKED
            task.completed_at = datetime.utcnow()
            session.commit()

            logger.info(f"Cancelled RL task: {task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False
        finally:
            session.close()

    def get_models(self) -> RLModelListResponse:
        """Get list of trained RL models."""
        session = self._get_session()
        try:
            tasks = (
                session.query(RLTaskModel)
                .filter(
                    RLTaskModel.task_type == "training",
                    RLTaskModel.status == RLTaskStatus.SUCCESS,
                    RLTaskModel.model_path.isnot(None),
                )
                .order_by(desc(RLTaskModel.completed_at))
                .all()
            )

            models = []
            for task in tasks:
                if task.model_path and Path(task.model_path).exists():
                    models.append(
                        RLModelInfo(
                            model_id=f"model_{task.task_id}",
                            path=task.model_path,
                            task_id=task.task_id,
                            created_at=task.completed_at or task.created_at,
                            metrics=task.result.get("metrics") if task.result else None,
                            config=task.config.get("config") if task.config else None,
                        )
                    )

            return RLModelListResponse(models=models, total=len(models))
        finally:
            session.close()

    def get_stats(self) -> RLStatsResponse:
        """Get RL training statistics."""
        session = self._get_session()
        try:
            from sqlalchemy import func

            total = session.query(RLTaskModel).filter_by(task_type="training").count()
            successful = (
                session.query(RLTaskModel)
                .filter_by(task_type="training", status=RLTaskStatus.SUCCESS)
                .count()
            )
            failed = (
                session.query(RLTaskModel)
                .filter_by(task_type="training", status=RLTaskStatus.FAILED)
                .count()
            )
            running = (
                session.query(RLTaskModel)
                .filter(
                    RLTaskModel.task_type == "training",
                    RLTaskModel.status.in_([RLTaskStatus.STARTED, RLTaskStatus.RUNNING]),
                )
                .count()
            )

            # Count models
            models_count = (
                session.query(RLTaskModel)
                .filter(
                    RLTaskModel.task_type == "training",
                    RLTaskModel.status == RLTaskStatus.SUCCESS,
                    RLTaskModel.model_path.isnot(None),
                )
                .count()
            )

            # Calculate average training time
            avg_time = None
            completed_tasks = (
                session.query(RLTaskModel)
                .filter(
                    RLTaskModel.task_type == "training",
                    RLTaskModel.status == RLTaskStatus.SUCCESS,
                    RLTaskModel.started_at.isnot(None),
                    RLTaskModel.completed_at.isnot(None),
                )
                .all()
            )

            if completed_tasks:
                total_seconds = sum(
                    (t.completed_at - t.started_at).total_seconds()
                    for t in completed_tasks
                )
                avg_time = total_seconds / len(completed_tasks)

            return RLStatsResponse(
                total_training_jobs=total,
                successful_jobs=successful,
                failed_jobs=failed,
                running_jobs=running,
                total_models=models_count,
                average_training_time_seconds=avg_time,
            )
        finally:
            session.close()

    def _task_to_response(self, task: RLTaskModel) -> RLTaskResponse:
        """Convert task model to response."""
        return RLTaskResponse(
            task_id=task.task_id,
            celery_task_id=task.celery_task_id or "",
            status=task.status,
            task_type=task.task_type,
            name=task.name,
            created_at=task.created_at,
            started_at=task.started_at,
            completed_at=task.completed_at,
            result=task.result,
            error=task.error,
        )


# Singleton instance
rl_service = RLService()
