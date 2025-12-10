"""Repository layer for database operations."""

import json
from datetime import datetime
from typing import Optional

import redis.asyncio as redis
from sqlalchemy import select, func, delete, update
from sqlalchemy.ext.asyncio import AsyncSession

from iqfmp.db.models import (
    FactorORM,
    ResearchTrialORM,
    StrategyORM,
    BacktestResultORM,
    FactorValueORM,
    MiningTaskORM,
)
from iqfmp.models.factor import Factor, FactorMetrics, FactorStatus, StabilityReport


class FactorRepository:
    """Repository for Factor database operations."""

    CACHE_PREFIX = "factor:"
    CACHE_TTL = 3600  # 1 hour

    def __init__(self, session: AsyncSession, redis_client: Optional[redis.Redis] = None):
        """Initialize repository with database session and optional Redis client."""
        self.session = session
        self.redis = redis_client

    # ==================== Cache Operations ====================

    async def _get_from_cache(self, factor_id: str) -> Optional[dict]:
        """Get factor from Redis cache."""
        if self.redis is None:
            return None
        try:
            data = await self.redis.get(f"{self.CACHE_PREFIX}{factor_id}")
            if data:
                return json.loads(data)
        except Exception:
            pass
        return None

    async def _set_cache(self, factor_id: str, data: dict) -> None:
        """Set factor in Redis cache."""
        if self.redis is None:
            return
        try:
            await self.redis.setex(
                f"{self.CACHE_PREFIX}{factor_id}",
                self.CACHE_TTL,
                json.dumps(data, default=str),
            )
        except Exception:
            pass

    async def _invalidate_cache(self, factor_id: str) -> None:
        """Invalidate factor cache."""
        if self.redis is None:
            return
        try:
            await self.redis.delete(f"{self.CACHE_PREFIX}{factor_id}")
        except Exception:
            pass

    # ==================== CRUD Operations ====================

    async def create(self, factor: Factor) -> Factor:
        """Create a new factor in the database."""
        orm_factor = FactorORM(
            id=factor.id,
            name=factor.name,
            family=factor.family,
            code=factor.code,
            code_hash=factor.code_hash,
            target_task=factor.target_task,
            status=factor.status.value if hasattr(factor.status, "value") else str(factor.status),
            cluster_id=factor.cluster_id,
            metrics=factor.metrics.model_dump() if factor.metrics else None,
            stability=factor.stability.model_dump() if factor.stability else None,
            experiment_number=factor.experiment_number,
            created_at=factor.created_at,
        )
        self.session.add(orm_factor)
        await self.session.flush()

        # Cache the factor
        await self._set_cache(factor.id, orm_factor.to_dict())

        return factor

    async def get_by_id(self, factor_id: str) -> Optional[Factor]:
        """Get factor by ID."""
        # Try cache first
        cached = await self._get_from_cache(factor_id)
        if cached:
            return self._orm_dict_to_factor(cached)

        # Query database
        result = await self.session.execute(
            select(FactorORM).where(FactorORM.id == factor_id)
        )
        orm_factor = result.scalar_one_or_none()

        if orm_factor is None:
            return None

        # Update cache
        await self._set_cache(factor_id, orm_factor.to_dict())

        return self._orm_to_factor(orm_factor)

    async def list_factors(
        self,
        page: int = 1,
        page_size: int = 10,
        family: Optional[str] = None,
        status: Optional[str] = None,
    ) -> tuple[list[Factor], int]:
        """List factors with pagination and filtering."""
        query = select(FactorORM)

        # Apply filters
        if family:
            query = query.where(FactorORM.family.contains([family]))
        if status:
            query = query.where(FactorORM.status == status)

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await self.session.execute(count_query)
        total = total_result.scalar() or 0

        # Apply pagination
        offset = (page - 1) * page_size
        query = query.order_by(FactorORM.created_at.desc()).offset(offset).limit(page_size)

        result = await self.session.execute(query)
        orm_factors = result.scalars().all()

        factors = [self._orm_to_factor(f) for f in orm_factors]
        return factors, total

    async def update(self, factor: Factor) -> Factor:
        """Update an existing factor."""
        await self.session.execute(
            update(FactorORM)
            .where(FactorORM.id == factor.id)
            .values(
                name=factor.name,
                family=factor.family,
                code=factor.code,
                status=factor.status.value if hasattr(factor.status, "value") else str(factor.status),
                cluster_id=factor.cluster_id,
                metrics=factor.metrics.model_dump() if factor.metrics else None,
                stability=factor.stability.model_dump() if factor.stability else None,
                experiment_number=factor.experiment_number,
            )
        )
        await self.session.flush()

        # Invalidate cache
        await self._invalidate_cache(factor.id)

        return factor

    async def delete(self, factor_id: str) -> bool:
        """Delete a factor by ID."""
        result = await self.session.execute(
            delete(FactorORM).where(FactorORM.id == factor_id)
        )
        await self.session.flush()

        # Invalidate cache
        await self._invalidate_cache(factor_id)

        return result.rowcount > 0

    async def get_by_code_hash(self, code_hash: str) -> Optional[Factor]:
        """Find factor by code hash (for deduplication)."""
        result = await self.session.execute(
            select(FactorORM).where(FactorORM.code_hash == code_hash)
        )
        orm_factor = result.scalar_one_or_none()

        if orm_factor is None:
            return None

        return self._orm_to_factor(orm_factor)

    async def count_by_status(self) -> dict[str, int]:
        """Count factors by status."""
        result = await self.session.execute(
            select(FactorORM.status, func.count())
            .group_by(FactorORM.status)
        )
        return {row[0]: row[1] for row in result.all()}

    # ==================== Helper Methods ====================

    def _orm_to_factor(self, orm: FactorORM) -> Factor:
        """Convert ORM model to Pydantic model."""
        metrics = None
        if orm.metrics:
            metrics = FactorMetrics(**orm.metrics)

        stability = None
        if orm.stability:
            stability = StabilityReport(**orm.stability)

        return Factor(
            id=orm.id,
            name=orm.name,
            family=orm.family or [],
            code=orm.code,
            code_hash=orm.code_hash,
            target_task=orm.target_task,
            status=FactorStatus(orm.status),
            cluster_id=orm.cluster_id,
            metrics=metrics,
            stability=stability,
            experiment_number=orm.experiment_number,
            created_at=orm.created_at,
        )

    def _orm_dict_to_factor(self, data: dict) -> Factor:
        """Convert ORM dict (from cache) to Pydantic model."""
        metrics = None
        if data.get("metrics"):
            metrics = FactorMetrics(**data["metrics"])

        stability = None
        if data.get("stability"):
            stability = StabilityReport(**data["stability"])

        return Factor(
            id=data["id"],
            name=data["name"],
            family=data.get("family", []),
            code=data["code"],
            code_hash=data.get("code_hash", ""),
            target_task=data.get("target_task", "prediction"),
            status=FactorStatus(data.get("status", "candidate")),
            cluster_id=data.get("cluster_id"),
            metrics=metrics,
            stability=stability,
            experiment_number=data.get("experiment_number", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
        )


class ResearchTrialRepository:
    """Repository for Research Trial database operations.

    This repository provides PostgreSQL persistence for research trials,
    with proper factor linking and automatic trial numbering.
    """

    def __init__(self, session: AsyncSession):
        """Initialize repository with database session."""
        self.session = session

    async def create(
        self,
        factor_id: str,
        factor_name: str,
        sharpe_ratio: float,
        threshold_used: float,
        passed_threshold: bool,
        factor_family: str = "unknown",
        ic_mean: Optional[float] = None,
        ir: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        win_rate: Optional[float] = None,
        evaluation_config: Optional[dict] = None,
    ) -> int:
        """Create a new research trial record.

        Args:
            factor_id: ID of the factor being evaluated
            factor_name: Name of the factor
            sharpe_ratio: Sharpe ratio achieved
            threshold_used: Dynamic threshold at time of evaluation
            passed_threshold: Whether trial passed the threshold
            factor_family: Factor family category
            ic_mean: Mean information coefficient
            ir: Information ratio
            max_drawdown: Maximum drawdown
            win_rate: Win rate percentage
            evaluation_config: Configuration used for evaluation

        Returns:
            Trial number (sequential ID)
        """
        import uuid

        # Get next trial number
        result = await self.session.execute(
            select(func.coalesce(func.max(ResearchTrialORM.trial_number), 0))
        )
        next_trial = result.scalar() + 1

        # Generate unique trial ID
        trial_id = str(uuid.uuid4())[:8]

        trial = ResearchTrialORM(
            trial_id=trial_id,
            trial_number=next_trial,
            factor_id=factor_id,
            factor_name=factor_name,
            factor_family=factor_family,
            sharpe_ratio=sharpe_ratio,
            ic_mean=ic_mean,
            ir=ir,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            threshold_used=threshold_used,
            passed_threshold=passed_threshold,
            evaluation_config=evaluation_config,
        )
        self.session.add(trial)
        await self.session.flush()

        return next_trial

    async def get_total_trials(self) -> int:
        """Get total number of trials."""
        result = await self.session.execute(
            select(func.count()).select_from(ResearchTrialORM)
        )
        return result.scalar() or 0

    async def get_recent_trials(self, limit: int = 50) -> list[dict]:
        """Get recent trials with all metrics."""
        result = await self.session.execute(
            select(ResearchTrialORM)
            .order_by(ResearchTrialORM.trial_number.desc())
            .limit(limit)
        )
        trials = result.scalars().all()

        return [t.to_dict() for t in trials]

    async def get_trials_by_factor(self, factor_id: str) -> list[dict]:
        """Get all trials for a specific factor."""
        result = await self.session.execute(
            select(ResearchTrialORM)
            .where(ResearchTrialORM.factor_id == factor_id)
            .order_by(ResearchTrialORM.trial_number.desc())
        )
        trials = result.scalars().all()
        return [t.to_dict() for t in trials]

    async def get_trials_by_family(self, family: str, limit: int = 100) -> list[dict]:
        """Get trials by factor family."""
        result = await self.session.execute(
            select(ResearchTrialORM)
            .where(ResearchTrialORM.factor_family == family)
            .order_by(ResearchTrialORM.trial_number.desc())
            .limit(limit)
        )
        trials = result.scalars().all()
        return [t.to_dict() for t in trials]

    async def calculate_dynamic_threshold(self, base_threshold: float = 2.0) -> float:
        """Calculate dynamic threshold using DSR formula."""
        total_trials = await self.get_total_trials()

        if total_trials <= 1:
            return base_threshold

        # E[max(Z_1, ..., Z_n)] ≈ sqrt(2 * ln(n)) for large n
        import math
        expected_max = math.sqrt(2 * math.log(total_trials))

        # Adjusted threshold = base + expected_max
        return base_threshold + expected_max * 0.1  # Scale factor

    async def get_statistics(self) -> dict:
        """Get research trial statistics."""
        # Total trials
        total = await self.get_total_trials()

        # Passing rate
        result = await self.session.execute(
            select(func.count()).select_from(ResearchTrialORM)
            .where(ResearchTrialORM.passed_threshold == True)
        )
        passed = result.scalar() or 0

        # Average Sharpe
        result = await self.session.execute(
            select(func.avg(ResearchTrialORM.sharpe_ratio))
        )
        avg_sharpe = result.scalar() or 0.0

        # Best Sharpe
        result = await self.session.execute(
            select(func.max(ResearchTrialORM.sharpe_ratio))
        )
        max_sharpe = result.scalar() or 0.0

        return {
            "total_trials": total,
            "passed_trials": passed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "avg_sharpe": float(avg_sharpe),
            "max_sharpe": float(max_sharpe),
            "current_threshold": await self.calculate_dynamic_threshold(),
        }


class StrategyRepository:
    """Repository for Strategy database operations."""

    CACHE_PREFIX = "strategy:"
    CACHE_TTL = 3600

    def __init__(self, session: AsyncSession, redis_client: Optional[redis.Redis] = None):
        """Initialize repository with database session and optional Redis client."""
        self.session = session
        self.redis = redis_client

    async def _get_from_cache(self, strategy_id: str) -> Optional[dict]:
        """Get strategy from Redis cache."""
        if self.redis is None:
            return None
        try:
            data = await self.redis.get(f"{self.CACHE_PREFIX}{strategy_id}")
            if data:
                return json.loads(data)
        except Exception:
            pass
        return None

    async def _set_cache(self, strategy_id: str, data: dict) -> None:
        """Set strategy in Redis cache."""
        if self.redis is None:
            return
        try:
            await self.redis.setex(
                f"{self.CACHE_PREFIX}{strategy_id}",
                self.CACHE_TTL,
                json.dumps(data, default=str),
            )
        except Exception:
            pass

    async def _invalidate_cache(self, strategy_id: str) -> None:
        """Invalidate strategy cache."""
        if self.redis is None:
            return
        try:
            await self.redis.delete(f"{self.CACHE_PREFIX}{strategy_id}")
        except Exception:
            pass

    async def create(
        self,
        strategy_id: str,
        name: str,
        description: Optional[str],
        factor_ids: list[str],
        factor_weights: Optional[dict],
        code: str,
        config: Optional[dict] = None,
        status: str = "draft",
    ) -> dict:
        """Create a new strategy in the database."""
        strategy = StrategyORM(
            id=strategy_id,
            name=name,
            description=description,
            factor_ids=factor_ids,
            factor_weights=factor_weights,
            code=code,
            config=config,
            status=status,
        )
        self.session.add(strategy)
        await self.session.flush()

        result = self._orm_to_dict(strategy)
        await self._set_cache(strategy_id, result)
        return result

    async def get_by_id(self, strategy_id: str) -> Optional[dict]:
        """Get strategy by ID."""
        cached = await self._get_from_cache(strategy_id)
        if cached:
            return cached

        result = await self.session.execute(
            select(StrategyORM).where(StrategyORM.id == strategy_id)
        )
        strategy = result.scalar_one_or_none()

        if strategy is None:
            return None

        data = self._orm_to_dict(strategy)
        await self._set_cache(strategy_id, data)
        return data

    async def list_strategies(
        self,
        page: int = 1,
        page_size: int = 10,
        status: Optional[str] = None,
    ) -> tuple[list[dict], int]:
        """List strategies with pagination."""
        query = select(StrategyORM)

        if status:
            query = query.where(StrategyORM.status == status)

        count_query = select(func.count()).select_from(query.subquery())
        total_result = await self.session.execute(count_query)
        total = total_result.scalar() or 0

        offset = (page - 1) * page_size
        query = query.order_by(StrategyORM.created_at.desc()).offset(offset).limit(page_size)

        result = await self.session.execute(query)
        strategies = result.scalars().all()

        return [self._orm_to_dict(s) for s in strategies], total

    async def update(
        self,
        strategy_id: str,
        **updates
    ) -> Optional[dict]:
        """Update an existing strategy."""
        await self.session.execute(
            update(StrategyORM)
            .where(StrategyORM.id == strategy_id)
            .values(**updates)
        )
        await self.session.flush()
        await self._invalidate_cache(strategy_id)

        return await self.get_by_id(strategy_id)

    async def delete(self, strategy_id: str) -> bool:
        """Delete a strategy by ID."""
        result = await self.session.execute(
            delete(StrategyORM).where(StrategyORM.id == strategy_id)
        )
        await self.session.flush()
        await self._invalidate_cache(strategy_id)
        return result.rowcount > 0

    def _orm_to_dict(self, orm: StrategyORM) -> dict:
        """Convert ORM model to dictionary."""
        return {
            "id": orm.id,
            "name": orm.name,
            "description": orm.description,
            "factor_ids": orm.factor_ids or [],
            "factor_weights": orm.factor_weights,
            "code": orm.code,
            "config": orm.config,
            "status": orm.status,
            "created_at": orm.created_at.isoformat() if orm.created_at else None,
            "updated_at": orm.updated_at.isoformat() if orm.updated_at else None,
        }


class BacktestResultRepository:
    """Repository for Backtest Result database operations."""

    def __init__(self, session: AsyncSession):
        """Initialize repository with database session."""
        self.session = session

    async def create(
        self,
        result_id: str,
        strategy_id: str,
        start_date: datetime,
        end_date: datetime,
        total_return: float,
        sharpe_ratio: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        win_rate: Optional[float] = None,
        profit_factor: Optional[float] = None,
        trade_count: Optional[int] = None,
        full_results: Optional[dict] = None,
    ) -> dict:
        """Create a new backtest result."""
        result = BacktestResultORM(
            id=result_id,
            strategy_id=strategy_id,
            start_date=start_date,
            end_date=end_date,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trade_count=trade_count,
            full_results=full_results,
        )
        self.session.add(result)
        await self.session.flush()
        return self._orm_to_dict(result)

    async def get_by_id(self, result_id: str) -> Optional[dict]:
        """Get backtest result by ID."""
        result = await self.session.execute(
            select(BacktestResultORM).where(BacktestResultORM.id == result_id)
        )
        backtest = result.scalar_one_or_none()
        return self._orm_to_dict(backtest) if backtest else None

    async def list_by_strategy(
        self,
        strategy_id: str,
        limit: int = 20,
    ) -> list[dict]:
        """List backtest results for a strategy."""
        result = await self.session.execute(
            select(BacktestResultORM)
            .where(BacktestResultORM.strategy_id == strategy_id)
            .order_by(BacktestResultORM.created_at.desc())
            .limit(limit)
        )
        results = result.scalars().all()
        return [self._orm_to_dict(r) for r in results]

    async def get_best_result(self, strategy_id: str) -> Optional[dict]:
        """Get best backtest result by Sharpe ratio."""
        result = await self.session.execute(
            select(BacktestResultORM)
            .where(BacktestResultORM.strategy_id == strategy_id)
            .order_by(BacktestResultORM.sharpe_ratio.desc().nullslast())
            .limit(1)
        )
        backtest = result.scalar_one_or_none()
        return self._orm_to_dict(backtest) if backtest else None

    def _orm_to_dict(self, orm: BacktestResultORM) -> dict:
        """Convert ORM model to dictionary."""
        return {
            "id": orm.id,
            "strategy_id": orm.strategy_id,
            "start_date": orm.start_date.isoformat() if orm.start_date else None,
            "end_date": orm.end_date.isoformat() if orm.end_date else None,
            "total_return": orm.total_return,
            "sharpe_ratio": orm.sharpe_ratio,
            "max_drawdown": orm.max_drawdown,
            "win_rate": orm.win_rate,
            "profit_factor": orm.profit_factor,
            "trade_count": orm.trade_count,
            "full_results": orm.full_results,
            "created_at": orm.created_at.isoformat() if orm.created_at else None,
        }


class FactorValueRepository:
    """Repository for Factor Value time-series database operations.

    Stores computed factor values in TimescaleDB hypertable for
    efficient time-series queries and historical analysis.
    """

    BATCH_SIZE = 1000  # Insert in batches for performance

    def __init__(self, session: AsyncSession):
        """Initialize repository with database session."""
        self.session = session

    async def save_factor_values(
        self,
        factor_id: str,
        values: "pd.DataFrame",
        symbol: str = "MULTI",
        timeframe: str = "1d",
    ) -> int:
        """Save computed factor values to database.

        Args:
            factor_id: Factor ID
            values: DataFrame with 'timestamp' index and 'value' column,
                   or Series with DatetimeIndex
            symbol: Symbol identifier
            timeframe: Timeframe (1m, 5m, 1h, 1d)

        Returns:
            Number of rows inserted
        """
        import pandas as pd

        # Convert Series to DataFrame if needed
        if isinstance(values, pd.Series):
            df = pd.DataFrame({"value": values})
            if isinstance(values.index, pd.DatetimeIndex):
                df["timestamp"] = values.index
            else:
                df["timestamp"] = pd.to_datetime(values.index)
        else:
            df = values.copy()
            if "timestamp" not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    df["timestamp"] = df.index
                else:
                    df["timestamp"] = pd.to_datetime(df.index)

        # Drop NaN values
        df = df.dropna(subset=["value"])

        if df.empty:
            return 0

        # Insert in batches
        rows_inserted = 0
        for i in range(0, len(df), self.BATCH_SIZE):
            batch = df.iloc[i:i + self.BATCH_SIZE]
            orm_objects = [
                FactorValueORM(
                    factor_id=factor_id,
                    symbol=symbol,
                    value=float(row["value"]),
                    timeframe=timeframe,
                    timestamp=row["timestamp"],
                )
                for _, row in batch.iterrows()
            ]
            self.session.add_all(orm_objects)
            rows_inserted += len(orm_objects)

        await self.session.flush()
        return rows_inserted

    async def get_factor_values(
        self,
        factor_id: str,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10000,
    ) -> list[dict]:
        """Get factor values from database.

        Args:
            factor_id: Factor ID
            symbol: Optional symbol filter
            start_time: Optional start time
            end_time: Optional end time
            limit: Maximum rows to return

        Returns:
            List of factor value records
        """
        query = select(FactorValueORM).where(FactorValueORM.factor_id == factor_id)

        if symbol:
            query = query.where(FactorValueORM.symbol == symbol)
        if start_time:
            query = query.where(FactorValueORM.timestamp >= start_time)
        if end_time:
            query = query.where(FactorValueORM.timestamp <= end_time)

        query = query.order_by(FactorValueORM.timestamp.desc()).limit(limit)

        result = await self.session.execute(query)
        records = result.scalars().all()

        return [
            {
                "factor_id": r.factor_id,
                "symbol": r.symbol,
                "value": r.value,
                "timeframe": r.timeframe,
                "timestamp": r.timestamp.isoformat(),
            }
            for r in records
        ]

    async def delete_factor_values(self, factor_id: str) -> int:
        """Delete all values for a factor.

        Args:
            factor_id: Factor ID

        Returns:
            Number of rows deleted
        """
        result = await self.session.execute(
            delete(FactorValueORM).where(FactorValueORM.factor_id == factor_id)
        )
        await self.session.flush()
        return result.rowcount

    async def get_latest_timestamp(self, factor_id: str) -> Optional[datetime]:
        """Get the latest timestamp for a factor.

        Args:
            factor_id: Factor ID

        Returns:
            Latest timestamp or None
        """
        result = await self.session.execute(
            select(func.max(FactorValueORM.timestamp))
            .where(FactorValueORM.factor_id == factor_id)
        )
        return result.scalar()


class MiningTaskRepository:
    """Repository for Mining Task database operations.

    Provides dual storage: TimescaleDB for persistence, Redis for real-time updates.
    """

    CACHE_PREFIX = "mining_task:"
    CACHE_TTL = 86400  # 24 hours

    def __init__(self, session: AsyncSession, redis_client: Optional[redis.Redis] = None):
        """Initialize repository with database session and optional Redis client."""
        self.session = session
        self.redis = redis_client

    async def create(
        self,
        task_id: str,
        name: str,
        description: Optional[str],
        factor_families: list[str],
        target_count: int,
        auto_evaluate: bool = True,
        celery_task_id: Optional[str] = None,
    ) -> dict:
        """Create a new mining task.

        Args:
            task_id: Task ID
            name: Task name
            description: Task description
            factor_families: Factor families to mine
            target_count: Target number of factors
            auto_evaluate: Whether to auto-evaluate
            celery_task_id: Optional Celery task ID

        Returns:
            Created task as dict
        """
        task = MiningTaskORM(
            id=task_id,
            name=name,
            description=description,
            factor_families=factor_families,
            target_count=target_count,
            auto_evaluate=auto_evaluate,
            celery_task_id=celery_task_id,
            status="pending",
        )
        self.session.add(task)
        await self.session.flush()

        result = task.to_dict()

        # Also cache in Redis for real-time access
        if self.redis:
            await self.redis.setex(
                f"{self.CACHE_PREFIX}{task_id}",
                self.CACHE_TTL,
                json.dumps(result, default=str),
            )

        return result

    async def get_by_id(self, task_id: str) -> Optional[dict]:
        """Get mining task by ID.

        Args:
            task_id: Task ID

        Returns:
            Task dict or None
        """
        # Try cache first for running tasks
        if self.redis:
            try:
                cached = await self.redis.get(f"{self.CACHE_PREFIX}{task_id}")
                if cached:
                    return json.loads(cached)
            except Exception:
                pass

        # Query database
        result = await self.session.execute(
            select(MiningTaskORM).where(MiningTaskORM.id == task_id)
        )
        task = result.scalar_one_or_none()

        if task:
            return task.to_dict()
        return None

    async def update_progress(
        self,
        task_id: str,
        generated_count: int,
        passed_count: int,
        failed_count: int,
        progress: float,
        status: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Optional[dict]:
        """Update mining task progress.

        Args:
            task_id: Task ID
            generated_count: Number of factors generated
            passed_count: Number passed evaluation
            failed_count: Number failed evaluation
            progress: Progress percentage (0-100)
            status: Optional new status
            error_message: Optional error message

        Returns:
            Updated task dict
        """
        updates = {
            "generated_count": generated_count,
            "passed_count": passed_count,
            "failed_count": failed_count,
            "progress": progress,
        }
        if status:
            updates["status"] = status
        if error_message:
            updates["error_message"] = error_message

        await self.session.execute(
            update(MiningTaskORM)
            .where(MiningTaskORM.id == task_id)
            .values(**updates)
        )
        await self.session.flush()

        # Update cache
        task = await self.get_by_id(task_id)
        if task and self.redis:
            await self.redis.setex(
                f"{self.CACHE_PREFIX}{task_id}",
                self.CACHE_TTL,
                json.dumps(task, default=str),
            )

        return task

    async def set_started(self, task_id: str) -> Optional[dict]:
        """Mark task as started.

        Args:
            task_id: Task ID

        Returns:
            Updated task dict
        """
        now = datetime.now()
        await self.session.execute(
            update(MiningTaskORM)
            .where(MiningTaskORM.id == task_id)
            .values(status="running", started_at=now)
        )
        await self.session.flush()
        return await self.get_by_id(task_id)

    async def set_completed(
        self,
        task_id: str,
        status: str = "completed",
        error_message: Optional[str] = None,
    ) -> Optional[dict]:
        """Mark task as completed or failed.

        Args:
            task_id: Task ID
            status: Final status (completed, failed, cancelled)
            error_message: Optional error message

        Returns:
            Updated task dict
        """
        now = datetime.now()
        updates = {
            "status": status,
            "completed_at": now,
        }
        if error_message:
            updates["error_message"] = error_message

        await self.session.execute(
            update(MiningTaskORM)
            .where(MiningTaskORM.id == task_id)
            .values(**updates)
        )
        await self.session.flush()

        # Update cache
        task = await self.get_by_id(task_id)
        if task and self.redis:
            await self.redis.setex(
                f"{self.CACHE_PREFIX}{task_id}",
                self.CACHE_TTL,
                json.dumps(task, default=str),
            )

        return task

    async def list_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """List mining tasks with pagination.

        Args:
            status: Optional status filter
            limit: Maximum tasks to return
            offset: Offset for pagination

        Returns:
            Tuple of (tasks, total_count)
        """
        query = select(MiningTaskORM)

        if status:
            query = query.where(MiningTaskORM.status == status)

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await self.session.execute(count_query)
        total = total_result.scalar() or 0

        # Apply pagination
        query = query.order_by(MiningTaskORM.created_at.desc()).offset(offset).limit(limit)

        result = await self.session.execute(query)
        tasks = result.scalars().all()

        return [t.to_dict() for t in tasks], total

    async def get_active_tasks(self) -> list[dict]:
        """Get all active (running) tasks.

        Returns:
            List of active task dicts
        """
        result = await self.session.execute(
            select(MiningTaskORM)
            .where(MiningTaskORM.status.in_(["pending", "running"]))
            .order_by(MiningTaskORM.created_at.desc())
        )
        tasks = result.scalars().all()
        return [t.to_dict() for t in tasks]

    async def get_statistics(self) -> dict:
        """Get mining task statistics.

        Returns:
            Statistics dict
        """
        # Total tasks
        total_result = await self.session.execute(
            select(func.count()).select_from(MiningTaskORM)
        )
        total = total_result.scalar() or 0

        # By status
        status_result = await self.session.execute(
            select(MiningTaskORM.status, func.count())
            .group_by(MiningTaskORM.status)
        )
        by_status = {row[0]: row[1] for row in status_result.all()}

        # Total factors generated
        gen_result = await self.session.execute(
            select(func.sum(MiningTaskORM.generated_count))
        )
        total_generated = gen_result.scalar() or 0

        # Total passed
        passed_result = await self.session.execute(
            select(func.sum(MiningTaskORM.passed_count))
        )
        total_passed = passed_result.scalar() or 0

        return {
            "total_tasks": total,
            "by_status": by_status,
            "total_generated": total_generated,
            "total_passed": total_passed,
            "pass_rate": total_passed / total_generated if total_generated > 0 else 0,
        }
