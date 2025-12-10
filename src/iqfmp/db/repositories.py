"""Repository layer for database operations."""

import json
from datetime import datetime
from typing import Optional

import redis.asyncio as redis
from sqlalchemy import select, func, delete, update
from sqlalchemy.ext.asyncio import AsyncSession

from iqfmp.db.models import FactorORM, ResearchTrialORM, StrategyORM, BacktestResultORM
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
