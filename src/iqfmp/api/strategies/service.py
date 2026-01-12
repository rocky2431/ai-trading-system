"""Strategy service for business logic with database support."""

import uuid
from datetime import datetime
from typing import Optional

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

from iqfmp.db.repositories import StrategyRepository, BacktestResultRepository


class StrategyService:
    """Service for strategy management with database persistence."""

    def __init__(
        self,
        session: AsyncSession,
        redis_client: Optional[redis.Redis] = None,
    ) -> None:
        """Initialize strategy service with database session."""
        self.session = session
        self.redis_client = redis_client
        self.strategy_repo = StrategyRepository(session, redis_client)
        self.backtest_repo = BacktestResultRepository(session)

    async def create_strategy(
        self,
        name: str,
        code: str,
        description: Optional[str] = None,
        factor_ids: Optional[list[str]] = None,
        factor_weights: Optional[dict[str, float]] = None,
        config: Optional[dict] = None,
    ) -> dict:
        """Create a new strategy.

        Args:
            name: Strategy name
            code: Strategy code
            description: Strategy description
            factor_ids: List of factor IDs used
            factor_weights: Weights for each factor
            config: Strategy configuration

        Returns:
            Created strategy dict
        """
        strategy_id = str(uuid.uuid4())
        return await self.strategy_repo.create(
            strategy_id=strategy_id,
            name=name,
            description=description,
            factor_ids=factor_ids or [],
            factor_weights=factor_weights,
            code=code,
            config=config,
            status="draft",
        )

    async def get_strategy(self, strategy_id: str) -> Optional[dict]:
        """Get strategy by ID."""
        return await self.strategy_repo.get_by_id(strategy_id)

    async def list_strategies(
        self,
        page: int = 1,
        page_size: int = 10,
        status: Optional[str] = None,
    ) -> tuple[list[dict], int]:
        """List strategies with pagination."""
        return await self.strategy_repo.list_strategies(page, page_size, status)

    async def update_strategy(
        self,
        strategy_id: str,
        **updates
    ) -> Optional[dict]:
        """Update a strategy."""
        # Filter out None values
        updates = {k: v for k, v in updates.items() if v is not None}
        if not updates:
            return await self.strategy_repo.get_by_id(strategy_id)
        return await self.strategy_repo.update(strategy_id, **updates)

    async def delete_strategy(self, strategy_id: str) -> bool:
        """Delete a strategy."""
        return await self.strategy_repo.delete(strategy_id)

    async def run_backtest(
        self,
        strategy_id: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
    ) -> dict:
        """Run a backtest for a strategy.

        WARNING: This is currently a SIMULATED backtest using deterministic
        random values. The results are NOT real backtest metrics.

        IMPLEMENTATION NOTE: Integration with UnifiedBacktestRunner pending.
        Requirements for real backtest:
        1. Factor signal data retrieval from factor_ids
        2. Price data retrieval for the date range
        3. UnifiedBacktestRunner execution (see core/unified_backtest.py)

        The simulated results use deterministic seeding so the same
        strategy + date range always produces consistent (but fake) results.
        """
        import random
        # Deterministic seed for consistent results per strategy/date
        random.seed(hash(strategy_id + str(start_date)))

        # SIMULATED RESULTS - NOT REAL BACKTEST DATA
        # These values are generated randomly but deterministically
        total_return = random.uniform(-0.2, 0.5)
        sharpe_ratio = random.uniform(0.5, 2.5)
        max_drawdown = random.uniform(0.05, 0.3)
        win_rate = random.uniform(0.4, 0.65)
        profit_factor = random.uniform(1.0, 2.5)
        trade_count = random.randint(50, 500)

        result_id = str(uuid.uuid4())
        return await self.backtest_repo.create(
            result_id=result_id,
            strategy_id=strategy_id,
            start_date=start_date,
            end_date=end_date,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trade_count=trade_count,
            full_results={
                "initial_capital": initial_capital,
                "commission": commission,
                "final_capital": initial_capital * (1 + total_return),
                # CRITICAL: Flag to indicate this is simulated data
                "is_simulated": True,
                "simulation_note": "Results are deterministically generated, not from real backtest execution.",
            },
        )

    async def get_backtest_results(
        self,
        strategy_id: str,
        limit: int = 20,
    ) -> list[dict]:
        """Get backtest results for a strategy."""
        return await self.backtest_repo.list_by_strategy(strategy_id, limit)

    async def get_best_backtest(self, strategy_id: str) -> Optional[dict]:
        """Get best backtest result for a strategy."""
        return await self.backtest_repo.get_best_result(strategy_id)
