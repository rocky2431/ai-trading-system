"""Backtest service for strategy management and backtesting.

Integrates with:
- StrategyRepository for strategy persistence (TimescaleDB)
- BacktestResultORM for backtest results (TimescaleDB)
- Redis for real-time backtest progress tracking
"""

import asyncio
import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

from iqfmp.api.backtest.schemas import (
    BacktestConfig,
    BacktestMetrics,
    BacktestResponse,
    BacktestDetailResponse,
    BacktestEquityCurve,
    BacktestTrade,
    BacktestStatsResponse,
    StrategyResponse,
    OptimizationConfig,
    OptimizationDetailResponse,
    OptimizationResponse,
    OptimizationTrialResult,
)
from iqfmp.db.repositories import StrategyRepository


class StrategyNotFoundError(Exception):
    """Raised when strategy is not found."""
    pass


class BacktestNotFoundError(Exception):
    """Raised when backtest is not found."""
    pass


# =============================================================================
# Helper Functions
# =============================================================================


def parse_datetime_or_default(
    data: dict, key: str, default: datetime | None = None
) -> datetime:
    """Parse datetime from dict, falling back to default or now()."""
    if data.get(key):
        return datetime.fromisoformat(data[key])
    return default if default is not None else datetime.now()


def parse_datetime_or_none(data: dict, key: str) -> datetime | None:
    """Parse datetime from dict, returning None if missing."""
    if data.get(key):
        return datetime.fromisoformat(data[key])
    return None


def parse_metrics(data: dict, key: str = "metrics") -> BacktestMetrics | None:
    """Parse BacktestMetrics from dict, returning None if missing."""
    if data.get(key):
        return BacktestMetrics(**data[key])
    return None


class BacktestService:
    """Service for strategy and backtest management.

    Uses:
    - StrategyRepository for strategy persistence (TimescaleDB primary, Redis cache)
    - Redis for real-time backtest progress tracking
    - BacktestResultORM for backtest results persistence
    """

    def __init__(
        self,
        session: AsyncSession,
        redis_client: Optional[redis.Redis] = None,
    ) -> None:
        """Initialize backtest service."""
        self.session = session
        self.redis_client = redis_client
        # Use StrategyRepository for strategy persistence (DB + Redis cache)
        self.strategy_repo = StrategyRepository(session, redis_client)

    # ==================== Strategy Methods ====================

    async def create_strategy(
        self,
        name: str,
        description: str,
        factor_ids: list[str],
        weighting_method: str,
        rebalance_frequency: str,
        universe: str,
        custom_universe: list[str],
        long_only: bool,
        max_positions: int,
    ) -> StrategyResponse:
        """Create a new strategy and persist to TimescaleDB."""
        strategy_id = str(uuid.uuid4())

        # Store strategy config in config field (JSONB)
        config = {
            "weighting_method": weighting_method,
            "rebalance_frequency": rebalance_frequency,
            "universe": universe,
            "custom_universe": custom_universe,
            "long_only": long_only,
            "max_positions": max_positions,
        }

        # Persist to TimescaleDB via StrategyRepository
        strategy_data = await self.strategy_repo.create(
            strategy_id=strategy_id,
            name=name,
            description=description,
            factor_ids=factor_ids,
            factor_weights={"method": weighting_method},
            code="",  # Strategy code (optional)
            config=config,
            status="draft",
        )
        await self.session.commit()

        # Parse datetime from ISO string
        created_at = parse_datetime_or_default(strategy_data, "created_at")
        updated_at = parse_datetime_or_default(strategy_data, "updated_at")

        return StrategyResponse(
            id=strategy_id,
            name=name,
            description=description,
            factor_ids=factor_ids,
            weighting_method=weighting_method,
            rebalance_frequency=rebalance_frequency,
            universe=universe,
            custom_universe=custom_universe,
            long_only=long_only,
            max_positions=max_positions,
            status="draft",
            created_at=created_at,
            updated_at=updated_at,
        )

    async def list_strategies(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
    ) -> tuple[list[StrategyResponse], int]:
        """List strategies from TimescaleDB with pagination."""
        # Query from TimescaleDB via StrategyRepository
        strategy_dicts, total = await self.strategy_repo.list_strategies(
            page=page,
            page_size=page_size,
            status=status,
        )

        strategies = []
        for data in strategy_dicts:
            config = data.get("config") or {}
            strategies.append(StrategyResponse(
                id=data["id"],
                name=data["name"],
                description=data.get("description", ""),
                factor_ids=data.get("factor_ids", []),
                weighting_method=config.get("weighting_method", "equal"),
                rebalance_frequency=config.get("rebalance_frequency", "daily"),
                universe=config.get("universe", "all"),
                custom_universe=config.get("custom_universe", []),
                long_only=config.get("long_only", False),
                max_positions=config.get("max_positions", 20),
                status=data.get("status", "draft"),
                created_at=parse_datetime_or_default(data, "created_at"),
                updated_at=parse_datetime_or_default(data, "updated_at"),
            ))

        return strategies, total

    async def get_strategy(self, strategy_id: str) -> Optional[StrategyResponse]:
        """Get strategy by ID from TimescaleDB (with Redis cache)."""
        # Query from TimescaleDB via StrategyRepository
        data = await self.strategy_repo.get_by_id(strategy_id)
        if data is None:
            return None

        config = data.get("config") or {}
        return StrategyResponse(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            factor_ids=data.get("factor_ids", []),
            weighting_method=config.get("weighting_method", "equal"),
            rebalance_frequency=config.get("rebalance_frequency", "daily"),
            universe=config.get("universe", "all"),
            custom_universe=config.get("custom_universe", []),
            long_only=config.get("long_only", False),
            max_positions=config.get("max_positions", 20),
            status=data.get("status", "draft"),
            created_at=parse_datetime_or_default(data, "created_at"),
            updated_at=parse_datetime_or_default(data, "updated_at"),
        )

    async def delete_strategy(self, strategy_id: str) -> bool:
        """Delete a strategy from TimescaleDB."""
        result = await self.strategy_repo.delete(strategy_id)
        await self.session.commit()
        return result

    # ==================== Backtest Methods ====================

    async def create_backtest(
        self,
        strategy_id: str,
        config: BacktestConfig,
        name: Optional[str] = None,
        description: str = "",
    ) -> str:
        """Create and start a backtest."""
        # Verify strategy exists
        strategy = await self.get_strategy(strategy_id)
        if not strategy:
            raise StrategyNotFoundError(f"Strategy {strategy_id} not found")

        backtest_id = str(uuid.uuid4())
        now = datetime.now()

        backtest_data = {
            "id": backtest_id,
            "strategy_id": strategy_id,
            "strategy_name": strategy.name,
            "name": name or f"Backtest {now.strftime('%Y%m%d_%H%M%S')}",
            "description": description,
            "config": config.model_dump(),
            "status": "pending",
            "progress": 0.0,
            "metrics": None,
            "error_message": None,
            "created_at": now.isoformat(),
            "started_at": None,
            "completed_at": None,
        }

        if self.redis_client:
            await self.redis_client.hset("backtests", backtest_id, json.dumps(backtest_data))
            await self.redis_client.sadd("backtests:active", backtest_id)

        # Start backtest in background
        asyncio.create_task(self._run_backtest(backtest_id, backtest_data, strategy))

        return backtest_id

    async def _run_backtest(
        self,
        backtest_id: str,
        backtest_data: dict,
        strategy: StrategyResponse,
    ) -> None:
        """Run backtest using real market data and factor signals."""
        try:
            backtest_data["status"] = "running"
            backtest_data["started_at"] = datetime.now().isoformat()
            await self._update_backtest(backtest_id, backtest_data)

            # Import real backtest engine and data/factor providers
            from iqfmp.core.backtest_engine import BacktestEngine
            from iqfmp.core.factor_engine import BUILTIN_FACTORS, FactorEngine
            from iqfmp.core.data_provider import DataProvider
            from iqfmp.db.repositories import FactorRepository

            config = BacktestConfig(**backtest_data["config"])

            # Update progress: loading data
            backtest_data["progress"] = 10.0
            await self._update_backtest(backtest_id, backtest_data)

            # Resolve factor codes from strategy factors (support multi-factor average)
            factor_codes: list[str] = []
            factor_repo = FactorRepository(self.session, self.redis_client)
            for fid in strategy.factor_ids:
                f = await factor_repo.get_by_id(fid)
                if f:
                    factor_codes.append(f.code)

            # Fallback to builtin if none found
            if not factor_codes:
                factor_codes = [BUILTIN_FACTORS.get("momentum_20d", BUILTIN_FACTORS["rsi_14"])]

            # Update progress: running backtest
            backtest_data["progress"] = 30.0
            await self._update_backtest(backtest_id, backtest_data)

            # Load market data from DB (fallback to CSV)
            symbols = config.symbols or ["ETH/USDT"]
            symbol = symbols[0]
            timeframe = config.timeframe or "1d"
            provider = DataProvider(session=self.session)
            df = await provider.load_ohlcv(symbol=symbol, timeframe=timeframe)

            # Compute factor values with real engine (DB data)
            factor_engine = FactorEngine(df=df, symbol=symbol.replace("/", ""), timeframe=timeframe)
            factor_value_series: list[pd.Series] = []
            for code in factor_codes:
                try:
                    factor_value_series.append(factor_engine.compute_factor(code, "signal"))
                except Exception as e:
                    logger.warning(f"Failed to compute factor '{code}': {e}")
                    continue

            if not factor_value_series:
                raise ValueError("No valid factors to backtest")

            # Simple equal-weight combination of multiple factors
            if len(factor_value_series) == 1:
                factor_values = factor_value_series[0]
                factor_code = factor_codes[0]
            else:
                factor_values = sum(factor_value_series) / len(factor_value_series)
                factor_code = "multi_factor_composite"

            # Run real backtest using DB data and computed factor values
            engine = BacktestEngine(df=df)
            result = engine.run_factor_backtest(
                factor_code=factor_code,
                factor_values=factor_values,
                initial_capital=config.initial_capital,
                start_date=config.start_date,
                end_date=config.end_date,
                rebalance_frequency=strategy.rebalance_frequency,
                long_only=strategy.long_only,
            )

            # Update progress: calculating metrics
            backtest_data["progress"] = 80.0
            await self._update_backtest(backtest_id, backtest_data)

            # Convert result to metrics
            metrics = BacktestMetrics(
                total_return=result.total_return,
                annual_return=result.annual_return,
                sharpe_ratio=result.sharpe_ratio,
                sortino_ratio=result.sortino_ratio,
                max_drawdown=result.max_drawdown,
                max_drawdown_duration=result.max_drawdown_duration,
                win_rate=result.win_rate,
                profit_factor=result.profit_factor,
                calmar_ratio=result.calmar_ratio,
                volatility=result.volatility,
                beta=0.0,  # Not calculated in simple backtest
                alpha=0.0,  # Not calculated in simple backtest
                information_ratio=0.0,  # Not calculated in simple backtest
                trade_count=result.trade_count,
                avg_trade_return=result.avg_trade_return,
                avg_holding_period=result.avg_holding_period,
            )

            # Store detailed results for later retrieval
            backtest_data["equity_curve"] = result.equity_curve
            backtest_data["trades"] = [
                {
                    "id": t.id,
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry_date": t.entry_date,
                    "entry_price": t.entry_price,
                    "exit_date": t.exit_date,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "holding_days": t.holding_days,
                }
                for t in result.trades
            ]
            backtest_data["monthly_returns"] = result.monthly_returns
            backtest_data["factor_contributions"] = result.factor_contributions

            backtest_data["metrics"] = metrics.model_dump()
            backtest_data["status"] = "completed"
            backtest_data["progress"] = 100.0
            backtest_data["completed_at"] = datetime.now().isoformat()

            # Persist to TimescaleDB (Task: backtest DB persistence)
            await self._save_backtest_to_db(
                backtest_id=backtest_id,
                strategy_id=strategy.id,
                config=config,
                result=result,
            )

        except Exception as e:
            logger.error(f"Backtest {backtest_id} failed: {e}")
            backtest_data["status"] = "failed"
            backtest_data["error_message"] = str(e)
            backtest_data["completed_at"] = datetime.now().isoformat()

        await self._update_backtest(backtest_id, backtest_data)

        if self.redis_client:
            await self.redis_client.srem("backtests:active", backtest_id)

    async def _save_backtest_to_db(
        self,
        backtest_id: str,
        strategy_id: str,
        config: BacktestConfig,
        result,
    ) -> None:
        """Persist backtest result to TimescaleDB.

        Args:
            backtest_id: Backtest ID
            strategy_id: Strategy ID
            config: Backtest configuration
            result: BacktestResult from backtest_engine
        """
        try:
            from iqfmp.db.models import BacktestResultORM

            # Parse dates
            start_date = datetime.fromisoformat(config.start_date) if config.start_date else datetime.now()
            end_date = datetime.fromisoformat(config.end_date) if config.end_date else datetime.now()

            # Create ORM object
            db_result = BacktestResultORM(
                id=backtest_id,
                strategy_id=strategy_id,
                start_date=start_date,
                end_date=end_date,
                total_return=result.total_return,
                sharpe_ratio=result.sharpe_ratio,
                max_drawdown=result.max_drawdown,
                win_rate=result.win_rate,
                profit_factor=result.profit_factor,
                trade_count=result.trade_count,
                full_results=result.to_dict(),
            )

            self.session.add(db_result)
            await self.session.commit()

        except Exception as e:
            # Log but don't fail - Redis backup exists
            import logging
            logging.getLogger(__name__).warning(f"Failed to save backtest to DB: {e}")
            await self.session.rollback()

    async def _update_backtest(self, backtest_id: str, backtest_data: dict) -> None:
        """Update backtest in Redis."""
        if self.redis_client:
            await self.redis_client.hset("backtests", backtest_id, json.dumps(backtest_data))

    async def list_backtests(
        self,
        strategy_id: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[BacktestResponse], int]:
        """List backtests with filtering and pagination."""
        backtests = []

        if self.redis_client:
            all_backtests = await self.redis_client.hgetall("backtests")
            for data_json in all_backtests.values():
                data = json.loads(data_json)

                if strategy_id and data["strategy_id"] != strategy_id:
                    continue
                if status and data["status"] != status:
                    continue

                backtests.append(BacktestResponse(
                    id=data["id"],
                    strategy_id=data["strategy_id"],
                    strategy_name=data["strategy_name"],
                    name=data["name"],
                    description=data.get("description", ""),
                    config=BacktestConfig(**data["config"]),
                    status=data["status"],
                    progress=data["progress"],
                    metrics=parse_metrics(data),
                    error_message=data.get("error_message"),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    started_at=parse_datetime_or_none(data, "started_at"),
                    completed_at=parse_datetime_or_none(data, "completed_at"),
                ))

        # Also get backtests from DB (for completed backtests not in Redis)
        db_backtests = await self._list_backtests_from_db(strategy_id, status)

        # Merge Redis and DB results (DB results not already in Redis)
        redis_ids = {b.id for b in backtests}
        for db_backtest in db_backtests:
            if db_backtest.id not in redis_ids:
                backtests.append(db_backtest)

        backtests.sort(key=lambda b: b.created_at, reverse=True)
        total = len(backtests)
        start = (page - 1) * page_size
        end = start + page_size

        return backtests[start:end], total

    async def _list_backtests_from_db(
        self,
        strategy_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[BacktestResponse]:
        """List backtests from TimescaleDB."""
        backtests = []
        try:
            from sqlalchemy import select
            from iqfmp.db.models import BacktestResultORM

            query = select(BacktestResultORM).order_by(BacktestResultORM.created_at.desc())

            if strategy_id:
                query = query.where(BacktestResultORM.strategy_id == strategy_id)

            result = await self.session.execute(query.limit(100))
            db_results = result.scalars().all()

            for db_result in db_results:
                full_results = db_result.full_results or {}

                metrics = BacktestMetrics(
                    total_return=db_result.total_return,
                    annual_return=full_results.get("annual_return", 0.0),
                    sharpe_ratio=db_result.sharpe_ratio or 0.0,
                    sortino_ratio=full_results.get("sortino_ratio", 0.0),
                    max_drawdown=db_result.max_drawdown or 0.0,
                    max_drawdown_duration=full_results.get("max_drawdown_duration", 0),
                    win_rate=db_result.win_rate or 0.0,
                    profit_factor=db_result.profit_factor or 0.0,
                    calmar_ratio=full_results.get("calmar_ratio", 0.0),
                    volatility=full_results.get("volatility", 0.0),
                    beta=0.0,
                    alpha=0.0,
                    information_ratio=0.0,
                    trade_count=db_result.trade_count or 0,
                    avg_trade_return=full_results.get("avg_trade_return", 0.0),
                    avg_holding_period=full_results.get("avg_holding_period", 0.0),
                )

                # Only include completed backtests from DB
                if status and status != "completed":
                    continue

                backtests.append(BacktestResponse(
                    id=db_result.id,
                    strategy_id=db_result.strategy_id,
                    strategy_name="",
                    name=f"Backtest {db_result.id[:8]}",
                    description="Loaded from database",
                    config=BacktestConfig(
                        start_date=db_result.start_date.isoformat() if db_result.start_date else None,
                        end_date=db_result.end_date.isoformat() if db_result.end_date else None,
                        initial_capital=100000.0,
                    ),
                    status="completed",
                    progress=100.0,
                    metrics=metrics,
                    error_message=None,
                    created_at=db_result.created_at,
                    started_at=db_result.created_at,
                    completed_at=db_result.created_at,
                ))

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to list backtests from DB: {e}")

        return backtests

    async def get_backtest(self, backtest_id: str) -> Optional[BacktestResponse]:
        """Get backtest by ID from Redis with DB fallback."""
        # Try Redis first (for running backtests)
        if self.redis_client:
            data_json = await self.redis_client.hget("backtests", backtest_id)
            if data_json:
                data = json.loads(data_json)

                return BacktestResponse(
                    id=data["id"],
                    strategy_id=data["strategy_id"],
                    strategy_name=data["strategy_name"],
                    name=data["name"],
                    description=data.get("description", ""),
                    config=BacktestConfig(**data["config"]),
                    status=data["status"],
                    progress=data["progress"],
                    metrics=parse_metrics(data),
                    error_message=data.get("error_message"),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    started_at=parse_datetime_or_none(data, "started_at"),
                    completed_at=parse_datetime_or_none(data, "completed_at"),
                )

        # Fallback to database for completed backtests
        return await self._get_backtest_from_db(backtest_id)

    async def _get_backtest_from_db(self, backtest_id: str) -> Optional[BacktestResponse]:
        """Get backtest from TimescaleDB."""
        try:
            from sqlalchemy import select
            from iqfmp.db.models import BacktestResultORM

            result = await self.session.execute(
                select(BacktestResultORM).where(BacktestResultORM.id == backtest_id)
            )
            db_result = result.scalar_one_or_none()

            if db_result:
                full_results = db_result.full_results or {}

                metrics = BacktestMetrics(
                    total_return=db_result.total_return,
                    annual_return=full_results.get("annual_return", 0.0),
                    sharpe_ratio=db_result.sharpe_ratio or 0.0,
                    sortino_ratio=full_results.get("sortino_ratio", 0.0),
                    max_drawdown=db_result.max_drawdown or 0.0,
                    max_drawdown_duration=full_results.get("max_drawdown_duration", 0),
                    win_rate=db_result.win_rate or 0.0,
                    profit_factor=db_result.profit_factor or 0.0,
                    calmar_ratio=full_results.get("calmar_ratio", 0.0),
                    volatility=full_results.get("volatility", 0.0),
                    beta=0.0,
                    alpha=0.0,
                    information_ratio=0.0,
                    trade_count=db_result.trade_count or 0,
                    avg_trade_return=full_results.get("avg_trade_return", 0.0),
                    avg_holding_period=full_results.get("avg_holding_period", 0.0),
                )

                return BacktestResponse(
                    id=db_result.id,
                    strategy_id=db_result.strategy_id,
                    strategy_name="",  # Not stored in DB
                    name=f"Backtest {db_result.id[:8]}",
                    description="Loaded from database",
                    config=BacktestConfig(
                        start_date=db_result.start_date.isoformat() if db_result.start_date else None,
                        end_date=db_result.end_date.isoformat() if db_result.end_date else None,
                        initial_capital=100000.0,
                    ),
                    status="completed",
                    progress=100.0,
                    metrics=metrics,
                    error_message=None,
                    created_at=db_result.created_at,
                    started_at=db_result.created_at,
                    completed_at=db_result.created_at,
                )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to get backtest from DB: {e}")

        return None

    async def get_backtest_detail(self, backtest_id: str) -> Optional[BacktestDetailResponse]:
        """Get detailed backtest results including real equity curve and trades."""
        backtest = await self.get_backtest(backtest_id)
        if not backtest or backtest.status != "completed":
            return None

        # Get stored backtest data
        if self.redis_client:
            data_json = await self.redis_client.hget("backtests", backtest_id)
            if data_json:
                data = json.loads(data_json)

                # Use stored equity curve (real data)
                stored_equity = data.get("equity_curve", [])
                equity_curve = [
                    BacktestEquityCurve(
                        date=ec["date"],
                        equity=ec["equity"],
                        drawdown=ec["drawdown"],
                        benchmark_equity=ec.get("benchmark_equity", ec["equity"]),
                    )
                    for ec in stored_equity[-500:]  # Limit to last 500 points
                ]

                # Use stored trades (real trades)
                stored_trades = data.get("trades", [])
                trades = [
                    BacktestTrade(
                        id=t["id"],
                        symbol=t["symbol"],
                        side=t["side"],
                        entry_date=t["entry_date"],
                        entry_price=t["entry_price"],
                        exit_date=t["exit_date"],
                        exit_price=t["exit_price"],
                        quantity=t["quantity"],
                        pnl=t["pnl"],
                        pnl_pct=t["pnl_pct"],
                        holding_days=t["holding_days"],
                    )
                    for t in stored_trades
                ]

                # Use stored monthly returns
                monthly_returns = data.get("monthly_returns", {})

                # Use stored factor contributions
                factor_contributions = data.get("factor_contributions", {"momentum": 100.0})

                return BacktestDetailResponse(
                    backtest=backtest,
                    equity_curve=equity_curve,
                    trades=trades,
                    monthly_returns=monthly_returns,
                    factor_contributions=factor_contributions,
                )

        # Fallback if no stored data
        return BacktestDetailResponse(
            backtest=backtest,
            equity_curve=[],
            trades=[],
            monthly_returns={},
            factor_contributions={},
        )

    async def delete_backtest(self, backtest_id: str) -> bool:
        """Delete a backtest."""
        if self.redis_client:
            result = await self.redis_client.hdel("backtests", backtest_id)
            await self.redis_client.srem("backtests:active", backtest_id)
            return result > 0
        return False

    # ==================== Optimization Methods ====================

    async def create_optimization(
        self,
        strategy_id: str,
        backtest_config: BacktestConfig,
        optimization_config: OptimizationConfig,
        name: Optional[str] = None,
        description: str = "",
    ) -> str:
        """Create and start a backtest parameter optimization.

        Uses Optuna for hyperparameter optimization of backtest parameters.

        Args:
            strategy_id: Strategy to optimize
            backtest_config: Base backtest configuration
            optimization_config: Optimization settings
            name: Optional optimization name
            description: Optional description

        Returns:
            Optimization ID
        """
        # Verify strategy exists
        strategy = await self.get_strategy(strategy_id)
        if not strategy:
            raise StrategyNotFoundError(f"Strategy {strategy_id} not found")

        optimization_id = str(uuid.uuid4())
        now = datetime.now()

        optimization_data = {
            "id": optimization_id,
            "strategy_id": strategy_id,
            "strategy_name": strategy.name,
            "name": name or f"Optimization {now.strftime('%Y%m%d_%H%M%S')}",
            "description": description,
            "backtest_config": backtest_config.model_dump(),
            "optimization_config": optimization_config.model_dump(),
            "status": "pending",
            "progress": 0.0,
            "current_trial": 0,
            "total_trials": optimization_config.n_trials,
            "best_trial_id": None,
            "best_params": None,
            "best_metrics": None,
            "top_trials": [],
            "all_trials": [],
            "param_importance": {},
            "convergence_history": [],
            "error_message": None,
            "created_at": now.isoformat(),
            "started_at": None,
            "completed_at": None,
        }

        if self.redis_client:
            await self.redis_client.hset(
                "optimizations", optimization_id, json.dumps(optimization_data)
            )
            await self.redis_client.sadd("optimizations:active", optimization_id)

        # Start optimization in background
        asyncio.create_task(
            self._run_optimization(optimization_id, optimization_data, strategy)
        )

        return optimization_id

    async def _run_optimization(
        self,
        optimization_id: str,
        optimization_data: dict,
        strategy: StrategyResponse,
    ) -> None:
        """Run Optuna optimization in background.

        Args:
            optimization_id: Optimization ID
            optimization_data: Optimization state dict
            strategy: Strategy to optimize
        """
        try:
            optimization_data["status"] = "running"
            optimization_data["started_at"] = datetime.now().isoformat()
            await self._update_optimization(optimization_id, optimization_data)

            # Import optimization components
            from iqfmp.evaluation.backtest_optimizer import (
                BacktestOptimizationConfig,
                BacktestOptimizationMetric,
                BacktestOptimizer,
                BacktestPrunerType,
                BacktestSamplerType,
            )
            from iqfmp.core.unified_backtest import (
                BacktestMode,
                UnifiedBacktestParams,
            )
            from iqfmp.core.data_provider import DataProvider

            opt_config = OptimizationConfig(**optimization_data["optimization_config"])
            bt_config = BacktestConfig(**optimization_data["backtest_config"])

            # Load market data
            symbols = bt_config.symbols or ["ETH/USDT"]
            symbol = symbols[0]
            provider = DataProvider(session=self.session)
            df = await provider.load_ohlcv(symbol=symbol, timeframe=bt_config.timeframe)

            # Create signals from strategy factors
            from iqfmp.core.factor_engine import FactorEngine
            from iqfmp.db.repositories import FactorRepository

            factor_repo = FactorRepository(self.session, self.redis_client)
            factor_engine = FactorEngine(
                df=df, symbol=symbol.replace("/", ""), timeframe=bt_config.timeframe
            )

            # Compute factor signals
            signals_list: list[pd.Series] = []
            for fid in strategy.factor_ids:
                f = await factor_repo.get_by_id(fid)
                if f:
                    try:
                        signals_list.append(
                            factor_engine.compute_factor(f.code, "signal")
                        )
                    except Exception as e:
                        logger.warning(f"Failed to compute factor '{fid}': {e}")
                        continue

            if not signals_list:
                raise ValueError("No valid factors to optimize")

            # Combine signals
            if len(signals_list) == 1:
                signals = signals_list[0].to_frame("signal")
            else:
                combined = sum(signals_list) / len(signals_list)
                signals = combined.to_frame("signal")

            # Map config to optimizer types
            sampler_map = {
                "tpe": BacktestSamplerType.TPE,
                "cmaes": BacktestSamplerType.CMAES,
                "random": BacktestSamplerType.RANDOM,
                "grid": BacktestSamplerType.GRID,
            }
            pruner_map = {
                "median": BacktestPrunerType.MEDIAN,
                "hyperband": BacktestPrunerType.HYPERBAND,
                "percentile": BacktestPrunerType.PERCENTILE,
                "none": BacktestPrunerType.NONE,
            }

            # Get primary metric
            primary_metric = BacktestOptimizationMetric.SHARPE
            if opt_config.metrics:
                metric_map = {
                    "sharpe": BacktestOptimizationMetric.SHARPE,
                    "calmar": BacktestOptimizationMetric.CALMAR,
                    "total_return": BacktestOptimizationMetric.TOTAL_RETURN,
                    "sortino": BacktestOptimizationMetric.SORTINO,
                    "ic": BacktestOptimizationMetric.IC,
                }
                primary_metric = metric_map.get(
                    opt_config.metrics[0].name, BacktestOptimizationMetric.SHARPE
                )

            # Create optimization config
            optimizer_config = BacktestOptimizationConfig(
                n_trials=opt_config.n_trials,
                n_jobs=opt_config.n_jobs,
                timeout=opt_config.timeout,
                metric=primary_metric,
                sampler=sampler_map.get(opt_config.sampler, BacktestSamplerType.TPE),
                pruner=pruner_map.get(opt_config.pruner, BacktestPrunerType.MEDIAN),
            )

            # Create base backtest params
            base_params = UnifiedBacktestParams(
                start_time=bt_config.start_date,
                end_time=bt_config.end_date,
                initial_capital=bt_config.initial_capital,
                commission_rate=bt_config.commission_rate,
                mode=BacktestMode.CRYPTO,
                optimization_run_id=optimization_id,
            )

            # Create optimizer
            optimizer = BacktestOptimizer(
                signals=signals,
                price_data=df,
                config=optimizer_config,
                base_params=base_params,
                redis_client=self.redis_client,
            )

            # Progress callback
            async def progress_callback(
                trial_id: int, value: float, params: dict
            ) -> None:
                optimization_data["current_trial"] = trial_id + 1
                optimization_data["progress"] = (
                    (trial_id + 1) / opt_config.n_trials
                ) * 100

                # Update best if improved
                if (
                    optimization_data["best_metrics"] is None
                    or value > optimization_data["best_metrics"].get("sharpe_ratio", 0)
                ):
                    optimization_data["best_trial_id"] = trial_id
                    optimization_data["best_params"] = params

                optimization_data["convergence_history"].append(
                    {"trial_id": trial_id, "value": value}
                )

                await self._update_optimization(optimization_id, optimization_data)

            # Run optimization (synchronous call in async context)
            import asyncio

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, optimizer.optimize)

            # Process results
            optimization_data["status"] = "completed"
            optimization_data["progress"] = 100.0
            optimization_data["completed_at"] = datetime.now().isoformat()

            if result.best_params:
                optimization_data["best_params"] = result.best_params
                optimization_data["best_trial_id"] = result.best_trial_number
                optimization_data["best_metrics"] = {
                    "total_return": result.best_value,
                    "annual_return": 0.0,
                    "sharpe_ratio": result.best_value
                    if primary_metric == BacktestOptimizationMetric.SHARPE
                    else 0.0,
                    "sortino_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "max_drawdown_duration": 0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "calmar_ratio": result.best_value
                    if primary_metric == BacktestOptimizationMetric.CALMAR
                    else 0.0,
                    "volatility": 0.0,
                    "beta": 0.0,
                    "alpha": 0.0,
                    "information_ratio": 0.0,
                    "trade_count": 0,
                    "avg_trade_return": 0.0,
                    "avg_holding_period": 0.0,
                }

            # Store all trials
            optimization_data["all_trials"] = [
                {
                    "trial_id": t.trial_id,
                    "params": t.params,
                    "metrics": {
                        "total_return": t.value,
                        "sharpe_ratio": t.value
                        if primary_metric == BacktestOptimizationMetric.SHARPE
                        else 0.0,
                        "calmar_ratio": t.value
                        if primary_metric == BacktestOptimizationMetric.CALMAR
                        else 0.0,
                        "max_drawdown": 0.0,
                        "win_rate": 0.0,
                        "profit_factor": 0.0,
                        "trade_count": 0,
                        "annual_return": 0.0,
                        "sortino_ratio": 0.0,
                        "max_drawdown_duration": 0,
                        "volatility": 0.0,
                        "beta": 0.0,
                        "alpha": 0.0,
                        "information_ratio": 0.0,
                        "avg_trade_return": 0.0,
                        "avg_holding_period": 0.0,
                    },
                    "duration_seconds": t.duration_seconds,
                    "rank": i + 1,
                    "pruned": t.pruned,
                }
                for i, t in enumerate(
                    sorted(result.all_trials, key=lambda x: x.value, reverse=True)
                )
            ]

            # Top 10 trials
            optimization_data["top_trials"] = optimization_data["all_trials"][:10]

            # Parameter importance
            optimization_data["param_importance"] = result.param_importance

        except Exception as e:
            logger.error(f"Optimization {optimization_id} failed: {e}")
            optimization_data["status"] = "failed"
            optimization_data["error_message"] = str(e)
            optimization_data["completed_at"] = datetime.now().isoformat()

        await self._update_optimization(optimization_id, optimization_data)

        if self.redis_client:
            await self.redis_client.srem("optimizations:active", optimization_id)

    async def _update_optimization(
        self, optimization_id: str, optimization_data: dict
    ) -> None:
        """Update optimization in Redis."""
        if self.redis_client:
            await self.redis_client.hset(
                "optimizations", optimization_id, json.dumps(optimization_data)
            )

    async def list_optimizations(
        self,
        strategy_id: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[OptimizationResponse], int]:
        """List optimizations with filtering and pagination."""
        optimizations = []

        if self.redis_client:
            all_optimizations = await self.redis_client.hgetall("optimizations")
            for data_json in all_optimizations.values():
                data = json.loads(data_json)

                if strategy_id and data["strategy_id"] != strategy_id:
                    continue
                if status and data["status"] != status:
                    continue

                top_trials = []
                for t in data.get("top_trials", [])[:5]:
                    trial_metrics = parse_metrics(t)
                    if trial_metrics:
                        top_trials.append(
                            OptimizationTrialResult(
                                trial_id=t["trial_id"],
                                params=t["params"],
                                metrics=trial_metrics,
                                duration_seconds=t["duration_seconds"],
                                rank=t["rank"],
                                pruned=t.get("pruned", False),
                            )
                        )

                optimizations.append(
                    OptimizationResponse(
                        id=data["id"],
                        strategy_id=data["strategy_id"],
                        name=data["name"],
                        description=data.get("description", ""),
                        status=data["status"],
                        progress=data["progress"],
                        current_trial=data["current_trial"],
                        total_trials=data["total_trials"],
                        best_trial_id=data.get("best_trial_id"),
                        best_params=data.get("best_params"),
                        best_metrics=parse_metrics(data, "best_metrics"),
                        top_trials=top_trials,
                        error_message=data.get("error_message"),
                        created_at=datetime.fromisoformat(data["created_at"]),
                        started_at=parse_datetime_or_none(data, "started_at"),
                        completed_at=parse_datetime_or_none(data, "completed_at"),
                    )
                )

        optimizations.sort(key=lambda o: o.created_at, reverse=True)
        total = len(optimizations)
        start = (page - 1) * page_size
        end = start + page_size

        return optimizations[start:end], total

    async def get_optimization(
        self, optimization_id: str
    ) -> Optional[OptimizationResponse]:
        """Get optimization by ID."""
        if self.redis_client:
            data_json = await self.redis_client.hget("optimizations", optimization_id)
            if data_json:
                data = json.loads(data_json)

                top_trials = []
                for t in data.get("top_trials", [])[:5]:
                    trial_metrics = parse_metrics(t)
                    if trial_metrics:
                        top_trials.append(
                            OptimizationTrialResult(
                                trial_id=t["trial_id"],
                                params=t["params"],
                                metrics=trial_metrics,
                                duration_seconds=t["duration_seconds"],
                                rank=t["rank"],
                                pruned=t.get("pruned", False),
                            )
                        )

                return OptimizationResponse(
                    id=data["id"],
                    strategy_id=data["strategy_id"],
                    name=data["name"],
                    description=data.get("description", ""),
                    status=data["status"],
                    progress=data["progress"],
                    current_trial=data["current_trial"],
                    total_trials=data["total_trials"],
                    best_trial_id=data.get("best_trial_id"),
                    best_params=data.get("best_params"),
                    best_metrics=parse_metrics(data, "best_metrics"),
                    top_trials=top_trials,
                    error_message=data.get("error_message"),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    started_at=parse_datetime_or_none(data, "started_at"),
                    completed_at=parse_datetime_or_none(data, "completed_at"),
                )

        return None

    async def get_optimization_detail(
        self, optimization_id: str
    ) -> Optional[OptimizationDetailResponse]:
        """Get detailed optimization results."""
        optimization = await self.get_optimization(optimization_id)
        if not optimization or optimization.status != "completed":
            return None

        if self.redis_client:
            data_json = await self.redis_client.hget("optimizations", optimization_id)
            if data_json:
                data = json.loads(data_json)

                all_trials = []
                for t in data.get("all_trials", []):
                    trial_metrics = parse_metrics(t)
                    if trial_metrics:
                        all_trials.append(
                            OptimizationTrialResult(
                                trial_id=t["trial_id"],
                                params=t["params"],
                                metrics=trial_metrics,
                                duration_seconds=t["duration_seconds"],
                                rank=t["rank"],
                                pruned=t.get("pruned", False),
                            )
                        )

                return OptimizationDetailResponse(
                    optimization=optimization,
                    all_trials=all_trials,
                    param_importance=data.get("param_importance", {}),
                    convergence_history=data.get("convergence_history", []),
                )

        return None

    async def cancel_optimization(self, optimization_id: str) -> bool:
        """Cancel a running optimization."""
        if self.redis_client:
            data_json = await self.redis_client.hget("optimizations", optimization_id)
            if data_json:
                data = json.loads(data_json)
                if data["status"] == "running":
                    data["status"] = "cancelled"
                    data["completed_at"] = datetime.now().isoformat()
                    await self.redis_client.hset(
                        "optimizations", optimization_id, json.dumps(data)
                    )
                    await self.redis_client.srem("optimizations:active", optimization_id)
                    return True
        return False

    async def delete_optimization(self, optimization_id: str) -> bool:
        """Delete an optimization."""
        if self.redis_client:
            result = await self.redis_client.hdel("optimizations", optimization_id)
            await self.redis_client.srem("optimizations:active", optimization_id)
            return result > 0
        return False

    # ==================== Stats Methods ====================

    async def get_stats(self) -> BacktestStatsResponse:
        """Get backtest statistics."""
        strategies, _ = await self.list_strategies(page=1, page_size=1000)
        backtests, _ = await self.list_backtests(page=1, page_size=1000)

        running = sum(1 for b in backtests if b.status in ("pending", "running"))
        today = datetime.now().date()
        completed_today = sum(
            1 for b in backtests
            if b.completed_at and b.completed_at.date() == today
        )

        sharpes = [b.metrics.sharpe_ratio for b in backtests if b.metrics]
        avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0

        best_backtest = max(
            (b for b in backtests if b.metrics),
            key=lambda b: b.metrics.sharpe_ratio if b.metrics else 0,
            default=None,
        )

        return BacktestStatsResponse(
            total_strategies=len(strategies),
            total_backtests=len(backtests),
            running_backtests=running,
            completed_today=completed_today,
            avg_sharpe=avg_sharpe,
            best_strategy_id=best_backtest.strategy_id if best_backtest else None,
            best_sharpe=best_backtest.metrics.sharpe_ratio if best_backtest and best_backtest.metrics else 0,
        )
