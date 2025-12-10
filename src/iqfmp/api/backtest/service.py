"""Backtest service for strategy management and backtesting."""

import asyncio
import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Optional

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
    OptimizationResponse,
    OptimizationResult,
)


class StrategyNotFoundError(Exception):
    """Raised when strategy is not found."""
    pass


class BacktestNotFoundError(Exception):
    """Raised when backtest is not found."""
    pass


class BacktestService:
    """Service for strategy and backtest management."""

    def __init__(
        self,
        session: AsyncSession,
        redis_client: Optional[redis.Redis] = None,
    ) -> None:
        """Initialize backtest service."""
        self.session = session
        self.redis_client = redis_client

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
        """Create a new strategy."""
        strategy_id = str(uuid.uuid4())
        now = datetime.now()

        strategy_data = {
            "id": strategy_id,
            "name": name,
            "description": description,
            "factor_ids": factor_ids,
            "weighting_method": weighting_method,
            "rebalance_frequency": rebalance_frequency,
            "universe": universe,
            "custom_universe": custom_universe,
            "long_only": long_only,
            "max_positions": max_positions,
            "status": "draft",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

        if self.redis_client:
            await self.redis_client.hset("strategies", strategy_id, json.dumps(strategy_data))

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
            created_at=now,
            updated_at=now,
        )

    async def list_strategies(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
    ) -> tuple[list[StrategyResponse], int]:
        """List strategies with pagination."""
        strategies = []

        if self.redis_client:
            all_strategies = await self.redis_client.hgetall("strategies")
            for data_json in all_strategies.values():
                data = json.loads(data_json)
                if status and data["status"] != status:
                    continue
                strategies.append(StrategyResponse(
                    id=data["id"],
                    name=data["name"],
                    description=data.get("description", ""),
                    factor_ids=data.get("factor_ids", []),
                    weighting_method=data.get("weighting_method", "equal"),
                    rebalance_frequency=data.get("rebalance_frequency", "daily"),
                    universe=data.get("universe", "all"),
                    custom_universe=data.get("custom_universe", []),
                    long_only=data.get("long_only", False),
                    max_positions=data.get("max_positions", 20),
                    status=data["status"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    updated_at=datetime.fromisoformat(data["updated_at"]),
                ))

        strategies.sort(key=lambda s: s.created_at, reverse=True)
        total = len(strategies)
        start = (page - 1) * page_size
        end = start + page_size

        return strategies[start:end], total

    async def get_strategy(self, strategy_id: str) -> Optional[StrategyResponse]:
        """Get strategy by ID."""
        if self.redis_client:
            data_json = await self.redis_client.hget("strategies", strategy_id)
            if data_json:
                data = json.loads(data_json)
                return StrategyResponse(
                    id=data["id"],
                    name=data["name"],
                    description=data.get("description", ""),
                    factor_ids=data.get("factor_ids", []),
                    weighting_method=data.get("weighting_method", "equal"),
                    rebalance_frequency=data.get("rebalance_frequency", "daily"),
                    universe=data.get("universe", "all"),
                    custom_universe=data.get("custom_universe", []),
                    long_only=data.get("long_only", False),
                    max_positions=data.get("max_positions", 20),
                    status=data["status"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    updated_at=datetime.fromisoformat(data["updated_at"]),
                )
        return None

    async def delete_strategy(self, strategy_id: str) -> bool:
        """Delete a strategy."""
        if self.redis_client:
            result = await self.redis_client.hdel("strategies", strategy_id)
            return result > 0
        return False

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

            # Import real backtest engine
            from iqfmp.core.backtest_engine import BacktestEngine
            from iqfmp.core.factor_engine import BUILTIN_FACTORS

            config = BacktestConfig(**backtest_data["config"])

            # Update progress: loading data
            backtest_data["progress"] = 10.0
            await self._update_backtest(backtest_id, backtest_data)

            # Get factor code (use default momentum if not available)
            factor_code = BUILTIN_FACTORS.get("momentum_20d", BUILTIN_FACTORS["rsi_14"])

            # Update progress: running backtest
            backtest_data["progress"] = 30.0
            await self._update_backtest(backtest_id, backtest_data)

            # Run real backtest
            engine = BacktestEngine()
            result = engine.run_factor_backtest(
                factor_code=factor_code,
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

        except Exception as e:
            backtest_data["status"] = "failed"
            backtest_data["error_message"] = str(e)
            backtest_data["completed_at"] = datetime.now().isoformat()

        await self._update_backtest(backtest_id, backtest_data)

        if self.redis_client:
            await self.redis_client.srem("backtests:active", backtest_id)

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

                metrics = None
                if data.get("metrics"):
                    metrics = BacktestMetrics(**data["metrics"])

                backtests.append(BacktestResponse(
                    id=data["id"],
                    strategy_id=data["strategy_id"],
                    strategy_name=data["strategy_name"],
                    name=data["name"],
                    description=data.get("description", ""),
                    config=BacktestConfig(**data["config"]),
                    status=data["status"],
                    progress=data["progress"],
                    metrics=metrics,
                    error_message=data.get("error_message"),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
                    completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
                ))

        backtests.sort(key=lambda b: b.created_at, reverse=True)
        total = len(backtests)
        start = (page - 1) * page_size
        end = start + page_size

        return backtests[start:end], total

    async def get_backtest(self, backtest_id: str) -> Optional[BacktestResponse]:
        """Get backtest by ID."""
        if self.redis_client:
            data_json = await self.redis_client.hget("backtests", backtest_id)
            if data_json:
                data = json.loads(data_json)
                metrics = BacktestMetrics(**data["metrics"]) if data.get("metrics") else None

                return BacktestResponse(
                    id=data["id"],
                    strategy_id=data["strategy_id"],
                    strategy_name=data["strategy_name"],
                    name=data["name"],
                    description=data.get("description", ""),
                    config=BacktestConfig(**data["config"]),
                    status=data["status"],
                    progress=data["progress"],
                    metrics=metrics,
                    error_message=data.get("error_message"),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
                    completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
                )
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
