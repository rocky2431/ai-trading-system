"""Strategy API router."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from iqfmp.db.database import get_db, get_redis
from iqfmp.api.strategies.schemas import (
    BacktestListResponse,
    BacktestRequest,
    BacktestResultResponse,
    StrategyCreateRequest,
    StrategyListResponse,
    StrategyResponse,
    StrategyUpdateRequest,
)
from iqfmp.api.strategies.service import StrategyService

router = APIRouter(tags=["strategies"])


async def get_strategy_service(
    session: AsyncSession = Depends(get_db),
    redis_client=Depends(get_redis),
) -> StrategyService:
    """Dependency injection for StrategyService."""
    return StrategyService(session, redis_client)


@router.post("", response_model=StrategyResponse, status_code=201)
async def create_strategy(
    request: StrategyCreateRequest,
    service: StrategyService = Depends(get_strategy_service),
) -> StrategyResponse:
    """Create a new strategy."""
    strategy = await service.create_strategy(
        name=request.name,
        code=request.code,
        description=request.description,
        factor_ids=request.factor_ids,
        factor_weights=request.factor_weights,
        config=request.config,
    )
    return StrategyResponse(**strategy)


@router.get("", response_model=StrategyListResponse)
async def list_strategies(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    status: Optional[str] = Query(default=None),
    service: StrategyService = Depends(get_strategy_service),
) -> StrategyListResponse:
    """List strategies with pagination."""
    strategies, total = await service.list_strategies(page, page_size, status)
    return StrategyListResponse(
        strategies=[StrategyResponse(**s) for s in strategies],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(
    strategy_id: str,
    service: StrategyService = Depends(get_strategy_service),
) -> StrategyResponse:
    """Get a strategy by ID."""
    strategy = await service.get_strategy(strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return StrategyResponse(**strategy)


@router.patch("/{strategy_id}", response_model=StrategyResponse)
async def update_strategy(
    strategy_id: str,
    request: StrategyUpdateRequest,
    service: StrategyService = Depends(get_strategy_service),
) -> StrategyResponse:
    """Update a strategy."""
    strategy = await service.update_strategy(
        strategy_id,
        name=request.name,
        description=request.description,
        factor_ids=request.factor_ids,
        factor_weights=request.factor_weights,
        code=request.code,
        config=request.config,
        status=request.status,
    )
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return StrategyResponse(**strategy)


@router.delete("/{strategy_id}", status_code=204)
async def delete_strategy(
    strategy_id: str,
    service: StrategyService = Depends(get_strategy_service),
) -> None:
    """Delete a strategy."""
    deleted = await service.delete_strategy(strategy_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Strategy not found")


@router.post("/{strategy_id}/backtest", response_model=BacktestResultResponse, status_code=201)
async def run_backtest(
    strategy_id: str,
    request: BacktestRequest,
    service: StrategyService = Depends(get_strategy_service),
) -> BacktestResultResponse:
    """Run a backtest for a strategy."""
    strategy = await service.get_strategy(strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")

    result = await service.run_backtest(
        strategy_id=strategy_id,
        start_date=request.start_date,
        end_date=request.end_date,
        initial_capital=request.initial_capital,
        commission=request.commission,
    )
    return BacktestResultResponse(**result)


@router.get("/{strategy_id}/backtests", response_model=BacktestListResponse)
async def list_backtests(
    strategy_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    service: StrategyService = Depends(get_strategy_service),
) -> BacktestListResponse:
    """List backtest results for a strategy."""
    results = await service.get_backtest_results(strategy_id, limit)
    return BacktestListResponse(
        results=[BacktestResultResponse(**r) for r in results],
        total=len(results),
    )
