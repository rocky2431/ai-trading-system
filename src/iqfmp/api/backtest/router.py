"""Backtest API router."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from iqfmp.api.backtest.schemas import (
    BacktestCreateRequest,
    BacktestDetailResponse,
    BacktestListResponse,
    BacktestResponse,
    BacktestStatsResponse,
    GenericResponse,
    StrategyCreateRequest,
    StrategyListResponse,
    StrategyResponse,
)
from iqfmp.api.backtest.service import (
    BacktestNotFoundError,
    BacktestService,
    StrategyNotFoundError,
)
from iqfmp.db.database import get_db, get_redis

router = APIRouter(tags=["backtest"])


async def get_backtest_service(
    session: AsyncSession = Depends(get_db),
    redis_client=Depends(get_redis),
) -> BacktestService:
    """Dependency injection for BacktestService."""
    return BacktestService(session, redis_client)


# ==================== Strategy Endpoints ====================


@router.post("/strategies", response_model=StrategyResponse, status_code=status.HTTP_201_CREATED)
async def create_strategy(
    request: StrategyCreateRequest,
    service: BacktestService = Depends(get_backtest_service),
) -> StrategyResponse:
    """Create a new strategy.

    Args:
        request: Strategy creation request
        service: Backtest service

    Returns:
        Created strategy
    """
    return await service.create_strategy(
        name=request.name,
        description=request.description,
        factor_ids=request.factor_ids,
        weighting_method=request.weighting_method,
        rebalance_frequency=request.rebalance_frequency,
        universe=request.universe,
        custom_universe=request.custom_universe,
        long_only=request.long_only,
        max_positions=request.max_positions,
    )


@router.get("/strategies", response_model=StrategyListResponse)
async def list_strategies(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: Optional[str] = Query(None, alias="status"),
    service: BacktestService = Depends(get_backtest_service),
) -> StrategyListResponse:
    """List strategies with pagination.

    Args:
        page: Page number
        page_size: Items per page
        status_filter: Filter by status
        service: Backtest service

    Returns:
        List of strategies
    """
    strategies, total = await service.list_strategies(
        page=page,
        page_size=page_size,
        status=status_filter,
    )
    return StrategyListResponse(
        strategies=strategies,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/strategies/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(
    strategy_id: str,
    service: BacktestService = Depends(get_backtest_service),
) -> StrategyResponse:
    """Get strategy by ID.

    Args:
        strategy_id: Strategy ID
        service: Backtest service

    Returns:
        Strategy details

    Raises:
        HTTPException: If strategy not found
    """
    strategy = await service.get_strategy(strategy_id)
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found",
        )
    return strategy


@router.delete("/strategies/{strategy_id}", response_model=GenericResponse)
async def delete_strategy(
    strategy_id: str,
    service: BacktestService = Depends(get_backtest_service),
) -> GenericResponse:
    """Delete a strategy.

    Args:
        strategy_id: Strategy ID
        service: Backtest service

    Returns:
        Success/failure response
    """
    success = await service.delete_strategy(strategy_id)
    if success:
        return GenericResponse(success=True, message=f"Strategy {strategy_id} deleted")
    return GenericResponse(success=False, message=f"Strategy {strategy_id} not found")


# ==================== Backtest Endpoints ====================


@router.post("/backtests", response_model=GenericResponse, status_code=status.HTTP_201_CREATED)
async def create_backtest(
    request: BacktestCreateRequest,
    service: BacktestService = Depends(get_backtest_service),
) -> GenericResponse:
    """Create and start a backtest.

    Args:
        request: Backtest creation request
        service: Backtest service

    Returns:
        Success response with backtest ID
    """
    try:
        backtest_id = await service.create_backtest(
            strategy_id=request.strategy_id,
            config=request.config,
            name=request.name,
            description=request.description,
        )
        return GenericResponse(
            success=True,
            message=f"Backtest created with ID: {backtest_id}",
        )
    except StrategyNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.get("/backtests", response_model=BacktestListResponse)
async def list_backtests(
    strategy_id: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None, alias="status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    service: BacktestService = Depends(get_backtest_service),
) -> BacktestListResponse:
    """List backtests with filtering and pagination.

    Args:
        strategy_id: Filter by strategy ID
        status_filter: Filter by status
        page: Page number
        page_size: Items per page
        service: Backtest service

    Returns:
        List of backtests
    """
    backtests, total = await service.list_backtests(
        strategy_id=strategy_id,
        status=status_filter,
        page=page,
        page_size=page_size,
    )
    return BacktestListResponse(
        backtests=backtests,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/backtests/{backtest_id}", response_model=BacktestResponse)
async def get_backtest(
    backtest_id: str,
    service: BacktestService = Depends(get_backtest_service),
) -> BacktestResponse:
    """Get backtest by ID.

    Args:
        backtest_id: Backtest ID
        service: Backtest service

    Returns:
        Backtest details

    Raises:
        HTTPException: If backtest not found
    """
    backtest = await service.get_backtest(backtest_id)
    if not backtest:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backtest {backtest_id} not found",
        )
    return backtest


@router.get("/backtests/{backtest_id}/detail", response_model=BacktestDetailResponse)
async def get_backtest_detail(
    backtest_id: str,
    service: BacktestService = Depends(get_backtest_service),
) -> BacktestDetailResponse:
    """Get detailed backtest results including equity curve and trades.

    Args:
        backtest_id: Backtest ID
        service: Backtest service

    Returns:
        Detailed backtest results

    Raises:
        HTTPException: If backtest not found or not completed
    """
    detail = await service.get_backtest_detail(backtest_id)
    if not detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backtest {backtest_id} not found or not completed",
        )
    return detail


@router.delete("/backtests/{backtest_id}", response_model=GenericResponse)
async def delete_backtest(
    backtest_id: str,
    service: BacktestService = Depends(get_backtest_service),
) -> GenericResponse:
    """Delete a backtest.

    Args:
        backtest_id: Backtest ID
        service: Backtest service

    Returns:
        Success/failure response
    """
    success = await service.delete_backtest(backtest_id)
    if success:
        return GenericResponse(success=True, message=f"Backtest {backtest_id} deleted")
    return GenericResponse(success=False, message=f"Backtest {backtest_id} not found")


# ==================== Stats Endpoint ====================


@router.get("/stats", response_model=BacktestStatsResponse)
async def get_backtest_stats(
    service: BacktestService = Depends(get_backtest_service),
) -> BacktestStatsResponse:
    """Get backtest statistics.

    Args:
        service: Backtest service

    Returns:
        Backtest statistics
    """
    return await service.get_stats()
