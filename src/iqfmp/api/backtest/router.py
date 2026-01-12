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
    CreateFromTemplateRequest,
    GenericResponse,
    OptimizationDetailResponse,
    OptimizationListResponse,
    OptimizationRequest,
    OptimizationResponse,
    StrategyResponse,
    StrategyTemplateListResponse,
    StrategyTemplateResponse,
)
from iqfmp.api.backtest.service import (
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


# ==================== Backtest Endpoints ====================
# NOTE: Strategy CRUD endpoints have been moved to /strategies router.
# Use /strategies/* instead of /backtest/strategies/*


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


# ==================== Optimization Endpoints ====================


@router.post(
    "/optimizations",
    response_model=GenericResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_optimization(
    request: OptimizationRequest,
    service: BacktestService = Depends(get_backtest_service),
) -> GenericResponse:
    """Create and start a backtest parameter optimization.

    Uses Optuna for hyperparameter optimization with configurable:
    - Sampler (TPE, CMAES, Random, Grid)
    - Pruner (Median, Hyperband, Percentile)
    - Metrics (Sharpe, Calmar, Return, etc.)

    Args:
        request: Optimization configuration
        service: Backtest service

    Returns:
        Success response with optimization ID
    """
    try:
        optimization_id = await service.create_optimization(
            strategy_id=request.strategy_id,
            backtest_config=request.backtest_config,
            optimization_config=request.optimization_config,
            name=request.name,
            description=request.description,
        )
        return GenericResponse(
            success=True,
            message=f"Optimization started with ID: {optimization_id}",
        )
    except StrategyNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.get("/optimizations", response_model=OptimizationListResponse)
async def list_optimizations(
    strategy_id: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None, alias="status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    service: BacktestService = Depends(get_backtest_service),
) -> OptimizationListResponse:
    """List optimizations with filtering and pagination.

    Args:
        strategy_id: Filter by strategy ID
        status_filter: Filter by status
        page: Page number
        page_size: Items per page
        service: Backtest service

    Returns:
        List of optimizations
    """
    optimizations, total = await service.list_optimizations(
        strategy_id=strategy_id,
        status=status_filter,
        page=page,
        page_size=page_size,
    )
    return OptimizationListResponse(
        optimizations=optimizations,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/optimizations/{optimization_id}", response_model=OptimizationResponse)
async def get_optimization(
    optimization_id: str,
    service: BacktestService = Depends(get_backtest_service),
) -> OptimizationResponse:
    """Get optimization by ID.

    Args:
        optimization_id: Optimization ID
        service: Backtest service

    Returns:
        Optimization details

    Raises:
        HTTPException: If optimization not found
    """
    optimization = await service.get_optimization(optimization_id)
    if not optimization:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Optimization {optimization_id} not found",
        )
    return optimization


@router.get(
    "/optimizations/{optimization_id}/detail",
    response_model=OptimizationDetailResponse,
)
async def get_optimization_detail(
    optimization_id: str,
    service: BacktestService = Depends(get_backtest_service),
) -> OptimizationDetailResponse:
    """Get detailed optimization results including all trials.

    Args:
        optimization_id: Optimization ID
        service: Backtest service

    Returns:
        Detailed optimization results

    Raises:
        HTTPException: If optimization not found or not completed
    """
    detail = await service.get_optimization_detail(optimization_id)
    if not detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Optimization {optimization_id} not found or not completed",
        )
    return detail


@router.post("/optimizations/{optimization_id}/cancel", response_model=GenericResponse)
async def cancel_optimization(
    optimization_id: str,
    service: BacktestService = Depends(get_backtest_service),
) -> GenericResponse:
    """Cancel a running optimization.

    Args:
        optimization_id: Optimization ID
        service: Backtest service

    Returns:
        Success/failure response
    """
    success = await service.cancel_optimization(optimization_id)
    if success:
        return GenericResponse(
            success=True,
            message=f"Optimization {optimization_id} cancelled",
        )
    return GenericResponse(
        success=False,
        message=f"Optimization {optimization_id} not found or already completed",
    )


@router.delete("/optimizations/{optimization_id}", response_model=GenericResponse)
async def delete_optimization(
    optimization_id: str,
    service: BacktestService = Depends(get_backtest_service),
) -> GenericResponse:
    """Delete an optimization.

    Args:
        optimization_id: Optimization ID
        service: Backtest service

    Returns:
        Success/failure response
    """
    success = await service.delete_optimization(optimization_id)
    if success:
        return GenericResponse(
            success=True,
            message=f"Optimization {optimization_id} deleted",
        )
    return GenericResponse(
        success=False,
        message=f"Optimization {optimization_id} not found",
    )


# ==================== Strategy Template Endpoints (P1-2) ====================


@router.get("/templates", response_model=StrategyTemplateListResponse)
async def list_strategy_templates(
    category: Optional[str] = Query(None),
    risk_level: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
) -> StrategyTemplateListResponse:
    """List available strategy templates.

    Args:
        category: Filter by category (momentum, mean_reversion, multi_factor, crypto)
        risk_level: Filter by risk level (conservative, moderate, aggressive)
        search: Search query for name, description, or tags

    Returns:
        List of strategy templates
    """
    from iqfmp.api.backtest.templates import (
        get_all_templates,
        search_templates,
    )

    # Start with all templates
    templates = get_all_templates()

    # Apply filters
    if category:
        templates = [t for t in templates if t.category == category]

    if risk_level:
        templates = [t for t in templates if t.risk_level == risk_level]

    if search:
        search_results = search_templates(search)
        template_ids = {t.id for t in search_results}
        templates = [t for t in templates if t.id in template_ids]

    # Convert to response format
    response_templates = [
        StrategyTemplateResponse(
            id=t.id,
            name=t.name,
            description=t.description,
            category=t.category,
            risk_level=t.risk_level,
            factors=t.factors,
            factor_descriptions=t.factor_descriptions,
            weighting_method=t.weighting_method,
            rebalance_frequency=t.rebalance_frequency,
            max_positions=t.max_positions,
            long_only=t.long_only,
            max_drawdown=t.max_drawdown,
            position_size_limit=t.position_size_limit,
            stop_loss_enabled=t.stop_loss_enabled,
            stop_loss_threshold=t.stop_loss_threshold,
            expected_sharpe=t.expected_sharpe,
            expected_annual_return=t.expected_annual_return,
            expected_max_drawdown=t.expected_max_drawdown,
            tags=t.tags,
            suitable_for=t.suitable_for,
            not_suitable_for=t.not_suitable_for,
        )
        for t in templates
    ]

    return StrategyTemplateListResponse(
        templates=response_templates,
        total=len(response_templates),
    )


@router.get("/templates/{template_id}", response_model=StrategyTemplateResponse)
async def get_strategy_template(template_id: str) -> StrategyTemplateResponse:
    """Get a specific strategy template by ID.

    Args:
        template_id: Template ID

    Returns:
        Strategy template details

    Raises:
        HTTPException: If template not found
    """
    from iqfmp.api.backtest.templates import get_template_by_id

    template = get_template_by_id(template_id)
    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template {template_id} not found",
        )

    return StrategyTemplateResponse(
        id=template.id,
        name=template.name,
        description=template.description,
        category=template.category,
        risk_level=template.risk_level,
        factors=template.factors,
        factor_descriptions=template.factor_descriptions,
        weighting_method=template.weighting_method,
        rebalance_frequency=template.rebalance_frequency,
        max_positions=template.max_positions,
        long_only=template.long_only,
        max_drawdown=template.max_drawdown,
        position_size_limit=template.position_size_limit,
        stop_loss_enabled=template.stop_loss_enabled,
        stop_loss_threshold=template.stop_loss_threshold,
        expected_sharpe=template.expected_sharpe,
        expected_annual_return=template.expected_annual_return,
        expected_max_drawdown=template.expected_max_drawdown,
        tags=template.tags,
        suitable_for=template.suitable_for,
        not_suitable_for=template.not_suitable_for,
    )


@router.post(
    "/strategies/from-template",
    response_model=StrategyResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_strategy_from_template(
    request: CreateFromTemplateRequest,
    service: BacktestService = Depends(get_backtest_service),
) -> StrategyResponse:
    """Create a new strategy from a template.

    Args:
        request: Template ID and optional customizations
        service: Backtest service

    Returns:
        Created strategy

    Raises:
        HTTPException: If template not found
    """
    from iqfmp.api.backtest.templates import get_template_by_id

    template = get_template_by_id(request.template_id)
    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template {request.template_id} not found",
        )

    # Use template values with optional overrides
    name = request.name or template.name
    description = request.description or template.description

    # Apply customizations if provided
    factors = template.factors
    weighting_method = template.weighting_method
    rebalance_frequency = template.rebalance_frequency
    max_positions = template.max_positions
    long_only = template.long_only

    if request.customizations:
        factors = request.customizations.get("factors", factors)
        weighting_method = request.customizations.get(
            "weighting_method", weighting_method
        )
        rebalance_frequency = request.customizations.get(
            "rebalance_frequency", rebalance_frequency
        )
        max_positions = request.customizations.get("max_positions", max_positions)
        long_only = request.customizations.get("long_only", long_only)

    return await service.create_strategy(
        name=name,
        description=description,
        factor_ids=factors,
        weighting_method=weighting_method,
        rebalance_frequency=rebalance_frequency,
        universe="all",
        custom_universe=[],
        long_only=long_only,
        max_positions=max_positions,
    )
