"""Factor API router with async database support."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from iqfmp.api.factors.schemas import (
    FactorCompareRequest,
    FactorCompareResponse,
    FactorCreateRequest,
    FactorEvaluateRequest,
    FactorEvaluateResponse,
    FactorGenerateRequest,
    FactorLibraryStats,
    FactorListResponse,
    FactorResponse,
    FactorStatsResponse,
    FactorStatusUpdateRequest,
    MetricsResponse,
    MiningTaskCancelResponse,
    MiningTaskCreateRequest,
    MiningTaskCreateResponse,
    MiningTaskListResponse,
    MiningTaskStatus,
    StabilityResponse,
)
from iqfmp.api.factors.service import FactorNotFoundError, FactorEvaluationError, FactorService
from iqfmp.db.database import get_db, get_redis
from iqfmp.models.factor import Factor

router = APIRouter(tags=["factors"])


def _factor_to_response(factor: Factor) -> FactorResponse:
    """Convert Factor model to FactorResponse."""
    metrics = None
    if factor.metrics:
        metrics = MetricsResponse(
            ic_mean=factor.metrics.ic_mean,
            ic_std=factor.metrics.ic_std,
            ir=factor.metrics.ir,
            sharpe=factor.metrics.sharpe,
            max_drawdown=factor.metrics.max_drawdown,
            turnover=factor.metrics.turnover,
            ic_by_split=factor.metrics.ic_by_split,
            sharpe_by_split=factor.metrics.sharpe_by_split,
        )

    stability = None
    if factor.stability:
        stability = StabilityResponse(
            time_stability=factor.stability.time_stability,
            market_stability=factor.stability.market_stability,
            regime_stability=factor.stability.regime_stability,
        )

    return FactorResponse(
        id=factor.id,
        name=factor.name,
        family=factor.family,
        code=factor.code,
        code_hash=factor.code_hash,
        target_task=factor.target_task,
        status=factor.status.value if hasattr(factor.status, "value") else str(factor.status),
        metrics=metrics,
        stability=stability,
        cluster_id=factor.cluster_id,
        experiment_number=factor.experiment_number,
        created_at=factor.created_at,
    )


async def get_factor_service(
    session: AsyncSession = Depends(get_db),
    redis_client=Depends(get_redis),
) -> FactorService:
    """Dependency injection for FactorService."""
    return FactorService(session, redis_client)


@router.post("/generate", response_model=FactorResponse, status_code=status.HTTP_201_CREATED)
async def generate_factor(
    request: FactorGenerateRequest,
    factor_service: FactorService = Depends(get_factor_service),
) -> FactorResponse:
    """Generate a new factor from description using LLM.

    Args:
        request: Factor generation request with description
        factor_service: Factor service

    Returns:
        Generated factor
    """
    try:
        factor = await factor_service.generate_factor(
            description=request.description,
            family=request.family,
            target_task=request.target_task,
        )
        return _factor_to_response(factor)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("", response_model=FactorResponse, status_code=status.HTTP_201_CREATED)
async def create_factor(
    request: FactorCreateRequest,
    factor_service: FactorService = Depends(get_factor_service),
) -> FactorResponse:
    """Create a new factor with provided code.

    Args:
        request: Factor creation request
        factor_service: Factor service

    Returns:
        Created factor
    """
    try:
        factor = await factor_service.create_factor(
            name=request.name,
            family=request.family,
            code=request.code,
            target_task=request.target_task,
        )
        return _factor_to_response(factor)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("", response_model=FactorListResponse)
async def list_factors(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    family: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None, alias="status"),
    factor_service: FactorService = Depends(get_factor_service),
) -> FactorListResponse:
    """List factors with pagination and filtering.

    Args:
        page: Page number (1-indexed)
        page_size: Items per page
        family: Filter by family
        status_filter: Filter by status
        factor_service: Factor service

    Returns:
        List of factors with pagination info
    """
    factors, total = await factor_service.list_factors(
        page=page,
        page_size=page_size,
        family=family,
        status=status_filter,
    )
    return FactorListResponse(
        factors=[_factor_to_response(f) for f in factors],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/stats", response_model=FactorStatsResponse)
async def get_factor_stats(
    factor_service: FactorService = Depends(get_factor_service),
) -> FactorStatsResponse:
    """Get comprehensive factor statistics for monitoring dashboard.

    Args:
        factor_service: Factor service

    Returns:
        Factor statistics including totals, status counts, thresholds, and metrics
    """
    stats = await factor_service.get_stats()
    return FactorStatsResponse(
        # Basic fields
        total_factors=stats["total_factors"],
        by_status=stats["by_status"],
        total_trials=stats["total_trials"],
        current_threshold=stats["current_threshold"],
        # Extended fields for monitoring dashboard
        evaluated_count=stats["evaluated_count"],
        pass_rate=stats["pass_rate"],
        avg_ic=stats["avg_ic"],
        avg_sharpe=stats["avg_sharpe"],
        pending_count=stats["pending_count"],
    )


# ==================== Mining Task Endpoints ====================
# NOTE: These routes MUST be defined BEFORE /{factor_id} to avoid route conflicts


@router.post("/mining", response_model=MiningTaskCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_mining_task(
    request: MiningTaskCreateRequest,
    factor_service: FactorService = Depends(get_factor_service),
) -> MiningTaskCreateResponse:
    """Create and start a factor mining task.

    Args:
        request: Mining task creation request
        factor_service: Factor service

    Returns:
        Mining task creation response with task ID
    """
    try:
        task_id = await factor_service.create_mining_task(
            name=request.name,
            description=request.description,
            factor_families=request.factor_families,
            target_count=request.target_count,
            auto_evaluate=request.auto_evaluate,
            # Advanced configuration (optional)
            data_config=request.data_config.model_dump() if request.data_config else None,
            benchmark_config=request.benchmark_config.model_dump() if request.benchmark_config else None,
            ml_config=request.ml_config.model_dump() if request.ml_config else None,
            robustness_config=request.robustness_config.model_dump() if request.robustness_config else None,
        )
        return MiningTaskCreateResponse(
            success=True,
            message=f"Mining task '{request.name}' created successfully",
            task_id=task_id,
        )
    except Exception as e:
        return MiningTaskCreateResponse(
            success=False,
            message=str(e),
            task_id=None,
        )


@router.get("/mining", response_model=MiningTaskListResponse)
async def list_mining_tasks(
    status_filter: Optional[str] = Query(None, alias="status"),
    limit: int = Query(20, ge=1, le=100),
    factor_service: FactorService = Depends(get_factor_service),
) -> MiningTaskListResponse:
    """List mining tasks with optional status filter.

    Args:
        status_filter: Filter by task status
        limit: Maximum number of tasks to return
        factor_service: Factor service

    Returns:
        List of mining tasks
    """
    tasks = await factor_service.list_mining_tasks(status=status_filter, limit=limit)
    return MiningTaskListResponse(tasks=tasks, total=len(tasks))


@router.get("/mining/{task_id}", response_model=MiningTaskStatus)
async def get_mining_task(
    task_id: str,
    factor_service: FactorService = Depends(get_factor_service),
) -> MiningTaskStatus:
    """Get mining task status by ID.

    Args:
        task_id: Mining task ID
        factor_service: Factor service

    Returns:
        Mining task status

    Raises:
        HTTPException: If task not found
    """
    task = await factor_service.get_mining_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mining task {task_id} not found",
        )
    return task


@router.delete("/mining/{task_id}", response_model=MiningTaskCancelResponse)
async def cancel_mining_task(
    task_id: str,
    factor_service: FactorService = Depends(get_factor_service),
) -> MiningTaskCancelResponse:
    """Cancel a running mining task.

    Args:
        task_id: Mining task ID
        factor_service: Factor service

    Returns:
        Cancel response
    """
    success = await factor_service.cancel_mining_task(task_id)
    if success:
        return MiningTaskCancelResponse(
            success=True,
            message=f"Mining task {task_id} cancelled successfully",
        )
    return MiningTaskCancelResponse(
        success=False,
        message=f"Failed to cancel mining task {task_id} (not found or already completed)",
    )


# ==================== Factor Library Endpoints ====================


@router.get("/library/stats", response_model=FactorLibraryStats)
async def get_library_stats(
    factor_service: FactorService = Depends(get_factor_service),
) -> FactorLibraryStats:
    """Get factor library statistics.

    Args:
        factor_service: Factor service

    Returns:
        Comprehensive library statistics
    """
    return await factor_service.get_library_stats()


@router.post("/compare", response_model=FactorCompareResponse)
async def compare_factors(
    request: FactorCompareRequest,
    factor_service: FactorService = Depends(get_factor_service),
) -> FactorCompareResponse:
    """Compare multiple factors.

    Args:
        request: Factor comparison request with factor IDs
        factor_service: Factor service

    Returns:
        Factor comparison with correlation matrix and ranking

    Raises:
        HTTPException: If any factor not found
    """
    try:
        factors, correlation_matrix, ranking = await factor_service.compare_factors(
            factor_ids=request.factor_ids
        )
        return FactorCompareResponse(
            factors=[_factor_to_response(f) for f in factors],
            correlation_matrix=correlation_matrix,
            ranking=ranking,
        )
    except FactorNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


# ==================== Individual Factor Endpoints ====================
# NOTE: These routes with /{factor_id} MUST be defined AFTER static routes


@router.get("/{factor_id}", response_model=FactorResponse)
async def get_factor(
    factor_id: str,
    factor_service: FactorService = Depends(get_factor_service),
) -> FactorResponse:
    """Get factor by ID.

    Args:
        factor_id: Factor ID
        factor_service: Factor service

    Returns:
        Factor details

    Raises:
        HTTPException: If factor not found
    """
    factor = await factor_service.get_factor(factor_id)
    if not factor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Factor {factor_id} not found",
        )
    return _factor_to_response(factor)


@router.post("/{factor_id}/evaluate", response_model=FactorEvaluateResponse)
async def evaluate_factor(
    factor_id: str,
    request: FactorEvaluateRequest,
    factor_service: FactorService = Depends(get_factor_service),
) -> FactorEvaluateResponse:
    """Evaluate a factor and record in research ledger.

    Args:
        factor_id: Factor ID
        request: Evaluation request with splits
        factor_service: Factor service

    Returns:
        Evaluation results with metrics, stability, and threshold check

    Raises:
        HTTPException: If factor not found
    """
    try:
        metrics, stability, passed, exp_num = await factor_service.evaluate_factor(
            factor_id=factor_id,
            splits=request.splits,
            market_splits=request.market_splits,
            symbol=request.symbol,
            timeframe=request.timeframe,
        )

        return FactorEvaluateResponse(
            factor_id=factor_id,
            metrics=MetricsResponse(
                ic_mean=metrics.ic_mean,
                ic_std=metrics.ic_std,
                ir=metrics.ir,
                sharpe=metrics.sharpe,
                max_drawdown=metrics.max_drawdown,
                turnover=metrics.turnover,
                ic_by_split=metrics.ic_by_split,
                sharpe_by_split=metrics.sharpe_by_split,
            ),
            stability=StabilityResponse(
                time_stability=stability.time_stability,
                market_stability=stability.market_stability,
                regime_stability=stability.regime_stability,
            ),
            passed_threshold=passed,
            experiment_number=exp_num,
        )
    except FactorNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Factor {factor_id} not found",
        )
    except FactorEvaluationError as e:
        # H1 FIX: Properly report data loading/evaluation failures
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )


@router.put("/{factor_id}/status", response_model=FactorResponse)
async def update_factor_status(
    factor_id: str,
    request: FactorStatusUpdateRequest,
    factor_service: FactorService = Depends(get_factor_service),
) -> FactorResponse:
    """Update factor status.

    Args:
        factor_id: Factor ID
        request: Status update request
        factor_service: Factor service

    Returns:
        Updated factor

    Raises:
        HTTPException: If factor not found
    """
    try:
        factor = await factor_service.update_status(factor_id, request.status)
        return _factor_to_response(factor)
    except FactorNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Factor {factor_id} not found",
        )


@router.delete("/{factor_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_factor(
    factor_id: str,
    factor_service: FactorService = Depends(get_factor_service),
) -> None:
    """Delete a factor.

    Args:
        factor_id: Factor ID
        factor_service: Factor service

    Raises:
        HTTPException: If factor not found
    """
    deleted = await factor_service.delete_factor(factor_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Factor {factor_id} not found",
        )
