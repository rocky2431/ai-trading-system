"""Factor API router."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from iqfmp.api.factors.schemas import (
    FactorEvaluateRequest,
    FactorEvaluateResponse,
    FactorGenerateRequest,
    FactorListResponse,
    FactorResponse,
    FactorStatusUpdateRequest,
    MetricsResponse,
    StabilityResponse,
)
from iqfmp.api.factors.service import (
    FactorNotFoundError,
    FactorService,
    get_factor_service,
)
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


@router.post("/generate", response_model=FactorResponse, status_code=status.HTTP_201_CREATED)
async def generate_factor(
    request: FactorGenerateRequest,
    factor_service: FactorService = Depends(get_factor_service),
) -> FactorResponse:
    """Generate a new factor from description.

    Args:
        request: Factor generation request
        factor_service: Factor service

    Returns:
        Generated factor
    """
    factor = factor_service.generate_factor(
        description=request.description,
        family=request.family,
        target_task=request.target_task,
    )
    return _factor_to_response(factor)


@router.get("", response_model=FactorListResponse)
async def list_factors(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    family: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    factor_service: FactorService = Depends(get_factor_service),
) -> FactorListResponse:
    """List factors with pagination and filtering.

    Args:
        page: Page number (1-indexed)
        page_size: Items per page
        family: Filter by family
        status: Filter by status
        factor_service: Factor service

    Returns:
        List of factors
    """
    factors, total = factor_service.list_factors(
        page=page,
        page_size=page_size,
        family=family,
        status=status,
    )
    return FactorListResponse(
        factors=[_factor_to_response(f) for f in factors],
        total=total,
        page=page,
        page_size=page_size,
    )


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
    factor = factor_service.get_factor(factor_id)
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
    """Evaluate a factor.

    Args:
        factor_id: Factor ID
        request: Evaluation request
        factor_service: Factor service

    Returns:
        Evaluation results

    Raises:
        HTTPException: If factor not found
    """
    try:
        metrics, stability, passed, exp_num = factor_service.evaluate_factor(
            factor_id=factor_id,
            splits=request.splits,
            market_splits=request.market_splits,
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
        factor = factor_service.update_status(factor_id, request.status)
        return _factor_to_response(factor)
    except FactorNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Factor {factor_id} not found",
        )
