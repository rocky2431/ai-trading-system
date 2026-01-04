"""Strategy API router."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from iqfmp.api.strategies.schemas import (
    BacktestListResponse,
    BacktestRequest,
    BacktestResultResponse,
    CreateFromTemplateRequest,
    StrategyCreateRequest,
    StrategyListResponse,
    StrategyResponse,
    StrategyTemplateListResponse,
    StrategyTemplateResponse,
    StrategyUpdateRequest,
)
from iqfmp.api.strategies.service import StrategyService
from iqfmp.api.strategies.templates import (
    get_all_templates,
    get_template_by_id,
    get_templates_by_category,
)
from iqfmp.db.database import get_db, get_redis

logger = logging.getLogger(__name__)

router = APIRouter(tags=["strategies"])


async def get_strategy_service(
    session: AsyncSession = Depends(get_db),
    redis_client: Any = Depends(get_redis),
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
    status: str | None = Query(default=None),
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


# Strategy Template Endpoints (must be before /{strategy_id} to avoid route conflicts)


@router.get("/templates", response_model=StrategyTemplateListResponse)
async def list_templates(
    category: str | None = Query(default=None),
) -> StrategyTemplateListResponse:
    """List all strategy templates, optionally filtered by category."""
    templates = get_templates_by_category(category) if category else get_all_templates()

    return StrategyTemplateListResponse(
        templates=[StrategyTemplateResponse(**t) for t in templates],
        total=len(templates),
    )


@router.get("/templates/{template_id}", response_model=StrategyTemplateResponse)
async def get_template(template_id: str) -> StrategyTemplateResponse:
    """Get a specific strategy template by ID."""
    template = get_template_by_id(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return StrategyTemplateResponse(**template)


@router.post(
    "/from-template/{template_id}",
    response_model=StrategyResponse,
    status_code=201,
    responses={404: {"description": "Template not found"}},
)
async def create_from_template(
    template_id: str,
    request: CreateFromTemplateRequest,
    service: StrategyService = Depends(get_strategy_service),
) -> StrategyResponse:
    """Create a new strategy from a template.

    Raises:
        HTTPException: 404 if template_id does not exist.
        HTTPException: 422 if template has no factors defined.
    """
    template = get_template_by_id(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    # Validate template has required fields and factors
    factors = template.get("factors", [])
    if not factors:
        logger.error("Template %s has no factors defined", template_id)
        raise HTTPException(
            status_code=422,
            detail=f"Template '{template_id}' has no factors defined",
        )

    # Generate strategy code from template
    code = _generate_strategy_code(template)

    # Use template values with optional overrides
    name = request.name or f"{template.get('name', template_id)} Strategy"
    description = request.description or template.get("description", "")

    # Build factor weights with division-by-zero guard
    factor_weights = request.factor_weights
    if not factor_weights:
        num_factors = len(factors)
        if template.get("weighting_method") == "equal":
            factor_weights = dict.fromkeys(factors, 1.0 / num_factors)
        else:
            # For non-equal methods, default to equal weights as placeholder
            factor_weights = dict.fromkeys(factors, 1.0 / num_factors)

    # Build config from template with safe access
    config = {
        "template_id": template_id,
        "weighting_method": template.get("weighting_method", "equal"),
        "rebalance_frequency": template.get("rebalance_frequency", "weekly"),
        "max_positions": template.get("max_positions", 20),
        "long_only": template.get("long_only", True),
        "max_drawdown": template.get("max_drawdown", 0.2),
        "position_size_limit": template.get("position_size_limit", 0.1),
        "stop_loss_enabled": template.get("stop_loss_enabled", False),
        "stop_loss_threshold": template.get("stop_loss_threshold"),
        "risk_level": template.get("risk_level", "moderate"),
        "category": template.get("category", "custom"),
    }

    strategy = await service.create_strategy(
        name=name,
        code=code,
        description=description,
        factor_ids=factors,
        factor_weights=factor_weights,
        config=config,
    )
    return StrategyResponse(**strategy)


def _generate_strategy_code(template: dict[str, Any]) -> str:
    """Generate strategy code from template configuration.

    Args:
        template: Strategy template dictionary containing configuration.

    Returns:
        Generated Python code string for the strategy class.
    """
    factors = template.get("factors", [])
    factors_str = ", ".join(f'"{f}"' for f in factors)
    template_id = template.get("id", "custom")
    class_name = template_id.title().replace("_", "")

    return f'''"""
Strategy: {template.get("name", "Custom Strategy")}
Category: {template.get("category", "custom")}
Risk Level: {template.get("risk_level", "moderate")}
Description: {template.get("description", "")}
"""

from iqfmp.strategies.base import BaseStrategy

class {class_name}Strategy(BaseStrategy):
    """Auto-generated strategy from template: {template_id}"""

    FACTORS = [{factors_str}]
    WEIGHTING_METHOD = "{template.get("weighting_method", "equal")}"
    REBALANCE_FREQUENCY = "{template.get("rebalance_frequency", "weekly")}"
    MAX_POSITIONS = {template.get("max_positions", 20)}
    LONG_ONLY = {template.get("long_only", True)}

    def generate_signals(self, factor_data):
        """Generate trading signals from factor data."""
        combined = self.combine_factors(factor_data, self.WEIGHTING_METHOD)
        return self.rank_and_select(combined, self.MAX_POSITIONS, self.LONG_ONLY)

    def combine_factors(self, factor_data, method):
        """Combine multiple factors using specified method.

        Note: ic_weighted and vol_inverse methods currently use equal weighting
        as a placeholder. Full implementation requires historical IC/volatility data.
        """
        if not factor_data:
            raise ValueError("factor_data cannot be empty")
        if method == "equal":
            return sum(factor_data.values()) / len(factor_data)
        elif method == "ic_weighted":
            # Placeholder: IC-weighted requires historical IC scores
            return sum(factor_data.values()) / len(factor_data)
        elif method == "vol_inverse":
            # Placeholder: Vol-inverse requires volatility estimates
            return sum(factor_data.values()) / len(factor_data)
        return sum(factor_data.values()) / len(factor_data)

    def rank_and_select(self, scores, max_positions, long_only):
        """Rank assets and select top positions."""
        if len(scores) == 0:
            return scores
        ranked = scores.rank(ascending=False, pct=True)
        long_threshold = max_positions / len(scores)
        signals = (ranked <= long_threshold).astype(float)
        if not long_only:
            short_threshold = 1 - long_threshold
            signals = signals - (ranked >= short_threshold).astype(float)
        return signals
'''


@router.get(
    "/{strategy_id}",
    response_model=StrategyResponse,
    responses={404: {"description": "Strategy not found"}},
)
async def get_strategy(
    strategy_id: str,
    service: StrategyService = Depends(get_strategy_service),
) -> StrategyResponse:
    """Get a strategy by ID.

    Raises:
        HTTPException: 404 if strategy_id does not exist.
    """
    strategy = await service.get_strategy(strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return StrategyResponse(**strategy)


@router.patch(
    "/{strategy_id}",
    response_model=StrategyResponse,
    responses={404: {"description": "Strategy not found"}},
)
async def update_strategy(
    strategy_id: str,
    request: StrategyUpdateRequest,
    service: StrategyService = Depends(get_strategy_service),
) -> StrategyResponse:
    """Update a strategy.

    Raises:
        HTTPException: 404 if strategy_id does not exist.
    """
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


@router.delete(
    "/{strategy_id}",
    status_code=204,
    responses={404: {"description": "Strategy not found"}},
)
async def delete_strategy(
    strategy_id: str,
    service: StrategyService = Depends(get_strategy_service),
) -> None:
    """Delete a strategy.

    Raises:
        HTTPException: 404 if strategy_id does not exist.
    """
    deleted = await service.delete_strategy(strategy_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Strategy not found")


@router.post(
    "/{strategy_id}/backtest",
    response_model=BacktestResultResponse,
    status_code=201,
    responses={404: {"description": "Strategy not found"}},
)
async def run_backtest(
    strategy_id: str,
    request: BacktestRequest,
    service: StrategyService = Depends(get_strategy_service),
) -> BacktestResultResponse:
    """Run a backtest for a strategy.

    Raises:
        HTTPException: 404 if strategy_id does not exist.
    """
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
