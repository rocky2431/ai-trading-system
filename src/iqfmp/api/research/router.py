"""Research API router."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query

from iqfmp.api.research.schemas import (
    LedgerListResponse,
    StatisticsResponse,
    StatsResponse,
    ThresholdConfigResponse,
    ThresholdHistoryItem,
    ThresholdResponse,
    TrialResponse,
)
from iqfmp.api.research.service import get_research_service

router = APIRouter(tags=["research"])
metrics_router = APIRouter(tags=["metrics"])


@router.get("/ledger", response_model=LedgerListResponse)
async def list_ledger(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=10, ge=1, le=100, description="Items per page"),
    family: Optional[str] = Query(default=None, description="Filter by factor family"),
    min_sharpe: Optional[float] = Query(
        default=None, description="Filter by minimum Sharpe ratio"
    ),
    start_date: Optional[datetime] = Query(
        default=None, description="Filter by start date"
    ),
    end_date: Optional[datetime] = Query(default=None, description="Filter by end date"),
) -> LedgerListResponse:
    """List research ledger trials with pagination and filtering.

    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page (max 100)
        family: Filter by factor family
        min_sharpe: Filter by minimum Sharpe ratio
        start_date: Filter by start date
        end_date: Filter by end date

    Returns:
        Paginated list of trial records
    """
    service = get_research_service()
    trials, total = service.list_trials(
        page=page,
        page_size=page_size,
        family=family if family else None,
        min_sharpe=min_sharpe,
        start_date=start_date,
        end_date=end_date,
    )

    return LedgerListResponse(
        trials=[
            TrialResponse(
                trial_id=t.trial_id,
                factor_name=t.factor_name,
                factor_family=t.factor_family,
                sharpe_ratio=t.sharpe_ratio,
                ic_mean=t.ic_mean,
                ir=t.ir,
                max_drawdown=t.max_drawdown,
                win_rate=t.win_rate,
                created_at=t.created_at,
                metadata=t.metadata,
            )
            for t in trials
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    group_by_family: bool = Query(
        default=False, description="Group statistics by factor family"
    ),
) -> StatsResponse:
    """Get research ledger statistics.

    Args:
        group_by_family: Whether to include family-grouped statistics

    Returns:
        Overall and optionally family-grouped statistics
    """
    service = get_research_service()
    overall_stats = service.get_statistics()

    overall = StatisticsResponse(
        total_trials=overall_stats.total_trials,
        mean_sharpe=overall_stats.mean_sharpe,
        std_sharpe=overall_stats.std_sharpe,
        max_sharpe=overall_stats.max_sharpe,
        min_sharpe=overall_stats.min_sharpe,
        median_sharpe=overall_stats.median_sharpe,
    )

    by_family = None
    if group_by_family:
        family_stats = service.get_statistics_by_family()
        by_family = {
            family: StatisticsResponse(
                total_trials=stats.total_trials,
                mean_sharpe=stats.mean_sharpe,
                std_sharpe=stats.std_sharpe,
                max_sharpe=stats.max_sharpe,
                min_sharpe=stats.min_sharpe,
                median_sharpe=stats.median_sharpe,
            )
            for family, stats in family_stats.items()
        }

    return StatsResponse(overall=overall, by_family=by_family)


@router.get("/thresholds", response_model=ThresholdResponse)
async def get_thresholds() -> ThresholdResponse:
    """Get current dynamic threshold information.

    Returns:
        Current threshold, configuration, and history
    """
    service = get_research_service()
    info = service.get_threshold_info()

    return ThresholdResponse(
        current_threshold=info["current_threshold"],
        n_trials=info["n_trials"],
        config=ThresholdConfigResponse(
            base_sharpe_threshold=info["config"]["base_sharpe_threshold"],
            confidence_level=info["config"]["confidence_level"],
            min_trials_for_adjustment=info["config"]["min_trials_for_adjustment"],
        ),
        threshold_history=[
            ThresholdHistoryItem(n_trials=h["n_trials"], threshold=h["threshold"])
            for h in info["threshold_history"]
        ],
    )


@metrics_router.get("/thresholds", response_model=ThresholdResponse)
async def get_metrics_thresholds() -> ThresholdResponse:
    """Get current metrics thresholds information.

    This is an alias endpoint for /research/thresholds under /metrics prefix.

    Returns:
        Current threshold, configuration, and history
    """
    return await get_thresholds()
