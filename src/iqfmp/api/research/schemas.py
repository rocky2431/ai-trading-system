"""Research API schemas."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class WalkForwardResultResponse(BaseModel):
    """Walk-forward validation result for overfitting detection."""

    avg_oos_ic: Optional[float] = None  # Average out-of-sample IC
    ic_degradation: Optional[float] = None  # OOS degradation ratio
    ic_consistency: Optional[float] = None  # IC stability score (0-1)
    passes_robustness: Optional[bool] = None  # Passes robustness test


class TrialResponse(BaseModel):
    """Trial record response schema."""

    trial_id: str
    factor_name: str
    factor_family: str
    sharpe_ratio: float
    ic_mean: Optional[float] = None
    ir: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    created_at: datetime
    metadata: dict = Field(default_factory=dict)
    # Enhanced fields for P1 data integrity
    duration_ms: Optional[int] = None  # Evaluation duration in milliseconds
    walk_forward: Optional[WalkForwardResultResponse] = None  # Walk-forward validation


class LedgerListResponse(BaseModel):
    """Ledger list response schema."""

    trials: list[TrialResponse]
    total: int
    page: int
    page_size: int


class StatisticsResponse(BaseModel):
    """Statistics response schema."""

    total_trials: int
    mean_sharpe: float = 0.0
    std_sharpe: float = 0.0
    max_sharpe: float = 0.0
    min_sharpe: float = 0.0
    median_sharpe: float = 0.0


class StatsResponse(BaseModel):
    """Overall stats response schema."""

    overall: StatisticsResponse
    by_family: Optional[dict[str, StatisticsResponse]] = None


class ThresholdConfigResponse(BaseModel):
    """Threshold configuration response schema."""

    base_sharpe_threshold: float
    confidence_level: float
    min_trials_for_adjustment: int


class ThresholdHistoryItem(BaseModel):
    """Threshold history item schema."""

    n_trials: int
    threshold: float


class ThresholdResponse(BaseModel):
    """Threshold response schema."""

    current_threshold: float
    n_trials: int
    config: ThresholdConfigResponse
    threshold_history: list[ThresholdHistoryItem] = Field(default_factory=list)
