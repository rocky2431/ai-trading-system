"""Strategy API schemas."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class StrategyCreateRequest(BaseModel):
    """Request to create a strategy."""

    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    factor_ids: list[str] = Field(default_factory=list)
    factor_weights: Optional[dict[str, float]] = None
    code: str = Field(..., min_length=1)
    config: Optional[dict] = None


class StrategyUpdateRequest(BaseModel):
    """Request to update a strategy."""

    name: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None
    factor_ids: Optional[list[str]] = None
    factor_weights: Optional[dict[str, float]] = None
    code: Optional[str] = None
    config: Optional[dict] = None
    status: Optional[str] = None


class StrategyResponse(BaseModel):
    """Strategy response model."""

    id: str
    name: str
    description: Optional[str] = None
    factor_ids: list[str]
    factor_weights: Optional[dict[str, float]] = None
    code: str
    config: Optional[dict] = None
    status: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class StrategyListResponse(BaseModel):
    """Response for listing strategies."""

    strategies: list[StrategyResponse]
    total: int
    page: int
    page_size: int


class BacktestRequest(BaseModel):
    """Request to run a backtest."""

    start_date: datetime
    end_date: datetime
    initial_capital: float = Field(default=100000.0, gt=0)
    commission: float = Field(default=0.001, ge=0)


class BacktestResultResponse(BaseModel):
    """Backtest result response model."""

    id: str
    strategy_id: str
    start_date: datetime
    end_date: datetime
    total_return: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    trade_count: Optional[int] = None
    created_at: Optional[datetime] = None


class BacktestListResponse(BaseModel):
    """Response for listing backtest results."""

    results: list[BacktestResultResponse]
    total: int
