"""Backtest API schemas."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ============== Strategy Schemas ==============

class StrategyCreateRequest(BaseModel):
    """Request to create a new strategy."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="")
    factor_ids: list[str] = Field(default_factory=list)
    weighting_method: str = Field(default="equal")  # equal, ic_weighted, optimization
    rebalance_frequency: str = Field(default="daily")  # daily, weekly, monthly
    universe: str = Field(default="all")  # all, top100, custom
    custom_universe: list[str] = Field(default_factory=list)
    long_only: bool = Field(default=False)
    max_positions: int = Field(default=20, ge=1, le=100)


class StrategyResponse(BaseModel):
    """Strategy response schema."""

    id: str
    name: str
    description: str
    factor_ids: list[str]
    weighting_method: str
    rebalance_frequency: str
    universe: str
    custom_universe: list[str]
    long_only: bool
    max_positions: int
    status: str  # draft, active, archived
    created_at: datetime
    updated_at: datetime


class StrategyListResponse(BaseModel):
    """Strategy list response."""

    strategies: list[StrategyResponse]
    total: int
    page: int
    page_size: int


# ============== Backtest Schemas ==============

class BacktestConfig(BaseModel):
    """Backtest configuration."""

    start_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    symbols: list[str] = Field(default_factory=lambda: ["ETH/USDT"])
    timeframe: str = Field(default="1d")
    initial_capital: float = Field(default=1000000.0, gt=0)
    commission_rate: float = Field(default=0.001, ge=0, le=0.1)  # 0.1%
    slippage: float = Field(default=0.001, ge=0, le=0.1)  # 0.1%
    benchmark: str = Field(default="BTC")
    risk_free_rate: float = Field(default=0.02, ge=0, le=0.2)  # 2% annual


class BacktestCreateRequest(BaseModel):
    """Request to create a backtest."""

    strategy_id: str
    config: BacktestConfig
    name: Optional[str] = None
    description: str = Field(default="")


class BacktestMetrics(BaseModel):
    """Backtest performance metrics."""

    total_return: float  # Total return percentage
    annual_return: float  # Annualized return
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # Days
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    volatility: float
    beta: float
    alpha: float
    information_ratio: float
    trade_count: int
    avg_trade_return: float
    avg_holding_period: float  # Days


class BacktestTrade(BaseModel):
    """Individual trade record."""

    id: str
    symbol: str
    side: str  # long, short
    entry_date: str
    entry_price: float
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    quantity: float
    pnl: float
    pnl_pct: float
    holding_days: int


class BacktestEquityCurve(BaseModel):
    """Equity curve data point."""

    date: str
    equity: float
    drawdown: float
    benchmark_equity: float


class BacktestResponse(BaseModel):
    """Backtest response schema."""

    id: str
    strategy_id: str
    strategy_name: str
    name: str
    description: str
    config: BacktestConfig
    status: str  # pending, running, completed, failed
    progress: float
    metrics: Optional[BacktestMetrics] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class BacktestListResponse(BaseModel):
    """Backtest list response."""

    backtests: list[BacktestResponse]
    total: int
    page: int
    page_size: int


class BacktestDetailResponse(BaseModel):
    """Detailed backtest response with equity curve and trades."""

    backtest: BacktestResponse
    equity_curve: list[BacktestEquityCurve]
    trades: list[BacktestTrade]
    monthly_returns: dict[str, float]  # "2024-01": 0.05
    factor_contributions: dict[str, float]  # factor_id: contribution


# ============== Optimization Schemas ==============

class OptimizationConfig(BaseModel):
    """Strategy optimization configuration."""

    method: str = Field(default="grid")  # grid, random, bayesian
    objective: str = Field(default="sharpe")  # sharpe, return, calmar
    param_ranges: dict[str, dict] = Field(default_factory=dict)
    max_iterations: int = Field(default=100, ge=1, le=1000)
    cross_validation_folds: int = Field(default=5, ge=2, le=10)


class OptimizationRequest(BaseModel):
    """Request to run strategy optimization."""

    strategy_id: str
    backtest_config: BacktestConfig
    optimization_config: OptimizationConfig


class OptimizationResult(BaseModel):
    """Single optimization result."""

    params: dict
    metrics: BacktestMetrics
    rank: int


class OptimizationResponse(BaseModel):
    """Optimization response."""

    id: str
    strategy_id: str
    status: str  # pending, running, completed, failed
    progress: float
    best_params: Optional[dict] = None
    best_metrics: Optional[BacktestMetrics] = None
    all_results: list[OptimizationResult]
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


# ============== Common Response Schemas ==============

class BacktestStatsResponse(BaseModel):
    """Backtest statistics response."""

    total_strategies: int
    total_backtests: int
    running_backtests: int
    completed_today: int
    avg_sharpe: float
    best_strategy_id: Optional[str] = None
    best_sharpe: float = 0.0


class GenericResponse(BaseModel):
    """Generic success/failure response."""

    success: bool
    message: str
