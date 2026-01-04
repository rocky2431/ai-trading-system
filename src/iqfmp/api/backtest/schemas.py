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

class OptimizationMetric(BaseModel):
    """Optimization metric configuration."""

    name: str = Field(default="sharpe")  # sharpe, calmar, total_return, sortino, ic
    direction: str = Field(default="maximize")  # maximize or minimize


class OptimizationConfig(BaseModel):
    """Strategy optimization configuration (Optuna-based)."""

    # Basic settings
    n_trials: int = Field(default=100, ge=1, le=10000)
    n_jobs: int = Field(default=1, ge=1, le=32)
    timeout: Optional[int] = Field(default=None, ge=60)  # seconds

    # Metrics to optimize
    metrics: list[OptimizationMetric] = Field(
        default_factory=lambda: [OptimizationMetric(name="sharpe")]
    )

    # Sampler configuration
    sampler: str = Field(default="tpe")  # tpe, cmaes, random, grid
    sampler_kwargs: dict = Field(default_factory=dict)

    # Pruner configuration
    pruner: str = Field(default="median")  # median, hyperband, percentile, none
    pruner_kwargs: dict = Field(default_factory=dict)

    # Search space (optional - can use defaults)
    custom_search_spaces: list[dict] = Field(default_factory=list)

    # Cross-validation settings
    cross_validation_folds: int = Field(default=5, ge=2, le=10)
    walk_forward_enabled: bool = Field(default=False)

    # Detection settings
    lookahead_check: bool = Field(default=True)
    lookahead_mode: str = Field(default="lenient")  # strict, lenient


class OptimizationRequest(BaseModel):
    """Request to run strategy optimization."""

    strategy_id: str
    backtest_config: BacktestConfig
    optimization_config: OptimizationConfig
    name: Optional[str] = None
    description: str = Field(default="")


class OptimizationTrialResult(BaseModel):
    """Single optimization trial result."""

    trial_id: int
    params: dict
    metrics: BacktestMetrics
    duration_seconds: float
    rank: int
    pruned: bool = False


class OptimizationProgressUpdate(BaseModel):
    """Real-time optimization progress update."""

    optimization_id: str
    current_trial: int
    total_trials: int
    best_value: Optional[float] = None
    best_params: Optional[dict] = None
    elapsed_seconds: float
    estimated_remaining_seconds: Optional[float] = None


class OptimizationResponse(BaseModel):
    """Optimization response."""

    id: str
    strategy_id: str
    name: str
    description: str
    status: str  # pending, running, completed, failed, cancelled
    progress: float
    current_trial: int
    total_trials: int
    best_trial_id: Optional[int] = None
    best_params: Optional[dict] = None
    best_metrics: Optional[BacktestMetrics] = None
    top_trials: list[OptimizationTrialResult] = Field(default_factory=list)
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class OptimizationDetailResponse(BaseModel):
    """Detailed optimization response with all trials."""

    optimization: OptimizationResponse
    all_trials: list[OptimizationTrialResult]
    param_importance: dict[str, float]  # parameter name -> importance score
    convergence_history: list[dict]  # trial_id, value pairs


class OptimizationListResponse(BaseModel):
    """Optimization list response."""

    optimizations: list[OptimizationResponse]
    total: int
    page: int
    page_size: int


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


# ============== Strategy Template Schemas (P1-2) ==============


class StrategyTemplateResponse(BaseModel):
    """Strategy template response."""

    id: str
    name: str
    description: str
    category: str  # momentum, mean_reversion, multi_factor, crypto
    risk_level: str  # conservative, moderate, aggressive

    # Strategy configuration
    factors: list[str]
    factor_descriptions: dict[str, str]
    weighting_method: str
    rebalance_frequency: str
    max_positions: int
    long_only: bool

    # Risk parameters
    max_drawdown: float
    position_size_limit: float
    stop_loss_enabled: bool
    stop_loss_threshold: Optional[float] = None

    # Expected performance
    expected_sharpe: float
    expected_annual_return: float
    expected_max_drawdown: float

    # Metadata
    tags: list[str]
    suitable_for: list[str]
    not_suitable_for: list[str]


class StrategyTemplateListResponse(BaseModel):
    """List of strategy templates."""

    templates: list[StrategyTemplateResponse]
    total: int


class CreateFromTemplateRequest(BaseModel):
    """Request to create a strategy from a template."""

    template_id: str
    name: Optional[str] = None  # Override template name
    description: Optional[str] = None  # Override description
    customizations: Optional[dict] = None  # Override specific fields
