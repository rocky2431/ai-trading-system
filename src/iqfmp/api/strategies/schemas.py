"""Strategy API schemas."""

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# Type aliases for strategy configuration
CategoryType = Literal["momentum", "mean_reversion", "multi_factor", "crypto", "custom"]
RiskLevelType = Literal["conservative", "moderate", "aggressive"]
WeightingMethodType = Literal["equal", "ic_weighted", "vol_inverse", "custom"]
RebalanceFrequencyType = Literal["daily", "weekly", "monthly", "quarterly"]


class StrategyCreateRequest(BaseModel):
    """Request to create a strategy."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    factor_ids: list[str] = Field(default_factory=list)
    factor_weights: dict[str, float] | None = None
    code: str = Field(..., min_length=1)
    config: dict[str, Any] | None = None


class StrategyUpdateRequest(BaseModel):
    """Request to update a strategy."""

    name: str | None = Field(None, max_length=100)
    description: str | None = None
    factor_ids: list[str] | None = None
    factor_weights: dict[str, float] | None = None
    code: str | None = None
    config: dict[str, Any] | None = None
    status: str | None = None


class StrategyResponse(BaseModel):
    """Strategy response model."""

    id: str
    name: str
    description: str | None = None
    factor_ids: list[str]
    factor_weights: dict[str, float] | None = None
    code: str
    config: dict[str, Any] | None = None
    status: str
    created_at: datetime | None = None
    updated_at: datetime | None = None


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
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    profit_factor: float | None = None
    trade_count: int | None = None
    created_at: datetime | None = None


class BacktestListResponse(BaseModel):
    """Response for listing backtest results."""

    results: list[BacktestResultResponse]
    total: int


# Strategy Template Schemas


class StrategyTemplateResponse(BaseModel):
    """Strategy template response model."""

    id: str = Field(..., min_length=1, max_length=50)
    name: str = Field(..., min_length=1, max_length=100)
    description: str
    category: CategoryType
    risk_level: RiskLevelType

    # Strategy configuration
    factors: list[str] = Field(..., min_length=1)
    factor_descriptions: dict[str, str]
    weighting_method: WeightingMethodType
    rebalance_frequency: RebalanceFrequencyType
    max_positions: int = Field(..., ge=1, le=500)
    long_only: bool

    # Risk parameters (using Decimal for financial precision)
    max_drawdown: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    position_size_limit: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    stop_loss_enabled: bool
    stop_loss_threshold: Decimal | None = Field(
        default=None, ge=Decimal("0"), le=Decimal("1")
    )

    # Expected performance
    expected_sharpe: Decimal = Field(..., ge=Decimal("-5"), le=Decimal("10"))
    expected_annual_return: Decimal = Field(..., ge=Decimal("-1"), le=Decimal("10"))
    expected_max_drawdown: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))

    # Metadata
    tags: list[str]
    suitable_for: list[str]
    not_suitable_for: list[str]

    @model_validator(mode="after")
    def validate_factor_consistency(self) -> "StrategyTemplateResponse":
        """Validate that factor_descriptions keys match factors list."""
        if set(self.factor_descriptions.keys()) != set(self.factors):
            missing = set(self.factors) - set(self.factor_descriptions.keys())
            extra = set(self.factor_descriptions.keys()) - set(self.factors)
            msg = []
            if missing:
                msg.append(f"Missing descriptions for: {missing}")
            if extra:
                msg.append(f"Extra descriptions for: {extra}")
            raise ValueError("; ".join(msg))
        return self


class StrategyTemplateListResponse(BaseModel):
    """Response for listing strategy templates."""

    templates: list[StrategyTemplateResponse]
    total: int


class CreateFromTemplateRequest(BaseModel):
    """Request to create strategy from template."""

    name: str | None = None  # Override template name
    description: str | None = None  # Override description
    factor_weights: dict[str, float] | None = None  # Custom weights
