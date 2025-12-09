"""Factor data models."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class FactorStatus(str, Enum):
    """Factor lifecycle status."""

    CANDIDATE = "candidate"
    REJECTED = "rejected"
    CORE = "core"
    REDUNDANT = "redundant"


class FactorMetrics(BaseModel):
    """Factor evaluation metrics."""

    ic_mean: float = Field(description="Mean Information Coefficient")
    ic_std: float = Field(description="IC Standard Deviation")
    ir: float = Field(description="Information Ratio (IC/IC_std)")
    sharpe: float = Field(description="Sharpe Ratio")
    max_drawdown: float = Field(description="Maximum Drawdown")
    turnover: float = Field(description="Factor Turnover")
    ic_by_split: dict[str, float] = Field(
        default_factory=dict, description="IC by data split"
    )
    sharpe_by_split: dict[str, float] = Field(
        default_factory=dict, description="Sharpe by data split"
    )


class StabilityReport(BaseModel):
    """Factor stability analysis report."""

    time_stability: dict[str, float] = Field(
        default_factory=dict, description="Time-based stability metrics"
    )
    market_stability: dict[str, float] = Field(
        default_factory=dict, description="Market-based stability metrics"
    )
    regime_stability: dict[str, float] = Field(
        default_factory=dict, description="Regime-based stability metrics"
    )


class Factor(BaseModel):
    """Factor entity model."""

    id: str = Field(description="Unique factor ID")
    name: str = Field(description="Factor name")
    family: list[str] = Field(description="Factor family tags")
    code: str = Field(description="Factor computation code")
    code_hash: str = Field(description="Code hash for deduplication")
    target_task: str = Field(description="Target prediction task")

    metrics: Optional[FactorMetrics] = Field(
        default=None, description="Evaluation metrics"
    )
    stability: Optional[StabilityReport] = Field(
        default=None, description="Stability analysis"
    )

    status: FactorStatus = Field(
        default=FactorStatus.CANDIDATE, description="Factor status"
    )
    cluster_id: Optional[str] = Field(
        default=None, description="Cluster ID for similar factors"
    )

    created_at: datetime = Field(default_factory=datetime.now)
    experiment_number: int = Field(
        default=0, description="Research ledger experiment number"
    )

    model_config = ConfigDict(use_enum_values=True)
