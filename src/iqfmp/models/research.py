"""Research experiment models."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from iqfmp.models.factor import FactorMetrics


class EvalConfig(BaseModel):
    """Factor evaluation configuration."""

    date_range: tuple[str, str] = Field(description="Evaluation date range")
    symbols: list[str] = Field(description="Symbols to evaluate")
    frequencies: list[str] = Field(
        default=["1h", "4h", "1d"], description="Time frequencies"
    )
    market_groups: list[str] = Field(
        default=["major", "mid", "small"], description="Market cap groups"
    )


class ResearchExperiment(BaseModel):
    """Research ledger experiment record."""

    id: str = Field(description="Experiment ID")
    timestamp: datetime = Field(default_factory=datetime.now)
    factor_id: str = Field(description="Associated factor ID")
    code_hash: str = Field(description="Factor code hash")
    prompt: str = Field(description="Generation prompt")
    config: EvalConfig = Field(description="Evaluation configuration")
    metrics: FactorMetrics = Field(description="Evaluation metrics")
    passed: bool = Field(description="Whether factor passed thresholds")
    rejection_reason: Optional[str] = Field(
        default=None, description="Reason for rejection if failed"
    )
