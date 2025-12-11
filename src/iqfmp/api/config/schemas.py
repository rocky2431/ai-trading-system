"""Config API schemas for IQFMP."""

from typing import Optional
from pydantic import BaseModel, Field


# ============== Config Status ==============

class FeaturesStatus(BaseModel):
    """Feature availability status."""
    factor_generation: bool = False
    factor_evaluation: bool = False
    strategy_assembly: bool = False
    backtest_optimization: bool = False
    live_trading: bool = False


class ConfigStatusResponse(BaseModel):
    """Configuration status response."""
    llm_configured: bool = False
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    exchange_configured: bool = False
    exchange_id: Optional[str] = None
    qlib_available: bool = False
    timescaledb_connected: bool = False
    redis_connected: bool = False
    features: FeaturesStatus = Field(default_factory=FeaturesStatus)


# ============== LLM Models ==============

class ModelInfo(BaseModel):
    """Model information."""
    id: str
    name: str
    provider: str
    context_length: Optional[int] = None
    use_case: Optional[str] = None  # factor_generation, strategy_design, etc.


class EmbeddingModelInfo(BaseModel):
    """Embedding model information."""
    id: str
    name: str
    dimensions: int
    context_length: Optional[int] = None  # Max input tokens


class AvailableModelsResponse(BaseModel):
    """Available models response."""
    models: dict[str, list[ModelInfo]] = Field(default_factory=dict)
    embedding_models: dict[str, list[EmbeddingModelInfo]] = Field(default_factory=dict)


# ============== API Keys ==============

class SavedAPIKeysResponse(BaseModel):
    """Saved API keys (masked)."""
    api_key: Optional[str] = None  # Masked: sk-or-v1-****
    model: Optional[str] = None
    embedding_model: Optional[str] = None
    exchange_id: Optional[str] = None
    exchange_api_key: Optional[str] = None  # Masked


class SetAPIKeysRequest(BaseModel):
    """Set API keys request."""
    provider: Optional[str] = "openrouter"
    api_key: Optional[str] = None
    model: Optional[str] = None
    embedding_model: Optional[str] = None
    exchange_id: Optional[str] = None
    exchange_api_key: Optional[str] = None
    exchange_secret: Optional[str] = None


class SetAPIKeysResponse(BaseModel):
    """Set API keys response."""
    success: bool
    message: str


# ============== LLM Test ==============

class TestLLMResponse(BaseModel):
    """Test LLM response."""
    success: bool
    model: Optional[str] = None
    response_time_ms: Optional[float] = None
    message: str


# ============== Exchange Test ==============

class TestExchangeResponse(BaseModel):
    """Test exchange response."""
    success: bool
    exchange_id: Optional[str] = None
    latest_btc_price: Optional[float] = None
    message: str


# ============== Agent Configuration ==============

class AgentModelConfig(BaseModel):
    """Agent-specific model configuration."""
    agent_id: str
    agent_name: str
    description: str
    model_id: str
    enabled: bool = True


class AgentConfigResponse(BaseModel):
    """Agent configuration response."""
    agents: list[AgentModelConfig] = Field(default_factory=list)


class SetAgentConfigRequest(BaseModel):
    """Set agent configuration request."""
    agent_id: str
    model_id: Optional[str] = None
    enabled: Optional[bool] = None


# ============== Data Config (Qlib/TimescaleDB) ==============

class FrequencyOption(BaseModel):
    """Frequency option."""
    id: str
    name: str
    description: str


class DataConfigResponse(BaseModel):
    """Data configuration response."""
    data_frequency: str = "1h"
    data_source: str = "timescaledb"  # timescaledb / ccxt / file
    symbols: list[str] = Field(default_factory=list)
    qlib_data_path: Optional[str] = None
    frequency_options: list[FrequencyOption] = Field(default_factory=list)


class SetDataConfigRequest(BaseModel):
    """Set data configuration request."""
    data_frequency: Optional[str] = None
    data_source: Optional[str] = None
    symbols: Optional[list[str]] = None


# ============== Factor Mining Config ==============

class FactorFamilyOption(BaseModel):
    """Factor family option."""
    id: str
    name: str
    description: str
    enabled: bool = True


class EvaluationConfig(BaseModel):
    """Factor evaluation configuration."""
    # Thresholds
    min_ic: float = 0.02
    min_ir: float = 0.5
    min_sharpe: float = 1.0
    max_turnover: float = 0.5
    # Cross-validation
    cv_folds: int = 5
    train_ratio: float = 0.6
    valid_ratio: float = 0.2
    test_ratio: float = 0.2
    # Anti-overfitting
    use_dynamic_threshold: bool = True
    deflation_rate: float = 0.1


class FactorMiningConfigResponse(BaseModel):
    """Factor mining configuration response."""
    factor_families: list[FactorFamilyOption] = Field(default_factory=list)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    max_concurrent_generation: int = 10
    code_execution_timeout: int = 60


class SetFactorMiningConfigRequest(BaseModel):
    """Set factor mining configuration request."""
    factor_families: Optional[list[FactorFamilyOption]] = None
    evaluation: Optional[EvaluationConfig] = None
    max_concurrent_generation: Optional[int] = None
    code_execution_timeout: Optional[int] = None


# ============== Risk Control Config ==============

class RiskControlConfig(BaseModel):
    """Risk control configuration."""
    max_single_loss_pct: float = 0.02      # 单笔最大亏损 2%
    max_daily_loss_pct: float = 0.05       # 日最大亏损 5%
    max_position_pct: float = 0.10         # 单策略最大仓位 10%
    max_total_position_pct: float = 0.50   # 总仓位上限 50%
    emergency_close_threshold: float = 0.08  # 紧急平仓阈值 8%


class RiskControlConfigResponse(BaseModel):
    """Risk control configuration response."""
    config: RiskControlConfig = Field(default_factory=RiskControlConfig)
    is_live_trading_enabled: bool = False


class SetRiskControlConfigRequest(BaseModel):
    """Set risk control configuration request."""
    max_single_loss_pct: Optional[float] = None
    max_daily_loss_pct: Optional[float] = None
    max_position_pct: Optional[float] = None
    max_total_position_pct: Optional[float] = None
    emergency_close_threshold: Optional[float] = None
