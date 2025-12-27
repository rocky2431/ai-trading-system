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
    """Agent-specific model configuration.

    Attributes:
        agent_id: Unique agent identifier (e.g., "factor_generation")
        agent_name: Display name (e.g., "Factor Generation")
        description: Agent description
        model_id: OpenRouter model ID (e.g., "deepseek/deepseek-coder-v3")
        enabled: Whether the agent is enabled
        temperature: Sampling temperature (0.0-2.0, lower = more deterministic)
        system_prompt: Custom system prompt override (None = use default)
    """
    agent_id: str
    agent_name: str
    description: str
    model_id: str
    enabled: bool = True
    temperature: float = 0.7
    system_prompt: Optional[str] = None  # Custom system prompt (None = use default)


class AgentConfigResponse(BaseModel):
    """Agent configuration response."""
    agents: list[AgentModelConfig] = Field(default_factory=list)


class SetAgentConfigRequest(BaseModel):
    """Set agent configuration request.

    Attributes:
        agent_id: Agent to configure
        model_id: OpenRouter model ID (optional)
        enabled: Enable/disable agent (optional)
        temperature: Sampling temperature (optional)
        system_prompt: Custom system prompt (optional, empty string = reset to default)
    """
    agent_id: str
    model_id: Optional[str] = None
    enabled: Optional[bool] = None
    temperature: Optional[float] = None
    system_prompt: Optional[str] = None  # Custom system prompt


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


# ============== P3: Sandbox Config ==============

class SandboxConfigResponse(BaseModel):
    """Sandbox execution configuration."""
    timeout_seconds: int = Field(60, description="Execution timeout in seconds")
    max_memory_mb: int = Field(512, description="Maximum memory in MB")
    max_cpu_seconds: int = Field(30, description="Maximum CPU time in seconds")
    use_subprocess: bool = Field(True, description="Use subprocess isolation")
    allowed_modules: list[str] = Field(
        default_factory=list,
        description="Whitelist of allowed Python modules",
    )


class SandboxConfigUpdate(BaseModel):
    """Update sandbox configuration."""
    timeout_seconds: Optional[int] = Field(None, ge=10, le=300)
    max_memory_mb: Optional[int] = Field(None, ge=128, le=4096)
    max_cpu_seconds: Optional[int] = Field(None, ge=5, le=120)
    use_subprocess: Optional[bool] = None
    allowed_modules: Optional[list[str]] = None


class ExecutionLogEntry(BaseModel):
    """Single execution log entry."""
    execution_id: str
    factor_name: Optional[str] = None
    status: str  # success, error, timeout, security_violation, resource_exceeded
    execution_time: float
    memory_used_mb: Optional[float] = None
    error_message: Optional[str] = None
    created_at: str


class ExecutionLogResponse(BaseModel):
    """Paginated execution log response."""
    items: list[ExecutionLogEntry]
    total: int
    page: int
    page_size: int
    has_next: bool


# ============== P3: Security Config ==============

class SecurityConfigResponse(BaseModel):
    """Security configuration settings."""
    research_ledger_strict: bool = Field(True, description="Require PostgreSQL for ResearchLedger")
    vector_strict_mode: bool = Field(True, description="Require Qdrant for vector storage")
    human_review_enabled: bool = Field(True, description="Enable human review gate for LLM code")
    ast_security_check: bool = Field(True, description="Enable AST security checking")
    sandbox_enabled: bool = Field(True, description="Enable sandboxed code execution")


class SecurityConfigUpdate(BaseModel):
    """Update security configuration."""
    research_ledger_strict: Optional[bool] = None
    vector_strict_mode: Optional[bool] = None
    human_review_enabled: Optional[bool] = None
    ast_security_check: Optional[bool] = None
    sandbox_enabled: Optional[bool] = None


# ============== P3: LLM Advanced Config ==============

class RateLimitConfigResponse(BaseModel):
    """Rate limiting configuration."""
    requests_per_minute: int = Field(60, ge=1, le=1000)
    tokens_per_minute: int = Field(100000, ge=1000, le=1000000)


class FallbackChainConfig(BaseModel):
    """Fallback model chain configuration."""
    models: list[str] = Field(default_factory=list, description="Ordered list of fallback models")
    max_retries: int = Field(3, ge=1, le=10)


class LLMAdvancedConfigResponse(BaseModel):
    """LLM provider advanced configuration."""
    default_model: str
    available_models: list[str]
    rate_limit: RateLimitConfigResponse
    fallback_chain: FallbackChainConfig
    cache_enabled: bool
    cache_ttl: int
    total_requests: int
    total_tokens: int
    total_cost: float
    cache_hit_rate: float


class LLMAdvancedConfigUpdate(BaseModel):
    """Update LLM advanced configuration."""
    rate_limit: Optional[RateLimitConfigResponse] = None
    fallback_chain: Optional[FallbackChainConfig] = None
    cache_enabled: Optional[bool] = None
    cache_ttl: Optional[int] = Field(None, ge=60, le=86400)


class LLMTraceEntry(BaseModel):
    """LLM API call trace entry."""
    trace_id: str
    execution_id: str
    conversation_id: Optional[str] = None
    agent: Optional[str] = None
    model: str
    prompt_id: Optional[str] = None
    prompt_version: Optional[str] = None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_estimate: float
    latency_ms: float
    cached: bool
    created_at: str


class LLMTraceResponse(BaseModel):
    """Paginated LLM trace response."""
    items: list[LLMTraceEntry]
    total: int
    page: int
    page_size: int
    has_next: bool


class LLMCostSummary(BaseModel):
    """LLM cost summary by agent/model."""
    total_cost: float
    total_tokens: int
    total_requests: int
    cache_hit_rate: float
    by_agent: dict[str, float]
    by_model: dict[str, float]
    hourly_costs: list[float]


# ============== P3: Derivative Data Config ==============

class DerivativeDataConfigResponse(BaseModel):
    """Derivative data download configuration."""
    funding_rate_enabled: bool = True
    open_interest_enabled: bool = True
    liquidation_enabled: bool = True
    long_short_ratio_enabled: bool = True
    mark_price_enabled: bool = False
    taker_buy_sell_enabled: bool = False
    data_source: str = Field("timescale", description="timescale or qlib")
    exchanges: list[str] = Field(default_factory=lambda: ["binance"])


class DerivativeDataConfigUpdate(BaseModel):
    """Update derivative data configuration."""
    funding_rate_enabled: Optional[bool] = None
    open_interest_enabled: Optional[bool] = None
    liquidation_enabled: Optional[bool] = None
    long_short_ratio_enabled: Optional[bool] = None
    mark_price_enabled: Optional[bool] = None
    taker_buy_sell_enabled: Optional[bool] = None
    data_source: Optional[str] = None
    exchanges: Optional[list[str]] = None


# ============== P3: Checkpoint Config ==============

class CheckpointThreadInfo(BaseModel):
    """Thread information for checkpointing."""
    thread_id: str
    name: Optional[str] = None
    created_at: str
    last_updated: str
    checkpoint_count: int
    current_phase: Optional[str] = None


class CheckpointInfo(BaseModel):
    """Checkpoint information."""
    checkpoint_id: str
    thread_id: str
    phase: str
    created_at: str
    metadata: dict


class CheckpointListResponse(BaseModel):
    """List of checkpoints for a thread."""
    thread_id: str
    checkpoints: list[CheckpointInfo]
    total: int


class CheckpointStateResponse(BaseModel):
    """Full checkpoint state."""
    checkpoint_id: str
    thread_id: str
    phase: str
    hypothesis: Optional[str] = None
    factors: list[dict]
    evaluation_results: dict
    strategy: Optional[dict] = None
    backtest_results: Optional[dict] = None
    messages: list[dict]
    created_at: str


# ============== P3: Alpha Benchmark Config ==============

class BenchmarkConfigResponse(BaseModel):
    """Alpha benchmark configuration."""
    benchmark_type: str = Field("alpha158", description="alpha158 or alpha360")
    enabled: bool = True
    auto_run_on_evaluation: bool = True
    novelty_threshold: float = Field(0.3, description="Correlation threshold for novelty")
    min_improvement_pct: float = Field(5.0, description="Minimum improvement over benchmark")


class BenchmarkConfigUpdate(BaseModel):
    """Update benchmark configuration."""
    benchmark_type: Optional[str] = None
    enabled: Optional[bool] = None
    auto_run_on_evaluation: Optional[bool] = None
    novelty_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_improvement_pct: Optional[float] = Field(None, ge=0.0, le=100.0)


class BenchmarkResultEntry(BaseModel):
    """Single benchmark result entry."""
    result_id: str
    factor_name: str
    factor_ic: float
    factor_ir: float
    factor_sharpe: float
    benchmark_avg_ic: float
    benchmark_avg_ir: float
    ic_improvement: float
    ir_improvement: float
    rank: int
    total_factors: int
    is_novel: bool
    created_at: str


class BenchmarkResultsResponse(BaseModel):
    """Paginated benchmark results."""
    items: list[BenchmarkResultEntry]
    total: int
    page: int
    page_size: int
    has_next: bool
