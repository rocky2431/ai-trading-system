"""Config API router for IQFMP."""

from fastapi import APIRouter, Depends

from iqfmp.api.config.schemas import (
    AgentConfigResponse,
    AvailableModelsResponse,
    ConfigStatusResponse,
    DataConfigResponse,
    FactorMiningConfigResponse,
    RiskControlConfigResponse,
    SavedAPIKeysResponse,
    SetAgentConfigRequest,
    SetAPIKeysRequest,
    SetAPIKeysResponse,
    SetDataConfigRequest,
    SetFactorMiningConfigRequest,
    SetRiskControlConfigRequest,
    TestExchangeResponse,
    TestLLMResponse,
    # P3: Sandbox/Security
    SandboxConfigResponse,
    SandboxConfigUpdate,
    ExecutionLogResponse,
    SecurityConfigResponse,
    SecurityConfigUpdate,
    # P3: LLM Advanced
    LLMAdvancedConfigResponse,
    LLMAdvancedConfigUpdate,
    LLMTraceResponse,
    LLMCostSummary,
    # P3: Derivative Data
    DerivativeDataConfigResponse,
    DerivativeDataConfigUpdate,
    # P3: Checkpoints
    CheckpointThreadInfo,
    CheckpointListResponse,
    CheckpointStateResponse,
    # P3: Alpha Benchmark
    BenchmarkConfigResponse,
    BenchmarkConfigUpdate,
    BenchmarkResultsResponse,
)
from iqfmp.api.config.service import ConfigService, get_config_service

router = APIRouter(tags=["config"])


# ============== Status ==============

@router.get("/status", response_model=ConfigStatusResponse)
async def get_status(
    service: ConfigService = Depends(get_config_service),
) -> ConfigStatusResponse:
    """Get current configuration status."""
    return service.get_status()


# ============== Models ==============

@router.get("/models", response_model=AvailableModelsResponse)
async def get_available_models(
    service: ConfigService = Depends(get_config_service),
) -> AvailableModelsResponse:
    """Get available LLM models."""
    return service.get_available_models()


# ============== API Keys ==============

@router.get("/api-keys", response_model=SavedAPIKeysResponse)
async def get_api_keys(
    service: ConfigService = Depends(get_config_service),
) -> SavedAPIKeysResponse:
    """Get saved API keys (masked)."""
    return service.get_saved_api_keys()


@router.post("/api-keys", response_model=SetAPIKeysResponse)
async def set_api_keys(
    request: SetAPIKeysRequest,
    service: ConfigService = Depends(get_config_service),
) -> SetAPIKeysResponse:
    """Set API keys."""
    return service.set_api_keys(
        provider=request.provider,
        api_key=request.api_key,
        model=request.model,
        embedding_model=request.embedding_model,
        exchange_id=request.exchange_id,
        exchange_api_key=request.exchange_api_key,
        exchange_secret=request.exchange_secret,
    )


@router.delete("/api-keys/{key_type}", response_model=SetAPIKeysResponse)
async def delete_api_keys(
    key_type: str,
    service: ConfigService = Depends(get_config_service),
) -> SetAPIKeysResponse:
    """Delete API keys by type.

    Args:
        key_type: "llm" to delete LLM-related keys, "exchange" for exchange keys
    """
    return service.delete_api_keys(key_type)


# ============== Test Connections ==============

@router.post("/test-llm", response_model=TestLLMResponse)
async def test_llm(
    service: ConfigService = Depends(get_config_service),
) -> TestLLMResponse:
    """Test LLM connection."""
    return await service.test_llm()


@router.post("/test-exchange", response_model=TestExchangeResponse)
async def test_exchange(
    service: ConfigService = Depends(get_config_service),
) -> TestExchangeResponse:
    """Test exchange connection."""
    return await service.test_exchange()


# ============== Agent Configuration ==============

@router.get("/agents", response_model=AgentConfigResponse)
async def get_agent_config(
    service: ConfigService = Depends(get_config_service),
) -> AgentConfigResponse:
    """Get agent configuration."""
    return service.get_agent_config()


@router.post("/agents", response_model=SetAPIKeysResponse)
async def set_agent_config(
    request: SetAgentConfigRequest,
    service: ConfigService = Depends(get_config_service),
) -> SetAPIKeysResponse:
    """Set agent configuration."""
    return service.set_agent_config(request)


# ============== Data Configuration ==============

@router.get("/data", response_model=DataConfigResponse)
async def get_data_config(
    service: ConfigService = Depends(get_config_service),
) -> DataConfigResponse:
    """Get data configuration."""
    return service.get_data_config()


@router.post("/data", response_model=SetAPIKeysResponse)
async def set_data_config(
    request: SetDataConfigRequest,
    service: ConfigService = Depends(get_config_service),
) -> SetAPIKeysResponse:
    """Set data configuration."""
    return service.set_data_config(request)


# ============== Factor Mining Configuration ==============

@router.get("/factor-mining", response_model=FactorMiningConfigResponse)
async def get_factor_mining_config(
    service: ConfigService = Depends(get_config_service),
) -> FactorMiningConfigResponse:
    """Get factor mining configuration."""
    return service.get_factor_mining_config()


@router.post("/factor-mining", response_model=SetAPIKeysResponse)
async def set_factor_mining_config(
    request: SetFactorMiningConfigRequest,
    service: ConfigService = Depends(get_config_service),
) -> SetAPIKeysResponse:
    """Set factor mining configuration."""
    return service.set_factor_mining_config(request)


# ============== Risk Control Configuration ==============

@router.get("/risk-control", response_model=RiskControlConfigResponse)
async def get_risk_control_config(
    service: ConfigService = Depends(get_config_service),
) -> RiskControlConfigResponse:
    """Get risk control configuration."""
    return service.get_risk_control_config()


@router.post("/risk-control", response_model=SetAPIKeysResponse)
async def set_risk_control_config(
    request: SetRiskControlConfigRequest,
    service: ConfigService = Depends(get_config_service),
) -> SetAPIKeysResponse:
    """Set risk control configuration."""
    return service.set_risk_control_config(request)


# ============== P3: Sandbox Configuration ==============

@router.get("/sandbox", response_model=SandboxConfigResponse)
async def get_sandbox_config(
    service: ConfigService = Depends(get_config_service),
) -> SandboxConfigResponse:
    """Get sandbox execution configuration."""
    return service.get_sandbox_config()


@router.put("/sandbox", response_model=SetAPIKeysResponse)
async def update_sandbox_config(
    request: SandboxConfigUpdate,
    service: ConfigService = Depends(get_config_service),
) -> SetAPIKeysResponse:
    """Update sandbox execution configuration."""
    return service.update_sandbox_config(request)


@router.get("/sandbox/executions", response_model=ExecutionLogResponse)
async def get_execution_logs(
    page: int = 1,
    page_size: int = 20,
    status: str | None = None,
    service: ConfigService = Depends(get_config_service),
) -> ExecutionLogResponse:
    """Get sandbox execution logs with pagination."""
    return await service.get_execution_logs(page, page_size, status)


# ============== P3: Security Configuration ==============

@router.get("/security", response_model=SecurityConfigResponse)
async def get_security_config(
    service: ConfigService = Depends(get_config_service),
) -> SecurityConfigResponse:
    """Get security configuration settings."""
    return service.get_security_config()


@router.put("/security", response_model=SetAPIKeysResponse)
async def update_security_config(
    request: SecurityConfigUpdate,
    service: ConfigService = Depends(get_config_service),
) -> SetAPIKeysResponse:
    """Update security configuration settings."""
    return service.update_security_config(request)


# ============== P3: LLM Advanced Configuration ==============

@router.get("/llm/advanced", response_model=LLMAdvancedConfigResponse)
async def get_llm_advanced_config(
    service: ConfigService = Depends(get_config_service),
) -> LLMAdvancedConfigResponse:
    """Get LLM provider advanced configuration including rate limits and fallback chain."""
    return await service.get_llm_advanced_config()


@router.put("/llm/advanced", response_model=SetAPIKeysResponse)
async def update_llm_advanced_config(
    request: LLMAdvancedConfigUpdate,
    service: ConfigService = Depends(get_config_service),
) -> SetAPIKeysResponse:
    """Update LLM advanced configuration."""
    return service.update_llm_advanced_config(request)


@router.get("/llm/traces", response_model=LLMTraceResponse)
async def get_llm_traces(
    page: int = 1,
    page_size: int = 20,
    agent: str | None = None,
    model: str | None = None,
    service: ConfigService = Depends(get_config_service),
) -> LLMTraceResponse:
    """Get LLM API call traces (audit log) with pagination."""
    return await service.get_llm_traces(page, page_size, agent, model)


@router.get("/llm/costs", response_model=LLMCostSummary)
async def get_llm_costs(
    hours: int = 24,
    service: ConfigService = Depends(get_config_service),
) -> LLMCostSummary:
    """Get LLM cost summary for the specified time period."""
    return await service.get_llm_costs(hours)


# ============== P3: Derivative Data Configuration ==============

@router.get("/derivative-data", response_model=DerivativeDataConfigResponse)
async def get_derivative_data_config(
    service: ConfigService = Depends(get_config_service),
) -> DerivativeDataConfigResponse:
    """Get derivative data download configuration."""
    return service.get_derivative_data_config()


@router.put("/derivative-data", response_model=SetAPIKeysResponse)
async def update_derivative_data_config(
    request: DerivativeDataConfigUpdate,
    service: ConfigService = Depends(get_config_service),
) -> SetAPIKeysResponse:
    """Update derivative data configuration."""
    return service.update_derivative_data_config(request)


# ============== P3: Checkpoint Management ==============

@router.get("/checkpoints/threads", response_model=list[CheckpointThreadInfo])
async def list_checkpoint_threads(
    limit: int = 20,
    service: ConfigService = Depends(get_config_service),
) -> list[CheckpointThreadInfo]:
    """List all checkpoint threads."""
    return await service.list_checkpoint_threads(limit)


@router.get("/checkpoints/{thread_id}", response_model=CheckpointListResponse)
async def list_checkpoints(
    thread_id: str,
    service: ConfigService = Depends(get_config_service),
) -> CheckpointListResponse:
    """List checkpoints for a specific thread."""
    return await service.list_checkpoints(thread_id)


@router.get("/checkpoints/{thread_id}/{checkpoint_id}", response_model=CheckpointStateResponse)
async def get_checkpoint_state(
    thread_id: str,
    checkpoint_id: str,
    service: ConfigService = Depends(get_config_service),
) -> CheckpointStateResponse:
    """Get full state of a specific checkpoint."""
    return await service.get_checkpoint_state(thread_id, checkpoint_id)


@router.post("/checkpoints/{thread_id}/restore/{checkpoint_id}", response_model=SetAPIKeysResponse)
async def restore_checkpoint(
    thread_id: str,
    checkpoint_id: str,
    service: ConfigService = Depends(get_config_service),
) -> SetAPIKeysResponse:
    """Restore pipeline to a specific checkpoint (time travel)."""
    return await service.restore_checkpoint(thread_id, checkpoint_id)


# ============== P3: Alpha Benchmark Configuration ==============

@router.get("/benchmark", response_model=BenchmarkConfigResponse)
async def get_benchmark_config(
    service: ConfigService = Depends(get_config_service),
) -> BenchmarkConfigResponse:
    """Get alpha benchmark configuration."""
    return service.get_benchmark_config()


@router.put("/benchmark", response_model=SetAPIKeysResponse)
async def update_benchmark_config(
    request: BenchmarkConfigUpdate,
    service: ConfigService = Depends(get_config_service),
) -> SetAPIKeysResponse:
    """Update alpha benchmark configuration."""
    return service.update_benchmark_config(request)


@router.get("/benchmark/results", response_model=BenchmarkResultsResponse)
async def get_benchmark_results(
    page: int = 1,
    page_size: int = 20,
    novel_only: bool = False,
    service: ConfigService = Depends(get_config_service),
) -> BenchmarkResultsResponse:
    """Get benchmark results with pagination."""
    return await service.get_benchmark_results(page, page_size, novel_only)


@router.post("/benchmark/run", response_model=SetAPIKeysResponse)
async def run_benchmark(
    factor_names: list[str] | None = None,
    service: ConfigService = Depends(get_config_service),
) -> SetAPIKeysResponse:
    """Run alpha benchmark on specified factors (or all pending factors)."""
    return await service.run_benchmark(factor_names)
