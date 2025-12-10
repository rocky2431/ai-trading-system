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
