"""Config service for managing IQFMP system configuration."""

import os
import json
import time
from pathlib import Path
from typing import Optional

from iqfmp.api.config.schemas import (
    AgentConfigResponse,
    AgentModelConfig,
    AvailableModelsResponse,
    ConfigStatusResponse,
    DataConfigResponse,
    EmbeddingModelInfo,
    EvaluationConfig,
    FactorFamilyOption,
    FactorMiningConfigResponse,
    FeaturesStatus,
    FrequencyOption,
    ModelInfo,
    RiskControlConfig,
    RiskControlConfigResponse,
    SavedAPIKeysResponse,
    SetAgentConfigRequest,
    SetAPIKeysResponse,
    SetDataConfigRequest,
    SetFactorMiningConfigRequest,
    SetRiskControlConfigRequest,
    TestExchangeResponse,
    TestLLMResponse,
)


class ConfigService:
    """Service for managing IQFMP system configuration."""

    # IQFMP Agents (from architecture.md)
    AGENTS = [
        {
            "agent_id": "factor_generation",
            "agent_name": "Factor Generation Agent",
            "description": "LLM 驱动的因子代码生成",
            "default_model": "deepseek/deepseek-coder",
        },
        {
            "agent_id": "factor_evaluation",
            "agent_name": "Factor Evaluation Agent",
            "description": "多维度验证 + 防过拟合",
            "default_model": "deepseek/deepseek-chat",
        },
        {
            "agent_id": "strategy_assembly",
            "agent_name": "Strategy Assembly Agent",
            "description": "策略组装与优化",
            "default_model": "anthropic/claude-3.5-sonnet",
        },
        {
            "agent_id": "backtest_optimization",
            "agent_name": "Backtest Optimization Agent",
            "description": "回测参数优化",
            "default_model": "openai/gpt-4o",
        },
        {
            "agent_id": "risk_check",
            "agent_name": "Risk Check Agent",
            "description": "风险检查与控制",
            "default_model": "deepseek/deepseek-chat",
        },
    ]

    # Factor families (from architecture.md)
    FACTOR_FAMILIES = [
        {"id": "momentum", "name": "Momentum", "description": "趋势动量因子"},
        {"id": "value", "name": "Value", "description": "价值因子"},
        {"id": "volatility", "name": "Volatility", "description": "波动率因子"},
        {"id": "liquidity", "name": "Liquidity", "description": "流动性因子"},
        {"id": "sentiment", "name": "Sentiment", "description": "情绪因子"},
        {"id": "funding", "name": "Funding Rate", "description": "资金费率因子"},
        {"id": "orderbook", "name": "Order Book", "description": "订单簿因子"},
    ]

    def __init__(self) -> None:
        """Initialize config service."""
        self._config_dir = Path.home() / ".iqfmp"
        self._config_file = self._config_dir / "config.json"
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._config = self._load_config()

    def _load_config(self) -> dict:
        """Load configuration from file."""
        if self._config_file.exists():
            try:
                with open(self._config_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_config(self) -> None:
        """Save configuration to file."""
        with open(self._config_file, "w") as f:
            json.dump(self._config, f, indent=2)

    def _mask_api_key(self, key: Optional[str]) -> Optional[str]:
        """Mask API key for display."""
        if not key:
            return None
        if len(key) <= 8:
            return "****"
        return f"{key[:8]}...{key[-4:]}"

    def get_status(self) -> ConfigStatusResponse:
        """Get current configuration status."""
        llm_configured = bool(self._config.get("api_key"))
        exchange_configured = bool(self._config.get("exchange_api_key"))

        # Check for Qlib availability
        qlib_available = False
        try:
            import qlib
            qlib_available = True
        except (ImportError, LookupError, Exception):
            # ImportError: qlib not installed
            # LookupError: qlib installed but setuptools-scm version detection failed
            # Exception: catch any other initialization errors
            pass

        # Check database connections
        timescaledb_connected = bool(os.getenv("DATABASE_URL"))
        redis_connected = bool(os.getenv("REDIS_URL"))

        features = FeaturesStatus(
            factor_generation=llm_configured,
            factor_evaluation=llm_configured and qlib_available,
            strategy_assembly=llm_configured,
            backtest_optimization=llm_configured and qlib_available,
            live_trading=exchange_configured,
        )

        return ConfigStatusResponse(
            llm_configured=llm_configured,
            llm_provider=self._config.get("provider") if llm_configured else None,
            llm_model=self._config.get("model") if llm_configured else None,
            exchange_configured=exchange_configured,
            exchange_id=self._config.get("exchange_id") if exchange_configured else None,
            qlib_available=qlib_available,
            timescaledb_connected=timescaledb_connected,
            redis_connected=redis_connected,
            features=features,
        )

    def get_available_models(self) -> AvailableModelsResponse:
        """Get available LLM models."""
        openrouter_models = [
            ModelInfo(
                id="deepseek/deepseek-coder",
                name="DeepSeek Coder",
                provider="DeepSeek",
                context_length=64000,
                use_case="factor_generation",
            ),
            ModelInfo(
                id="deepseek/deepseek-chat-v3-0324",
                name="DeepSeek Chat V3",
                provider="DeepSeek",
                context_length=64000,
                use_case="general",
            ),
            ModelInfo(
                id="deepseek/deepseek-r1",
                name="DeepSeek R1",
                provider="DeepSeek",
                context_length=64000,
                use_case="reasoning",
            ),
            ModelInfo(
                id="anthropic/claude-3.5-sonnet",
                name="Claude 3.5 Sonnet",
                provider="Anthropic",
                context_length=200000,
                use_case="strategy_design",
            ),
            ModelInfo(
                id="openai/gpt-4o",
                name="GPT-4o",
                provider="OpenAI",
                context_length=128000,
                use_case="code_review",
            ),
            ModelInfo(
                id="google/gemini-2.0-flash-exp",
                name="Gemini 2.0 Flash",
                provider="Google",
                context_length=1000000,
                use_case="analysis",
            ),
        ]

        embedding_models = [
            EmbeddingModelInfo(
                id="openai/text-embedding-3-large",
                name="OpenAI text-embedding-3-large",
                dimensions=3072,
            ),
            EmbeddingModelInfo(
                id="openai/text-embedding-3-small",
                name="OpenAI text-embedding-3-small",
                dimensions=1536,
            ),
        ]

        return AvailableModelsResponse(
            models={"openrouter": openrouter_models},
            embedding_models={"openrouter": embedding_models},
        )

    def get_saved_api_keys(self) -> SavedAPIKeysResponse:
        """Get saved API keys (masked)."""
        return SavedAPIKeysResponse(
            api_key=self._mask_api_key(self._config.get("api_key")),
            model=self._config.get("model"),
            embedding_model=self._config.get("embedding_model"),
            exchange_id=self._config.get("exchange_id"),
            exchange_api_key=self._mask_api_key(self._config.get("exchange_api_key")),
        )

    def set_api_keys(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        exchange_id: Optional[str] = None,
        exchange_api_key: Optional[str] = None,
        exchange_secret: Optional[str] = None,
    ) -> SetAPIKeysResponse:
        """Set API keys."""
        updated = []

        if provider:
            self._config["provider"] = provider
            updated.append("provider")

        if api_key:
            self._config["api_key"] = api_key
            updated.append("api_key")
            os.environ["OPENROUTER_API_KEY"] = api_key

        if model:
            self._config["model"] = model
            updated.append("model")

        if embedding_model:
            self._config["embedding_model"] = embedding_model
            updated.append("embedding_model")

        if exchange_id:
            self._config["exchange_id"] = exchange_id
            updated.append("exchange_id")

        if exchange_api_key:
            self._config["exchange_api_key"] = exchange_api_key
            updated.append("exchange_api_key")

        if exchange_secret:
            self._config["exchange_secret"] = exchange_secret
            updated.append("exchange_secret")

        self._save_config()

        return SetAPIKeysResponse(
            success=True,
            message=f"Configuration updated: {', '.join(updated)}",
        )

    async def test_llm(self) -> TestLLMResponse:
        """Test LLM connection."""
        api_key = self._config.get("api_key")
        model = self._config.get("model", "deepseek/deepseek-chat-v3-0324")

        if not api_key:
            return TestLLMResponse(
                success=False,
                message="No API key configured",
            )

        try:
            import httpx

            start_time = time.time()

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "Say 'test ok' in 2 words"}],
                        "max_tokens": 10,
                    },
                    timeout=30.0,
                )

            elapsed_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                return TestLLMResponse(
                    success=True,
                    model=model,
                    response_time_ms=round(elapsed_ms, 2),
                    message="LLM connection successful",
                )
            else:
                return TestLLMResponse(
                    success=False,
                    message=f"API error: {response.status_code} - {response.text}",
                )
        except Exception as e:
            return TestLLMResponse(
                success=False,
                message=f"Connection error: {str(e)}",
            )

    async def test_exchange(self) -> TestExchangeResponse:
        """Test exchange connection."""
        exchange_api_key = self._config.get("exchange_api_key")
        exchange_id = self._config.get("exchange_id", "binance")

        if not exchange_api_key:
            return TestExchangeResponse(
                success=False,
                message="No exchange API key configured",
            )

        try:
            import ccxt

            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({
                "apiKey": exchange_api_key,
                "secret": self._config.get("exchange_secret", ""),
            })

            ticker = exchange.fetch_ticker("BTC/USDT")

            return TestExchangeResponse(
                success=True,
                exchange_id=exchange_id,
                latest_btc_price=ticker.get("last"),
                message="Exchange connection successful",
            )
        except Exception as e:
            return TestExchangeResponse(
                success=False,
                exchange_id=exchange_id,
                message=f"Connection error: {str(e)}",
            )

    # ============== Agent Configuration ==============

    def get_agent_config(self) -> AgentConfigResponse:
        """Get agent configuration."""
        agent_config = self._config.get("agents", {})
        agents = []

        for agent_def in self.AGENTS:
            agent_id = agent_def["agent_id"]
            saved_config = agent_config.get(agent_id, {})
            agents.append(AgentModelConfig(
                agent_id=agent_id,
                agent_name=agent_def["agent_name"],
                description=agent_def["description"],
                model_id=saved_config.get("model_id", agent_def["default_model"]),
                enabled=saved_config.get("enabled", True),
            ))

        return AgentConfigResponse(agents=agents)

    def set_agent_config(self, request: SetAgentConfigRequest) -> SetAPIKeysResponse:
        """Set agent configuration."""
        if "agents" not in self._config:
            self._config["agents"] = {}

        if request.agent_id not in self._config["agents"]:
            self._config["agents"][request.agent_id] = {}

        updated = []

        if request.model_id is not None:
            self._config["agents"][request.agent_id]["model_id"] = request.model_id
            updated.append("model_id")

        if request.enabled is not None:
            self._config["agents"][request.agent_id]["enabled"] = request.enabled
            updated.append("enabled")

        self._save_config()

        return SetAPIKeysResponse(
            success=True,
            message=f"Agent '{request.agent_id}' updated: {', '.join(updated)}",
        )

    # ============== Data Configuration ==============

    def get_data_config(self) -> DataConfigResponse:
        """Get data configuration."""
        data_config = self._config.get("data", {})

        frequency_options = [
            FrequencyOption(id="1m", name="1 Minute", description="1分钟K线"),
            FrequencyOption(id="5m", name="5 Minutes", description="5分钟K线"),
            FrequencyOption(id="15m", name="15 Minutes", description="15分钟K线"),
            FrequencyOption(id="1h", name="1 Hour", description="1小时K线"),
            FrequencyOption(id="4h", name="4 Hours", description="4小时K线"),
            FrequencyOption(id="1d", name="Daily", description="日K线"),
        ]

        return DataConfigResponse(
            data_frequency=data_config.get("data_frequency", "1h"),
            data_source=data_config.get("data_source", "timescaledb"),
            symbols=data_config.get("symbols", ["BTC/USDT", "ETH/USDT"]),
            qlib_data_path=data_config.get("qlib_data_path"),
            frequency_options=frequency_options,
        )

    def set_data_config(self, request: SetDataConfigRequest) -> SetAPIKeysResponse:
        """Set data configuration."""
        if "data" not in self._config:
            self._config["data"] = {}

        updated = []

        if request.data_frequency is not None:
            self._config["data"]["data_frequency"] = request.data_frequency
            updated.append("data_frequency")

        if request.data_source is not None:
            self._config["data"]["data_source"] = request.data_source
            updated.append("data_source")

        if request.symbols is not None:
            self._config["data"]["symbols"] = request.symbols
            updated.append("symbols")

        self._save_config()

        return SetAPIKeysResponse(
            success=True,
            message=f"Data configuration updated: {', '.join(updated)}",
        )

    # ============== Factor Mining Configuration ==============

    def get_factor_mining_config(self) -> FactorMiningConfigResponse:
        """Get factor mining configuration."""
        mining_config = self._config.get("factor_mining", {})

        factor_families = []
        saved_families = mining_config.get("factor_families", {})
        for family_def in self.FACTOR_FAMILIES:
            family_id = family_def["id"]
            enabled = saved_families.get(family_id, {}).get("enabled", True)
            factor_families.append(FactorFamilyOption(
                id=family_id,
                name=family_def["name"],
                description=family_def["description"],
                enabled=enabled,
            ))

        eval_config = mining_config.get("evaluation", {})
        evaluation = EvaluationConfig(**eval_config) if eval_config else EvaluationConfig()

        return FactorMiningConfigResponse(
            factor_families=factor_families,
            evaluation=evaluation,
            max_concurrent_generation=mining_config.get("max_concurrent_generation", 10),
            code_execution_timeout=mining_config.get("code_execution_timeout", 60),
        )

    def set_factor_mining_config(self, request: SetFactorMiningConfigRequest) -> SetAPIKeysResponse:
        """Set factor mining configuration."""
        if "factor_mining" not in self._config:
            self._config["factor_mining"] = {}

        updated = []

        if request.factor_families is not None:
            self._config["factor_mining"]["factor_families"] = {
                f.id: {"enabled": f.enabled} for f in request.factor_families
            }
            updated.append("factor_families")

        if request.evaluation is not None:
            self._config["factor_mining"]["evaluation"] = request.evaluation.model_dump()
            updated.append("evaluation")

        if request.max_concurrent_generation is not None:
            self._config["factor_mining"]["max_concurrent_generation"] = request.max_concurrent_generation
            updated.append("max_concurrent_generation")

        if request.code_execution_timeout is not None:
            self._config["factor_mining"]["code_execution_timeout"] = request.code_execution_timeout
            updated.append("code_execution_timeout")

        self._save_config()

        return SetAPIKeysResponse(
            success=True,
            message=f"Factor mining configuration updated: {', '.join(updated)}",
        )

    # ============== Risk Control Configuration ==============

    def get_risk_control_config(self) -> RiskControlConfigResponse:
        """Get risk control configuration."""
        risk_config = self._config.get("risk_control", {})
        config = RiskControlConfig(**risk_config) if risk_config else RiskControlConfig()

        return RiskControlConfigResponse(
            config=config,
            is_live_trading_enabled=self._config.get("live_trading_enabled", False),
        )

    def set_risk_control_config(self, request: SetRiskControlConfigRequest) -> SetAPIKeysResponse:
        """Set risk control configuration."""
        if "risk_control" not in self._config:
            self._config["risk_control"] = {}

        updated = []

        for field in ["max_single_loss_pct", "max_daily_loss_pct", "max_position_pct",
                      "max_total_position_pct", "emergency_close_threshold"]:
            value = getattr(request, field, None)
            if value is not None:
                self._config["risk_control"][field] = value
                updated.append(field)

        self._save_config()

        return SetAPIKeysResponse(
            success=True,
            message=f"Risk control configuration updated: {', '.join(updated)}",
        )


# Singleton instance
_config_service: Optional[ConfigService] = None


def get_config_service() -> ConfigService:
    """Get or create config service singleton."""
    global _config_service
    if _config_service is None:
        _config_service = ConfigService()
    return _config_service
