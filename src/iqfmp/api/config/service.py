"""Config service for managing IQFMP system configuration."""

import os
import json
import time
import httpx
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

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

    # Cache for OpenRouter models (TTL: 5 minutes)
    _models_cache: list = []
    _models_cache_time: Optional[datetime] = None
    _embedding_models_cache: list = []
    _embedding_models_cache_time: Optional[datetime] = None
    _cache_ttl = timedelta(minutes=5)

    # Preferred providers to show (filter from 500+ models)
    PREFERRED_PROVIDERS = [
        "deepseek",
        "anthropic",
        "openai",
        "google",
        "x-ai",
        "z-ai",        # 智谱 GLM
        "moonshotai",  # Kimi
        "mistralai",
    ]

    # IQFMP Agents (from architecture.md) - Updated with latest models 2024-2025
    AGENTS = [
        {
            "agent_id": "factor_generation",
            "agent_name": "Factor Generation Agent",
            "description": "LLM-driven factor code generation",
            "default_model": "deepseek/deepseek-coder-v3",  # Best for code
        },
        {
            "agent_id": "factor_evaluation",
            "agent_name": "Factor Evaluation Agent",
            "description": "Multi-dimensional validation + Anti-overfitting",
            "default_model": "deepseek/deepseek-r1",  # Best for reasoning
        },
        {
            "agent_id": "strategy_assembly",
            "agent_name": "Strategy Assembly Agent",
            "description": "Strategy assembly and optimization",
            "default_model": "anthropic/claude-sonnet-4",  # Latest Claude
        },
        {
            "agent_id": "backtest_optimization",
            "agent_name": "Backtest Optimization Agent",
            "description": "Backtest parameter optimization",
            "default_model": "openai/gpt-4.1",  # Latest GPT
        },
        {
            "agent_id": "risk_check",
            "agent_name": "Risk Check Agent",
            "description": "Risk check and control",
            "default_model": "google/gemini-2.5-flash",  # Fast + 1M context
        },
    ]

    # Factor families (from architecture.md)
    FACTOR_FAMILIES = [
        {"id": "momentum", "name": "Momentum", "description": "Trend momentum factors"},
        {"id": "value", "name": "Value", "description": "Value factors"},
        {"id": "volatility", "name": "Volatility", "description": "Volatility factors"},
        {"id": "liquidity", "name": "Liquidity", "description": "Liquidity factors"},
        {"id": "sentiment", "name": "Sentiment", "description": "Sentiment factors"},
        {"id": "funding", "name": "Funding Rate", "description": "Funding rate factors"},
        {"id": "orderbook", "name": "Order Book", "description": "Order book factors"},
    ]

    def __init__(self) -> None:
        """Initialize config service."""
        self._config_dir = Path.home() / ".iqfmp"
        self._config_file = self._config_dir / "config.json"
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._config = self._load_config()
        # Restore API key from config to environment on startup
        self._restore_env_from_config()

    def _restore_env_from_config(self) -> None:
        """Restore environment variables from saved config on startup.

        This ensures API keys persist across server restarts.
        """
        # Restore OpenRouter API key
        api_key = self._config.get("api_key")
        if api_key and not os.getenv("OPENROUTER_API_KEY"):
            os.environ["OPENROUTER_API_KEY"] = api_key
            print(f"Restored OPENROUTER_API_KEY from config: {self._mask_api_key(api_key)}")

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
        # Check LLM configuration from multiple sources
        llm_api_key = (
            self._config.get("api_key")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("LLM_API_KEY")
        )
        llm_configured = bool(llm_api_key)
        llm_provider = self._config.get("provider") or "openrouter" if llm_configured else None
        llm_model = (
            self._config.get("model")
            or os.getenv("OPENROUTER_MODEL")
            or os.getenv("LLM_MODEL")
        )

        # Check exchange configuration
        exchange_api_key = self._config.get("exchange_api_key") or os.getenv("EXCHANGE_API_KEY")
        exchange_configured = bool(exchange_api_key)
        exchange_id = self._config.get("exchange_id") or os.getenv("EXCHANGE_ID")

        # Check for Qlib availability (M2 FIX: check actual initialization status)
        qlib_available = False
        try:
            from iqfmp.core.qlib_init import is_qlib_initialized
            qlib_available = is_qlib_initialized()
        except (ImportError, LookupError, Exception):
            # Fallback to simple import check
            try:
                import qlib
                qlib_available = True
            except Exception:
                pass

        # Check TimescaleDB connection - support both DATABASE_URL and individual PG* vars
        timescaledb_connected = self._check_timescaledb_connection()

        # Check Redis connection - support both REDIS_URL and individual vars
        redis_connected = self._check_redis_connection()

        features = FeaturesStatus(
            factor_generation=llm_configured,
            factor_evaluation=llm_configured and qlib_available,
            strategy_assembly=llm_configured,
            backtest_optimization=llm_configured and qlib_available,
            live_trading=exchange_configured,
        )

        return ConfigStatusResponse(
            llm_configured=llm_configured,
            llm_provider=llm_provider,
            llm_model=llm_model,
            exchange_configured=exchange_configured,
            exchange_id=exchange_id if exchange_configured else None,
            qlib_available=qlib_available,
            timescaledb_connected=timescaledb_connected,
            redis_connected=redis_connected,
            features=features,
        )

    def _check_timescaledb_connection(self) -> bool:
        """Check TimescaleDB connection status."""
        try:
            import psycopg2
            from urllib.parse import urlparse

            # Parse DATABASE_URL if available
            db_url = os.getenv("DATABASE_URL")
            if db_url:
                parsed = urlparse(db_url)
                conn = psycopg2.connect(
                    host=parsed.hostname or "localhost",
                    port=parsed.port or 5433,
                    user=parsed.username or "iqfmp",
                    password=parsed.password or "iqfmp",
                    dbname=parsed.path.lstrip("/") or "iqfmp",
                    connect_timeout=3,
                )
            else:
                # Use individual PG* environment variables with same defaults as database.py
                conn = psycopg2.connect(
                    host=os.getenv("PGHOST", "localhost"),
                    port=int(os.getenv("PGPORT", "5433")),
                    user=os.getenv("PGUSER", "iqfmp"),
                    password=os.getenv("PGPASSWORD", "iqfmp"),
                    dbname=os.getenv("PGDATABASE", "iqfmp"),
                    connect_timeout=3,
                )
            conn.close()
            return True
        except Exception:
            return False

    def _check_redis_connection(self) -> bool:
        """Check Redis connection status."""
        # Check if REDIS_URL or individual vars exist
        has_config = bool(
            os.getenv("REDIS_URL")
            or os.getenv("REDIS_HOST")
        )
        if not has_config:
            # Default to localhost:6379
            pass

        # Try actual connection
        try:
            import redis
            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                client = redis.from_url(redis_url, socket_timeout=2)
            else:
                client = redis.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", "6379")),
                    password=os.getenv("REDIS_PASSWORD"),
                    socket_timeout=2,
                )
            client.ping()
            return True
        except Exception:
            return False

    def _fetch_openrouter_models(self) -> list[ModelInfo]:
        """Fetch models from OpenRouter API with caching.

        Returns filtered list of models from preferred providers.
        Uses 5-minute cache to avoid excessive API calls.
        """
        # Check cache
        if (
            self._models_cache
            and self._models_cache_time
            and datetime.now() - self._models_cache_time < self._cache_ttl
        ):
            return self._models_cache

        try:
            # Fetch from OpenRouter API
            with httpx.Client(timeout=10.0) as client:
                response = client.get("https://openrouter.ai/api/v1/models")
                response.raise_for_status()
                data = response.json()

            models = []
            for m in data.get("data", []):
                model_id = m.get("id", "")
                # Skip free/beta variants and filter by preferred providers
                if ":free" in model_id or ":beta" in model_id:
                    continue

                provider = model_id.split("/")[0] if "/" in model_id else ""
                if provider not in self.PREFERRED_PROVIDERS:
                    continue

                # Extract context length from pricing info or default
                context_length = m.get("context_length", 128000)

                # Map provider to display name
                provider_names = {
                    "deepseek": "DeepSeek",
                    "anthropic": "Anthropic",
                    "openai": "OpenAI",
                    "google": "Google",
                    "x-ai": "xAI",
                    "z-ai": "智谱 GLM",
                    "moonshotai": "Kimi",
                    "mistralai": "Mistral",
                }

                models.append(ModelInfo(
                    id=model_id,
                    name=m.get("name", model_id),
                    provider=provider_names.get(provider, provider.title()),
                    context_length=context_length,
                    use_case=self._infer_use_case(model_id, m.get("name", "")),
                ))

            # Sort by provider, then by name
            models.sort(key=lambda x: (x.provider, x.name))

            # Update cache
            ConfigService._models_cache = models
            ConfigService._models_cache_time = datetime.now()

            return models

        except Exception as e:
            # Fallback to cached or minimal list
            if self._models_cache:
                return self._models_cache
            return self._get_fallback_models()

    def _infer_use_case(self, model_id: str, name: str) -> str:
        """Infer use case from model ID and name."""
        lower_id = model_id.lower()
        lower_name = name.lower()

        if "coder" in lower_id or "code" in lower_name:
            return "code_generation"
        elif "r1" in lower_id or "o1" in lower_id or "o3" in lower_id or "reasoning" in lower_name:
            return "reasoning"
        elif "flash" in lower_id or "haiku" in lower_id or "mini" in lower_id:
            return "fast_tasks"
        elif "opus" in lower_id or "pro" in lower_id:
            return "complex_analysis"
        elif "sonnet" in lower_id:
            return "strategy_design"
        else:
            return "general"

    def _get_fallback_models(self) -> list[ModelInfo]:
        """Return fallback models if API call fails."""
        return [
            ModelInfo(id="deepseek/deepseek-chat", name="DeepSeek Chat", provider="DeepSeek", context_length=64000, use_case="general"),
            ModelInfo(id="anthropic/claude-sonnet-4", name="Claude Sonnet 4", provider="Anthropic", context_length=200000, use_case="strategy_design"),
            ModelInfo(id="openai/gpt-4o", name="GPT-4o", provider="OpenAI", context_length=128000, use_case="general"),
            ModelInfo(id="google/gemini-2.0-flash", name="Gemini 2.0 Flash", provider="Google", context_length=1000000, use_case="fast_tasks"),
        ]

    def _fetch_openrouter_embedding_models(self) -> list[EmbeddingModelInfo]:
        """Fetch embedding models from OpenRouter API with caching.

        Returns list of embedding models from /api/v1/embeddings/models.
        Uses 5-minute cache to avoid excessive API calls.
        """
        # Check cache
        if (
            self._embedding_models_cache
            and self._embedding_models_cache_time
            and datetime.now() - self._embedding_models_cache_time < self._cache_ttl
        ):
            return self._embedding_models_cache

        try:
            # Fetch from OpenRouter API
            with httpx.Client(timeout=10.0) as client:
                response = client.get("https://openrouter.ai/api/v1/embeddings/models")
                response.raise_for_status()
                data = response.json()

            models = []
            for m in data.get("data", []):
                model_id = m.get("id", "")
                name = m.get("name", model_id)
                context_length = m.get("context_length", 8192)

                # Extract dimensions from description or use default based on model
                dimensions = self._infer_embedding_dimensions(model_id, m.get("description", ""))

                models.append(EmbeddingModelInfo(
                    id=model_id,
                    name=name,
                    dimensions=dimensions,
                    context_length=context_length,
                ))

            # Sort by name
            models.sort(key=lambda x: x.name)

            # Update cache
            ConfigService._embedding_models_cache = models
            ConfigService._embedding_models_cache_time = datetime.now()

            return models

        except Exception as e:
            # Fallback to cached or static list
            if self._embedding_models_cache:
                return self._embedding_models_cache
            return self._get_fallback_embedding_models()

    def _infer_embedding_dimensions(self, model_id: str, description: str) -> int:
        """Infer embedding dimensions from model ID or description."""
        lower_id = model_id.lower()
        lower_desc = description.lower()

        # Known dimensions based on model families
        if "text-embedding-3-large" in lower_id:
            return 3072
        elif "text-embedding-3-small" in lower_id:
            return 1536
        elif "text-embedding-ada" in lower_id:
            return 1536
        elif "qwen3-embedding-8b" in lower_id or "qwen3-embedding-4b" in lower_id:
            return 4096  # Qwen3 embedding models
        elif "gemini-embedding" in lower_id:
            return 768  # Google Gemini
        elif "mistral-embed" in lower_id or "codestral-embed" in lower_id:
            return 1024  # Mistral embedding
        elif "bge-m3" in lower_id or "bge-large" in lower_id:
            return 1024
        elif "bge-base" in lower_id:
            return 768
        elif "gte-large" in lower_id or "e5-large" in lower_id:
            return 1024
        elif "gte-base" in lower_id or "e5-base" in lower_id:
            return 768
        elif "minilm" in lower_id:
            return 384
        elif "mpnet" in lower_id:
            return 768

        # Try to extract from description
        for dim in [3072, 2048, 1536, 1024, 768, 512, 384]:
            if f"{dim}-dimensional" in lower_desc or f"{dim}d" in lower_desc:
                return dim

        # Default
        return 1024

    def _get_fallback_embedding_models(self) -> list[EmbeddingModelInfo]:
        """Return fallback embedding models if API call fails."""
        return [
            EmbeddingModelInfo(id="openai/text-embedding-3-large", name="OpenAI Text Embedding 3 Large", dimensions=3072, context_length=8192),
            EmbeddingModelInfo(id="openai/text-embedding-3-small", name="OpenAI Text Embedding 3 Small", dimensions=1536, context_length=8192),
            EmbeddingModelInfo(id="openai/text-embedding-ada-002", name="OpenAI Text Embedding Ada 002", dimensions=1536, context_length=8192),
            EmbeddingModelInfo(id="google/gemini-embedding-001", name="Google Gemini Embedding 001", dimensions=768, context_length=20000),
            EmbeddingModelInfo(id="mistralai/mistral-embed-2312", name="Mistral Embed 2312", dimensions=1024, context_length=8192),
        ]

    def get_available_models(self) -> AvailableModelsResponse:
        """Get available LLM models dynamically from OpenRouter API.

        Models are fetched from https://openrouter.ai/api/v1/models
        and filtered by preferred providers (DeepSeek, Claude, GPT, Gemini, etc.)
        Embedding models are fetched from https://openrouter.ai/api/v1/embeddings/models
        Results are cached for 5 minutes.
        """
        # Fetch models dynamically
        openrouter_models = self._fetch_openrouter_models()

        # Fetch embedding models dynamically from OpenRouter
        embedding_models = self._fetch_openrouter_embedding_models()

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

    def delete_api_keys(
        self,
        key_type: str,  # "llm" or "exchange"
    ) -> SetAPIKeysResponse:
        """Delete API keys by type.

        Args:
            key_type: "llm" to delete LLM-related keys, "exchange" for exchange keys
        """
        deleted = []

        if key_type == "llm":
            # Delete LLM-related config
            if "api_key" in self._config:
                del self._config["api_key"]
                deleted.append("api_key")
            # ALWAYS clear from environment (even if not in config file)
            # API key might have been loaded from .env file via dotenv
            if "OPENROUTER_API_KEY" in os.environ:
                del os.environ["OPENROUTER_API_KEY"]
                if "api_key" not in deleted:
                    deleted.append("api_key (env)")
            if "LLM_API_KEY" in os.environ:
                del os.environ["LLM_API_KEY"]
                deleted.append("LLM_API_KEY (env)")
            if "model" in self._config:
                del self._config["model"]
                deleted.append("model")
            if "embedding_model" in self._config:
                del self._config["embedding_model"]
                deleted.append("embedding_model")
            if "provider" in self._config:
                del self._config["provider"]
                deleted.append("provider")

        elif key_type == "exchange":
            # Delete exchange-related config
            if "exchange_id" in self._config:
                del self._config["exchange_id"]
                deleted.append("exchange_id")
            if "exchange_api_key" in self._config:
                del self._config["exchange_api_key"]
                deleted.append("exchange_api_key")
            if "exchange_secret" in self._config:
                del self._config["exchange_secret"]
                deleted.append("exchange_secret")

        else:
            return SetAPIKeysResponse(
                success=False,
                message=f"Unknown key type: {key_type}. Use 'llm' or 'exchange'.",
            )

        if not deleted:
            return SetAPIKeysResponse(
                success=True,
                message=f"No {key_type} keys to delete.",
            )

        self._save_config()

        return SetAPIKeysResponse(
            success=True,
            message=f"Deleted: {', '.join(deleted)}",
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
            FrequencyOption(id="1m", name="1 Minute", description="1-minute candlestick"),
            FrequencyOption(id="5m", name="5 Minutes", description="5-minute candlestick"),
            FrequencyOption(id="15m", name="15 Minutes", description="15-minute candlestick"),
            FrequencyOption(id="1h", name="1 Hour", description="1-hour candlestick"),
            FrequencyOption(id="4h", name="4 Hours", description="4-hour candlestick"),
            FrequencyOption(id="1d", name="Daily", description="Daily candlestick"),
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
