"""Agent Model Configuration for IQFMP.

Centralized LLM model configuration for all agents.
Integrates with ConfigService to support frontend-selected models via OpenRouter API.

Model Selection Rationale:
- DeepSeek V3: Best for code generation (factor_generation, backtest)
- Claude Sonnet 4: Best for reasoning and creativity (hypothesis, strategy)
- GPT-4o: Best for analysis and judgment (evaluation, risk)
- Gemini 2.5 Flash: Fast with 1M context (risk check)

Usage:
    # Get model for an agent (checks ConfigService first, then falls back to defaults)
    from iqfmp.agents.model_config import get_agent_model_config, get_agent_full_config

    # Basic usage - get model ID and temperature
    model_id, temperature = get_agent_model_config("factor_generation")
    # model_id is OpenRouter model ID like "deepseek/deepseek-coder-v3"

    # Full config - includes custom system prompt from frontend
    model_id, temperature, system_prompt = get_agent_full_config("factor_generation")
    # system_prompt is None if using default, or custom prompt string
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    """Agent types in IQFMP pipeline."""

    HYPOTHESIS = "hypothesis"
    FACTOR_GENERATION = "factor_generation"
    EVALUATION = "factor_evaluation"  # Match ConfigService naming
    STRATEGY = "strategy_assembly"  # Match ConfigService naming
    BACKTEST = "backtest_optimization"  # Match ConfigService naming
    RISK = "risk_check"  # Match ConfigService naming


# Mapping from ConfigService agent IDs to AgentType
AGENT_ID_MAP = {
    "hypothesis": AgentType.HYPOTHESIS,
    "factor_generation": AgentType.FACTOR_GENERATION,
    "factor_evaluation": AgentType.EVALUATION,
    "strategy_assembly": AgentType.STRATEGY,
    "backtest_optimization": AgentType.BACKTEST,
    "risk_check": AgentType.RISK,
}


@dataclass
class AgentModelConfig:
    """Configuration for agent-specific LLM model settings.

    Attributes:
        agent_type: Type of agent
        model_id: OpenRouter model ID (e.g., "deepseek/deepseek-coder-v3")
        temperature: Sampling temperature (lower for code, higher for creativity)
        max_tokens: Maximum response tokens
        description: Why this model is chosen for this agent
        system_prompt: Custom system prompt override (None = use default)
    """

    agent_type: AgentType
    model_id: str  # OpenRouter model ID
    temperature: float = 0.7
    max_tokens: int = 4096
    description: str = ""
    system_prompt: Optional[str] = None  # Custom system prompt (None = use default)


# Default model configurations for each agent type (OpenRouter model IDs)
# These are used as fallbacks when no frontend config exists
DEFAULT_AGENT_MODELS: dict[AgentType, AgentModelConfig] = {
    AgentType.HYPOTHESIS: AgentModelConfig(
        agent_type=AgentType.HYPOTHESIS,
        model_id="anthropic/claude-sonnet-4",
        temperature=0.9,  # Higher for creativity
        max_tokens=2048,
        description="Claude excels at creative hypothesis generation and reasoning",
    ),
    AgentType.FACTOR_GENERATION: AgentModelConfig(
        agent_type=AgentType.FACTOR_GENERATION,
        model_id="deepseek/deepseek-coder-v3",
        temperature=0.3,  # Lower for precise code generation
        max_tokens=4096,
        description="DeepSeek Coder V3 is optimized for code generation with Qlib syntax",
    ),
    AgentType.EVALUATION: AgentModelConfig(
        agent_type=AgentType.EVALUATION,
        model_id="deepseek/deepseek-r1",
        temperature=0.5,
        max_tokens=2048,
        description="DeepSeek R1 provides strong reasoning for factor evaluation",
    ),
    AgentType.STRATEGY: AgentModelConfig(
        agent_type=AgentType.STRATEGY,
        model_id="anthropic/claude-sonnet-4",
        temperature=0.7,
        max_tokens=3072,
        description="Claude is strong at strategic planning and portfolio design",
    ),
    AgentType.BACKTEST: AgentModelConfig(
        agent_type=AgentType.BACKTEST,
        model_id="openai/gpt-4.1",
        temperature=0.3,
        max_tokens=4096,
        description="GPT-4.1 for efficient backtest code and optimization",
    ),
    AgentType.RISK: AgentModelConfig(
        agent_type=AgentType.RISK,
        model_id="google/gemini-2.5-flash",
        temperature=0.4,  # Conservative for risk assessment
        max_tokens=2048,
        description="Gemini 2.5 Flash for fast risk analysis with 1M context",
    ),
}


def get_config_service():
    """Get ConfigService singleton (lazy import to avoid circular imports)."""
    try:
        from iqfmp.api.config.service import get_config_service as _get_config_service
        return _get_config_service()
    except ImportError:
        logger.warning("ConfigService not available, using default configs")
        return None


class AgentModelRegistry:
    """Registry for managing agent model configurations.

    Integrates with ConfigService to load frontend-configured models.
    Falls back to defaults if no config exists.

    Usage:
        registry = AgentModelRegistry()

        # Get config (checks ConfigService first)
        config = registry.get_config(AgentType.FACTOR_GENERATION)

        # Override for specific agent
        registry.set_config(AgentType.FACTOR_GENERATION, AgentModelConfig(
            agent_type=AgentType.FACTOR_GENERATION,
            model_id="openai/gpt-4o",
            temperature=0.2,
        ))
    """

    def __init__(self, load_from_config_service: bool = True) -> None:
        """Initialize registry with default configurations.

        Args:
            load_from_config_service: If True, load configs from ConfigService on init
        """
        self._configs: dict[AgentType, AgentModelConfig] = DEFAULT_AGENT_MODELS.copy()
        self._config_service = None

        if load_from_config_service:
            self._load_from_config_service()

    def _load_from_config_service(self) -> None:
        """Load agent configurations from ConfigService.

        Loads model_id, temperature, and system_prompt from ConfigService.
        Falls back to defaults if ConfigService is unavailable or values are not set.
        """
        try:
            self._config_service = get_config_service()
            if self._config_service is None:
                return

            agent_config = self._config_service.get_agent_config()

            for agent in agent_config.agents:
                # Map agent_id to AgentType
                agent_type = AGENT_ID_MAP.get(agent.agent_id)
                if agent_type is None:
                    continue

                # Only update if a model is configured and agent is enabled
                if agent.model_id and agent.enabled:
                    default_config = DEFAULT_AGENT_MODELS[agent_type]

                    # Use ConfigService temperature if set, otherwise use default
                    temperature = (
                        agent.temperature
                        if hasattr(agent, "temperature") and agent.temperature is not None
                        else default_config.temperature
                    )

                    # Use ConfigService system_prompt if set (None means use default)
                    system_prompt = (
                        agent.system_prompt
                        if hasattr(agent, "system_prompt")
                        else None
                    )

                    self._configs[agent_type] = AgentModelConfig(
                        agent_type=agent_type,
                        model_id=agent.model_id,
                        temperature=temperature,
                        max_tokens=default_config.max_tokens,
                        description=agent.description or default_config.description,
                        system_prompt=system_prompt,
                    )
                    logger.info(
                        f"Loaded config for {agent.agent_id}: "
                        f"model={agent.model_id}, temp={temperature}, "
                        f"prompt={'custom' if system_prompt else 'default'}"
                    )

        except Exception as e:
            logger.warning(f"Failed to load from ConfigService: {e}, using defaults")

    def reload_from_config_service(self) -> None:
        """Force reload configurations from ConfigService."""
        self._configs = DEFAULT_AGENT_MODELS.copy()
        self._load_from_config_service()

    def get_config(self, agent_type: AgentType) -> AgentModelConfig:
        """Get model configuration for an agent type.

        Args:
            agent_type: Type of agent

        Returns:
            Model configuration for the agent
        """
        return self._configs.get(agent_type, DEFAULT_AGENT_MODELS[agent_type])

    def set_config(self, agent_type: AgentType, config: AgentModelConfig) -> None:
        """Set custom model configuration for an agent type.

        Args:
            agent_type: Type of agent
            config: Custom model configuration
        """
        self._configs[agent_type] = config

    def get_model_id(self, agent_type: AgentType) -> str:
        """Get OpenRouter model ID for an agent.

        Args:
            agent_type: Type of agent

        Returns:
            OpenRouter model ID (e.g., "deepseek/deepseek-coder-v3")
        """
        return self.get_config(agent_type).model_id

    def get_temperature(self, agent_type: AgentType) -> float:
        """Get temperature setting for an agent.

        Args:
            agent_type: Type of agent

        Returns:
            Temperature value for the agent
        """
        return self.get_config(agent_type).temperature

    def get_system_prompt(self, agent_type: AgentType) -> Optional[str]:
        """Get custom system prompt for an agent.

        Args:
            agent_type: Type of agent

        Returns:
            Custom system prompt or None if using default
        """
        return self.get_config(agent_type).system_prompt

    def reset_to_defaults(self) -> None:
        """Reset all configurations to defaults."""
        self._configs = DEFAULT_AGENT_MODELS.copy()

    def get_all_configs(self) -> dict[AgentType, AgentModelConfig]:
        """Get all current configurations.

        Returns:
            Dictionary of all agent configurations
        """
        return self._configs.copy()

    def is_agent_enabled(self, agent_id: str) -> bool:
        """Check if an agent is enabled in ConfigService.

        Args:
            agent_id: Agent ID string (e.g., "factor_generation")

        Returns:
            True if enabled, False otherwise
        """
        if self._config_service is None:
            return True  # Default to enabled

        try:
            agent_config = self._config_service.get_agent_config()
            for agent in agent_config.agents:
                if agent.agent_id == agent_id:
                    return agent.enabled
            return True
        except Exception:
            return True


# Global registry instance (singleton pattern)
_global_registry: Optional[AgentModelRegistry] = None


def get_model_registry() -> AgentModelRegistry:
    """Get the global agent model registry.

    Returns:
        Global AgentModelRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentModelRegistry()
    return _global_registry


def reload_model_registry() -> None:
    """Reload the global registry from ConfigService.

    Call this after frontend config changes to pick up new settings.
    """
    global _global_registry
    if _global_registry is not None:
        _global_registry.reload_from_config_service()
    else:
        _global_registry = AgentModelRegistry()


def get_agent_model(agent_type: AgentType) -> str:
    """Convenience function to get OpenRouter model ID for an agent type.

    Args:
        agent_type: Type of agent

    Returns:
        OpenRouter model ID for the agent
    """
    return get_model_registry().get_model_id(agent_type)


def get_agent_temperature(agent_type: AgentType) -> float:
    """Convenience function to get temperature for an agent type.

    Args:
        agent_type: Type of agent

    Returns:
        Temperature value for the agent
    """
    return get_model_registry().get_temperature(agent_type)


def get_agent_model_config(agent_id: str) -> tuple[str, float]:
    """Get model configuration for an agent by ID string.

    This is the main entry point for agents to get their configured model.

    Args:
        agent_id: Agent ID string (e.g., "factor_generation")

    Returns:
        Tuple of (model_id, temperature)

    Example:
        model_id, temperature = get_agent_model_config("factor_generation")
        # model_id = "deepseek/deepseek-coder-v3"
        # temperature = 0.3
    """
    agent_type = AGENT_ID_MAP.get(agent_id)
    if agent_type is None:
        # Try to find by matching enum value
        for at in AgentType:
            if at.value == agent_id:
                agent_type = at
                break

    if agent_type is None:
        logger.warning(f"Unknown agent_id: {agent_id}, using default")
        return "deepseek/deepseek-chat", 0.7

    config = get_model_registry().get_config(agent_type)
    return config.model_id, config.temperature


def get_agent_full_config(agent_id: str) -> tuple[str, float, Optional[str]]:
    """Get full model configuration for an agent by ID string.

    This function returns all configurable settings including system_prompt.
    Use this when you need to apply custom prompts from frontend configuration.

    Args:
        agent_id: Agent ID string (e.g., "factor_generation")

    Returns:
        Tuple of (model_id, temperature, system_prompt)
        system_prompt is None if using default

    Example:
        model_id, temperature, system_prompt = get_agent_full_config("factor_generation")
        # model_id = "deepseek/deepseek-coder-v3"
        # temperature = 0.3
        # system_prompt = None or "Custom prompt..."
    """
    agent_type = AGENT_ID_MAP.get(agent_id)
    if agent_type is None:
        # Try to find by matching enum value
        for at in AgentType:
            if at.value == agent_id:
                agent_type = at
                break

    if agent_type is None:
        logger.warning(f"Unknown agent_id: {agent_id}, using default")
        return "deepseek/deepseek-chat", 0.7, None

    config = get_model_registry().get_config(agent_type)
    return config.model_id, config.temperature, config.system_prompt


def get_agent_system_prompt(agent_id: str) -> Optional[str]:
    """Convenience function to get custom system prompt for an agent.

    Args:
        agent_id: Agent ID string (e.g., "factor_generation")

    Returns:
        Custom system prompt or None if using default
    """
    agent_type = AGENT_ID_MAP.get(agent_id)
    if agent_type is None:
        for at in AgentType:
            if at.value == agent_id:
                agent_type = at
                break

    if agent_type is None:
        return None

    return get_model_registry().get_system_prompt(agent_type)
