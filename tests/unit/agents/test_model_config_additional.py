"""Additional tests for model configuration registry."""

from __future__ import annotations

from iqfmp.agents import model_config
from iqfmp.agents.model_config import AgentModelRegistry, AgentType


def test_get_agent_model_config_defaults() -> None:
    model_id, temperature = model_config.get_agent_model_config("factor_generation")
    assert isinstance(model_id, str)
    assert 0.0 <= temperature <= 1.0


def test_registry_set_and_get() -> None:
    registry = AgentModelRegistry(load_from_config_service=False)
    config = registry.get_config(AgentType.HYPOTHESIS)
    assert config is not None

    registry.set_config(AgentType.HYPOTHESIS, config)
    assert registry.get_config(AgentType.HYPOTHESIS).model_id == config.model_id


def test_registry_helpers() -> None:
    registry = AgentModelRegistry(load_from_config_service=False)
    model_id = registry.get_model_id(AgentType.EVALUATION)
    temperature = registry.get_temperature(AgentType.EVALUATION)
    assert isinstance(model_id, str)
    assert isinstance(temperature, float)


def test_unknown_agent_id_fallback() -> None:
    model_id, temperature = model_config.get_agent_model_config("unknown_agent")
    assert model_id
    assert temperature == 0.7


def test_get_agent_full_config() -> None:
    model_id, temperature, system_prompt = model_config.get_agent_full_config("factor_generation")
    assert isinstance(model_id, str)
    assert system_prompt is None or isinstance(system_prompt, str)
