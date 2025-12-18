"""IQFMP Prompt Templates for LLM Agents.

This module provides crypto-optimized prompt templates for various agents:
- FactorGenerationPrompt: Generate quantitative factors
- HypothesisGenerationPrompt: Generate trading hypotheses
- StrategyGenerationPrompt: Design trading strategies
- RiskManagementPrompt: Risk analysis and recommendations

Example usage:
    from iqfmp.llm.prompts import FactorGenerationPrompt

    prompt = FactorGenerationPrompt()
    system = prompt.get_system_prompt()
    user_prompt = prompt.render(
        user_request="Create a funding rate momentum factor",
        factor_family="funding",
    )
"""

from .base import (
    AgentType,
    BasePromptTemplate,
    CryptoDataFields,
    CryptoMarketContext,
    CryptoMarketType,
    PromptRole,
)
from .factor_generation import (
    FactorGenerationPrompt,
    FactorRefinementPrompt,
)
from .hypothesis import (
    FeedbackAnalysisPrompt,
    HypothesisGenerationPrompt,
    HypothesisToCodePrompt,
    ResearchPlanPrompt,
)
from .strategy import (
    BacktestAnalysisPrompt,
    RiskManagementPrompt,
    StrategyGenerationPrompt,
)

__all__ = [
    # Base classes
    "AgentType",
    "BasePromptTemplate",
    "CryptoDataFields",
    "CryptoMarketContext",
    "CryptoMarketType",
    "PromptRole",
    # Factor generation
    "FactorGenerationPrompt",
    "FactorRefinementPrompt",
    # Hypothesis
    "HypothesisGenerationPrompt",
    "HypothesisToCodePrompt",
    "FeedbackAnalysisPrompt",
    "ResearchPlanPrompt",
    # Strategy and risk
    "StrategyGenerationPrompt",
    "RiskManagementPrompt",
    "BacktestAnalysisPrompt",
]


def get_prompt_for_agent(agent_type: AgentType) -> BasePromptTemplate:
    """Factory function to get appropriate prompt for an agent type.

    Args:
        agent_type: Type of agent

    Returns:
        Appropriate prompt template instance
    """
    prompts = {
        AgentType.FACTOR_GENERATION: FactorGenerationPrompt,
        AgentType.HYPOTHESIS: HypothesisGenerationPrompt,
        AgentType.STRATEGY: StrategyGenerationPrompt,
        AgentType.RISK: RiskManagementPrompt,
        AgentType.BACKTEST: BacktestAnalysisPrompt,
    }

    prompt_class = prompts.get(agent_type)
    if prompt_class is None:
        raise ValueError(f"Unknown agent type: {agent_type}")

    return prompt_class()
