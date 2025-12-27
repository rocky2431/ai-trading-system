"""Prompts API Service.

Service layer for managing LLM prompt templates and system configuration.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from iqfmp.api.prompts.schemas import (
    PromptHistoryEntry,
    PromptHistoryList,
    PromptTemplate,
    PromptTemplateList,
    SystemModeConfig,
    SystemModeConfigResponse,
)

logger = logging.getLogger(__name__)


# Default prompt templates by agent
DEFAULT_PROMPTS: dict[str, dict[str, Any]] = {
    "factor_generation": {
        "agent_name": "Factor Generation",
        "prompt_id": "factor_gen_v1",
        "version": "1.2.0",
        "description": "Generates Qlib-compatible alpha factor code",
        "system_prompt": """You are an expert quantitative researcher specializing in cryptocurrency alpha factor development.

Your task is to generate Python code that computes alpha factors using the Qlib framework.

## Requirements:
1. Use only pandas operations (no external data sources)
2. Handle missing data gracefully with .fillna() or .dropna()
3. Return a pd.Series with the same index as input
4. Document the factor's economic rationale
5. Consider crypto-specific patterns:
   - 24/7 trading (no market close effects)
   - Funding rate dynamics
   - Liquidation cascades
   - Exchange-specific behaviors

## Output Format:
```python
def calculate(data: pd.DataFrame) -> pd.Series:
    '''
    Factor Name: <name>
    Description: <what it measures>
    Hypothesis: <why it should predict returns>
    '''
    # Implementation
    return result
```""",
    },
    "factor_evaluation": {
        "agent_name": "Factor Evaluation",
        "prompt_id": "factor_eval_v1",
        "version": "1.1.0",
        "description": "Evaluates factor quality metrics",
        "system_prompt": """You are an expert quantitative analyst specializing in factor evaluation.

Your task is to analyze alpha factors and provide comprehensive evaluation.

## Evaluation Dimensions:
1. **Statistical Quality**: IC, IR, t-statistic
2. **Economic Intuition**: Does the factor make economic sense?
3. **Robustness**: Cross-validation, out-of-sample performance
4. **Decay Analysis**: How quickly does alpha decay?
5. **Correlation**: Correlation with existing factors
6. **Turnover**: Trading costs impact

## Output Requirements:
- Provide quantitative metrics
- Flag potential overfitting risks
- Suggest improvements if applicable
- Compare to benchmark factors""",
    },
    "strategy_assembly": {
        "agent_name": "Strategy Assembly",
        "prompt_id": "strategy_v1",
        "version": "1.0.0",
        "description": "Assembles trading strategies from factors",
        "system_prompt": """You are an expert portfolio manager specializing in cryptocurrency trading strategies.

Your task is to assemble optimal trading strategies from validated alpha factors.

## Strategy Design Principles:
1. **Diversification**: Combine uncorrelated factors
2. **Risk Management**: Position sizing and stop-losses
3. **Execution**: Consider slippage and market impact
4. **Crypto-Specific**:
   - Funding rate optimization
   - Cross-exchange arbitrage
   - Liquidation avoidance

## Output Requirements:
- Factor weights and rebalancing frequency
- Risk parameters (max position, stop-loss)
- Expected performance metrics
- Implementation notes""",
    },
    "backtest_optimization": {
        "agent_name": "Backtest Optimization",
        "prompt_id": "backtest_v1",
        "version": "1.0.0",
        "description": "Optimizes backtest parameters",
        "system_prompt": """You are an expert in backtesting and strategy optimization.

Your task is to optimize strategy parameters and validate performance.

## Optimization Guidelines:
1. **Avoid Overfitting**: Use cross-validation
2. **Transaction Costs**: Include realistic costs
3. **Slippage Model**: Account for market impact
4. **Walk-Forward**: Use out-of-sample testing

## Output Requirements:
- Optimized parameters with confidence intervals
- Performance metrics (Sharpe, Sortino, Max DD)
- Sensitivity analysis
- Recommendations for live trading""",
    },
    "risk_check": {
        "agent_name": "Risk Check",
        "prompt_id": "risk_v1",
        "version": "1.0.0",
        "description": "Performs risk assessment",
        "system_prompt": """You are an expert risk manager for cryptocurrency trading systems.

Your task is to assess and validate strategy risk profiles.

## Risk Dimensions:
1. **Market Risk**: Volatility, tail risk
2. **Liquidity Risk**: Slippage, execution risk
3. **Operational Risk**: System failures
4. **Crypto-Specific**:
   - Exchange counterparty risk
   - Smart contract risk
   - Regulatory risk

## Output Requirements:
- Risk score (1-10)
- Key risk factors identified
- Mitigation recommendations
- Go/No-Go decision with rationale""",
    },
    "hypothesis": {
        "agent_name": "Hypothesis Generation",
        "prompt_id": "hypothesis_v1",
        "version": "1.0.0",
        "description": "Generates trading hypotheses",
        "system_prompt": """You are a creative quantitative researcher generating novel trading hypotheses.

Your task is to propose testable alpha hypotheses based on market observations.

## Hypothesis Guidelines:
1. **Novelty**: Avoid well-known factors
2. **Testability**: Must be implementable with available data
3. **Economic Rationale**: Clear causal mechanism
4. **Crypto Focus**:
   - On-chain data signals
   - Derivative market dynamics
   - Social sentiment indicators

## Output Requirements:
- Clear hypothesis statement
- Expected data requirements
- Predicted signal characteristics
- Testing methodology""",
    },
}


class PromptsService:
    """Service for managing prompt templates."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._config_dir = Path.home() / ".iqfmp"
        self._config_file = self._config_dir / "config.json"
        self._history_file = self._config_dir / "prompt_history.json"
        self._ensure_config_dir()
        logger.info("PromptsService initialized")

    def _ensure_config_dir(self) -> None:
        """Ensure config directory exists."""
        self._config_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> dict:
        """Load configuration from file."""
        if self._config_file.exists():
            with open(self._config_file) as f:
                return json.load(f)
        return {}

    def _save_config(self, config: dict) -> None:
        """Save configuration to file."""
        with open(self._config_file, "w") as f:
            json.dump(config, f, indent=2)

    def _load_history(self) -> list[dict]:
        """Load prompt history from file."""
        if self._history_file.exists():
            with open(self._history_file) as f:
                return json.load(f)
        return []

    def _save_history(self, history: list[dict]) -> None:
        """Save prompt history to file."""
        with open(self._history_file, "w") as f:
            json.dump(history, f, indent=2, default=str)

    def get_templates(self) -> PromptTemplateList:
        """Get all prompt templates with custom overrides."""
        config = self._load_config()
        agents_config = config.get("agents", {})

        templates = []
        for agent_id, default in DEFAULT_PROMPTS.items():
            agent_config = agents_config.get(agent_id, {})
            custom_prompt = agent_config.get("system_prompt")

            templates.append(
                PromptTemplate(
                    agent_id=agent_id,
                    agent_name=default["agent_name"],
                    prompt_id=default["prompt_id"],
                    version=default["version"],
                    system_prompt=custom_prompt or default["system_prompt"],
                    description=default["description"],
                    is_custom=bool(custom_prompt),
                )
            )

        return PromptTemplateList(templates=templates, total=len(templates))

    def get_template(self, agent_id: str) -> PromptTemplate | None:
        """Get a specific prompt template."""
        if agent_id not in DEFAULT_PROMPTS:
            return None

        default = DEFAULT_PROMPTS[agent_id]
        config = self._load_config()
        agent_config = config.get("agents", {}).get(agent_id, {})
        custom_prompt = agent_config.get("system_prompt")

        return PromptTemplate(
            agent_id=agent_id,
            agent_name=default["agent_name"],
            prompt_id=default["prompt_id"],
            version=default["version"],
            system_prompt=custom_prompt or default["system_prompt"],
            description=default["description"],
            is_custom=bool(custom_prompt),
        )

    def get_default_template(self, agent_id: str) -> PromptTemplate | None:
        """Get the default (non-custom) template."""
        if agent_id not in DEFAULT_PROMPTS:
            return None

        default = DEFAULT_PROMPTS[agent_id]
        return PromptTemplate(
            agent_id=agent_id,
            agent_name=default["agent_name"],
            prompt_id=default["prompt_id"],
            version=default["version"],
            system_prompt=default["system_prompt"],
            description=default["description"],
            is_custom=False,
        )

    def update_template(
        self, agent_id: str, system_prompt: str | None
    ) -> PromptTemplate | None:
        """Update a prompt template with custom override.

        Args:
            agent_id: Agent identifier
            system_prompt: Custom prompt (None or empty string to reset)

        Returns:
            Updated template or None if agent not found
        """
        if agent_id not in DEFAULT_PROMPTS:
            return None

        config = self._load_config()
        if "agents" not in config:
            config["agents"] = {}
        if agent_id not in config["agents"]:
            config["agents"][agent_id] = {}

        # Get old prompt for history
        old_prompt = config["agents"][agent_id].get("system_prompt")

        # Update or reset
        if system_prompt and system_prompt.strip():
            config["agents"][agent_id]["system_prompt"] = system_prompt
            change_type = "updated" if old_prompt else "created"
        else:
            config["agents"][agent_id].pop("system_prompt", None)
            change_type = "reset"

        self._save_config(config)

        # Record history
        self._record_history(
            agent_id=agent_id,
            old_prompt=old_prompt,
            new_prompt=system_prompt if system_prompt else None,
            change_type=change_type,
        )

        return self.get_template(agent_id)

    def _record_history(
        self,
        agent_id: str,
        old_prompt: str | None,
        new_prompt: str | None,
        change_type: str,
    ) -> None:
        """Record prompt change in history."""
        history = self._load_history()
        history.append(
            {
                "id": f"hist_{uuid.uuid4().hex[:12]}",
                "agent_id": agent_id,
                "old_prompt": old_prompt,
                "new_prompt": new_prompt,
                "changed_by": "api",
                "changed_at": datetime.utcnow().isoformat(),
                "change_type": change_type,
            }
        )
        # Keep last 100 entries
        if len(history) > 100:
            history = history[-100:]
        self._save_history(history)

    def get_history(
        self, agent_id: str | None = None, limit: int = 50
    ) -> PromptHistoryList:
        """Get prompt change history."""
        history = self._load_history()

        if agent_id:
            history = [h for h in history if h["agent_id"] == agent_id]

        # Sort by date descending
        history = sorted(history, key=lambda x: x["changed_at"], reverse=True)[:limit]

        entries = [
            PromptHistoryEntry(
                id=h["id"],
                agent_id=h["agent_id"],
                old_prompt=h.get("old_prompt"),
                new_prompt=h.get("new_prompt"),
                changed_by=h.get("changed_by", "system"),
                changed_at=datetime.fromisoformat(h["changed_at"]),
                change_type=h["change_type"],
            )
            for h in history
        ]

        return PromptHistoryList(entries=entries, total=len(entries))

    def get_system_mode(self) -> SystemModeConfigResponse:
        """Get system mode configuration."""
        config = self._load_config()
        system_mode = config.get("system_mode", {})

        current = SystemModeConfig(
            strict_mode_enabled=system_mode.get("strict_mode_enabled", True),
            vector_strict_mode=system_mode.get("vector_strict_mode", True),
            sandbox_enabled=system_mode.get("sandbox_enabled", True),
            sandbox_timeout_seconds=system_mode.get("sandbox_timeout_seconds", 60),
            sandbox_memory_limit_mb=system_mode.get("sandbox_memory_limit_mb", 512),
            sandbox_network_allowed=system_mode.get("sandbox_network_allowed", False),
            human_review_enabled=system_mode.get("human_review_enabled", True),
            auto_reject_timeout_seconds=system_mode.get(
                "auto_reject_timeout_seconds", 3600
            ),
            ml_signal_enabled=system_mode.get("ml_signal_enabled", False),
            tool_context_enabled=system_mode.get("tool_context_enabled", False),
            checkpoint_enabled=system_mode.get("checkpoint_enabled", True),
        )

        return SystemModeConfigResponse(
            config=current,
            defaults=SystemModeConfig(),  # Default values
        )

    def update_system_mode(self, updates: dict) -> SystemModeConfigResponse:
        """Update system mode configuration."""
        config = self._load_config()
        if "system_mode" not in config:
            config["system_mode"] = {}

        # Update only provided fields
        for key, value in updates.items():
            if value is not None:
                config["system_mode"][key] = value

        self._save_config(config)
        return self.get_system_mode()


# Singleton instance
prompts_service = PromptsService()
