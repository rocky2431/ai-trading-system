"""IQFMP Reinforcement Learning Module.

This module provides:
- Trading environments (single asset, portfolio)
- RL agents (PPO, A2C)
- Training utilities and evaluation tools

Example usage:

```python
from iqfmp.rl import (
    SingleAssetTradingEnv,
    TradingEnvConfig,
    PPOAgent,
    AgentConfig,
    train_trading_agent,
    backtest_agent,
)

# Quick training
agent, result = train_trading_agent(
    df=price_data,
    agent_type="ppo",
    total_timesteps=50000,
)

# Backtest
metrics = backtest_agent(agent, test_data)
print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
```
"""

from .envs import (
    # Environments
    TradingAction,
    TradingEnvConfig,
    BaseTradingEnv,
    SingleAssetTradingEnv,
    PortfolioEnv,
    create_trading_env,
    # Availability
    GYM_AVAILABLE,
)

from .agents import (
    # Config and buffer
    AgentConfig,
    RolloutBuffer,
    # Networks
    MLP,
    ActorCriticNetwork,
    # Agents
    BaseAgent,
    PPOAgent,
    A2CAgent,
    create_agent,
    # Availability
    TORCH_AVAILABLE,
)

from .trainer import (
    # Config and results
    TrainingConfig,
    TrainingResult,
    # Trainer
    RLTrainer,
    # Convenience functions
    train_trading_agent,
    backtest_agent,
)

__all__ = [
    # Environment classes
    "TradingAction",
    "TradingEnvConfig",
    "BaseTradingEnv",
    "SingleAssetTradingEnv",
    "PortfolioEnv",
    "create_trading_env",
    # Agent classes
    "AgentConfig",
    "RolloutBuffer",
    "MLP",
    "ActorCriticNetwork",
    "BaseAgent",
    "PPOAgent",
    "A2CAgent",
    "create_agent",
    # Training classes
    "TrainingConfig",
    "TrainingResult",
    "RLTrainer",
    # Convenience functions
    "train_trading_agent",
    "backtest_agent",
    # Availability flags
    "GYM_AVAILABLE",
    "TORCH_AVAILABLE",
]
