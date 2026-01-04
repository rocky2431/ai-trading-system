"""IQFMP Reinforcement Learning Module.

⚠️ EXPERIMENTAL/FUTURE FEATURE - NOT YET INTEGRATED INTO MAIN PIPELINE ⚠️

This module provides:
- Trading environments (single asset, portfolio)
- RL agents (PPO, A2C, SAC)
- Training utilities and evaluation tools

STATUS: This module is IMPLEMENTED but NOT CONNECTED to the main agent pipeline.
The main pipeline uses rule-based feedback loops instead of RL.

INTEGRATION PLAN (P2 Priority):
1. Replace fixed slippage with RL-learned execution (Qlib SOP)
2. Use RL for dynamic position sizing
3. Integrate with Qlib's RL framework (qlib.contrib.rl)

See: architecture.md for integration roadmap.

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

import warnings

# Issue deprecation warning when module is imported
warnings.warn(
    "iqfmp.rl module is EXPERIMENTAL and not yet integrated into the main pipeline. "
    "The main pipeline currently uses rule-based feedback loops. "
    "RL integration is planned for P2 priority. "
    "See architecture.md for the integration roadmap.",
    category=FutureWarning,
    stacklevel=2,
)

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
    ReplayBuffer,
    # Networks
    MLP,
    ActorCriticNetwork,
    GaussianActor,
    TwinQNetwork,
    # Agents
    BaseAgent,
    PPOAgent,
    A2CAgent,
    SACAgent,
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
    "ReplayBuffer",
    "MLP",
    "ActorCriticNetwork",
    "GaussianActor",
    "TwinQNetwork",
    "BaseAgent",
    "PPOAgent",
    "A2CAgent",
    "SACAgent",
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
