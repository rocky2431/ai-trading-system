"""RL Training Utilities for Trading.

This module provides:
- Training loop for trading environments
- Evaluation utilities
- Logging and checkpointing
- Hyperparameter search integration
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .agents import AgentConfig, BaseAgent, RolloutBuffer, create_agent
from .envs import BaseTradingEnv, TradingEnvConfig, create_trading_env


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for RL training."""
    # Training duration
    total_timesteps: int = 100000
    n_eval_episodes: int = 10
    eval_freq: int = 5000  # Evaluate every N timesteps

    # Logging
    log_interval: int = 1000
    save_freq: int = 10000  # Save checkpoint every N timesteps
    save_path: Optional[str] = None

    # Early stopping
    early_stopping_patience: int = 10  # Stop after N evals without improvement
    min_improvement: float = 0.01  # Minimum improvement to count as progress

    # Curriculum learning
    use_curriculum: bool = False
    curriculum_stages: List[float] = field(default_factory=lambda: [0.2, 0.5, 0.8, 1.0])


@dataclass
class TrainingResult:
    """Result of training run."""
    final_reward: float
    best_reward: float
    total_timesteps: int
    training_time: float
    eval_rewards: List[float]
    eval_timesteps: List[int]
    final_metrics: Dict[str, float]


class RLTrainer:
    """Trainer for RL trading agents."""

    def __init__(
        self,
        env: BaseTradingEnv,
        agent: BaseAgent,
        config: Optional[TrainingConfig] = None,
        eval_env: Optional[BaseTradingEnv] = None,
    ):
        """Initialize trainer.

        Args:
            env: Training environment
            agent: RL agent
            config: Training configuration
            eval_env: Separate evaluation environment (optional)
        """
        self.env = env
        self.agent = agent
        self.config = config or TrainingConfig()
        self.eval_env = eval_env or env

        # Training state
        self.total_timesteps = 0
        self.best_reward = float("-inf")
        self.no_improvement_count = 0
        self.eval_rewards: List[float] = []
        self.eval_timesteps: List[int] = []

        # Curriculum state
        self.curriculum_stage = 0

    def _collect_rollout(self, n_steps: int) -> Tuple[RolloutBuffer, float]:
        """Collect experience from environment.

        Args:
            n_steps: Number of steps to collect

        Returns:
            (buffer, next_value)
        """
        buffer = RolloutBuffer()
        obs, _ = self.env.reset()

        for _ in range(n_steps):
            action, log_prob, value = self.agent.select_action(obs)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            buffer.add(obs, action, reward, value, log_prob, done)

            if done:
                obs, _ = self.env.reset()
            else:
                obs = next_obs

            self.total_timesteps += 1

        # Get next value for GAE
        with np.errstate(all="ignore"):
            _, _, next_value = self.agent.select_action(obs)

        return buffer, next_value

    def _evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate agent on evaluation environment.

        Args:
            n_episodes: Number of episodes to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        episode_pnls = []
        n_trades_list = []

        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                action, _, _ = self.agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_pnls.append(info.get("episode_pnl", 0))
            n_trades_list.append(info.get("n_trades", 0))

        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "mean_pnl": np.mean(episode_pnls),
            "mean_trades": np.mean(n_trades_list),
            "sharpe": np.mean(episode_rewards) / (np.std(episode_rewards) + 1e-8),
        }

    def _should_stop(self, current_reward: float) -> bool:
        """Check if training should stop early.

        Args:
            current_reward: Current evaluation reward

        Returns:
            True if should stop
        """
        if current_reward > self.best_reward + self.config.min_improvement:
            self.best_reward = current_reward
            self.no_improvement_count = 0
            return False
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.config.early_stopping_patience:
                logger.info(f"Early stopping: no improvement for {self.no_improvement_count} evaluations")
                return True
            return False

    def _update_curriculum(self):
        """Update curriculum learning stage."""
        if not self.config.use_curriculum:
            return

        if self.curriculum_stage >= len(self.config.curriculum_stages) - 1:
            return

        # Progress to next stage if reward improved
        if self.no_improvement_count == 0:
            next_stage = self.curriculum_stage + 1
            if next_stage < len(self.config.curriculum_stages):
                self.curriculum_stage = next_stage
                # Update environment difficulty (episode length)
                data_fraction = self.config.curriculum_stages[self.curriculum_stage]
                if hasattr(self.env, "config"):
                    original_length = self.env.config.episode_length or 1000
                    self.env.config.episode_length = int(original_length * data_fraction)
                logger.info(f"Curriculum: Stage {self.curriculum_stage}, data fraction {data_fraction}")

    def train(
        self,
        callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ) -> TrainingResult:
        """Run training loop.

        Args:
            callback: Optional callback(timesteps, metrics) called during training

        Returns:
            TrainingResult with training metrics
        """
        start_time = time.time()

        logger.info(f"Starting training for {self.config.total_timesteps} timesteps")

        while self.total_timesteps < self.config.total_timesteps:
            # Collect rollout
            buffer, next_value = self._collect_rollout(self.agent.config.n_steps)

            # Update agent
            update_metrics = self.agent.update(buffer, next_value)

            # Logging
            if self.total_timesteps % self.config.log_interval == 0:
                logger.info(
                    f"Timesteps: {self.total_timesteps}, "
                    f"Loss: {update_metrics.get('loss', 0):.4f}, "
                    f"Policy Loss: {update_metrics.get('policy_loss', 0):.4f}, "
                    f"Value Loss: {update_metrics.get('value_loss', 0):.4f}"
                )

            # Evaluation
            if self.total_timesteps % self.config.eval_freq == 0:
                eval_metrics = self._evaluate(self.config.n_eval_episodes)
                self.eval_rewards.append(eval_metrics["mean_reward"])
                self.eval_timesteps.append(self.total_timesteps)

                logger.info(
                    f"Eval at {self.total_timesteps}: "
                    f"Reward: {eval_metrics['mean_reward']:.4f} +/- {eval_metrics['std_reward']:.4f}, "
                    f"PnL: {eval_metrics['mean_pnl']:.4f}, "
                    f"Sharpe: {eval_metrics['sharpe']:.4f}"
                )

                if callback:
                    callback(self.total_timesteps, eval_metrics)

                # Early stopping check
                if self._should_stop(eval_metrics["mean_reward"]):
                    break

                # Curriculum update
                self._update_curriculum()

            # Save checkpoint
            if self.config.save_path and self.total_timesteps % self.config.save_freq == 0:
                checkpoint_path = f"{self.config.save_path}/checkpoint_{self.total_timesteps}.pt"
                self.agent.save(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

        training_time = time.time() - start_time

        # Final evaluation
        final_metrics = self._evaluate(self.config.n_eval_episodes)

        # Save final model
        if self.config.save_path:
            final_path = f"{self.config.save_path}/final_model.pt"
            self.agent.save(final_path)
            logger.info(f"Saved final model to {final_path}")

        return TrainingResult(
            final_reward=final_metrics["mean_reward"],
            best_reward=self.best_reward,
            total_timesteps=self.total_timesteps,
            training_time=training_time,
            eval_rewards=self.eval_rewards,
            eval_timesteps=self.eval_timesteps,
            final_metrics=final_metrics,
        )


def train_trading_agent(
    df: pd.DataFrame,
    agent_type: str = "ppo",
    total_timesteps: int = 100000,
    agent_config: Optional[AgentConfig] = None,
    env_config: Optional[TradingEnvConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    discrete_actions: bool = True,
    callback: Optional[Callable] = None,
) -> Tuple[BaseAgent, TrainingResult]:
    """Convenience function to train a trading agent.

    Args:
        df: Price data DataFrame
        agent_type: "ppo" or "a2c"
        total_timesteps: Total training timesteps
        agent_config: Agent configuration
        env_config: Environment configuration
        training_config: Training configuration
        discrete_actions: Use discrete actions
        callback: Training callback

    Returns:
        (trained_agent, training_result)
    """
    # Create environment
    env_config = env_config or TradingEnvConfig()
    env = create_trading_env(df, "single", env_config, discrete_actions=discrete_actions)

    # Create agent
    agent_config = agent_config or AgentConfig()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if discrete_actions else env.action_space.shape[0]
    agent = create_agent(agent_type, obs_dim, action_dim, agent_config, discrete_actions)

    # Create trainer
    training_config = training_config or TrainingConfig(total_timesteps=total_timesteps)
    trainer = RLTrainer(env, agent, training_config)

    # Train
    result = trainer.train(callback)

    return agent, result


def backtest_agent(
    agent: BaseAgent,
    df: pd.DataFrame,
    env_config: Optional[TradingEnvConfig] = None,
    discrete_actions: bool = True,
) -> Dict[str, Any]:
    """Backtest a trained agent.

    Args:
        agent: Trained RL agent
        df: Price data DataFrame
        env_config: Environment configuration
        discrete_actions: Use discrete actions

    Returns:
        Backtest results dictionary
    """
    env_config = env_config or TradingEnvConfig()
    env_config.episode_length = len(df) - env_config.lookback_window - 1

    env = create_trading_env(df, "single", env_config, discrete_actions=discrete_actions)

    obs, _ = env.reset()
    done = False

    actions = []
    rewards = []
    portfolio_values = []
    positions = []

    while not done:
        action, _, _ = agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        actions.append(action)
        rewards.append(reward)
        portfolio_values.append(info.get("portfolio_value", 0))
        positions.append(info.get("position", 0))

    # Calculate metrics
    returns = np.diff(portfolio_values) / (np.array(portfolio_values[:-1]) + 1e-10)
    cumulative_return = (portfolio_values[-1] / portfolio_values[0] - 1) if portfolio_values else 0

    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)  # Annualized

    max_drawdown = 0
    peak = portfolio_values[0]
    for pv in portfolio_values:
        if pv > peak:
            peak = pv
        drawdown = (peak - pv) / peak
        max_drawdown = max(max_drawdown, drawdown)

    return {
        "total_return": cumulative_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "n_trades": info.get("n_trades", 0),
        "final_pnl": info.get("total_pnl", 0),
        "portfolio_values": portfolio_values,
        "positions": positions,
        "actions": actions,
    }
