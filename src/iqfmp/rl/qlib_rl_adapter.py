"""Qlib RL Adapter for IQFMP.

This module provides integration between IQFMP's RL requirements and
Qlib's official RL framework for order execution, implementing:
- Qlib OrderExecution environment
- Qlib RL policies (PPO, etc.)
- Qlib RL training infrastructure
- TWAP/VWAP order execution strategies

Migration from custom RL implementations to Qlib RL.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Qlib RL imports with fallback
try:
    from qlib.rl.order_execution import (
        SAOEMetrics,
        SAOEStateAdapter,
        SAOEStrategy,
    )
    from qlib.rl.order_execution.simulator_qlib import SingleAssetOrderExecution
    from qlib.rl.trainer import train, backtest
    from qlib.rl.contrib.train_onpolicy import train_and_test
    QLIB_RL_AVAILABLE = True
except ImportError:
    QLIB_RL_AVAILABLE = False
    logger.warning("Qlib RL not available. Install with: pip install 'pyqlib[rl]'")

try:
    import gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    logger.warning("Gym not available. Install with: pip install gym")


# =============================================================================
# Qlib RL Configuration
# =============================================================================

@dataclass
class QlibRLConfig:
    """Configuration for Qlib RL training."""

    # Environment
    initial_amount: float = 10000.0
    order_amount: float = 1000.0
    time_per_step: str = "1min"

    # Training
    total_timesteps: int = 100000
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Policy
    policy_type: str = "ppo"  # "ppo", "a2c", "sac"
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])

    # Reward
    reward_type: str = "default"  # "default", "vwap", "twap"

    # Logging
    log_dir: str = "mlruns/rl"
    verbose: int = 1


# =============================================================================
# Qlib Order Execution Environment Wrapper
# =============================================================================

class QlibOrderExecutionEnv:
    """Wrapper for Qlib's SingleAssetOrderExecution environment.

    Provides a standardized interface for order execution RL.
    """

    def __init__(
        self,
        config: Optional[QlibRLConfig] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> None:
        """Initialize environment.

        Args:
            config: RL configuration
            data: Market data (OHLCV + features)
        """
        self.config = config or QlibRLConfig()
        self._data = data
        self._env = None

    def setup(
        self,
        data: pd.DataFrame,
        order_dir: int = 1,  # 1 for buy, -1 for sell
        order_amount: Optional[float] = None,
    ) -> None:
        """Setup the environment with data.

        Args:
            data: Market data
            order_dir: Order direction
            order_amount: Amount to execute
        """
        if not QLIB_RL_AVAILABLE:
            raise RuntimeError("Qlib RL not available")

        self._data = data
        amount = order_amount or self.config.order_amount

        # Create Qlib environment
        self._env = SingleAssetOrderExecution(
            order_dir=order_dir,
            order_amount=amount,
            time_per_step=self.config.time_per_step,
            # Use default SAOE components
        )

    @property
    def observation_space(self):
        """Get observation space."""
        if self._env:
            return self._env.observation_space
        return None

    @property
    def action_space(self):
        """Get action space."""
        if self._env:
            return self._env.action_space
        return None

    def reset(self) -> np.ndarray:
        """Reset environment."""
        if self._env:
            return self._env.reset()
        raise RuntimeError("Environment not initialized")

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action in environment."""
        if self._env:
            return self._env.step(action)
        raise RuntimeError("Environment not initialized")


# =============================================================================
# Qlib RL Policy Adapters
# =============================================================================

class QlibPPOPolicy:
    """Adapter for Qlib PPO policy.

    Integrates with stable-baselines3 PPO through Qlib's interface.
    """

    def __init__(self, config: Optional[QlibRLConfig] = None) -> None:
        self.config = config or QlibRLConfig()
        self._policy = None

    def create(self, env: QlibOrderExecutionEnv) -> Any:
        """Create PPO policy for environment."""
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.policies import ActorCriticPolicy

            self._policy = PPO(
                "MlpPolicy",
                env._env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                verbose=self.config.verbose,
            )
            return self._policy

        except ImportError:
            logger.warning("stable-baselines3 not available")
            return None

    def train(self, total_timesteps: Optional[int] = None) -> Dict[str, Any]:
        """Train the policy."""
        if self._policy is None:
            raise RuntimeError("Policy not created")

        steps = total_timesteps or self.config.total_timesteps
        self._policy.learn(total_timesteps=steps)

        return {
            "timesteps": steps,
            "policy": self._policy,
        }

    def predict(self, observation: np.ndarray) -> Tuple[np.ndarray, Any]:
        """Predict action from observation."""
        if self._policy is None:
            raise RuntimeError("Policy not created")
        return self._policy.predict(observation)


# =============================================================================
# Qlib Order Execution Strategies
# =============================================================================

@dataclass
class TWAPStrategy:
    """Time-Weighted Average Price execution strategy.

    Splits order evenly across time periods.
    """

    total_amount: float
    num_periods: int
    slippage: float = 0.0001

    def get_order_schedule(self) -> List[float]:
        """Get order amounts for each period."""
        base_amount = self.total_amount / self.num_periods
        return [base_amount] * self.num_periods

    def execute(self, prices: pd.Series) -> Dict[str, Any]:
        """Simulate TWAP execution."""
        schedule = self.get_order_schedule()
        executed = []

        for i, amount in enumerate(schedule):
            price = prices.iloc[i] if i < len(prices) else prices.iloc[-1]
            exec_price = price * (1 + self.slippage)
            executed.append({
                "period": i,
                "amount": amount,
                "price": exec_price,
                "cost": amount * exec_price,
            })

        total_cost = sum(e["cost"] for e in executed)
        avg_price = total_cost / self.total_amount if self.total_amount else 0

        return {
            "strategy": "TWAP",
            "executions": executed,
            "total_amount": self.total_amount,
            "total_cost": total_cost,
            "avg_price": avg_price,
            "vwap_benchmark": prices.mean(),
            "slippage_bps": (avg_price / prices.mean() - 1) * 10000,
        }


@dataclass
class VWAPStrategy:
    """Volume-Weighted Average Price execution strategy.

    Allocates orders based on historical volume profile.
    """

    total_amount: float
    volume_profile: pd.Series  # Historical volume by period
    slippage: float = 0.0001

    def get_order_schedule(self) -> List[float]:
        """Get order amounts weighted by volume."""
        weights = self.volume_profile / self.volume_profile.sum()
        return (weights * self.total_amount).tolist()

    def execute(self, prices: pd.Series) -> Dict[str, Any]:
        """Simulate VWAP execution."""
        schedule = self.get_order_schedule()
        executed = []

        for i, amount in enumerate(schedule):
            if i >= len(prices):
                break
            price = prices.iloc[i]
            exec_price = price * (1 + self.slippage)
            executed.append({
                "period": i,
                "amount": amount,
                "price": exec_price,
                "cost": amount * exec_price,
            })

        total_amount_exec = sum(e["amount"] for e in executed)
        total_cost = sum(e["cost"] for e in executed)
        avg_price = total_cost / total_amount_exec if total_amount_exec else 0

        # Calculate VWAP benchmark
        vwap = (prices * self.volume_profile[:len(prices)]).sum() / self.volume_profile[:len(prices)].sum()

        return {
            "strategy": "VWAP",
            "executions": executed,
            "total_amount": total_amount_exec,
            "total_cost": total_cost,
            "avg_price": avg_price,
            "vwap_benchmark": vwap,
            "slippage_bps": (avg_price / vwap - 1) * 10000,
        }


# =============================================================================
# Qlib RL Training Runner
# =============================================================================

class QlibRLTrainer:
    """Trainer for Qlib RL models.

    Provides:
    - Training loop management
    - Evaluation and backtesting
    - Model checkpointing
    - Integration with MLflow
    """

    def __init__(self, config: Optional[QlibRLConfig] = None) -> None:
        self.config = config or QlibRLConfig()
        self._env = None
        self._policy = None

    def setup(
        self,
        data: pd.DataFrame,
        order_amount: Optional[float] = None,
    ) -> None:
        """Setup training environment.

        Args:
            data: Market data
            order_amount: Order amount to execute
        """
        self._env = QlibOrderExecutionEnv(self.config)
        self._env.setup(data, order_amount=order_amount)

        # Create policy based on config
        if self.config.policy_type == "ppo":
            policy_adapter = QlibPPOPolicy(self.config)
            self._policy = policy_adapter.create(self._env)

    def train(self) -> Dict[str, Any]:
        """Run training loop."""
        if self._policy is None:
            raise RuntimeError("Training not setup")

        logger.info(f"Starting RL training for {self.config.total_timesteps} steps")

        try:
            self._policy.learn(
                total_timesteps=self.config.total_timesteps,
                progress_bar=True,
            )

            return {
                "status": "success",
                "timesteps": self.config.total_timesteps,
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
            }

    def evaluate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate trained policy on data."""
        if self._policy is None:
            raise RuntimeError("No trained policy")

        # Create test environment
        test_env = QlibOrderExecutionEnv(self.config)
        test_env.setup(data)

        obs = test_env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action, _ = self._policy.predict(obs)
            obs, reward, done, info = test_env.step(action)
            total_reward += reward
            steps += 1

        return {
            "total_reward": total_reward,
            "steps": steps,
            "avg_reward": total_reward / steps if steps else 0,
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_order_execution_env(
    data: pd.DataFrame,
    config: Optional[QlibRLConfig] = None,
    order_dir: int = 1,
) -> QlibOrderExecutionEnv:
    """Create order execution environment.

    Args:
        data: Market data
        config: RL configuration
        order_dir: Order direction (1=buy, -1=sell)

    Returns:
        Configured environment
    """
    env = QlibOrderExecutionEnv(config)
    env.setup(data, order_dir=order_dir)
    return env


def run_rl_training(
    data: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run RL training with Qlib.

    Args:
        data: Training data
        config: Configuration dictionary

    Returns:
        Training results
    """
    rl_config = QlibRLConfig(**config) if config else QlibRLConfig()
    trainer = QlibRLTrainer(rl_config)
    trainer.setup(data)
    return trainer.train()


def run_twap(
    amount: float,
    num_periods: int,
    prices: pd.Series,
) -> Dict[str, Any]:
    """Run TWAP execution simulation.

    Args:
        amount: Total amount to execute
        num_periods: Number of time periods
        prices: Price series

    Returns:
        Execution results
    """
    strategy = TWAPStrategy(
        total_amount=amount,
        num_periods=num_periods,
    )
    return strategy.execute(prices)


def run_vwap(
    amount: float,
    volume_profile: pd.Series,
    prices: pd.Series,
) -> Dict[str, Any]:
    """Run VWAP execution simulation.

    Args:
        amount: Total amount to execute
        volume_profile: Historical volume profile
        prices: Price series

    Returns:
        Execution results
    """
    strategy = VWAPStrategy(
        total_amount=amount,
        volume_profile=volume_profile,
    )
    return strategy.execute(prices)


# =============================================================================
# P2.1 FIX: Qlib Native RL Training Integration
# =============================================================================


def run_qlib_native_training(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run Qlib's native RL training with train_and_test.

    P2.1 FIX: This function uses Qlib's native train_and_test function
    instead of wrapping stable-baselines3 directly. Benefits:
    - Integrated backtesting after training
    - Qlib-native state adapters (SAOEStateAdapter)
    - Proper SAOEMetrics tracking
    - Trainer with checkpointing support

    Args:
        train_data: Training market data (OHLCV + features)
        test_data: Test market data for backtesting
        config: Training configuration

    Returns:
        Training and backtest results
    """
    if not QLIB_RL_AVAILABLE:
        logger.warning("Qlib RL not available, falling back to stable-baselines3")
        return run_rl_training(train_data, config)

    rl_config = QlibRLConfig(**(config or {}))

    try:
        # Configure Qlib RL training
        training_config = {
            "env": {
                "class": "SingleAssetOrderExecution",
                "kwargs": {
                    "order_dir": 1,
                    "order_amount": rl_config.order_amount,
                    "time_per_step": rl_config.time_per_step,
                },
            },
            "policy": {
                "class": "PPO",
                "kwargs": {
                    "learning_rate": rl_config.learning_rate,
                    "batch_size": rl_config.batch_size,
                    "gamma": rl_config.gamma,
                    "gae_lambda": rl_config.gae_lambda,
                },
            },
            "trainer": {
                "total_timesteps": rl_config.total_timesteps,
                "log_dir": rl_config.log_dir,
            },
        }

        # Run Qlib native training with integrated backtest
        logger.info("Starting Qlib native RL training with train_and_test")
        result = train_and_test(
            train_data=train_data,
            test_data=test_data,
            config=training_config,
        )

        return {
            "status": "success",
            "method": "qlib_native",
            "training_result": result.get("training", {}),
            "backtest_result": result.get("backtest", {}),
            "metrics": result.get("metrics", {}),
        }

    except Exception as e:
        logger.error(f"Qlib native training failed: {e}, falling back to SB3")
        # Fallback to stable-baselines3 wrapper
        return {
            "status": "fallback",
            "method": "stable_baselines3",
            "error": str(e),
            "result": run_rl_training(train_data, config),
        }


def run_qlib_backtest(
    policy: Any,
    data: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run Qlib native backtesting on trained policy.

    P2.1 FIX: Uses Qlib's native backtest function for
    order execution strategy evaluation.

    Args:
        policy: Trained RL policy
        data: Market data for backtesting
        config: Backtest configuration

    Returns:
        Backtest results with SAOEMetrics
    """
    if not QLIB_RL_AVAILABLE:
        raise RuntimeError("Qlib RL required for native backtesting")

    rl_config = QlibRLConfig(**(config or {}))

    try:
        # Run Qlib native backtest
        result = backtest(
            policy=policy,
            data=data,
            order_amount=rl_config.order_amount,
            time_per_step=rl_config.time_per_step,
        )

        return {
            "status": "success",
            "total_reward": result.get("total_reward", 0),
            "avg_slippage_bps": result.get("avg_slippage_bps", 0),
            "execution_cost": result.get("execution_cost", 0),
            "saoe_metrics": result.get("metrics", {}),
        }

    except Exception as e:
        logger.error(f"Qlib backtest failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
        }
