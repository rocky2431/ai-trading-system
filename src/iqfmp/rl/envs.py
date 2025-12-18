"""Trading Environments for Reinforcement Learning.

This module provides Gym-compatible trading environments for:
- Single asset trading
- Portfolio management
- Market making

Environments follow the OpenAI Gym interface for compatibility
with standard RL libraries (Stable-Baselines3, etc.).
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False
        gym = None
        spaces = None


logger = logging.getLogger(__name__)


class TradingAction(IntEnum):
    """Discrete trading actions."""
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class TradingEnvConfig:
    """Configuration for trading environment."""
    initial_balance: float = 100000.0
    max_position: float = 1.0  # Maximum position size (fraction of portfolio)
    transaction_cost: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    reward_scaling: float = 1e-4
    lookback_window: int = 30  # Number of historical steps for observation
    use_funding_rate: bool = True  # Include funding rate in observation
    normalize_obs: bool = True
    episode_length: Optional[int] = None  # None = use all data


class BaseTradingEnv(ABC):
    """Abstract base class for trading environments."""

    def __init__(self, config: TradingEnvConfig):
        self.config = config
        self._reset_state()

    def _reset_state(self):
        """Reset internal state."""
        self.balance = self.config.initial_balance
        self.position = 0.0
        self.position_avg_price = 0.0
        self.total_pnl = 0.0
        self.episode_pnl = 0.0
        self.current_step = 0
        self.trades = []

    @abstractmethod
    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """Reset environment and return initial observation."""
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action and return (obs, reward, terminated, truncated, info)."""
        pass


class SingleAssetTradingEnv(BaseTradingEnv):
    """Single asset trading environment.

    Observation space:
    - Price features (OHLCV normalized)
    - Technical indicators
    - Position info
    - Funding rate (if crypto)

    Action space (discrete):
    - 0: Hold
    - 1: Buy (go long)
    - 2: Sell (go short)

    Or continuous:
    - [-1, 1]: Target position
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[TradingEnvConfig] = None,
        discrete_actions: bool = True,
    ):
        """Initialize trading environment.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                Optional: 'funding_rate', 'returns', feature columns
            config: Environment configuration
            discrete_actions: Use discrete (True) or continuous (False) actions
        """
        if not GYM_AVAILABLE:
            raise ImportError("Gymnasium/Gym not installed. Install with: pip install gymnasium")

        config = config or TradingEnvConfig()
        super().__init__(config)

        self.df = df.copy()
        self.discrete_actions = discrete_actions
        self._prepare_data()
        self._setup_spaces()

    def _prepare_data(self):
        """Prepare and normalize data."""
        # Ensure required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Calculate returns if not present
        if 'returns' not in self.df.columns:
            self.df['returns'] = self.df['close'].pct_change()

        # Fill NaN
        self.df = self.df.fillna(0)

        # Extract feature columns (all except OHLCV)
        self.feature_cols = [c for c in self.df.columns
                           if c not in ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'date']]

        # Normalize features if requested
        if self.config.normalize_obs:
            for col in self.feature_cols:
                if col != 'returns':  # Don't normalize returns again
                    std = self.df[col].std()
                    if std > 0:
                        self.df[col] = (self.df[col] - self.df[col].mean()) / std

        # Pre-compute normalized OHLCV
        self.df['norm_close'] = self.df['close'] / self.df['close'].iloc[0]
        self.df['norm_volume'] = np.log1p(self.df['volume']) / np.log1p(self.df['volume'].max() + 1)

        self.n_steps = len(self.df)

    def _setup_spaces(self):
        """Setup observation and action spaces."""
        # Observation: [price_features, indicators, position_info]
        n_price_features = 5  # normalized OHLCV
        n_indicators = len(self.feature_cols)
        n_position_info = 3  # position, unrealized_pnl, balance_ratio

        obs_dim = n_price_features + n_indicators + n_position_info

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        if self.discrete_actions:
            self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        else:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32,
            )

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        row = self.df.iloc[self.current_step]

        # Price features (normalized)
        price_features = [
            row['norm_close'],
            row['returns'],
            (row['high'] - row['low']) / (row['close'] + 1e-10),  # Volatility
            (row['close'] - row['open']) / (row['close'] + 1e-10),  # Direction
            row['norm_volume'],
        ]

        # Indicator features
        indicator_features = [row[col] for col in self.feature_cols]

        # Position info
        current_price = row['close']
        unrealized_pnl = 0.0
        if self.position != 0:
            unrealized_pnl = (current_price - self.position_avg_price) * self.position

        position_info = [
            self.position / self.config.max_position,  # Normalized position
            unrealized_pnl / self.config.initial_balance,  # Normalized PnL
            self.balance / self.config.initial_balance,  # Balance ratio
        ]

        obs = np.array(price_features + indicator_features + position_info, dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _execute_trade(self, target_position: float) -> float:
        """Execute trade and return transaction cost."""
        current_price = self.df.iloc[self.current_step]['close']
        position_change = target_position - self.position

        if abs(position_change) < 1e-6:
            return 0.0

        # Apply slippage
        if position_change > 0:
            exec_price = current_price * (1 + self.config.slippage)
        else:
            exec_price = current_price * (1 - self.config.slippage)

        # Calculate transaction cost
        trade_value = abs(position_change) * exec_price
        transaction_cost = trade_value * self.config.transaction_cost

        # Update position
        if position_change > 0:
            # Buying
            new_cost = position_change * exec_price
            if self.position > 0:
                # Averaging up
                total_cost = self.position_avg_price * self.position + new_cost
                self.position_avg_price = total_cost / (self.position + position_change)
            else:
                self.position_avg_price = exec_price
        else:
            # Selling
            if target_position <= 0 and self.position > 0:
                # Realized PnL from closing long
                realized_pnl = (exec_price - self.position_avg_price) * self.position
                self.total_pnl += realized_pnl
                self.episode_pnl += realized_pnl
                if target_position < 0:
                    self.position_avg_price = exec_price  # New short position
            elif target_position >= 0 and self.position < 0:
                # Realized PnL from closing short
                realized_pnl = (self.position_avg_price - exec_price) * abs(self.position)
                self.total_pnl += realized_pnl
                self.episode_pnl += realized_pnl
                if target_position > 0:
                    self.position_avg_price = exec_price  # New long position

        self.position = target_position
        self.balance -= transaction_cost

        # Record trade
        self.trades.append({
            'step': self.current_step,
            'price': exec_price,
            'position_change': position_change,
            'new_position': target_position,
            'cost': transaction_cost,
        })

        return transaction_cost

    def _calculate_reward(self, old_portfolio_value: float) -> float:
        """Calculate step reward."""
        current_price = self.df.iloc[self.current_step]['close']

        # Portfolio value = balance + position value
        position_value = self.position * current_price
        new_portfolio_value = self.balance + position_value

        # Reward = change in portfolio value (scaled)
        reward = (new_portfolio_value - old_portfolio_value) * self.config.reward_scaling

        return reward

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment."""
        super()._reset_state()

        # Random start position (after lookback window)
        start_range = self.n_steps - (self.config.episode_length or self.n_steps // 2)
        start_range = max(self.config.lookback_window, start_range)

        if seed is not None:
            np.random.seed(seed)

        self.current_step = np.random.randint(
            self.config.lookback_window,
            start_range
        )

        self.start_step = self.current_step
        self.max_step = min(
            self.current_step + (self.config.episode_length or self.n_steps),
            self.n_steps - 1
        )

        obs = self._get_observation()
        info = {
            'balance': self.balance,
            'position': self.position,
            'step': self.current_step,
        }

        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step."""
        # Calculate old portfolio value
        current_price = self.df.iloc[self.current_step]['close']
        old_portfolio_value = self.balance + self.position * current_price

        # Convert action to target position
        if self.discrete_actions:
            if action == TradingAction.HOLD:
                target_position = self.position
            elif action == TradingAction.BUY:
                target_position = self.config.max_position
            else:  # SELL
                target_position = -self.config.max_position
        else:
            target_position = float(action[0]) * self.config.max_position

        # Execute trade
        transaction_cost = self._execute_trade(target_position)

        # Move to next step
        self.current_step += 1

        # Calculate reward
        reward = self._calculate_reward(old_portfolio_value)

        # Check termination
        terminated = self.current_step >= self.max_step
        truncated = self.balance <= 0  # Bankrupt

        # Get new observation
        if not (terminated or truncated):
            obs = self._get_observation()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        # Calculate final portfolio value
        final_price = self.df.iloc[min(self.current_step, self.n_steps - 1)]['close']
        portfolio_value = self.balance + self.position * final_price

        info = {
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': portfolio_value,
            'total_pnl': self.total_pnl,
            'episode_pnl': self.episode_pnl,
            'n_trades': len(self.trades),
            'transaction_cost': transaction_cost,
            'step': self.current_step,
        }

        return obs, reward, terminated, truncated, info


class PortfolioEnv(BaseTradingEnv):
    """Multi-asset portfolio management environment.

    Manages a portfolio of N assets with continuous allocation.

    Observation space:
    - Per-asset features (returns, volatility, etc.)
    - Portfolio state (weights, cash)

    Action space:
    - Target weights for each asset [0, 1]^N
    """

    def __init__(
        self,
        dfs: dict[str, pd.DataFrame],
        config: Optional[TradingEnvConfig] = None,
    ):
        """Initialize portfolio environment.

        Args:
            dfs: Dictionary of {symbol: DataFrame} with OHLCV data
            config: Environment configuration
        """
        if not GYM_AVAILABLE:
            raise ImportError("Gymnasium/Gym not installed. Install with: pip install gymnasium")

        config = config or TradingEnvConfig()
        super().__init__(config)

        self.dfs = {k: v.copy() for k, v in dfs.items()}
        self.symbols = list(dfs.keys())
        self.n_assets = len(self.symbols)
        self._prepare_data()
        self._setup_spaces()

    def _prepare_data(self):
        """Align and prepare data for all assets."""
        # Align dataframes by index
        common_index = self.dfs[self.symbols[0]].index
        for symbol in self.symbols[1:]:
            common_index = common_index.intersection(self.dfs[symbol].index)

        for symbol in self.symbols:
            self.dfs[symbol] = self.dfs[symbol].loc[common_index]
            self.dfs[symbol]['returns'] = self.dfs[symbol]['close'].pct_change().fillna(0)

        self.n_steps = len(common_index)

    def _setup_spaces(self):
        """Setup observation and action spaces."""
        # Observation: per-asset features + portfolio state
        n_asset_features = 3  # returns, volatility, momentum
        n_portfolio_features = self.n_assets + 1  # weights + cash

        obs_dim = self.n_assets * n_asset_features + n_portfolio_features

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Action: target weights for each asset (will be normalized)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )

        # Portfolio state
        self.weights = np.zeros(self.n_assets)
        self.cash = 1.0  # Start with 100% cash

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        asset_features = []

        for symbol in self.symbols:
            df = self.dfs[symbol]
            row = df.iloc[self.current_step]

            # Calculate features
            returns = row['returns']

            # Rolling volatility (use past 20 steps)
            start = max(0, self.current_step - 20)
            volatility = df['returns'].iloc[start:self.current_step + 1].std()

            # Momentum (20-day return)
            if self.current_step >= 20:
                momentum = (df['close'].iloc[self.current_step] /
                           df['close'].iloc[self.current_step - 20] - 1)
            else:
                momentum = 0.0

            asset_features.extend([returns, volatility, momentum])

        # Portfolio state
        portfolio_features = list(self.weights) + [self.cash]

        obs = np.array(asset_features + portfolio_features, dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _rebalance(self, target_weights: np.ndarray) -> float:
        """Rebalance portfolio to target weights."""
        # Normalize weights
        total = target_weights.sum()
        if total > 1:
            target_weights = target_weights / total
            cash_weight = 0.0
        else:
            cash_weight = 1.0 - total

        # Calculate turnover
        turnover = np.sum(np.abs(target_weights - self.weights))
        transaction_cost = turnover * self.config.transaction_cost

        # Update weights and cash
        self.weights = target_weights
        self.cash = cash_weight
        self.balance -= transaction_cost * self.balance

        return transaction_cost

    def _calculate_portfolio_return(self) -> float:
        """Calculate portfolio return for current step."""
        portfolio_return = 0.0

        for i, symbol in enumerate(self.symbols):
            asset_return = self.dfs[symbol].iloc[self.current_step]['returns']
            portfolio_return += self.weights[i] * asset_return

        return portfolio_return

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment."""
        super()._reset_state()
        self.weights = np.zeros(self.n_assets)
        self.cash = 1.0

        if seed is not None:
            np.random.seed(seed)

        self.current_step = np.random.randint(
            self.config.lookback_window,
            self.n_steps - (self.config.episode_length or self.n_steps // 2)
        )

        self.start_step = self.current_step
        self.max_step = min(
            self.current_step + (self.config.episode_length or self.n_steps),
            self.n_steps - 1
        )

        obs = self._get_observation()
        info = {'balance': self.balance, 'weights': self.weights.copy()}

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step."""
        # Rebalance portfolio
        transaction_cost = self._rebalance(action)

        # Move to next step
        self.current_step += 1

        # Calculate portfolio return
        portfolio_return = self._calculate_portfolio_return()
        self.balance *= (1 + portfolio_return)

        # Reward: risk-adjusted return (Sharpe-like)
        reward = portfolio_return * self.config.reward_scaling

        # Check termination
        terminated = self.current_step >= self.max_step
        truncated = self.balance <= 0

        obs = self._get_observation() if not (terminated or truncated) else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )

        info = {
            'balance': self.balance,
            'weights': self.weights.copy(),
            'portfolio_return': portfolio_return,
            'transaction_cost': transaction_cost,
            'step': self.current_step,
        }

        return obs, reward, terminated, truncated, info


# =============================================================================
# Factory Functions
# =============================================================================


def create_trading_env(
    df: pd.DataFrame,
    env_type: str = "single",
    config: Optional[TradingEnvConfig] = None,
    **kwargs,
) -> BaseTradingEnv:
    """Factory function to create trading environment.

    Args:
        df: Price data DataFrame
        env_type: "single" for single asset, "portfolio" for multi-asset
        config: Environment configuration
        **kwargs: Additional arguments

    Returns:
        Trading environment instance
    """
    config = config or TradingEnvConfig()

    if env_type == "single":
        return SingleAssetTradingEnv(df, config, **kwargs)
    elif env_type == "portfolio":
        if not isinstance(df, dict):
            raise ValueError("Portfolio env requires dict of DataFrames")
        return PortfolioEnv(df, config)
    else:
        raise ValueError(f"Unknown env type: {env_type}")
