"""Reinforcement Learning Agents for Trading.

This module provides:
- PPO (Proximal Policy Optimization) - On-policy, supports discrete/continuous
- A2C (Advantage Actor-Critic) - On-policy, supports discrete/continuous
- SAC (Soft Actor-Critic) - Off-policy, continuous actions only

Key components:
- RolloutBuffer: For on-policy algorithms (PPO, A2C)
- ReplayBuffer: For off-policy algorithms (SAC)
- GaussianActor: Gaussian policy network with reparameterization
- TwinQNetwork: Twin Q-networks for SAC

Implementations use PyTorch.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # noqa: F401

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical, Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for RL agents."""
    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"  # relu, tanh, leaky_relu, elu, gelu

    # Training hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_range: float = 0.2  # PPO clip range
    entropy_coef: float = 0.01  # Entropy bonus
    value_coef: float = 0.5  # Value loss coefficient
    max_grad_norm: float = 0.5  # Gradient clipping

    # PPO-specific
    n_epochs: int = 10  # Number of PPO epochs
    batch_size: int = 64
    n_steps: int = 2048  # Steps per update

    # SAC-specific
    tau: float = 0.005  # Soft update coefficient
    alpha: float = 0.2  # Temperature parameter
    auto_alpha: bool = True  # Automatically tune alpha
    replay_buffer_capacity: int = 100000  # Replay buffer size

    # Device
    device: str = "auto"  # auto, cpu, cuda

    def get_device(self) -> "torch.device":
        """Get PyTorch device."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed")

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data."""
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def clear(self) -> None:
        """Clear buffer."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Add transition to buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def __len__(self) -> int:
        return len(self.observations)


@dataclass
class ReplayBuffer:
    """Replay buffer for off-policy algorithms (SAC, DDPG, TD3).

    Stores transitions in a circular buffer with efficient sampling.
    """

    capacity: int = 100000
    _observations: np.ndarray = field(init=False, repr=False)
    _actions: np.ndarray = field(init=False, repr=False)
    _rewards: np.ndarray = field(init=False, repr=False)
    _next_observations: np.ndarray = field(init=False, repr=False)
    _dones: np.ndarray = field(init=False, repr=False)
    _size: int = field(default=0, init=False)
    _ptr: int = field(default=0, init=False)
    _initialized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Validate capacity at construction."""
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity}")

    def _init_buffers(self, obs_dim: int, action_dim: int) -> None:
        """Initialize numpy arrays for storage."""
        self._observations = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._next_observations = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self._dones = np.zeros(self.capacity, dtype=np.float32)
        self._initialized = True

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add transition to buffer."""
        if not self._initialized:
            obs_dim = obs.shape[0] if obs.ndim > 0 else 1
            action_dim = action.shape[0] if action.ndim > 0 else 1
            self._init_buffers(obs_dim, action_dim)

        self._observations[self._ptr] = obs
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._next_observations[self._ptr] = next_obs
        self._dones[self._ptr] = float(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample batch of transitions.

        Returns:
            (observations, actions, rewards, next_observations, dones)

        Raises:
            ValueError: If buffer has fewer samples than batch_size.
        """
        if not self.is_ready(batch_size):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer "
                f"with only {self._size} samples"
            )
        indices = np.random.randint(0, self._size, size=batch_size)
        return (
            self._observations[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_observations[indices],
            self._dones[indices],
        )

    def __len__(self) -> int:
        return self._size

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self._size >= batch_size


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not installed")

    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
    }
    if name not in activations:
        raise ValueError(
            f"Unknown activation: '{name}'. Supported: {list(activations.keys())}"
        )
    return activations[name]


class MLP(nn.Module):
    """Multi-layer perceptron."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        output_activation: Optional[str] = None,
    ):
        super().__init__()

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(get_activation(activation))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation:
            layers.append(get_activation(output_activation))

        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)  # type: ignore[no-any-return]


class ActorCriticNetwork(nn.Module):
    """Shared actor-critic network for PPO/A2C."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        discrete: bool = True,
    ):
        super().__init__()

        self.discrete = discrete
        self.action_dim = action_dim

        # Shared feature extractor
        self.shared = MLP(obs_dim, hidden_dims[-1], hidden_dims[:-1], activation, activation)

        # Actor head
        if discrete:
            self.actor = nn.Linear(hidden_dims[-1], action_dim)
        else:
            self.actor_mean = nn.Linear(hidden_dims[-1], action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.critic = nn.Linear(hidden_dims[-1], 1)

    def forward(self, obs: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Forward pass returning action logits/mean and value."""
        features = self.shared(obs)

        if self.discrete:
            action_logits = self.actor(features)
            return action_logits, self.critic(features)
        else:
            action_mean = self.actor_mean(features)
            return action_mean, self.critic(features)

    def get_action_and_value(
        self,
        obs: "torch.Tensor",
        action: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Get action, log prob, entropy, and value."""
        features = self.shared(obs)
        value = self.critic(features)

        if self.discrete:
            logits = self.actor(features)
            dist = Categorical(logits=logits)

            if action is None:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        else:
            mean = self.actor_mean(features)
            std = torch.exp(self.actor_log_std)
            dist = Normal(mean, std)

            if action is None:
                action = dist.sample()
                action = torch.tanh(action)  # Squash to [-1, 1]

            # Log prob (no tanh correction - simplified for PPO/A2C)
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)

        return action, log_prob, entropy, value.squeeze(-1)


# =============================================================================
# SAC-specific Networks
# =============================================================================


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class GaussianActor(nn.Module):
    """Gaussian policy network for SAC with reparameterization trick.

    Outputs mean and log_std for a Gaussian distribution, then samples
    using the reparameterization trick and applies tanh squashing.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
    ):
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")

        super().__init__()

        self.action_dim = action_dim

        # Shared feature extractor
        self.shared = MLP(obs_dim, hidden_dims[-1], hidden_dims[:-1], activation, activation)

        # Mean and log_std heads
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

    def forward(
        self,
        obs: "torch.Tensor",
        deterministic: bool = False,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Forward pass returning action and log_prob.

        Args:
            obs: Observation tensor
            deterministic: If True, return mean action without sampling

        Returns:
            (action, log_prob) - action is squashed to [-1, 1]
        """
        features = self.shared(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        if deterministic:
            action = torch.tanh(mean)
            # Log prob is 0 for deterministic (not used)
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            # Reparameterization trick
            dist = Normal(mean, std)
            x_t = dist.rsample()  # Differentiable sampling
            action = torch.tanh(x_t)

            # Log prob with tanh correction (Appendix C of SAC paper)
            log_prob = dist.log_prob(x_t).sum(-1)
            # Correction for tanh squashing
            log_prob -= (2 * (np.log(2) - x_t - F.softplus(-2 * x_t))).sum(-1)

        return action, log_prob

    def get_action(self, obs: "torch.Tensor", deterministic: bool = False) -> "torch.Tensor":
        """Get action without log_prob (for inference)."""
        action, _ = self.forward(obs, deterministic)
        return action


class TwinQNetwork(nn.Module):
    """Twin Q-network for SAC (min of two Q-values).

    Uses two independent Q-networks and takes the minimum to reduce
    overestimation bias.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
    ):
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")

        super().__init__()

        input_dim = obs_dim + action_dim

        # Two independent Q-networks
        self.q1 = MLP(input_dim, 1, hidden_dims, activation)
        self.q2 = MLP(input_dim, 1, hidden_dims, activation)

    def forward(
        self, obs: "torch.Tensor", action: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Forward pass returning Q-values from both networks."""
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, obs: "torch.Tensor", action: "torch.Tensor") -> "torch.Tensor":
        """Forward pass through Q1 only (for policy update)."""
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x)


class BaseAgent(ABC):
    """Abstract base class for RL agents."""

    def __init__(self, obs_dim: int, action_dim: int, config: AgentConfig):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Install with: pip install torch")

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.device = config.get_device()

    @abstractmethod
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action given observation."""
        pass

    @abstractmethod
    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Update agent from rollout buffer."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent to file."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent from file."""
        pass


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization agent."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Optional[AgentConfig] = None,
        discrete: bool = True,
    ):
        config = config or AgentConfig()
        super().__init__(obs_dim, action_dim, config)

        self.discrete = discrete

        # Create actor-critic network
        self.network = ActorCriticNetwork(
            obs_dim,
            action_dim,
            config.hidden_dims,
            config.activation,
            discrete,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
        )

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """Select action and return (action, log_prob, value)."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            if deterministic:
                if self.discrete:
                    logits, value = self.network(obs_tensor)
                    action = logits.argmax(dim=-1)
                    log_prob = torch.zeros(1)
                else:
                    mean, value = self.network(obs_tensor)
                    action = torch.tanh(mean)
                    log_prob = torch.zeros(1)
            else:
                action, log_prob, _, value = self.network.get_action_and_value(obs_tensor)

            action = action.cpu().numpy().squeeze()
            log_prob = log_prob.cpu().numpy().item()
            value = value.cpu().numpy().item()

        return action, log_prob, value

    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns."""
        n_steps = len(rewards)
        advantages = np.zeros(n_steps)
        returns = np.zeros(n_steps)

        gae = 0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def update(self, buffer: RolloutBuffer, next_value: float = 0.0) -> Dict[str, float]:
        """Update agent using PPO."""
        # Convert buffer to numpy arrays
        observations = np.array(buffer.observations)
        actions = np.array(buffer.actions)
        rewards = np.array(buffer.rewards)
        values = np.array(buffer.values)
        log_probs = np.array(buffer.log_probs)
        dones = np.array(buffer.dones)

        # Compute advantages
        advantages, returns = self._compute_gae(rewards, values, dones, next_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        obs_tensor = torch.FloatTensor(observations).to(self.device)
        action_tensor = torch.LongTensor(actions) if self.discrete else torch.FloatTensor(actions)
        action_tensor = action_tensor.to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # PPO training loop
        n_samples = len(observations)
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.config.n_epochs):
            # Shuffle indices
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, self.config.batch_size):
                end = min(start + self.config.batch_size, n_samples)
                batch_indices = indices[start:end]

                # Get batch
                batch_obs = obs_tensor[batch_indices]
                batch_actions = action_tensor[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # Forward pass
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    batch_obs, batch_actions
                )

                # Policy loss (PPO clip)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

        n_updates = self.config.n_epochs * (n_samples // self.config.batch_size + 1)
        return {
            "loss": total_loss / n_updates,
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def save(self, path: str) -> None:
        """Save agent to file."""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }, path)

    def load(self, path: str) -> None:
        """Load agent from file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


class A2CAgent(BaseAgent):
    """Advantage Actor-Critic agent."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Optional[AgentConfig] = None,
        discrete: bool = True,
    ):
        config = config or AgentConfig()
        super().__init__(obs_dim, action_dim, config)

        self.discrete = discrete

        # Create actor-critic network
        self.network = ActorCriticNetwork(
            obs_dim,
            action_dim,
            config.hidden_dims,
            config.activation,
            discrete,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.RMSprop(
            self.network.parameters(),
            lr=config.learning_rate,
            alpha=0.99,
            eps=1e-5,
        )

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """Select action and return (action, log_prob, value)."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            if deterministic:
                if self.discrete:
                    logits, value = self.network(obs_tensor)
                    action = logits.argmax(dim=-1)
                    log_prob = torch.zeros(1)
                else:
                    mean, value = self.network(obs_tensor)
                    action = torch.tanh(mean)
                    log_prob = torch.zeros(1)
            else:
                action, log_prob, _, value = self.network.get_action_and_value(obs_tensor)

            action = action.cpu().numpy().squeeze()
            log_prob = log_prob.cpu().numpy().item()
            value = value.cpu().numpy().item()

        return action, log_prob, value

    def _compute_returns(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        next_value: float,
    ) -> np.ndarray:
        """Compute discounted returns."""
        n_steps = len(rewards)
        returns = np.zeros(n_steps)

        R = next_value
        for t in reversed(range(n_steps)):
            R = rewards[t] + self.config.gamma * R * (1 - dones[t])
            returns[t] = R

        return returns

    def update(self, buffer: RolloutBuffer, next_value: float = 0.0) -> Dict[str, float]:
        """Update agent using A2C."""
        # Convert buffer to numpy arrays
        observations = np.array(buffer.observations)
        actions = np.array(buffer.actions)
        rewards = np.array(buffer.rewards)
        values = np.array(buffer.values)
        dones = np.array(buffer.dones)

        # Compute returns
        returns = self._compute_returns(rewards, dones, next_value)

        # Compute advantages
        advantages = returns - values

        # Convert to tensors
        obs_tensor = torch.FloatTensor(observations).to(self.device)
        action_tensor = torch.LongTensor(actions) if self.discrete else torch.FloatTensor(actions)
        action_tensor = action_tensor.to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Forward pass
        _, log_probs, entropy, new_values = self.network.get_action_and_value(
            obs_tensor, action_tensor
        )

        # Policy loss (vanilla policy gradient with advantage)
        policy_loss = -(log_probs * advantages_tensor).mean()

        # Value loss
        value_loss = F.mse_loss(new_values, returns_tensor)

        # Entropy bonus
        entropy_loss = -entropy.mean()

        # Total loss
        loss = (
            policy_loss
            + self.config.value_coef * value_loss
            + self.config.entropy_coef * entropy_loss
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
        }

    def save(self, path: str) -> None:
        """Save agent to file."""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }, path)

    def load(self, path: str) -> None:
        """Load agent from file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


class SACAgent(BaseAgent):
    """Soft Actor-Critic agent.

    Off-policy maximum entropy RL algorithm with automatic temperature tuning.
    Uses twin Q-networks to reduce overestimation and a Gaussian policy.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Optional[AgentConfig] = None,
    ):
        """Initialize SAC agent.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            config: Agent configuration (uses SAC-specific params: tau, alpha, auto_alpha)
        """
        config = config or AgentConfig()
        super().__init__(obs_dim, action_dim, config)

        # Actor (Gaussian policy)
        self.actor = GaussianActor(
            obs_dim,
            action_dim,
            config.hidden_dims,
            config.activation,
        ).to(self.device)

        # Twin Q-networks
        self.critic = TwinQNetwork(
            obs_dim,
            action_dim,
            config.hidden_dims,
            config.activation,
        ).to(self.device)

        # Target Q-networks (for soft updates)
        self.critic_target = TwinQNetwork(
            obs_dim,
            action_dim,
            config.hidden_dims,
            config.activation,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.learning_rate,
        )

        # Entropy temperature (alpha)
        self.auto_alpha = config.auto_alpha
        if self.auto_alpha:
            # Target entropy: -dim(A) (heuristic from SAC paper)
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.learning_rate)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = config.alpha

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=config.replay_buffer_capacity)

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select action given observation.

        Args:
            obs: Observation
            deterministic: If True, return mean action

        Returns:
            Action array
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action = self.actor.get_action(obs_tensor, deterministic)
            return action.cpu().numpy().squeeze()

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        self.replay_buffer.add(obs, action, reward, next_obs, done)

    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """Update agent from internal replay buffer.

        Args:
            batch_size: Batch size for sampling

        Returns:
            Dictionary of training metrics
        """
        if not self.replay_buffer.is_ready(batch_size):
            logger.debug(
                "Skipping SAC update: replay buffer has %d samples, requires %d",
                len(self.replay_buffer),
                batch_size,
            )
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "alpha": self.alpha,
                "alpha_loss": 0.0,
                "skipped": True,
            }

        # Sample batch
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(batch_size)

        # Convert to tensors
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)

        # Update critic
        with torch.no_grad():
            # Sample action for next state
            next_actions, next_log_probs = self.actor(next_obs)
            next_log_probs = next_log_probs.unsqueeze(-1)

            # Target Q-values
            target_q1, target_q2 = self.critic_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + self.config.gamma * (1 - dones) * target_q

        # Current Q-values
        current_q1, current_q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()

        # Update actor
        new_actions, log_probs = self.actor(obs)
        q1_new = self.critic.q1_forward(obs, new_actions)
        # Fix shape: log_probs is (batch,), q1_new is (batch, 1) - squeeze q1_new
        actor_loss = (self.alpha * log_probs - q1_new.squeeze(-1)).mean()

        # Check for numerical instability
        if torch.isnan(critic_loss) or torch.isinf(critic_loss):
            logger.error(
                "Numerical instability: critic_loss=%s, skipping update",
                critic_loss.item() if not torch.isnan(critic_loss) else "nan",
            )
            return {
                "actor_loss": float("nan"),
                "critic_loss": float("nan"),
                "alpha": self.alpha,
                "alpha_loss": 0.0,
                "numerical_error": True,
            }

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()

        # Update alpha (entropy temperature)
        alpha_loss_value = 0.0
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            alpha_loss_value = alpha_loss.item()

        # Soft update target networks
        self._soft_update()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "alpha": self.alpha,
            "alpha_loss": alpha_loss_value,
        }

    def _soft_update(self) -> None:
        """Soft update target networks."""
        tau = self.config.tau
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, path: str) -> None:
        """Save agent to file.

        Args:
            path: Path to save checkpoint

        Raises:
            IOError: If save fails (permissions, disk space, etc.)
        """
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": self.config,
            "auto_alpha": self.auto_alpha,
        }
        if self.auto_alpha:
            checkpoint["log_alpha"] = self.log_alpha
            checkpoint["alpha_optimizer"] = self.alpha_optimizer.state_dict()

        try:
            torch.save(checkpoint, path)
            logger.info("Saved SAC agent to %s", path)
        except (OSError, IOError) as e:
            raise IOError(
                f"Failed to save SAC agent to '{path}': {e}. "
                f"Check directory exists and has write permissions."
            ) from e

    def load(self, path: str) -> None:
        """Load agent from file.

        Args:
            path: Path to checkpoint file

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint is corrupted or incompatible
        """
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Checkpoint not found at '{path}'. "
                f"Ensure the path is correct and the file exists."
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load checkpoint from '{path}': {e}. "
                f"The file may be corrupted or incompatible."
            ) from e

        # Validate required keys
        required_keys = ["actor", "critic", "critic_target", "actor_optimizer", "critic_optimizer"]
        missing = [k for k in required_keys if k not in checkpoint]
        if missing:
            raise ValueError(
                f"Invalid checkpoint at '{path}': missing keys {missing}. "
                f"Expected a checkpoint saved by SACAgent.save()."
            )

        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

        # Handle auto_alpha state
        if self.auto_alpha:
            if "log_alpha" not in checkpoint:
                logger.warning(
                    "Loading checkpoint into auto_alpha=True agent, but checkpoint "
                    "has no log_alpha. Using default alpha=%.4f. This may indicate "
                    "checkpoint was saved with auto_alpha=False.",
                    self.alpha,
                )
            else:
                # Fix device mismatch
                self.log_alpha = checkpoint["log_alpha"].to(self.device)
                if "alpha_optimizer" in checkpoint:
                    self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
                else:
                    logger.warning(
                        "Checkpoint has log_alpha but no alpha_optimizer state. "
                        "Alpha optimizer will start fresh."
                    )
                self.alpha = self.log_alpha.exp().item()

        logger.info("Loaded SAC agent from %s", path)


# =============================================================================
# Factory Functions
# =============================================================================


def create_agent(
    agent_type: str,
    obs_dim: int,
    action_dim: int,
    config: Optional[AgentConfig] = None,
    discrete: bool = True,
) -> BaseAgent:
    """Factory function to create RL agent.

    Args:
        agent_type: "ppo", "a2c", or "sac"
        obs_dim: Observation dimension
        action_dim: Action dimension
        config: Agent configuration
        discrete: Whether actions are discrete (ignored for SAC, which uses continuous)

    Returns:
        RL agent instance
    """
    config = config or AgentConfig()

    if agent_type.lower() == "ppo":
        return PPOAgent(obs_dim, action_dim, config, discrete)
    elif agent_type.lower() == "a2c":
        return A2CAgent(obs_dim, action_dim, config, discrete)
    elif agent_type.lower() == "sac":
        if discrete:
            logger.warning("SAC requires continuous actions. Ignoring discrete=True.")
        return SACAgent(obs_dim, action_dim, config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. Supported: ppo, a2c, sac")
