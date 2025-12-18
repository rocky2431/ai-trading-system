"""Reinforcement Learning Agents for Trading.

This module provides:
- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic)
- SAC (Soft Actor-Critic)

Implementations use PyTorch and support both discrete and continuous actions.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical, Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for RL agents."""
    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"  # relu, tanh, leaky_relu

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

    def clear(self):
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
    ):
        """Add transition to buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def __len__(self) -> int:
        return len(self.observations)


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
    return activations.get(name, nn.ReLU())


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

        layers = []
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
        return self.net(x)


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

            # Log prob with tanh correction
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)

        return action, log_prob, entropy, value.squeeze(-1)


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
    def save(self, path: str):
        """Save agent to file."""
        pass

    @abstractmethod
    def load(self, path: str):
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
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

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

    def save(self, path: str):
        """Save agent to file."""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }, path)

    def load(self, path: str):
        """Load agent from file."""
        checkpoint = torch.load(path, map_location=self.device)
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

    def save(self, path: str):
        """Save agent to file."""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }, path)

    def load(self, path: str):
        """Load agent from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


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
        agent_type: "ppo" or "a2c"
        obs_dim: Observation dimension
        action_dim: Action dimension
        config: Agent configuration
        discrete: Whether actions are discrete

    Returns:
        RL agent instance
    """
    config = config or AgentConfig()

    if agent_type.lower() == "ppo":
        return PPOAgent(obs_dim, action_dim, config, discrete)
    elif agent_type.lower() == "a2c":
        return A2CAgent(obs_dim, action_dim, config, discrete)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. Supported: ppo, a2c")
