"""Unit tests for SAC (Soft Actor-Critic) implementation.

Tests cover:
- ReplayBuffer: add, sample, capacity, is_ready
- GaussianActor: forward, get_action, deterministic/stochastic modes
- TwinQNetwork: forward, q1_forward
- SACAgent: select_action, store_transition, update, save/load
- create_agent factory with "sac" type

NOTE: Some PyTorch-based tests may segfault under pytest due to a known
compatibility issue with pytest + PyTorch + Python 3.13 on macOS.
Use run_sac_tests.py for comprehensive testing of PyTorch components.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from iqfmp.rl.agents import (
    AgentConfig,
    ReplayBuffer,
    TORCH_AVAILABLE,
    create_agent,
    get_activation,
)

# Conditional imports for PyTorch components
if TORCH_AVAILABLE:
    import torch

    from iqfmp.rl.agents import (
        GaussianActor,
        TwinQNetwork,
        SACAgent,
    )


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def obs_dim() -> int:
    """Observation dimension for tests."""
    return 10


@pytest.fixture
def action_dim() -> int:
    """Action dimension for tests."""
    return 3


@pytest.fixture
def hidden_dims() -> list[int]:
    """Hidden layer dimensions for networks."""
    return [64, 64]


@pytest.fixture
def small_buffer() -> ReplayBuffer:
    """Small replay buffer for testing."""
    return ReplayBuffer(capacity=100)


@pytest.fixture
def filled_buffer(obs_dim: int, action_dim: int) -> ReplayBuffer:
    """Replay buffer pre-filled with data."""
    buffer = ReplayBuffer(capacity=100)
    rng = np.random.default_rng(seed=42)
    for _ in range(50):
        obs = rng.random(obs_dim).astype(np.float32)
        action = rng.random(action_dim).astype(np.float32)
        reward = rng.random()
        next_obs = rng.random(obs_dim).astype(np.float32)
        done = rng.random() > 0.9
        buffer.add(obs, action, reward, next_obs, done)
    return buffer


# =============================================================================
# ReplayBuffer Tests
# =============================================================================


class TestReplayBuffer:
    """Tests for ReplayBuffer class."""

    def test_buffer_creation(self) -> None:
        """Test creating a replay buffer with default capacity."""
        buffer = ReplayBuffer()
        assert buffer.capacity == 100000
        assert len(buffer) == 0
        assert not buffer._initialized

    def test_buffer_custom_capacity(self) -> None:
        """Test creating a buffer with custom capacity."""
        buffer = ReplayBuffer(capacity=500)
        assert buffer.capacity == 500

    def test_add_single_transition(self, obs_dim: int, action_dim: int) -> None:
        """Test adding a single transition initializes the buffer."""
        buffer = ReplayBuffer(capacity=10)
        obs = np.zeros(obs_dim, dtype=np.float32)
        action = np.zeros(action_dim, dtype=np.float32)

        buffer.add(obs, action, 1.0, obs, False)

        assert buffer._initialized
        assert len(buffer) == 1
        assert buffer._observations.shape == (10, obs_dim)
        assert buffer._actions.shape == (10, action_dim)

    def test_add_multiple_transitions(
        self, small_buffer: ReplayBuffer, obs_dim: int, action_dim: int
    ) -> None:
        """Test adding multiple transitions."""
        rng = np.random.default_rng(seed=0)

        for i in range(20):
            obs = rng.random(obs_dim).astype(np.float32)
            action = rng.random(action_dim).astype(np.float32)
            small_buffer.add(obs, action, float(i), obs, i == 19)

        assert len(small_buffer) == 20

    def test_circular_buffer_overflow(
        self, obs_dim: int, action_dim: int
    ) -> None:
        """Test that buffer overwrites oldest entries when full."""
        buffer = ReplayBuffer(capacity=10)
        rng = np.random.default_rng(seed=1)

        for i in range(15):
            obs = rng.random(obs_dim).astype(np.float32)
            action = rng.random(action_dim).astype(np.float32)
            buffer.add(obs, action, float(i), obs, False)

        assert len(buffer) == 10
        assert buffer._ptr == 5  # Wrapped around

    def test_is_ready(self, small_buffer: ReplayBuffer, obs_dim: int, action_dim: int) -> None:
        """Test is_ready check for batch size."""
        obs = np.zeros(obs_dim, dtype=np.float32)
        action = np.zeros(action_dim, dtype=np.float32)

        assert not small_buffer.is_ready(10)

        for _ in range(5):
            small_buffer.add(obs, action, 0.0, obs, False)

        assert not small_buffer.is_ready(10)
        assert small_buffer.is_ready(5)
        assert small_buffer.is_ready(3)

    def test_sample_batch(self, filled_buffer: ReplayBuffer) -> None:
        """Test sampling a batch from the buffer."""
        batch_size = 16
        obs, actions, rewards, next_obs, dones = filled_buffer.sample(batch_size)

        assert obs.shape[0] == batch_size
        assert actions.shape[0] == batch_size
        assert rewards.shape[0] == batch_size
        assert next_obs.shape[0] == batch_size
        assert dones.shape[0] == batch_size

    def test_sample_returns_valid_indices(
        self, filled_buffer: ReplayBuffer
    ) -> None:
        """Test that sampled data is valid (no uninitialized data)."""
        obs, actions, rewards, next_obs, dones = filled_buffer.sample(32)

        # Should not contain any NaN or inf
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))
        assert not np.any(np.isnan(rewards))

    def test_invalid_capacity_raises_error(self) -> None:
        """Test that zero or negative capacity raises ValueError."""
        with pytest.raises(ValueError, match="capacity must be positive"):
            ReplayBuffer(capacity=0)
        with pytest.raises(ValueError, match="capacity must be positive"):
            ReplayBuffer(capacity=-1)

    def test_sample_raises_when_not_ready(self, obs_dim: int, action_dim: int) -> None:
        """Test sample raises ValueError when buffer has insufficient samples."""
        buffer = ReplayBuffer(capacity=100)
        obs = np.zeros(obs_dim, dtype=np.float32)
        action = np.zeros(action_dim, dtype=np.float32)

        # Add only 5 samples
        for _ in range(5):
            buffer.add(obs, action, 0.0, obs, False)

        with pytest.raises(ValueError, match="Cannot sample 10 transitions"):
            buffer.sample(10)


# =============================================================================
# get_activation Tests
# =============================================================================


class TestGetActivation:
    """Tests for get_activation utility function."""

    def test_valid_activations(self) -> None:
        """Test that valid activation names return nn.Module instances."""
        valid_names = ["relu", "tanh", "leaky_relu", "elu", "gelu"]
        for name in valid_names:
            activation = get_activation(name)
            assert callable(activation), f"{name} should return callable"

    def test_invalid_activation_raises_error(self) -> None:
        """Test that invalid activation name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown activation"):
            get_activation("invalid_activation")
        with pytest.raises(ValueError, match="Unknown activation"):
            get_activation("nonexistent")


# =============================================================================
# GaussianActor Tests
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestGaussianActor:
    """Tests for GaussianActor network."""

    def test_actor_creation(
        self, obs_dim: int, action_dim: int, hidden_dims: list[int]
    ) -> None:
        """Test creating a Gaussian actor network."""
        actor = GaussianActor(obs_dim, action_dim, hidden_dims)

        assert actor.action_dim == action_dim
        assert actor.mean_head.out_features == action_dim
        assert actor.log_std_head.out_features == action_dim

    def test_forward_stochastic(
        self, obs_dim: int, action_dim: int, hidden_dims: list[int]
    ) -> None:
        """Test stochastic forward pass."""
        actor = GaussianActor(obs_dim, action_dim, hidden_dims)
        obs = torch.randn(4, obs_dim)  # Batch of 4

        action, log_prob = actor(obs, deterministic=False)

        assert action.shape == (4, action_dim)
        assert log_prob.shape == (4,)
        # Actions should be bounded by tanh: [-1, 1]
        assert torch.all(action >= -1.0)
        assert torch.all(action <= 1.0)

    def test_forward_deterministic(
        self, obs_dim: int, action_dim: int, hidden_dims: list[int]
    ) -> None:
        """Test deterministic forward pass."""
        actor = GaussianActor(obs_dim, action_dim, hidden_dims)
        obs = torch.randn(1, obs_dim)

        action1, log_prob1 = actor(obs, deterministic=True)
        action2, log_prob2 = actor(obs, deterministic=True)

        # Deterministic should produce same action
        assert torch.allclose(action1, action2)
        # Log prob is zero for deterministic
        assert torch.allclose(log_prob1, torch.zeros(1))

    def test_get_action(
        self, obs_dim: int, action_dim: int, hidden_dims: list[int]
    ) -> None:
        """Test get_action convenience method."""
        actor = GaussianActor(obs_dim, action_dim, hidden_dims)
        obs = torch.randn(2, obs_dim)

        action = actor.get_action(obs, deterministic=True)

        assert action.shape == (2, action_dim)

    def test_stochastic_produces_different_samples(
        self, obs_dim: int, action_dim: int, hidden_dims: list[int]
    ) -> None:
        """Test that stochastic mode produces different samples."""
        actor = GaussianActor(obs_dim, action_dim, hidden_dims)
        obs = torch.randn(1, obs_dim)

        actions = []
        for _ in range(10):
            action, _ = actor(obs, deterministic=False)
            actions.append(action.detach())

        # At least some actions should be different
        all_same = all(torch.allclose(actions[0], a) for a in actions[1:])
        assert not all_same, "Stochastic samples should differ"

    def test_invalid_obs_dim_raises_error(
        self, action_dim: int, hidden_dims: list[int]
    ) -> None:
        """Test that invalid obs_dim raises ValueError."""
        with pytest.raises(ValueError, match="obs_dim must be positive"):
            GaussianActor(0, action_dim, hidden_dims)
        with pytest.raises(ValueError, match="obs_dim must be positive"):
            GaussianActor(-1, action_dim, hidden_dims)

    def test_invalid_action_dim_raises_error(
        self, obs_dim: int, hidden_dims: list[int]
    ) -> None:
        """Test that invalid action_dim raises ValueError."""
        with pytest.raises(ValueError, match="action_dim must be positive"):
            GaussianActor(obs_dim, 0, hidden_dims)
        with pytest.raises(ValueError, match="action_dim must be positive"):
            GaussianActor(obs_dim, -1, hidden_dims)

    def test_empty_hidden_dims_raises_error(
        self, obs_dim: int, action_dim: int
    ) -> None:
        """Test that empty hidden_dims raises ValueError."""
        with pytest.raises(ValueError, match="hidden_dims must be non-empty"):
            GaussianActor(obs_dim, action_dim, [])


# =============================================================================
# TwinQNetwork Tests
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestTwinQNetwork:
    """Tests for TwinQNetwork class."""

    def test_network_creation(
        self, obs_dim: int, action_dim: int, hidden_dims: list[int]
    ) -> None:
        """Test creating twin Q-network."""
        twin_q = TwinQNetwork(obs_dim, action_dim, hidden_dims)

        # Both networks should be initialized
        assert twin_q.q1 is not None
        assert twin_q.q2 is not None

    def test_forward_returns_two_q_values(
        self, obs_dim: int, action_dim: int, hidden_dims: list[int]
    ) -> None:
        """Test forward pass returns two Q-values."""
        twin_q = TwinQNetwork(obs_dim, action_dim, hidden_dims)
        obs = torch.randn(8, obs_dim)
        action = torch.randn(8, action_dim)

        q1, q2 = twin_q(obs, action)

        assert q1.shape == (8, 1)
        assert q2.shape == (8, 1)

    def test_q_values_are_independent(
        self, obs_dim: int, action_dim: int, hidden_dims: list[int]
    ) -> None:
        """Test that Q1 and Q2 produce different values."""
        twin_q = TwinQNetwork(obs_dim, action_dim, hidden_dims)
        obs = torch.randn(4, obs_dim)
        action = torch.randn(4, action_dim)

        q1, q2 = twin_q(obs, action)

        # With random init, Q1 and Q2 should differ
        assert not torch.allclose(q1, q2), "Q1 and Q2 should be independent"

    def test_q1_forward(
        self, obs_dim: int, action_dim: int, hidden_dims: list[int]
    ) -> None:
        """Test q1_forward method."""
        twin_q = TwinQNetwork(obs_dim, action_dim, hidden_dims)
        obs = torch.randn(4, obs_dim)
        action = torch.randn(4, action_dim)

        q1_only = twin_q.q1_forward(obs, action)
        q1_from_forward, _ = twin_q(obs, action)

        assert q1_only.shape == (4, 1)
        assert torch.allclose(q1_only, q1_from_forward)

    def test_invalid_obs_dim_raises_error(
        self, action_dim: int, hidden_dims: list[int]
    ) -> None:
        """Test that invalid obs_dim raises ValueError."""
        with pytest.raises(ValueError, match="obs_dim must be positive"):
            TwinQNetwork(0, action_dim, hidden_dims)
        with pytest.raises(ValueError, match="obs_dim must be positive"):
            TwinQNetwork(-1, action_dim, hidden_dims)

    def test_invalid_action_dim_raises_error(
        self, obs_dim: int, hidden_dims: list[int]
    ) -> None:
        """Test that invalid action_dim raises ValueError."""
        with pytest.raises(ValueError, match="action_dim must be positive"):
            TwinQNetwork(obs_dim, 0, hidden_dims)
        with pytest.raises(ValueError, match="action_dim must be positive"):
            TwinQNetwork(obs_dim, -1, hidden_dims)

    def test_empty_hidden_dims_raises_error(
        self, obs_dim: int, action_dim: int
    ) -> None:
        """Test that empty hidden_dims raises ValueError."""
        with pytest.raises(ValueError, match="hidden_dims must be non-empty"):
            TwinQNetwork(obs_dim, action_dim, [])


# =============================================================================
# SACAgent Tests
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestSACAgent:
    """Tests for SACAgent class."""

    def test_agent_creation_default_config(
        self, obs_dim: int, action_dim: int
    ) -> None:
        """Test creating SAC agent with default config."""
        agent = SACAgent(obs_dim, action_dim)

        assert agent.obs_dim == obs_dim
        assert agent.action_dim == action_dim
        assert agent.actor is not None
        assert agent.critic is not None
        assert agent.critic_target is not None
        assert agent.replay_buffer is not None

    def test_agent_creation_custom_config(
        self, obs_dim: int, action_dim: int
    ) -> None:
        """Test creating SAC agent with custom config."""
        config = AgentConfig(
            learning_rate=1e-4,
            gamma=0.95,
            tau=0.01,
            alpha=0.1,
            auto_alpha=False,
        )
        agent = SACAgent(obs_dim, action_dim, config)

        assert agent.config.learning_rate == 1e-4
        assert agent.config.gamma == 0.95
        assert agent.alpha == 0.1
        assert not agent.auto_alpha

    def test_agent_auto_alpha(
        self, obs_dim: int, action_dim: int
    ) -> None:
        """Test automatic entropy tuning setup."""
        config = AgentConfig(auto_alpha=True)
        agent = SACAgent(obs_dim, action_dim, config)

        assert agent.auto_alpha
        assert agent.target_entropy == -action_dim
        assert hasattr(agent, "log_alpha")
        assert hasattr(agent, "alpha_optimizer")

    def test_select_action(self, obs_dim: int, action_dim: int) -> None:
        """Test selecting an action."""
        agent = SACAgent(obs_dim, action_dim)
        obs = np.random.randn(obs_dim).astype(np.float32)

        action = agent.select_action(obs)

        assert action.shape == (action_dim,)
        # Actions should be bounded by tanh
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)

    def test_select_action_deterministic(
        self, obs_dim: int, action_dim: int
    ) -> None:
        """Test deterministic action selection."""
        agent = SACAgent(obs_dim, action_dim)
        obs = np.random.randn(obs_dim).astype(np.float32)

        action1 = agent.select_action(obs, deterministic=True)
        action2 = agent.select_action(obs, deterministic=True)

        assert np.allclose(action1, action2)

    def test_store_transition(self, obs_dim: int, action_dim: int) -> None:
        """Test storing transitions in replay buffer."""
        agent = SACAgent(obs_dim, action_dim)
        obs = np.random.randn(obs_dim).astype(np.float32)
        action = np.random.randn(action_dim).astype(np.float32)
        next_obs = np.random.randn(obs_dim).astype(np.float32)

        assert len(agent.replay_buffer) == 0

        agent.store_transition(obs, action, 1.0, next_obs, False)

        assert len(agent.replay_buffer) == 1

    def test_update_returns_zeros_when_buffer_not_ready(
        self, obs_dim: int, action_dim: int
    ) -> None:
        """Test update returns zeros when buffer doesn't have enough samples."""
        agent = SACAgent(obs_dim, action_dim)

        metrics = agent.update(batch_size=256)

        assert metrics["actor_loss"] == 0.0
        assert metrics["critic_loss"] == 0.0

    def test_update_runs_when_buffer_ready(
        self, obs_dim: int, action_dim: int
    ) -> None:
        """Test update runs successfully with enough samples."""
        agent = SACAgent(obs_dim, action_dim)
        rng = np.random.default_rng(seed=42)

        # Fill buffer with enough samples
        for _ in range(300):
            obs = rng.random(obs_dim).astype(np.float32)
            action = rng.random(action_dim).astype(np.float32)
            next_obs = rng.random(obs_dim).astype(np.float32)
            reward = rng.random() - 0.5
            done = rng.random() > 0.95
            agent.store_transition(obs, action, reward, next_obs, done)

        metrics = agent.update(batch_size=64)

        # Should have non-zero losses after update
        assert "actor_loss" in metrics
        assert "critic_loss" in metrics
        assert "alpha" in metrics

    def test_update_with_fixed_alpha(
        self, obs_dim: int, action_dim: int
    ) -> None:
        """Test update with auto_alpha=False uses fixed alpha."""
        fixed_alpha = 0.5
        config = AgentConfig(alpha=fixed_alpha, auto_alpha=False)
        agent = SACAgent(obs_dim, action_dim, config)
        rng = np.random.default_rng(seed=42)

        # Fill buffer with enough samples
        for _ in range(300):
            obs = rng.random(obs_dim).astype(np.float32)
            action = rng.random(action_dim).astype(np.float32)
            next_obs = rng.random(obs_dim).astype(np.float32)
            reward = rng.random() - 0.5
            done = rng.random() > 0.95
            agent.store_transition(obs, action, reward, next_obs, done)

        # Run multiple updates
        for _ in range(10):
            metrics = agent.update(batch_size=64)

        # Alpha should remain fixed
        assert agent.alpha == fixed_alpha
        assert metrics["alpha"] == fixed_alpha
        # alpha_loss should be 0.0 when not tuning
        assert metrics["alpha_loss"] == 0.0

    def test_save_and_load(self, obs_dim: int, action_dim: int) -> None:
        """Test saving and loading agent."""
        agent = SACAgent(obs_dim, action_dim)

        # Do some updates to change weights
        rng = np.random.default_rng(seed=123)
        for _ in range(100):
            obs = rng.random(obs_dim).astype(np.float32)
            action = rng.random(action_dim).astype(np.float32)
            next_obs = rng.random(obs_dim).astype(np.float32)
            agent.store_transition(obs, action, 0.5, next_obs, False)

        agent.update(batch_size=32)

        # Get action before save
        test_obs = rng.random(obs_dim).astype(np.float32)
        action_before = agent.select_action(test_obs, deterministic=True)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sac_agent.pt"
            agent.save(str(path))

            # Create new agent and load
            new_agent = SACAgent(obs_dim, action_dim)
            new_agent.load(str(path))

            action_after = new_agent.select_action(test_obs, deterministic=True)

        assert np.allclose(action_before, action_after)

    def test_load_nonexistent_file_raises_error(
        self, obs_dim: int, action_dim: int
    ) -> None:
        """Test loading from nonexistent file raises FileNotFoundError."""
        agent = SACAgent(obs_dim, action_dim)

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            agent.load("/nonexistent/path/to/agent.pt")

    def test_load_invalid_checkpoint_raises_error(
        self, obs_dim: int, action_dim: int
    ) -> None:
        """Test loading corrupted checkpoint raises ValueError."""
        agent = SACAgent(obs_dim, action_dim)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.pt"
            # Write invalid data to file
            path.write_text("not a valid checkpoint")

            with pytest.raises(ValueError, match="Failed to load checkpoint"):
                agent.load(str(path))

    def test_load_missing_keys_raises_error(
        self, obs_dim: int, action_dim: int
    ) -> None:
        """Test loading checkpoint with missing keys raises ValueError."""
        import torch as pt

        agent = SACAgent(obs_dim, action_dim)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "incomplete.pt"
            # Save checkpoint with missing keys
            pt.save({"actor": agent.actor.state_dict()}, str(path))

            with pytest.raises(ValueError, match="Invalid checkpoint.*missing keys"):
                agent.load(str(path))


# =============================================================================
# Factory Function Tests
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestCreateAgent:
    """Tests for create_agent factory function."""

    def test_create_sac_agent(self, obs_dim: int, action_dim: int) -> None:
        """Test creating SAC agent via factory."""
        agent = create_agent("sac", obs_dim, action_dim)

        assert isinstance(agent, SACAgent)

    def test_create_sac_case_insensitive(
        self, obs_dim: int, action_dim: int
    ) -> None:
        """Test factory is case-insensitive."""
        agent1 = create_agent("SAC", obs_dim, action_dim)
        agent2 = create_agent("Sac", obs_dim, action_dim)

        assert isinstance(agent1, SACAgent)
        assert isinstance(agent2, SACAgent)

    def test_create_sac_with_config(
        self, obs_dim: int, action_dim: int
    ) -> None:
        """Test creating SAC with custom config."""
        config = AgentConfig(learning_rate=5e-4, tau=0.01)
        agent = create_agent("sac", obs_dim, action_dim, config=config)

        assert agent.config.learning_rate == 5e-4
        assert agent.config.tau == 0.01

    def test_create_sac_ignores_discrete_flag(
        self, obs_dim: int, action_dim: int
    ) -> None:
        """Test SAC factory ignores discrete flag (SAC is always continuous)."""
        # This should work without error, just logs a warning
        agent = create_agent("sac", obs_dim, action_dim, discrete=True)

        assert isinstance(agent, SACAgent)

    def test_create_invalid_agent_raises(
        self, obs_dim: int, action_dim: int
    ) -> None:
        """Test invalid agent type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown agent type"):
            create_agent("invalid", obs_dim, action_dim)


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestSACIntegration:
    """Integration tests for SAC components working together."""

    def test_training_loop_simulation(
        self, obs_dim: int, action_dim: int
    ) -> None:
        """Test simulated training loop with SAC."""
        agent = SACAgent(obs_dim, action_dim)
        rng = np.random.default_rng(seed=0)

        # Simulate environment interaction
        obs = rng.random(obs_dim).astype(np.float32)
        total_reward = 0.0

        for step in range(500):
            # Select action
            action = agent.select_action(obs)

            # Simulate environment step
            next_obs = rng.random(obs_dim).astype(np.float32)
            reward = rng.random() - 0.5
            done = step == 499

            # Store transition
            agent.store_transition(obs, action, reward, next_obs, done)
            total_reward += reward

            # Update agent
            if step >= 64:  # Start updating after collecting some samples
                agent.update(batch_size=32)

            obs = next_obs

        # Should complete without errors
        assert agent.replay_buffer.is_ready(32)

    def test_alpha_adaptation(self, obs_dim: int, action_dim: int) -> None:
        """Test that alpha adapts during training with auto_alpha=True."""
        config = AgentConfig(auto_alpha=True)
        agent = SACAgent(obs_dim, action_dim, config)
        initial_alpha = agent.alpha

        rng = np.random.default_rng(seed=1)

        # Fill buffer and do multiple updates
        for _ in range(200):
            obs = rng.random(obs_dim).astype(np.float32)
            action = rng.random(action_dim).astype(np.float32)
            next_obs = rng.random(obs_dim).astype(np.float32)
            agent.store_transition(obs, action, rng.random(), next_obs, False)

        for _ in range(50):
            agent.update(batch_size=32)

        # Alpha should have changed (adapted)
        assert agent.alpha != initial_alpha
