#!/usr/bin/env python
"""Direct test runner for SAC components (bypasses pytest segfault issue)."""

import warnings
warnings.filterwarnings('ignore')

import tempfile
from pathlib import Path

import numpy as np


def run_tests():
    """Run SAC tests directly."""
    from iqfmp.rl.agents import (
        SACAgent,
        AgentConfig,
        GaussianActor,
        TwinQNetwork,
        create_agent,
        TORCH_AVAILABLE,
    )

    if not TORCH_AVAILABLE:
        print("SKIP: PyTorch not available")
        return

    import torch

    obs_dim = 10
    action_dim = 3
    hidden_dims = [64, 64]

    passed = 0
    failed = 0

    # Test GaussianActor
    print("Test: GaussianActor creation")
    try:
        actor = GaussianActor(obs_dim, action_dim, hidden_dims)
        assert actor.action_dim == action_dim
        passed += 1
        print("  PASSED")
    except Exception as e:
        failed += 1
        print(f"  FAILED: {e}")

    print("Test: GaussianActor forward stochastic")
    try:
        actor = GaussianActor(obs_dim, action_dim, hidden_dims)
        obs = torch.randn(4, obs_dim)
        action, log_prob = actor(obs, deterministic=False)
        assert action.shape == (4, action_dim)
        assert log_prob.shape == (4,)
        assert torch.all(action >= -1.0)
        assert torch.all(action <= 1.0)
        passed += 1
        print("  PASSED")
    except Exception as e:
        failed += 1
        print(f"  FAILED: {e}")

    print("Test: GaussianActor deterministic")
    try:
        actor = GaussianActor(obs_dim, action_dim, hidden_dims)
        obs = torch.randn(1, obs_dim)
        action1, _ = actor(obs, deterministic=True)
        action2, _ = actor(obs, deterministic=True)
        assert torch.allclose(action1, action2)
        passed += 1
        print("  PASSED")
    except Exception as e:
        failed += 1
        print(f"  FAILED: {e}")

    # Test TwinQNetwork
    print("Test: TwinQNetwork creation")
    try:
        twin_q = TwinQNetwork(obs_dim, action_dim, hidden_dims)
        assert twin_q.q1 is not None
        assert twin_q.q2 is not None
        passed += 1
        print("  PASSED")
    except Exception as e:
        failed += 1
        print(f"  FAILED: {e}")

    print("Test: TwinQNetwork forward")
    try:
        twin_q = TwinQNetwork(obs_dim, action_dim, hidden_dims)
        obs = torch.randn(8, obs_dim)
        action = torch.randn(8, action_dim)
        q1, q2 = twin_q(obs, action)
        assert q1.shape == (8, 1)
        assert q2.shape == (8, 1)
        passed += 1
        print("  PASSED")
    except Exception as e:
        failed += 1
        print(f"  FAILED: {e}")

    # Test SACAgent
    print("Test: SACAgent creation with default config")
    try:
        agent = SACAgent(obs_dim, action_dim)
        assert agent.obs_dim == obs_dim
        assert agent.action_dim == action_dim
        passed += 1
        print("  PASSED")
    except Exception as e:
        failed += 1
        print(f"  FAILED: {e}")

    print("Test: SACAgent creation with custom config")
    try:
        config = AgentConfig(learning_rate=1e-4, gamma=0.95, tau=0.01, alpha=0.1, auto_alpha=False)
        agent = SACAgent(obs_dim, action_dim, config)
        assert agent.config.learning_rate == 1e-4
        assert agent.alpha == 0.1
        passed += 1
        print("  PASSED")
    except Exception as e:
        failed += 1
        print(f"  FAILED: {e}")

    print("Test: SACAgent select_action")
    try:
        agent = SACAgent(obs_dim, action_dim)
        obs = np.random.randn(obs_dim).astype(np.float32)
        action = agent.select_action(obs)
        assert action.shape == (action_dim,)
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)
        passed += 1
        print("  PASSED")
    except Exception as e:
        failed += 1
        print(f"  FAILED: {e}")

    print("Test: SACAgent deterministic action")
    try:
        agent = SACAgent(obs_dim, action_dim)
        obs = np.random.randn(obs_dim).astype(np.float32)
        action1 = agent.select_action(obs, deterministic=True)
        action2 = agent.select_action(obs, deterministic=True)
        assert np.allclose(action1, action2)
        passed += 1
        print("  PASSED")
    except Exception as e:
        failed += 1
        print(f"  FAILED: {e}")

    print("Test: SACAgent store_transition")
    try:
        agent = SACAgent(obs_dim, action_dim)
        obs = np.random.randn(obs_dim).astype(np.float32)
        action = np.random.randn(action_dim).astype(np.float32)
        next_obs = np.random.randn(obs_dim).astype(np.float32)
        agent.store_transition(obs, action, 1.0, next_obs, False)
        assert len(agent.replay_buffer) == 1
        passed += 1
        print("  PASSED")
    except Exception as e:
        failed += 1
        print(f"  FAILED: {e}")

    print("Test: SACAgent update returns zeros when buffer not ready")
    try:
        agent = SACAgent(obs_dim, action_dim)
        metrics = agent.update(batch_size=256)
        assert metrics["actor_loss"] == 0.0
        passed += 1
        print("  PASSED")
    except Exception as e:
        failed += 1
        print(f"  FAILED: {e}")

    print("Test: SACAgent update runs when buffer ready")
    try:
        agent = SACAgent(obs_dim, action_dim)
        rng = np.random.default_rng(seed=42)
        for _ in range(300):
            obs = rng.random(obs_dim).astype(np.float32)
            action = rng.random(action_dim).astype(np.float32)
            next_obs = rng.random(obs_dim).astype(np.float32)
            agent.store_transition(obs, action, rng.random() - 0.5, next_obs, rng.random() > 0.95)
        metrics = agent.update(batch_size=64)
        assert "actor_loss" in metrics
        assert "critic_loss" in metrics
        passed += 1
        print("  PASSED")
    except Exception as e:
        failed += 1
        print(f"  FAILED: {e}")

    print("Test: SACAgent save and load")
    try:
        agent = SACAgent(obs_dim, action_dim)
        rng = np.random.default_rng(seed=123)
        for _ in range(100):
            obs = rng.random(obs_dim).astype(np.float32)
            action = rng.random(action_dim).astype(np.float32)
            next_obs = rng.random(obs_dim).astype(np.float32)
            agent.store_transition(obs, action, 0.5, next_obs, False)
        agent.update(batch_size=32)

        test_obs = rng.random(obs_dim).astype(np.float32)
        action_before = agent.select_action(test_obs, deterministic=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sac_agent.pt"
            agent.save(str(path))
            new_agent = SACAgent(obs_dim, action_dim)
            new_agent.load(str(path))
            action_after = new_agent.select_action(test_obs, deterministic=True)

        assert np.allclose(action_before, action_after)
        passed += 1
        print("  PASSED")
    except Exception as e:
        failed += 1
        print(f"  FAILED: {e}")

    print("Test: create_agent factory for SAC")
    try:
        agent = create_agent("sac", obs_dim, action_dim)
        assert isinstance(agent, SACAgent)
        passed += 1
        print("  PASSED")
    except Exception as e:
        failed += 1
        print(f"  FAILED: {e}")

    print("Test: create_agent case insensitive")
    try:
        agent1 = create_agent("SAC", obs_dim, action_dim)
        agent2 = create_agent("Sac", obs_dim, action_dim)
        assert isinstance(agent1, SACAgent)
        assert isinstance(agent2, SACAgent)
        passed += 1
        print("  PASSED")
    except Exception as e:
        failed += 1
        print(f"  FAILED: {e}")

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
