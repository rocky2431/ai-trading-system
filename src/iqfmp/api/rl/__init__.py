"""RL Training API module.

Provides REST endpoints for Qlib RL training and backtesting.
"""

from iqfmp.api.rl.router import router

__all__ = ["router"]
