"""Trading API module.

Provides real-time trading capabilities:
- Position management
- Order management
- Account information
- Risk monitoring
"""

from iqfmp.api.trading.router import router

__all__ = ["router"]
