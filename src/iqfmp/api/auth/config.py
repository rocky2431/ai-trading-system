"""JWT configuration for authentication."""

import os
import secrets
from dataclasses import dataclass, field


@dataclass
class JWTConfig:
    """JWT configuration settings."""

    secret_key: str = field(
        default_factory=lambda: os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
