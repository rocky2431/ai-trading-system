"""Token service for JWT operations."""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import jwt

from iqfmp.api.auth.config import JWTConfig


class TokenExpiredError(Exception):
    """Raised when token has expired."""

    pass


class InvalidTokenError(Exception):
    """Raised when token is invalid."""

    pass


class TokenService:
    """Service for JWT token operations."""

    def __init__(self, config: JWTConfig) -> None:
        """Initialize token service.

        Args:
            config: JWT configuration
        """
        self.config = config

    def create_access_token(
        self,
        data: dict[str, Any],
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """Create access token.

        Args:
            data: Data to encode in token
            expires_delta: Custom expiration time

        Returns:
            Encoded JWT token
        """
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                minutes=self.config.access_token_expire_minutes
            )

        to_encode.update({"exp": expire})

        return jwt.encode(
            to_encode,
            self.config.secret_key,
            algorithm=self.config.algorithm,
        )

    def create_refresh_token(
        self,
        data: dict[str, Any],
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """Create refresh token.

        Args:
            data: Data to encode in token
            expires_delta: Custom expiration time

        Returns:
            Encoded JWT token
        """
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                days=self.config.refresh_token_expire_days
            )

        to_encode.update({"exp": expire})

        return jwt.encode(
            to_encode,
            self.config.secret_key,
            algorithm=self.config.algorithm,
        )

    def decode_token(self, token: str) -> dict[str, Any]:
        """Decode and validate token.

        Args:
            token: JWT token to decode

        Returns:
            Decoded token payload

        Raises:
            TokenExpiredError: If token has expired
            InvalidTokenError: If token is invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise TokenExpiredError("Token has expired")
        except jwt.InvalidTokenError:
            raise InvalidTokenError("Invalid token")
