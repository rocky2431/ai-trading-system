"""Authentication dependencies for FastAPI."""

from typing import Callable, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from iqfmp.api.auth.config import JWTConfig
from iqfmp.api.auth.models import User, UserRole
from iqfmp.api.auth.service import UserService
from iqfmp.api.auth.token import InvalidTokenError, TokenExpiredError, TokenService

# Shared instances (in production, use dependency injection)
_jwt_config = JWTConfig()
_token_service = TokenService(_jwt_config)
_user_service = UserService()

security = HTTPBearer(auto_error=False)


def get_token_service() -> TokenService:
    """Get token service instance."""
    return _token_service


def get_user_service() -> UserService:
    """Get user service instance."""
    return _user_service


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    token_service: TokenService = Depends(get_token_service),
    user_service: UserService = Depends(get_user_service),
) -> User:
    """Get current authenticated user.

    Args:
        credentials: HTTP Bearer credentials
        token_service: Token service
        user_service: User service

    Returns:
        Current user

    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = token_service.decode_token(credentials.credentials)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )

        user = user_service.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is inactive",
            )

        return user

    except TokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_role(role: UserRole) -> Callable:
    """Create a dependency that requires a specific role.

    Args:
        role: Required role

    Returns:
        Dependency function
    """

    async def role_checker(
        current_user: User = Depends(get_current_user),
    ) -> User:
        """Check if user has required role."""
        if current_user.role != role and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role.value}' required",
            )
        return current_user

    return role_checker
