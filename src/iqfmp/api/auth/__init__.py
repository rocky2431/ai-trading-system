"""Authentication module for IQFMP API.

Provides JWT-based authentication:
- JWTConfig: JWT configuration
- TokenService: Token creation and validation
- PasswordService: Password hashing and verification
- UserService: User management
- AuthDependencies: FastAPI dependencies for auth
"""

from iqfmp.api.auth.config import JWTConfig
from iqfmp.api.auth.dependencies import get_current_user, require_role
from iqfmp.api.auth.models import User, UserRole
from iqfmp.api.auth.password import PasswordService
from iqfmp.api.auth.router import router
from iqfmp.api.auth.schemas import (
    LoginRequest,
    TokenResponse,
    UserCreate,
    UserResponse,
)
from iqfmp.api.auth.service import UserAlreadyExistsError, UserService
from iqfmp.api.auth.token import InvalidTokenError, TokenExpiredError, TokenService

__all__ = [
    # Config
    "JWTConfig",
    # Services
    "TokenService",
    "PasswordService",
    "UserService",
    # Models
    "User",
    "UserRole",
    # Schemas
    "LoginRequest",
    "TokenResponse",
    "UserCreate",
    "UserResponse",
    # Dependencies
    "get_current_user",
    "require_role",
    # Exceptions
    "InvalidTokenError",
    "TokenExpiredError",
    "UserAlreadyExistsError",
    # Router
    "router",
]
