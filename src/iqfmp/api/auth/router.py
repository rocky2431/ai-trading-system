"""Authentication router for FastAPI."""

from fastapi import APIRouter, Depends, HTTPException, status

from iqfmp.api.auth.config import JWTConfig
from iqfmp.api.auth.dependencies import (
    get_current_user,
    get_token_service,
    get_user_service,
)
from iqfmp.api.auth.models import User
from iqfmp.api.auth.schemas import (
    LoginRequest,
    RefreshTokenRequest,
    TokenResponse,
    UserCreate,
    UserResponse,
)
from iqfmp.api.auth.service import UserAlreadyExistsError, UserService
from iqfmp.api.auth.token import InvalidTokenError, TokenExpiredError, TokenService

router = APIRouter(tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    user_service: UserService = Depends(get_user_service),
) -> UserResponse:
    """Register a new user.

    Args:
        user_data: User registration data
        user_service: User service

    Returns:
        Created user information
    """
    try:
        user = user_service.create_user(user_data)
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            role=user.role.value,
            is_active=user.is_active,
        )
    except UserAlreadyExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )


@router.post("/login", response_model=TokenResponse)
async def login(
    login_data: LoginRequest,
    user_service: UserService = Depends(get_user_service),
    token_service: TokenService = Depends(get_token_service),
) -> TokenResponse:
    """Login and get access token.

    Args:
        login_data: Login credentials
        user_service: User service
        token_service: Token service

    Returns:
        Access and refresh tokens
    """
    user = user_service.authenticate(login_data.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    access_token = token_service.create_access_token(
        data={
            "sub": user.id,
            "username": user.username,
            "role": user.role.value,
            "type": "access",
        }
    )
    refresh_token = token_service.create_refresh_token(
        data={
            "sub": user.id,
            "type": "refresh",
        }
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_data: RefreshTokenRequest,
    token_service: TokenService = Depends(get_token_service),
    user_service: UserService = Depends(get_user_service),
) -> TokenResponse:
    """Refresh access token using refresh token.

    Args:
        refresh_data: Refresh token data
        token_service: Token service
        user_service: User service

    Returns:
        New access token
    """
    try:
        payload = token_service.decode_token(refresh_data.refresh_token)

        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
            )

        user_id = payload.get("sub")
        user = user_service.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )

        access_token = token_service.create_access_token(
            data={
                "sub": user.id,
                "username": user.username,
                "role": user.role.value,
                "type": "access",
            }
        )

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
        )

    except TokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired",
        )
    except InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    """Get current user information.

    Args:
        current_user: Current authenticated user

    Returns:
        Current user information
    """
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        role=current_user.role.value,
        is_active=current_user.is_active,
    )
