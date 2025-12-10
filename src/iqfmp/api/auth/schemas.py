"""Pydantic schemas for authentication."""

from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    """Schema for user creation."""

    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)


class UserResponse(BaseModel):
    """Schema for user response (no password)."""

    id: str
    username: str
    email: str
    role: str
    is_active: bool


class LoginRequest(BaseModel):
    """Schema for login request."""

    username: str
    password: str

    def __repr__(self) -> str:
        """Hide password in repr."""
        return f"LoginRequest(username={self.username!r}, password='***')"


class TokenResponse(BaseModel):
    """Schema for token response."""

    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"


class RefreshTokenRequest(BaseModel):
    """Schema for refresh token request."""

    refresh_token: str
