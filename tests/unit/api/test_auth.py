"""Tests for authentication module.

Tests cover:
- JWTConfig
- TokenService
- PasswordService
- AuthDependencies
- Auth Router
"""

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


# ==================== JWTConfig Tests ====================


class TestJWTConfig:
    """Tests for JWT configuration."""

    def test_default_config(self):
        """Test default JWT configuration."""
        from iqfmp.api.auth.config import JWTConfig

        config = JWTConfig()
        assert config.secret_key is not None
        assert config.algorithm == "HS256"
        assert config.access_token_expire_minutes == 30
        assert config.refresh_token_expire_days == 7

    def test_custom_config(self):
        """Test custom JWT configuration."""
        from iqfmp.api.auth.config import JWTConfig

        config = JWTConfig(
            secret_key="custom-secret",
            algorithm="HS512",
            access_token_expire_minutes=60,
            refresh_token_expire_days=14,
        )
        assert config.secret_key == "custom-secret"
        assert config.algorithm == "HS512"
        assert config.access_token_expire_minutes == 60
        assert config.refresh_token_expire_days == 14


# ==================== TokenService Tests ====================


class TestTokenService:
    """Tests for token service."""

    @pytest.fixture
    def token_service(self):
        """Create token service with test config."""
        from iqfmp.api.auth.config import JWTConfig
        from iqfmp.api.auth.token import TokenService

        config = JWTConfig(secret_key="test-secret-key-for-testing")
        return TokenService(config)

    def test_create_access_token(self, token_service):
        """Test creating access token."""
        token = token_service.create_access_token(
            data={"sub": "user123", "role": "admin"}
        )
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 50  # JWT tokens are long

    def test_create_refresh_token(self, token_service):
        """Test creating refresh token."""
        token = token_service.create_refresh_token(data={"sub": "user123"})
        assert token is not None
        assert isinstance(token, str)

    def test_decode_valid_token(self, token_service):
        """Test decoding valid token."""
        original_data = {"sub": "user123", "role": "admin"}
        token = token_service.create_access_token(data=original_data)

        decoded = token_service.decode_token(token)
        assert decoded["sub"] == "user123"
        assert decoded["role"] == "admin"
        assert "exp" in decoded

    def test_decode_expired_token(self, token_service):
        """Test decoding expired token raises error."""
        from iqfmp.api.auth.token import TokenExpiredError

        # Create token that expires immediately
        token = token_service.create_access_token(
            data={"sub": "user123"},
            expires_delta=timedelta(seconds=-1),  # Already expired
        )

        with pytest.raises(TokenExpiredError):
            token_service.decode_token(token)

    def test_decode_invalid_token(self, token_service):
        """Test decoding invalid token raises error."""
        from iqfmp.api.auth.token import InvalidTokenError

        with pytest.raises(InvalidTokenError):
            token_service.decode_token("invalid.token.here")

    def test_decode_tampered_token(self, token_service):
        """Test decoding tampered token raises error."""
        from iqfmp.api.auth.token import InvalidTokenError

        token = token_service.create_access_token(data={"sub": "user123"})
        # Tamper with the token
        tampered = token[:-5] + "xxxxx"

        with pytest.raises(InvalidTokenError):
            token_service.decode_token(tampered)

    def test_token_contains_expiry(self, token_service):
        """Test token contains expiry time."""
        token = token_service.create_access_token(data={"sub": "user123"})
        decoded = token_service.decode_token(token)
        assert "exp" in decoded

    def test_verify_token_type(self, token_service):
        """Test verifying token type."""
        access_token = token_service.create_access_token(
            data={"sub": "user123", "type": "access"}
        )
        refresh_token = token_service.create_refresh_token(
            data={"sub": "user123", "type": "refresh"}
        )

        # Both should decode successfully
        access_decoded = token_service.decode_token(access_token)
        refresh_decoded = token_service.decode_token(refresh_token)

        assert access_decoded["type"] == "access"
        assert refresh_decoded["type"] == "refresh"


# ==================== PasswordService Tests ====================


class TestPasswordService:
    """Tests for password service."""

    @pytest.fixture
    def password_service(self):
        """Create password service."""
        from iqfmp.api.auth.password import PasswordService

        return PasswordService()

    def test_hash_password(self, password_service):
        """Test password hashing."""
        password = "secure_password123"
        hashed = password_service.hash_password(password)

        assert hashed is not None
        assert hashed != password
        assert len(hashed) > 20

    def test_verify_correct_password(self, password_service):
        """Test verifying correct password."""
        password = "secure_password123"
        hashed = password_service.hash_password(password)

        assert password_service.verify_password(password, hashed) is True

    def test_verify_incorrect_password(self, password_service):
        """Test verifying incorrect password."""
        password = "secure_password123"
        hashed = password_service.hash_password(password)

        assert password_service.verify_password("wrong_password", hashed) is False

    def test_different_passwords_different_hashes(self, password_service):
        """Test same password produces different hashes (salt)."""
        password = "same_password"
        hash1 = password_service.hash_password(password)
        hash2 = password_service.hash_password(password)

        assert hash1 != hash2  # Bcrypt adds random salt

    def test_hash_empty_password(self, password_service):
        """Test hashing empty password."""
        hashed = password_service.hash_password("")
        assert hashed is not None
        assert password_service.verify_password("", hashed) is True


# ==================== User Model Tests ====================


class TestUserModel:
    """Tests for user model."""

    def test_user_creation(self):
        """Test creating user model."""
        from iqfmp.api.auth.models import User, UserRole

        user = User(
            id="user-123",
            username="testuser",
            email="test@example.com",
            hashed_password="hashed",
            role=UserRole.USER,
        )
        assert user.id == "user-123"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == UserRole.USER
        assert user.is_active is True

    def test_user_roles(self):
        """Test user role enum."""
        from iqfmp.api.auth.models import UserRole

        assert UserRole.ADMIN.value == "admin"
        assert UserRole.USER.value == "user"
        assert UserRole.VIEWER.value == "viewer"

    def test_user_create_schema(self):
        """Test user creation schema."""
        from iqfmp.api.auth.schemas import UserCreate

        user_data = UserCreate(
            username="newuser",
            email="new@example.com",
            password="secure123",
        )
        assert user_data.username == "newuser"
        assert user_data.email == "new@example.com"
        assert user_data.password == "secure123"

    def test_user_response_schema(self):
        """Test user response schema (no password)."""
        from iqfmp.api.auth.schemas import UserResponse

        response = UserResponse(
            id="user-123",
            username="testuser",
            email="test@example.com",
            role="user",
            is_active=True,
        )
        # Password should not be in response
        assert not hasattr(response, "password")
        assert not hasattr(response, "hashed_password")


# ==================== Token Schema Tests ====================


class TestTokenSchemas:
    """Tests for token schemas."""

    def test_token_response(self):
        """Test token response schema."""
        from iqfmp.api.auth.schemas import TokenResponse

        response = TokenResponse(
            access_token="access.token.here",
            refresh_token="refresh.token.here",
            token_type="bearer",
        )
        assert response.access_token == "access.token.here"
        assert response.token_type == "bearer"

    def test_login_request(self):
        """Test login request schema."""
        from iqfmp.api.auth.schemas import LoginRequest

        request = LoginRequest(
            username="testuser",
            password="password123",
        )
        assert request.username == "testuser"
        assert request.password == "password123"


# ==================== Auth Dependencies Tests ====================


class TestAuthDependencies:
    """Tests for authentication dependencies."""

    @pytest.fixture
    def mock_token_service(self):
        """Create mock token service."""
        from iqfmp.api.auth.token import TokenService

        service = MagicMock(spec=TokenService)
        return service

    def test_get_current_user_valid_token(self, mock_token_service):
        """Test getting current user with valid token."""
        from iqfmp.api.auth.dependencies import get_current_user
        from iqfmp.api.auth.models import User, UserRole

        mock_token_service.decode_token.return_value = {
            "sub": "user-123",
            "username": "testuser",
            "role": "user",
        }

        # This would need async test context
        # For now, just verify the function exists
        assert callable(get_current_user)

    def test_require_admin_role(self):
        """Test admin role requirement."""
        from iqfmp.api.auth.dependencies import require_role
        from iqfmp.api.auth.models import UserRole

        admin_checker = require_role(UserRole.ADMIN)
        assert callable(admin_checker)


# ==================== Auth Router Tests ====================


class TestAuthRouter:
    """Tests for authentication router."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from iqfmp.api.auth.router import router as auth_router

        app = FastAPI()
        app.include_router(auth_router, prefix="/api/v1/auth")
        return TestClient(app)

    def test_register_endpoint_exists(self, client):
        """Test register endpoint exists."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "newuser",
                "email": "new@example.com",
                "password": "secure123",
            },
        )
        # Should not be 404
        assert response.status_code != 404

    def test_login_endpoint_exists(self, client):
        """Test login endpoint exists."""
        response = client.post(
            "/api/v1/auth/login",
            json={
                "username": "testuser",
                "password": "password123",
            },
        )
        # Should not be 404
        assert response.status_code != 404

    def test_refresh_endpoint_exists(self, client):
        """Test refresh token endpoint exists."""
        response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": "some.token.here"},
        )
        # Should not be 404
        assert response.status_code != 404

    def test_me_endpoint_exists(self, client):
        """Test me endpoint exists."""
        response = client.get("/api/v1/auth/me")
        # Should not be 404 (might be 401 unauthorized)
        assert response.status_code != 404


# ==================== UserService Tests ====================


class TestUserService:
    """Tests for user service."""

    @pytest.fixture
    def user_service(self):
        """Create user service with mock storage."""
        from iqfmp.api.auth.service import UserService

        return UserService()

    def test_create_user(self, user_service):
        """Test creating a new user."""
        from iqfmp.api.auth.schemas import UserCreate

        user_data = UserCreate(
            username="newuser",
            email="new@example.com",
            password="secure123",
        )

        user = user_service.create_user(user_data)
        assert user.username == "newuser"
        assert user.email == "new@example.com"
        assert user.id is not None

    def test_get_user_by_username(self, user_service):
        """Test getting user by username."""
        from iqfmp.api.auth.schemas import UserCreate

        user_data = UserCreate(
            username="findme",
            email="find@example.com",
            password="secure123",
        )
        user_service.create_user(user_data)

        found = user_service.get_by_username("findme")
        assert found is not None
        assert found.username == "findme"

    def test_get_nonexistent_user(self, user_service):
        """Test getting nonexistent user returns None."""
        found = user_service.get_by_username("nonexistent")
        assert found is None

    def test_authenticate_valid_credentials(self, user_service):
        """Test authenticating with valid credentials."""
        from iqfmp.api.auth.schemas import UserCreate

        user_data = UserCreate(
            username="authuser",
            email="auth@example.com",
            password="correct_password",
        )
        user_service.create_user(user_data)

        authenticated = user_service.authenticate("authuser", "correct_password")
        assert authenticated is not None
        assert authenticated.username == "authuser"

    def test_authenticate_invalid_password(self, user_service):
        """Test authenticating with invalid password."""
        from iqfmp.api.auth.schemas import UserCreate

        user_data = UserCreate(
            username="authuser2",
            email="auth2@example.com",
            password="correct_password",
        )
        user_service.create_user(user_data)

        authenticated = user_service.authenticate("authuser2", "wrong_password")
        assert authenticated is None

    def test_duplicate_username_raises_error(self, user_service):
        """Test creating duplicate username raises error."""
        from iqfmp.api.auth.schemas import UserCreate
        from iqfmp.api.auth.service import UserAlreadyExistsError

        user_data = UserCreate(
            username="duplicate",
            email="first@example.com",
            password="secure123",
        )
        user_service.create_user(user_data)

        duplicate_data = UserCreate(
            username="duplicate",
            email="second@example.com",
            password="secure123",
        )

        with pytest.raises(UserAlreadyExistsError):
            user_service.create_user(duplicate_data)


# ==================== Integration Tests ====================


class TestAuthIntegration:
    """Integration tests for auth flow."""

    @pytest.fixture
    def full_client(self):
        """Create full app test client."""
        from iqfmp.api.main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        return TestClient(app)

    def test_health_check(self, full_client):
        """Test health check endpoint."""
        response = full_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


# ==================== Security Tests ====================


class TestSecurityMeasures:
    """Security tests for auth module."""

    def test_password_not_in_logs(self):
        """Test password is not logged."""
        from iqfmp.api.auth.schemas import LoginRequest

        request = LoginRequest(username="user", password="secret123")
        # __repr__ should not contain password
        repr_str = repr(request)
        assert "secret123" not in repr_str

    def test_token_has_expiry(self):
        """Test tokens always have expiry."""
        from iqfmp.api.auth.config import JWTConfig
        from iqfmp.api.auth.token import TokenService

        config = JWTConfig(secret_key="test-secret")
        service = TokenService(config)

        token = service.create_access_token(data={"sub": "user"})
        decoded = service.decode_token(token)

        assert "exp" in decoded


# ==================== Boundary Tests ====================


class TestAuthBoundary:
    """Boundary tests for auth module."""

    def test_very_long_username(self):
        """Test handling very long username."""
        from iqfmp.api.auth.schemas import UserCreate
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            UserCreate(
                username="a" * 256,  # Too long
                email="test@example.com",
                password="secure123",
            )

    def test_empty_password(self):
        """Test handling empty password."""
        from iqfmp.api.auth.schemas import UserCreate
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            UserCreate(
                username="user",
                email="test@example.com",
                password="",  # Empty
            )

    def test_invalid_email(self):
        """Test handling invalid email."""
        from iqfmp.api.auth.schemas import UserCreate
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            UserCreate(
                username="user",
                email="not-an-email",
                password="secure123",
            )
