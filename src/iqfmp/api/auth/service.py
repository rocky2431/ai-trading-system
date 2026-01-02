"""User service for authentication with Redis persistence."""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from iqfmp.api.auth.models import User, UserRole
from iqfmp.api.auth.password import PasswordService
from iqfmp.api.auth.schemas import UserCreate

logger = logging.getLogger(__name__)


class UserAlreadyExistsError(Exception):
    """Raised when user already exists."""

    pass


class UserServiceError(Exception):
    """Raised when user service operations fail."""

    pass


class UserService:
    """Service for user management with Redis persistence.

    Critical state per CLAUDE.md: User data must be persistent to survive
    service restarts. Uses Redis for persistence.
    """

    REDIS_KEY_PREFIX = "iqfmp:users:"
    REDIS_USERNAME_INDEX = "iqfmp:users:by_username"
    REDIS_EMAIL_INDEX = "iqfmp:users:by_email"

    def __init__(self, redis_client: Any = None) -> None:
        """Initialize user service with Redis persistence.

        Args:
            redis_client: Optional Redis client for dependency injection (testing)

        Raises:
            UserServiceError: If Redis is unavailable and no client injected
        """
        self._redis = redis_client if redis_client is not None else self._get_redis_client()
        self._password_service = PasswordService()

    def _get_redis_client(self) -> Any:
        """Get Redis client. Raises if unavailable (critical state requires persistence)."""
        try:
            from iqfmp.db import get_redis_client
            client = get_redis_client()
            if client is None:
                raise UserServiceError(
                    "Redis unavailable. User data requires persistent storage "
                    "per CLAUDE.md critical state rules."
                )
            return client
        except UserServiceError:
            raise
        except Exception as e:
            raise UserServiceError(
                f"Failed to connect to Redis for user storage: {e}"
            ) from e

    def _serialize_user(self, user: User) -> str:
        """Serialize User to JSON for Redis storage."""
        return json.dumps({
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "hashed_password": user.hashed_password,
            "role": user.role.value,
            "is_active": user.is_active,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "updated_at": user.updated_at.isoformat() if user.updated_at else None,
        })

    def _deserialize_user(self, data: str) -> User:
        """Deserialize JSON to User."""
        obj = json.loads(data)
        return User(
            id=obj["id"],
            username=obj["username"],
            email=obj["email"],
            hashed_password=obj["hashed_password"],
            role=UserRole(obj["role"]),
            is_active=obj["is_active"],
            created_at=datetime.fromisoformat(obj["created_at"]) if obj.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(obj["updated_at"]) if obj.get("updated_at") else None,
        )

    def create_user(self, user_data: UserCreate) -> User:
        """Create a new user.

        Args:
            user_data: User creation data

        Returns:
            Created user

        Raises:
            UserAlreadyExistsError: If username or email already exists
        """
        # Check for duplicate username
        if self.get_by_username(user_data.username):
            raise UserAlreadyExistsError(
                f"User with username '{user_data.username}' already exists"
            )

        # Check for duplicate email
        if self.get_by_email(user_data.email):
            raise UserAlreadyExistsError(
                f"User with email '{user_data.email}' already exists"
            )

        user_id = str(uuid.uuid4())
        hashed_password = self._password_service.hash_password(user_data.password)

        user = User(
            id=user_id,
            username=user_data.username,
            email=user_data.email,
            hashed_password=hashed_password,
            role=UserRole.USER,
        )

        # Store user in Redis
        try:
            user_key = f"{self.REDIS_KEY_PREFIX}{user_id}"
            self._redis.set(user_key, self._serialize_user(user))
            # Create username -> user_id index
            self._redis.hset(self.REDIS_USERNAME_INDEX, user.username, user_id)
            # Create email -> user_id index
            self._redis.hset(self.REDIS_EMAIL_INDEX, user.email, user_id)
            logger.info(f"Created user {user.username} with ID {user_id}")
        except Exception as e:
            logger.error(f"Failed to persist user {user.username}: {e}")
            raise UserServiceError(f"Failed to create user: {e}") from e

        return user

    def get_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID.

        Args:
            user_id: User ID

        Returns:
            User if found, None otherwise
        """
        try:
            user_key = f"{self.REDIS_KEY_PREFIX}{user_id}"
            data = self._redis.get(user_key)
            if data:
                return self._deserialize_user(data)
        except Exception as e:
            logger.warning(f"Failed to get user by ID {user_id}: {e}")
        return None

    def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username.

        Args:
            username: Username to search

        Returns:
            User if found, None otherwise
        """
        try:
            user_id = self._redis.hget(self.REDIS_USERNAME_INDEX, username)
            if user_id:
                return self.get_by_id(user_id)
        except Exception as e:
            logger.warning(f"Failed to get user by username {username}: {e}")
        return None

    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email.

        Args:
            email: Email to search

        Returns:
            User if found, None otherwise
        """
        try:
            user_id = self._redis.hget(self.REDIS_EMAIL_INDEX, email)
            if user_id:
                return self.get_by_id(user_id)
        except Exception as e:
            logger.warning(f"Failed to get user by email {email}: {e}")
        return None

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with credentials.

        Args:
            username: Username
            password: Plain text password

        Returns:
            User if authentication successful, None otherwise
        """
        user = self.get_by_username(username)
        if not user:
            return None

        if not self._password_service.verify_password(password, user.hashed_password):
            return None

        if not user.is_active:
            return None

        return user

    def update_user(self, user: User) -> User:
        """Update user data.

        Args:
            user: User to update

        Returns:
            Updated user
        """
        user.updated_at = datetime.now()
        try:
            user_key = f"{self.REDIS_KEY_PREFIX}{user.id}"
            self._redis.set(user_key, self._serialize_user(user))
            logger.info(f"Updated user {user.username}")
        except Exception as e:
            logger.error(f"Failed to update user {user.username}: {e}")
            raise UserServiceError(f"Failed to update user: {e}") from e
        return user

    def delete_user(self, user_id: str) -> bool:
        """Delete user by ID.

        Args:
            user_id: User ID to delete

        Returns:
            True if deleted, False if not found
        """
        user = self.get_by_id(user_id)
        if not user:
            return False

        try:
            user_key = f"{self.REDIS_KEY_PREFIX}{user_id}"
            self._redis.delete(user_key)
            self._redis.hdel(self.REDIS_USERNAME_INDEX, user.username)
            self._redis.hdel(self.REDIS_EMAIL_INDEX, user.email)
            logger.info(f"Deleted user {user.username}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete user {user_id}: {e}")
            raise UserServiceError(f"Failed to delete user: {e}") from e
