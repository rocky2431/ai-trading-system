"""User service for authentication."""

import uuid
from typing import Optional

from iqfmp.api.auth.models import User, UserRole
from iqfmp.api.auth.password import PasswordService
from iqfmp.api.auth.schemas import UserCreate


class UserAlreadyExistsError(Exception):
    """Raised when user already exists."""

    pass


class UserService:
    """Service for user management."""

    def __init__(self) -> None:
        """Initialize user service."""
        self._users: dict[str, User] = {}
        self._password_service = PasswordService()

    def create_user(self, user_data: UserCreate) -> User:
        """Create a new user.

        Args:
            user_data: User creation data

        Returns:
            Created user

        Raises:
            UserAlreadyExistsError: If username already exists
        """
        # Check for duplicate username
        if self.get_by_username(user_data.username):
            raise UserAlreadyExistsError(
                f"User with username '{user_data.username}' already exists"
            )

        # Check for duplicate email
        for user in self._users.values():
            if user.email == user_data.email:
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

        self._users[user_id] = user
        return user

    def get_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID.

        Args:
            user_id: User ID

        Returns:
            User if found, None otherwise
        """
        return self._users.get(user_id)

    def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username.

        Args:
            username: Username to search

        Returns:
            User if found, None otherwise
        """
        for user in self._users.values():
            if user.username == username:
                return user
        return None

    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email.

        Args:
            email: Email to search

        Returns:
            User if found, None otherwise
        """
        for user in self._users.values():
            if user.email == email:
                return user
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
