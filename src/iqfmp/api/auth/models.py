"""User models for authentication."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class UserRole(Enum):
    """User role enumeration."""

    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


@dataclass
class User:
    """User model."""

    id: str
    username: str
    email: str
    hashed_password: str
    role: UserRole = UserRole.USER
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
