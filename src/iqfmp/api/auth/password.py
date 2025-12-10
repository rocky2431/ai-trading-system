"""Password service for hashing and verification."""

import bcrypt


class PasswordService:
    """Service for password hashing and verification."""

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt.

        Args:
            password: Plain text password

        Returns:
            Hashed password
        """
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed.decode("utf-8")

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash.

        Args:
            password: Plain text password to verify
            hashed: Hashed password to compare against

        Returns:
            True if password matches, False otherwise
        """
        return bcrypt.checkpw(
            password.encode("utf-8"),
            hashed.encode("utf-8"),
        )
