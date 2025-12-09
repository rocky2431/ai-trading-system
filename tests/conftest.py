"""Pytest configuration and fixtures."""

import pytest
from fastapi.testclient import TestClient

from iqfmp.api.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def sample_factor_data() -> dict:
    """Sample factor data for testing."""
    return {
        "id": "test-factor-001",
        "name": "momentum_20d",
        "family": ["momentum", "trend"],
        "code": "def factor(df): return df['close'].pct_change(20)",
        "code_hash": "abc123",
        "target_task": "1d_trend",
    }
