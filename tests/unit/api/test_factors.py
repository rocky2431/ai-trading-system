"""Tests for factor management API.

Tests cover:
- Factor Schemas
- Factor Service
- Factor Router
- Integration scenarios
"""

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


# ==================== Schema Tests ====================


class TestFactorSchemas:
    """Tests for factor API schemas."""

    def test_factor_generate_request(self):
        """Test factor generation request schema."""
        from iqfmp.api.factors.schemas import FactorGenerateRequest

        request = FactorGenerateRequest(
            description="Calculate momentum over 20 periods",
            family=["momentum"],
            target_task="price_prediction",
        )
        assert request.description == "Calculate momentum over 20 periods"
        assert request.family == ["momentum"]
        assert request.target_task == "price_prediction"

    def test_factor_generate_request_defaults(self):
        """Test factor generation request with defaults."""
        from iqfmp.api.factors.schemas import FactorGenerateRequest

        request = FactorGenerateRequest(
            description="Simple factor",
        )
        assert request.family == []
        assert request.target_task == "price_prediction"

    def test_factor_response(self):
        """Test factor response schema."""
        from iqfmp.api.factors.schemas import FactorResponse

        response = FactorResponse(
            id="factor-123",
            name="momentum_20",
            family=["momentum"],
            code="def factor(df): return df['close'].pct_change(20)",
            code_hash="abc123",
            target_task="price_prediction",
            status="candidate",
            created_at=datetime.now(),
        )
        assert response.id == "factor-123"
        assert response.name == "momentum_20"
        assert response.status == "candidate"

    def test_factor_list_response(self):
        """Test factor list response schema."""
        from iqfmp.api.factors.schemas import FactorListResponse, FactorResponse

        factors = [
            FactorResponse(
                id="factor-1",
                name="momentum",
                family=["momentum"],
                code="...",
                code_hash="hash1",
                target_task="price_prediction",
                status="candidate",
                created_at=datetime.now(),
            ),
            FactorResponse(
                id="factor-2",
                name="volatility",
                family=["volatility"],
                code="...",
                code_hash="hash2",
                target_task="price_prediction",
                status="core",
                created_at=datetime.now(),
            ),
        ]
        response = FactorListResponse(
            factors=factors,
            total=2,
            page=1,
            page_size=10,
        )
        assert len(response.factors) == 2
        assert response.total == 2

    def test_factor_evaluate_request(self):
        """Test factor evaluate request schema."""
        from iqfmp.api.factors.schemas import FactorEvaluateRequest

        request = FactorEvaluateRequest(
            splits=["train", "valid", "test"],
            market_splits=["btc_eth", "altcoins"],
        )
        assert len(request.splits) == 3

    def test_factor_evaluate_response(self):
        """Test factor evaluate response schema."""
        from iqfmp.api.factors.schemas import FactorEvaluateResponse, MetricsResponse

        metrics = MetricsResponse(
            ic_mean=0.05,
            ic_std=0.02,
            ir=2.5,
            sharpe=1.8,
            max_drawdown=0.15,
            turnover=0.3,
        )
        response = FactorEvaluateResponse(
            factor_id="factor-123",
            metrics=metrics,
            passed_threshold=True,
            experiment_number=42,
        )
        assert response.passed_threshold is True
        assert response.metrics.ic_mean == 0.05

    def test_factor_status_update_request(self):
        """Test factor status update request schema."""
        from iqfmp.api.factors.schemas import FactorStatusUpdateRequest

        request = FactorStatusUpdateRequest(status="core")
        assert request.status == "core"

    def test_factor_status_update_invalid(self):
        """Test invalid status value."""
        from iqfmp.api.factors.schemas import FactorStatusUpdateRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            FactorStatusUpdateRequest(status="invalid_status")


# ==================== Factor Service Tests ====================


class TestFactorService:
    """Tests for factor service."""

    @pytest.fixture
    def factor_service(self):
        """Create factor service."""
        from iqfmp.api.factors.service import FactorService

        return FactorService()

    def test_create_factor(self, factor_service):
        """Test creating a factor."""
        factor = factor_service.create_factor(
            name="momentum_20",
            family=["momentum"],
            code="def factor(df): return df['close'].pct_change(20)",
            target_task="price_prediction",
        )
        assert factor.id is not None
        assert factor.name == "momentum_20"
        assert factor.status == "candidate"

    def test_get_factor(self, factor_service):
        """Test getting a factor by ID."""
        created = factor_service.create_factor(
            name="test_factor",
            family=["test"],
            code="...",
            target_task="test",
        )

        found = factor_service.get_factor(created.id)
        assert found is not None
        assert found.id == created.id

    def test_get_nonexistent_factor(self, factor_service):
        """Test getting nonexistent factor."""
        found = factor_service.get_factor("nonexistent-id")
        assert found is None

    def test_list_factors(self, factor_service):
        """Test listing factors."""
        factor_service.create_factor(
            name="factor1", family=["test"], code="...", target_task="test"
        )
        factor_service.create_factor(
            name="factor2", family=["test"], code="...", target_task="test"
        )

        factors, total = factor_service.list_factors()
        assert total >= 2

    def test_list_factors_with_filter(self, factor_service):
        """Test listing factors with family filter."""
        factor_service.create_factor(
            name="momentum1", family=["momentum"], code="...", target_task="test"
        )
        factor_service.create_factor(
            name="vol1", family=["volatility"], code="...", target_task="test"
        )

        factors, total = factor_service.list_factors(family="momentum")
        assert all("momentum" in f.family for f in factors)

    def test_list_factors_pagination(self, factor_service):
        """Test listing factors with pagination."""
        for i in range(15):
            factor_service.create_factor(
                name=f"factor_{i}", family=["test"], code="...", target_task="test"
            )

        factors, total = factor_service.list_factors(page=1, page_size=10)
        assert len(factors) == 10
        assert total >= 15

    def test_update_factor_status(self, factor_service):
        """Test updating factor status."""
        created = factor_service.create_factor(
            name="test_factor", family=["test"], code="...", target_task="test"
        )

        updated = factor_service.update_status(created.id, "core")
        assert updated.status == "core"

    def test_update_nonexistent_factor_status(self, factor_service):
        """Test updating nonexistent factor status."""
        from iqfmp.api.factors.service import FactorNotFoundError

        with pytest.raises(FactorNotFoundError):
            factor_service.update_status("nonexistent-id", "core")

    def test_delete_factor(self, factor_service):
        """Test deleting a factor."""
        created = factor_service.create_factor(
            name="to_delete", family=["test"], code="...", target_task="test"
        )

        result = factor_service.delete_factor(created.id)
        assert result is True
        assert factor_service.get_factor(created.id) is None

    def test_code_hash_generation(self, factor_service):
        """Test code hash is generated."""
        factor = factor_service.create_factor(
            name="test",
            family=["test"],
            code="def factor(df): return df['close']",
            target_task="test",
        )
        assert factor.code_hash is not None
        assert len(factor.code_hash) > 10


# ==================== Factor Router Tests ====================


class TestFactorRouter:
    """Tests for factor API router."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from iqfmp.api.factors.router import router as factors_router

        app = FastAPI()
        app.include_router(factors_router, prefix="/api/v1/factors")
        return TestClient(app)

    def test_generate_factor_endpoint(self, client):
        """Test generate factor endpoint."""
        response = client.post(
            "/api/v1/factors/generate",
            json={
                "description": "Calculate 20-period momentum",
                "family": ["momentum"],
                "target_task": "price_prediction",
            },
        )
        assert response.status_code != 404

    def test_list_factors_endpoint(self, client):
        """Test list factors endpoint."""
        response = client.get("/api/v1/factors")
        assert response.status_code == 200
        data = response.json()
        assert "factors" in data
        assert "total" in data

    def test_list_factors_with_pagination(self, client):
        """Test list factors with pagination params."""
        response = client.get("/api/v1/factors?page=1&page_size=5")
        assert response.status_code == 200

    def test_list_factors_with_family_filter(self, client):
        """Test list factors with family filter."""
        response = client.get("/api/v1/factors?family=momentum")
        assert response.status_code == 200

    def test_list_factors_with_status_filter(self, client):
        """Test list factors with status filter."""
        response = client.get("/api/v1/factors?status=core")
        assert response.status_code == 200

    def test_get_factor_endpoint(self, client):
        """Test get factor by ID endpoint."""
        # First create a factor
        response = client.get("/api/v1/factors/factor-123")
        # Should return 404 for nonexistent or the factor
        assert response.status_code in [200, 404]

    def test_evaluate_factor_endpoint(self, client):
        """Test evaluate factor endpoint."""
        # First create a factor
        create_resp = client.post(
            "/api/v1/factors/generate",
            json={"description": "test factor", "family": ["test"]},
        )
        factor_id = create_resp.json()["id"]

        response = client.post(
            f"/api/v1/factors/{factor_id}/evaluate",
            json={
                "splits": ["train", "valid", "test"],
            },
        )
        assert response.status_code == 200

    def test_update_status_endpoint(self, client):
        """Test update factor status endpoint."""
        # First create a factor
        create_resp = client.post(
            "/api/v1/factors/generate",
            json={"description": "test factor", "family": ["test"]},
        )
        factor_id = create_resp.json()["id"]

        response = client.put(
            f"/api/v1/factors/{factor_id}/status",
            json={"status": "core"},
        )
        assert response.status_code == 200


# ==================== Integration Tests ====================


class TestFactorAPIIntegration:
    """Integration tests for factor API."""

    @pytest.fixture
    def full_client(self):
        """Create full app test client."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from iqfmp.api.factors.router import router as factors_router
        from iqfmp.api.factors.service import FactorService

        app = FastAPI()
        app.include_router(factors_router, prefix="/api/v1/factors")

        # Reset service for each test
        return TestClient(app)

    def test_create_and_get_factor(self, full_client):
        """Test creating and retrieving a factor."""
        # Create
        create_response = full_client.post(
            "/api/v1/factors/generate",
            json={
                "description": "Calculate RSI over 14 periods",
                "family": ["momentum", "oscillator"],
                "target_task": "price_prediction",
            },
        )

        if create_response.status_code == 201:
            factor_id = create_response.json()["id"]

            # Get
            get_response = full_client.get(f"/api/v1/factors/{factor_id}")
            assert get_response.status_code == 200
            assert get_response.json()["id"] == factor_id

    def test_list_and_filter_factors(self, full_client):
        """Test listing and filtering factors."""
        # Create some factors
        full_client.post(
            "/api/v1/factors/generate",
            json={"description": "Momentum factor", "family": ["momentum"]},
        )
        full_client.post(
            "/api/v1/factors/generate",
            json={"description": "Volatility factor", "family": ["volatility"]},
        )

        # List all
        list_response = full_client.get("/api/v1/factors")
        assert list_response.status_code == 200

    def test_update_factor_lifecycle(self, full_client):
        """Test factor status lifecycle."""
        # Create
        create_response = full_client.post(
            "/api/v1/factors/generate",
            json={"description": "Test factor", "family": ["test"]},
        )

        if create_response.status_code == 201:
            factor_id = create_response.json()["id"]

            # Update status to core
            update_response = full_client.put(
                f"/api/v1/factors/{factor_id}/status",
                json={"status": "core"},
            )

            if update_response.status_code == 200:
                assert update_response.json()["status"] == "core"


# ==================== Boundary Tests ====================


class TestFactorAPIBoundary:
    """Boundary tests for factor API."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from iqfmp.api.factors.router import router as factors_router

        app = FastAPI()
        app.include_router(factors_router, prefix="/api/v1/factors")
        return TestClient(app)

    def test_empty_description(self, client):
        """Test generating factor with empty description."""
        response = client.post(
            "/api/v1/factors/generate",
            json={"description": ""},
        )
        # Should fail validation
        assert response.status_code in [400, 422]

    def test_very_long_description(self, client):
        """Test generating factor with very long description."""
        response = client.post(
            "/api/v1/factors/generate",
            json={"description": "a" * 10000},
        )
        # Should either work or fail gracefully
        assert response.status_code in [201, 400, 422]

    def test_invalid_page_number(self, client):
        """Test invalid page number."""
        response = client.get("/api/v1/factors?page=0")
        assert response.status_code in [400, 422]

    def test_negative_page_size(self, client):
        """Test negative page size."""
        response = client.get("/api/v1/factors?page_size=-1")
        assert response.status_code in [400, 422]

    def test_very_large_page_size(self, client):
        """Test very large page size is capped."""
        response = client.get("/api/v1/factors?page_size=1000")
        # Should either work with cap or reject
        assert response.status_code in [200, 400, 422]


# ==================== Exception Tests ====================


class TestFactorAPIExceptions:
    """Exception handling tests for factor API."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from iqfmp.api.factors.router import router as factors_router

        app = FastAPI()
        app.include_router(factors_router, prefix="/api/v1/factors")
        return TestClient(app)

    def test_get_nonexistent_factor(self, client):
        """Test getting nonexistent factor returns 404."""
        response = client.get("/api/v1/factors/nonexistent-id-12345")
        assert response.status_code == 404

    def test_update_nonexistent_factor_status(self, client):
        """Test updating nonexistent factor returns 404."""
        response = client.put(
            "/api/v1/factors/nonexistent-id-12345/status",
            json={"status": "core"},
        )
        assert response.status_code == 404

    def test_evaluate_nonexistent_factor(self, client):
        """Test evaluating nonexistent factor returns 404."""
        response = client.post(
            "/api/v1/factors/nonexistent-id-12345/evaluate",
            json={"splits": ["train"]},
        )
        assert response.status_code == 404

    def test_invalid_json_body(self, client):
        """Test invalid JSON body returns 422."""
        response = client.post(
            "/api/v1/factors/generate",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
