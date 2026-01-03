"""Tests for pipeline API.

Tests cover:
- Pipeline Schemas
- Pipeline Service
- Pipeline Router
- WebSocket functionality
- Integration scenarios
"""

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


# ==================== Schema Tests ====================


class TestPipelineSchemas:
    """Tests for pipeline API schemas."""

    def test_pipeline_run_request(self):
        """Test pipeline run request schema."""
        from iqfmp.api.pipeline.schemas import PipelineRunRequest

        request = PipelineRunRequest(
            pipeline_type="factor_evaluation",
            config={
                "factor_id": "factor-123",
                "date_range": ["2024-01-01", "2024-12-31"],
            },
        )
        assert request.pipeline_type == "factor_evaluation"
        assert "factor_id" in request.config

    def test_pipeline_run_request_all_types(self):
        """Test all pipeline types are valid."""
        from iqfmp.api.pipeline.schemas import PipelineRunRequest

        for ptype in ["factor_evaluation", "strategy_backtest", "full_pipeline"]:
            request = PipelineRunRequest(pipeline_type=ptype)
            assert request.pipeline_type == ptype

    def test_pipeline_run_request_invalid_type(self):
        """Test invalid pipeline type is rejected."""
        from pydantic import ValidationError

        from iqfmp.api.pipeline.schemas import PipelineRunRequest

        with pytest.raises(ValidationError):
            PipelineRunRequest(pipeline_type="invalid_type")

    def test_pipeline_run_response(self):
        """Test pipeline run response schema."""
        from iqfmp.api.pipeline.schemas import PipelineRunResponse

        response = PipelineRunResponse(
            run_id="run-123",
            status="pending",
            created_at=datetime.now(),
        )
        assert response.run_id == "run-123"
        assert response.status == "pending"

    def test_pipeline_status_response(self):
        """Test pipeline status response schema."""
        from iqfmp.api.pipeline.schemas import PipelineStatusResponse

        response = PipelineStatusResponse(
            run_id="run-123",
            status="running",
            progress=0.5,
            current_step="evaluating_factors",
            started_at=datetime.now(),
        )
        assert response.run_id == "run-123"
        assert response.progress == 0.5

    def test_pipeline_status_response_completed(self):
        """Test completed pipeline status response."""
        from iqfmp.api.pipeline.schemas import PipelineStatusResponse

        response = PipelineStatusResponse(
            run_id="run-123",
            status="completed",
            progress=1.0,
            current_step="done",
            started_at=datetime.now(),
            completed_at=datetime.now(),
            result={"sharpe": 1.5, "max_dd": 0.1},
        )
        assert response.status == "completed"
        assert response.result is not None

    def test_pipeline_status_response_failed(self):
        """Test failed pipeline status response."""
        from iqfmp.api.pipeline.schemas import PipelineStatusResponse

        response = PipelineStatusResponse(
            run_id="run-123",
            status="failed",
            progress=0.3,
            current_step="evaluating_factors",
            started_at=datetime.now(),
            error="Factor evaluation failed",
        )
        assert response.status == "failed"
        assert response.error is not None

    def test_pipeline_config_schema(self):
        """Test pipeline config schema."""
        from iqfmp.api.pipeline.schemas import PipelineConfig

        config = PipelineConfig(
            factor_id="factor-123",
            strategy_id="strategy-456",
            date_range=["2024-01-01", "2024-12-31"],
            symbols=["BTC/USDT", "ETH/USDT"],
        )
        assert config.factor_id == "factor-123"
        assert len(config.symbols) == 2


# ==================== Pipeline Service Tests ====================


class TestPipelineService:
    """Tests for pipeline service."""

    @pytest.fixture
    def pipeline_service(self):
        """Create pipeline service."""
        from iqfmp.api.pipeline.service import PipelineService

        return PipelineService()

    def test_create_run(self, pipeline_service):
        """Test creating a pipeline run."""
        run = pipeline_service.create_run(
            pipeline_type="factor_evaluation",
            config={"factor_id": "factor-123"},
        )
        assert run.run_id is not None
        assert run.status == "pending"

    def test_get_run_status(self, pipeline_service):
        """Test getting run status."""
        # Create a run first
        run = pipeline_service.create_run(
            pipeline_type="factor_evaluation",
            config={},
        )

        status = pipeline_service.get_run_status(run.run_id)
        assert status is not None
        assert status.run_id == run.run_id

    def test_get_nonexistent_run_status(self, pipeline_service):
        """Test getting nonexistent run status."""
        status = pipeline_service.get_run_status("nonexistent-id")
        assert status is None

    def test_update_run_progress(self, pipeline_service):
        """Test updating run progress."""
        run = pipeline_service.create_run(
            pipeline_type="factor_evaluation",
            config={},
        )

        pipeline_service.update_run_progress(
            run_id=run.run_id,
            progress=0.5,
            current_step="evaluating",
        )

        status = pipeline_service.get_run_status(run.run_id)
        assert status.progress == 0.5
        assert status.current_step == "evaluating"

    def test_start_run(self, pipeline_service):
        """Test starting a run."""
        run = pipeline_service.create_run(
            pipeline_type="factor_evaluation",
            config={},
        )

        pipeline_service.start_run(run.run_id)

        status = pipeline_service.get_run_status(run.run_id)
        assert status.status == "running"
        assert status.started_at is not None

    def test_complete_run(self, pipeline_service):
        """Test completing a run."""
        run = pipeline_service.create_run(
            pipeline_type="factor_evaluation",
            config={},
        )
        pipeline_service.start_run(run.run_id)

        result = {"sharpe": 1.5}
        pipeline_service.complete_run(run.run_id, result=result)

        status = pipeline_service.get_run_status(run.run_id)
        assert status.status == "completed"
        assert status.result == result
        assert status.completed_at is not None

    def test_fail_run(self, pipeline_service):
        """Test failing a run."""
        run = pipeline_service.create_run(
            pipeline_type="factor_evaluation",
            config={},
        )
        pipeline_service.start_run(run.run_id)

        pipeline_service.fail_run(run.run_id, error="Test error")

        status = pipeline_service.get_run_status(run.run_id)
        assert status.status == "failed"
        assert status.error == "Test error"

    def test_list_runs(self, pipeline_service):
        """Test listing runs."""
        pipeline_service.create_run("factor_evaluation", {})
        pipeline_service.create_run("strategy_backtest", {})

        runs = pipeline_service.list_runs()
        assert len(runs) >= 2

    def test_list_runs_by_status(self, pipeline_service):
        """Test listing runs filtered by status."""
        run1 = pipeline_service.create_run("factor_evaluation", {})
        run2 = pipeline_service.create_run("strategy_backtest", {})
        pipeline_service.start_run(run2.run_id)

        pending_runs = pipeline_service.list_runs(status="pending")
        running_runs = pipeline_service.list_runs(status="running")

        assert any(r.run_id == run1.run_id for r in pending_runs)
        assert any(r.run_id == run2.run_id for r in running_runs)


# ==================== Pipeline Router Tests ====================


class TestPipelineRouter:
    """Tests for pipeline API router."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from iqfmp.api.pipeline.router import router as pipeline_router

        app = FastAPI()
        app.include_router(pipeline_router, prefix="/api/v1/pipeline")
        return TestClient(app)

    def test_run_pipeline_endpoint(self, client):
        """Test run pipeline endpoint."""
        response = client.post(
            "/api/v1/pipeline/run",
            json={
                "pipeline_type": "factor_evaluation",
                "config": {"factor_id": "factor-123"},
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert "run_id" in data
        assert data["status"] == "pending"

    def test_run_pipeline_all_types(self, client):
        """Test running all pipeline types."""
        for ptype in ["factor_evaluation", "strategy_backtest", "full_pipeline"]:
            response = client.post(
                "/api/v1/pipeline/run",
                json={"pipeline_type": ptype},
            )
            assert response.status_code == 201

    def test_run_pipeline_invalid_type(self, client):
        """Test running pipeline with invalid type."""
        response = client.post(
            "/api/v1/pipeline/run",
            json={"pipeline_type": "invalid"},
        )
        assert response.status_code == 422

    def test_get_status_endpoint(self, client):
        """Test get status endpoint."""
        # Create a run first
        create_resp = client.post(
            "/api/v1/pipeline/run",
            json={"pipeline_type": "factor_evaluation"},
        )
        run_id = create_resp.json()["run_id"]

        response = client.get(f"/api/v1/pipeline/{run_id}/status")
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == run_id

    def test_get_status_nonexistent(self, client):
        """Test get status for nonexistent run."""
        response = client.get("/api/v1/pipeline/nonexistent-id/status")
        assert response.status_code == 404

    def test_list_runs_endpoint(self, client):
        """Test list runs endpoint."""
        # Create some runs
        client.post(
            "/api/v1/pipeline/run",
            json={"pipeline_type": "factor_evaluation"},
        )

        response = client.get("/api/v1/pipeline/runs")
        assert response.status_code == 200
        data = response.json()
        assert "runs" in data

    def test_list_runs_with_status_filter(self, client):
        """Test list runs with status filter."""
        response = client.get("/api/v1/pipeline/runs?status=pending")
        assert response.status_code == 200

    def test_cancel_run_endpoint(self, client):
        """Test cancel run endpoint."""
        # Create a run first
        create_resp = client.post(
            "/api/v1/pipeline/run",
            json={"pipeline_type": "factor_evaluation"},
        )
        run_id = create_resp.json()["run_id"]

        response = client.post(f"/api/v1/pipeline/{run_id}/cancel")
        assert response.status_code in [200, 400]  # 400 if already completed


# ==================== WebSocket Tests ====================


class TestPipelineWebSocket:
    """Tests for pipeline WebSocket functionality."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked token validation."""
        from unittest.mock import patch

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from iqfmp.api.pipeline.router import router as pipeline_router

        app = FastAPI()
        app.include_router(pipeline_router, prefix="/api/v1/pipeline")

        # Mock TokenService.decode_token to always return a valid payload
        with patch("iqfmp.api.pipeline.router.TokenService") as mock_token_service:
            mock_token_service.return_value.decode_token.return_value = {"sub": "test_user"}
            yield TestClient(app)

    def test_websocket_connection(self, client):
        """Test WebSocket connection."""
        # Create a run first
        create_resp = client.post(
            "/api/v1/pipeline/run",
            json={"pipeline_type": "factor_evaluation"},
        )
        run_id = create_resp.json()["run_id"]

        # Test WebSocket connection with test token
        with client.websocket_connect(f"/api/v1/pipeline/{run_id}/ws?token=test_token") as websocket:
            # Connection should be established
            assert websocket is not None

    def test_websocket_receives_status(self, client):
        """Test WebSocket receives status message."""
        # Create a run first
        create_resp = client.post(
            "/api/v1/pipeline/run",
            json={"pipeline_type": "factor_evaluation"},
        )
        run_id = create_resp.json()["run_id"]

        with client.websocket_connect(f"/api/v1/pipeline/{run_id}/ws?token=test_token") as websocket:
            # Should receive initial status message
            data = websocket.receive_json()
            assert "type" in data
            assert data["type"] == "status"


# ==================== Integration Tests ====================


class TestPipelineAPIIntegration:
    """Integration tests for pipeline API."""

    @pytest.fixture
    def full_client(self):
        """Create full app test client."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from iqfmp.api.pipeline.router import router as pipeline_router

        app = FastAPI()
        app.include_router(pipeline_router, prefix="/api/v1/pipeline")
        return TestClient(app)

    def test_create_and_check_status(self, full_client):
        """Test creating run and checking status."""
        # Create
        create_resp = full_client.post(
            "/api/v1/pipeline/run",
            json={
                "pipeline_type": "factor_evaluation",
                "config": {"factor_id": "test-factor"},
            },
        )
        assert create_resp.status_code == 201
        run_id = create_resp.json()["run_id"]

        # Check status
        status_resp = full_client.get(f"/api/v1/pipeline/{run_id}/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["run_id"] == run_id

    def test_run_lifecycle(self, full_client):
        """Test full run lifecycle."""
        # Create
        create_resp = full_client.post(
            "/api/v1/pipeline/run",
            json={"pipeline_type": "strategy_backtest"},
        )
        run_id = create_resp.json()["run_id"]

        # Initial status should be pending
        status_resp = full_client.get(f"/api/v1/pipeline/{run_id}/status")
        assert status_resp.json()["status"] == "pending"


# ==================== Boundary Tests ====================


class TestPipelineAPIBoundary:
    """Boundary tests for pipeline API."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from iqfmp.api.pipeline.router import router as pipeline_router

        app = FastAPI()
        app.include_router(pipeline_router, prefix="/api/v1/pipeline")
        return TestClient(app)

    def test_empty_config(self, client):
        """Test running pipeline with empty config."""
        response = client.post(
            "/api/v1/pipeline/run",
            json={"pipeline_type": "factor_evaluation", "config": {}},
        )
        # Should work with empty config
        assert response.status_code == 201

    def test_large_config(self, client):
        """Test running pipeline with large config."""
        large_config = {
            "symbols": [f"SYMBOL_{i}/USDT" for i in range(100)],
            "extra_data": "x" * 10000,
        }
        response = client.post(
            "/api/v1/pipeline/run",
            json={"pipeline_type": "factor_evaluation", "config": large_config},
        )
        # Should handle large config
        assert response.status_code in [201, 400]

    def test_invalid_run_id_format(self, client):
        """Test getting status with invalid run ID format."""
        response = client.get("/api/v1/pipeline/invalid-format!/status")
        # Should return 404 (not found) not 500 (server error)
        assert response.status_code == 404


# ==================== Exception Tests ====================


class TestPipelineAPIExceptions:
    """Exception handling tests for pipeline API."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from iqfmp.api.pipeline.router import router as pipeline_router

        app = FastAPI()
        app.include_router(pipeline_router, prefix="/api/v1/pipeline")
        return TestClient(app)

    def test_missing_pipeline_type(self, client):
        """Test request missing pipeline_type."""
        response = client.post(
            "/api/v1/pipeline/run",
            json={"config": {}},
        )
        assert response.status_code == 422

    def test_invalid_json(self, client):
        """Test request with invalid JSON."""
        response = client.post(
            "/api/v1/pipeline/run",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_cancel_nonexistent_run(self, client):
        """Test cancelling nonexistent run."""
        response = client.post("/api/v1/pipeline/nonexistent-id/cancel")
        assert response.status_code == 404
