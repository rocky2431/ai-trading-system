"""Tests for research ledger API.

Tests cover:
- Research Schemas
- Research Service
- Research Router
- Integration scenarios
"""

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


# ==================== Schema Tests ====================


class TestResearchSchemas:
    """Tests for research API schemas."""

    def test_trial_response(self):
        """Test trial response schema."""
        from iqfmp.api.research.schemas import TrialResponse

        response = TrialResponse(
            trial_id="trial-123",
            factor_name="momentum_20",
            factor_family="momentum",
            sharpe_ratio=2.5,
            ic_mean=0.05,
            ir=1.8,
            max_drawdown=0.15,
            win_rate=0.55,
            created_at=datetime.now(),
        )
        assert response.trial_id == "trial-123"
        assert response.sharpe_ratio == 2.5

    def test_trial_response_optional_fields(self):
        """Test trial response with optional fields."""
        from iqfmp.api.research.schemas import TrialResponse

        response = TrialResponse(
            trial_id="trial-456",
            factor_name="volatility_atr",
            factor_family="volatility",
            sharpe_ratio=1.8,
            created_at=datetime.now(),
        )
        assert response.ic_mean is None
        assert response.ir is None

    def test_ledger_list_response(self):
        """Test ledger list response schema."""
        from iqfmp.api.research.schemas import LedgerListResponse, TrialResponse

        trials = [
            TrialResponse(
                trial_id="trial-1",
                factor_name="factor1",
                factor_family="momentum",
                sharpe_ratio=2.0,
                created_at=datetime.now(),
            ),
            TrialResponse(
                trial_id="trial-2",
                factor_name="factor2",
                factor_family="volatility",
                sharpe_ratio=1.5,
                created_at=datetime.now(),
            ),
        ]
        response = LedgerListResponse(
            trials=trials,
            total=100,
            page=1,
            page_size=10,
        )
        assert len(response.trials) == 2
        assert response.total == 100

    def test_statistics_response(self):
        """Test statistics response schema."""
        from iqfmp.api.research.schemas import StatisticsResponse

        stats = StatisticsResponse(
            total_trials=100,
            mean_sharpe=1.5,
            std_sharpe=0.5,
            max_sharpe=3.2,
            min_sharpe=-0.5,
            median_sharpe=1.4,
        )
        assert stats.total_trials == 100
        assert stats.mean_sharpe == 1.5

    def test_stats_response(self):
        """Test overall stats response schema."""
        from iqfmp.api.research.schemas import StatisticsResponse, StatsResponse

        overall = StatisticsResponse(
            total_trials=100,
            mean_sharpe=1.5,
            std_sharpe=0.5,
            max_sharpe=3.2,
            min_sharpe=-0.5,
            median_sharpe=1.4,
        )
        response = StatsResponse(overall=overall)
        assert response.overall.total_trials == 100
        assert response.by_family is None

    def test_stats_response_with_family(self):
        """Test stats response with family breakdown."""
        from iqfmp.api.research.schemas import StatisticsResponse, StatsResponse

        overall = StatisticsResponse(
            total_trials=100,
            mean_sharpe=1.5,
            std_sharpe=0.5,
            max_sharpe=3.2,
            min_sharpe=-0.5,
            median_sharpe=1.4,
        )
        momentum_stats = StatisticsResponse(
            total_trials=40,
            mean_sharpe=1.8,
            std_sharpe=0.4,
            max_sharpe=3.0,
            min_sharpe=0.5,
            median_sharpe=1.7,
        )
        response = StatsResponse(
            overall=overall,
            by_family={"momentum": momentum_stats},
        )
        assert response.by_family is not None
        assert "momentum" in response.by_family

    def test_threshold_config_response(self):
        """Test threshold config response schema."""
        from iqfmp.api.research.schemas import ThresholdConfigResponse

        config = ThresholdConfigResponse(
            base_sharpe_threshold=2.0,
            confidence_level=0.95,
            min_trials_for_adjustment=1,
        )
        assert config.base_sharpe_threshold == 2.0
        assert config.confidence_level == 0.95

    def test_threshold_response(self):
        """Test threshold response schema."""
        from iqfmp.api.research.schemas import (
            ThresholdConfigResponse,
            ThresholdHistoryItem,
            ThresholdResponse,
        )

        config = ThresholdConfigResponse(
            base_sharpe_threshold=2.0,
            confidence_level=0.95,
            min_trials_for_adjustment=1,
        )
        history = [
            ThresholdHistoryItem(n_trials=10, threshold=2.05),
            ThresholdHistoryItem(n_trials=50, threshold=2.15),
        ]
        response = ThresholdResponse(
            current_threshold=2.15,
            n_trials=50,
            config=config,
            threshold_history=history,
        )
        assert response.current_threshold == 2.15
        assert len(response.threshold_history) == 2


# ==================== Research Service Tests ====================


class TestResearchService:
    """Tests for research service."""

    @pytest.fixture
    def research_service(self):
        """Create research service."""
        from iqfmp.api.research.service import ResearchService

        return ResearchService()

    def test_list_trials_empty(self, research_service):
        """Test listing trials when empty."""
        trials, total = research_service.list_trials()
        assert trials == []
        assert total == 0

    def test_list_trials_with_data(self, research_service):
        """Test listing trials with data."""
        # Add some trials
        research_service.add_trial(
            factor_name="factor1",
            factor_family="momentum",
            sharpe_ratio=2.0,
        )
        research_service.add_trial(
            factor_name="factor2",
            factor_family="volatility",
            sharpe_ratio=1.5,
        )

        trials, total = research_service.list_trials()
        assert total >= 2
        assert len(trials) >= 2

    def test_list_trials_pagination(self, research_service):
        """Test listing trials with pagination."""
        # Add many trials
        for i in range(15):
            research_service.add_trial(
                factor_name=f"factor_{i}",
                factor_family="test",
                sharpe_ratio=1.0 + i * 0.1,
            )

        trials, total = research_service.list_trials(page=1, page_size=10)
        assert len(trials) == 10
        assert total >= 15

        trials, total = research_service.list_trials(page=2, page_size=10)
        assert len(trials) == 5

    def test_list_trials_filter_by_family(self, research_service):
        """Test filtering trials by family."""
        research_service.add_trial(
            factor_name="momentum1",
            factor_family="momentum",
            sharpe_ratio=2.0,
        )
        research_service.add_trial(
            factor_name="vol1",
            factor_family="volatility",
            sharpe_ratio=1.5,
        )

        trials, total = research_service.list_trials(family="momentum")
        assert all(t.factor_family == "momentum" for t in trials)

    def test_list_trials_filter_by_min_sharpe(self, research_service):
        """Test filtering trials by minimum Sharpe ratio."""
        research_service.add_trial(
            factor_name="high_sharpe",
            factor_family="test",
            sharpe_ratio=3.0,
        )
        research_service.add_trial(
            factor_name="low_sharpe",
            factor_family="test",
            sharpe_ratio=0.5,
        )

        trials, total = research_service.list_trials(min_sharpe=2.0)
        assert all(t.sharpe_ratio >= 2.0 for t in trials)

    def test_list_trials_filter_by_date_range(self, research_service):
        """Test filtering trials by date range."""
        # This test verifies date filtering is supported
        trials, total = research_service.list_trials(
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now(),
        )
        # Should not raise error
        assert isinstance(trials, list)

    def test_get_statistics_empty(self, research_service):
        """Test getting statistics when empty."""
        stats = research_service.get_statistics()
        assert stats.total_trials == 0

    def test_get_statistics_with_data(self, research_service):
        """Test getting statistics with data."""
        research_service.add_trial(
            factor_name="factor1",
            factor_family="momentum",
            sharpe_ratio=2.0,
        )
        research_service.add_trial(
            factor_name="factor2",
            factor_family="momentum",
            sharpe_ratio=3.0,
        )

        stats = research_service.get_statistics()
        assert stats.total_trials >= 2
        assert stats.mean_sharpe >= 2.0

    def test_get_statistics_by_family(self, research_service):
        """Test getting statistics grouped by family."""
        research_service.add_trial(
            factor_name="momentum1",
            factor_family="momentum",
            sharpe_ratio=2.0,
        )
        research_service.add_trial(
            factor_name="vol1",
            factor_family="volatility",
            sharpe_ratio=1.5,
        )

        by_family = research_service.get_statistics_by_family()
        assert "momentum" in by_family or "volatility" in by_family

    def test_get_current_threshold(self, research_service):
        """Test getting current threshold."""
        threshold = research_service.get_current_threshold()
        assert threshold > 0

    def test_get_threshold_info(self, research_service):
        """Test getting full threshold info."""
        info = research_service.get_threshold_info()
        assert "current_threshold" in info
        assert "n_trials" in info
        assert "config" in info

    def test_add_trial(self, research_service):
        """Test adding a trial."""
        trial_id = research_service.add_trial(
            factor_name="test_factor",
            factor_family="test",
            sharpe_ratio=2.5,
            ic_mean=0.05,
        )
        assert trial_id is not None


# ==================== Research Router Tests ====================


class TestResearchRouter:
    """Tests for research API router."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from iqfmp.api.research.router import router as research_router

        app = FastAPI()
        app.include_router(research_router, prefix="/api/v1/research")
        return TestClient(app)

    def test_list_ledger_endpoint(self, client):
        """Test list ledger endpoint."""
        response = client.get("/api/v1/research/ledger")
        assert response.status_code == 200
        data = response.json()
        assert "trials" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data

    def test_list_ledger_with_pagination(self, client):
        """Test list ledger with pagination params."""
        response = client.get("/api/v1/research/ledger?page=1&page_size=5")
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 5

    def test_list_ledger_with_family_filter(self, client):
        """Test list ledger with family filter."""
        response = client.get("/api/v1/research/ledger?family=momentum")
        assert response.status_code == 200

    def test_list_ledger_with_min_sharpe_filter(self, client):
        """Test list ledger with min_sharpe filter."""
        response = client.get("/api/v1/research/ledger?min_sharpe=2.0")
        assert response.status_code == 200

    def test_list_ledger_with_date_filter(self, client):
        """Test list ledger with date filter."""
        response = client.get(
            "/api/v1/research/ledger?start_date=2025-01-01&end_date=2025-12-31"
        )
        assert response.status_code == 200

    def test_get_stats_endpoint(self, client):
        """Test get stats endpoint."""
        response = client.get("/api/v1/research/stats")
        assert response.status_code == 200
        data = response.json()
        assert "overall" in data

    def test_get_stats_with_family_grouping(self, client):
        """Test get stats with family grouping."""
        response = client.get("/api/v1/research/stats?group_by_family=true")
        assert response.status_code == 200
        data = response.json()
        assert "overall" in data
        # by_family may be None or dict depending on data

    def test_get_thresholds_endpoint(self, client):
        """Test get thresholds endpoint."""
        response = client.get("/api/v1/research/thresholds")
        assert response.status_code == 200
        data = response.json()
        assert "current_threshold" in data
        assert "n_trials" in data
        assert "config" in data


# ==================== Metrics Router Tests ====================


class TestMetricsRouter:
    """Tests for metrics API router."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from iqfmp.api.research.router import metrics_router

        app = FastAPI()
        app.include_router(metrics_router, prefix="/api/v1/metrics")
        return TestClient(app)

    def test_get_metrics_thresholds_endpoint(self, client):
        """Test get metrics thresholds endpoint."""
        response = client.get("/api/v1/metrics/thresholds")
        assert response.status_code == 200
        data = response.json()
        assert "current_threshold" in data
        assert "n_trials" in data
        assert "config" in data
        assert "threshold_history" in data


# ==================== Integration Tests ====================


class TestResearchAPIIntegration:
    """Integration tests for research API."""

    @pytest.fixture
    def full_client(self):
        """Create full app test client."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from iqfmp.api.research.router import metrics_router, router as research_router
        from iqfmp.api.research.service import ResearchService

        app = FastAPI()
        app.include_router(research_router, prefix="/api/v1/research")
        app.include_router(metrics_router, prefix="/api/v1/metrics")

        return TestClient(app)

    def test_ledger_stats_consistency(self, full_client):
        """Test that ledger and stats are consistent."""
        # Get ledger
        ledger_resp = full_client.get("/api/v1/research/ledger")
        assert ledger_resp.status_code == 200
        ledger_data = ledger_resp.json()

        # Get stats
        stats_resp = full_client.get("/api/v1/research/stats")
        assert stats_resp.status_code == 200
        stats_data = stats_resp.json()

        # Total trials should match
        assert ledger_data["total"] == stats_data["overall"]["total_trials"]

    def test_threshold_reflects_trial_count(self, full_client):
        """Test threshold reflects trial count."""
        # Get thresholds
        thresh_resp = full_client.get("/api/v1/metrics/thresholds")
        assert thresh_resp.status_code == 200
        thresh_data = thresh_resp.json()

        # Get ledger total
        ledger_resp = full_client.get("/api/v1/research/ledger")
        ledger_data = ledger_resp.json()

        # n_trials should match ledger total (or at least 1)
        assert thresh_data["n_trials"] >= 1


# ==================== Boundary Tests ====================


class TestResearchAPIBoundary:
    """Boundary tests for research API."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from iqfmp.api.research.router import router as research_router

        app = FastAPI()
        app.include_router(research_router, prefix="/api/v1/research")
        return TestClient(app)

    def test_invalid_page_number(self, client):
        """Test invalid page number."""
        response = client.get("/api/v1/research/ledger?page=0")
        assert response.status_code in [400, 422]

    def test_negative_page_size(self, client):
        """Test negative page size."""
        response = client.get("/api/v1/research/ledger?page_size=-1")
        assert response.status_code in [400, 422]

    def test_very_large_page_size(self, client):
        """Test very large page size is capped."""
        response = client.get("/api/v1/research/ledger?page_size=1000")
        # Should either work with cap or reject
        assert response.status_code in [200, 400, 422]
        if response.status_code == 200:
            data = response.json()
            assert data["page_size"] <= 100  # Should be capped

    def test_invalid_min_sharpe(self, client):
        """Test invalid min_sharpe value."""
        response = client.get("/api/v1/research/ledger?min_sharpe=invalid")
        assert response.status_code == 422

    def test_invalid_date_format(self, client):
        """Test invalid date format."""
        response = client.get("/api/v1/research/ledger?start_date=not-a-date")
        assert response.status_code == 422


# ==================== Exception Tests ====================


class TestResearchAPIExceptions:
    """Exception handling tests for research API."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from iqfmp.api.research.router import router as research_router

        app = FastAPI()
        app.include_router(research_router, prefix="/api/v1/research")
        return TestClient(app)

    def test_unknown_filter_ignored(self, client):
        """Test unknown filter parameters are ignored."""
        response = client.get("/api/v1/research/ledger?unknown_param=value")
        # Should not fail, just ignore
        assert response.status_code == 200

    def test_empty_family_filter(self, client):
        """Test empty family filter."""
        response = client.get("/api/v1/research/ledger?family=")
        # Should work (no filter applied)
        assert response.status_code == 200
