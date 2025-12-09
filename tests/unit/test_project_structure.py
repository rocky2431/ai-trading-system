"""Test project structure and configuration."""

from pathlib import Path

import pytest


class TestProjectStructure:
    """Test that project structure is correctly set up."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent.parent

    def test_src_directory_exists(self, project_root: Path) -> None:
        """Test that src directory exists."""
        assert (project_root / "src").exists()
        assert (project_root / "src").is_dir()

    def test_iqfmp_package_exists(self, project_root: Path) -> None:
        """Test that iqfmp package exists with __init__.py."""
        iqfmp_dir = project_root / "src" / "iqfmp"
        assert iqfmp_dir.exists()
        assert (iqfmp_dir / "__init__.py").exists()

    def test_required_submodules_exist(self, project_root: Path) -> None:
        """Test that all required submodules exist."""
        iqfmp_dir = project_root / "src" / "iqfmp"
        required_modules = [
            "agents",
            "core",
            "data",
            "llm",
            "api",
            "models",
            "utils",
        ]
        for module in required_modules:
            module_dir = iqfmp_dir / module
            assert module_dir.exists(), f"Module {module} does not exist"
            assert (
                module_dir / "__init__.py"
            ).exists(), f"Module {module} missing __init__.py"

    def test_tests_directory_structure(self, project_root: Path) -> None:
        """Test that tests directory structure is correct."""
        tests_dir = project_root / "tests"
        assert tests_dir.exists()
        assert (tests_dir / "unit").exists()
        assert (tests_dir / "integration").exists()
        assert (tests_dir / "e2e").exists()

    def test_pyproject_toml_exists(self, project_root: Path) -> None:
        """Test that pyproject.toml exists."""
        assert (project_root / "pyproject.toml").exists()

    def test_docker_compose_exists(self, project_root: Path) -> None:
        """Test that docker-compose.yml exists."""
        assert (project_root / "docker-compose.yml").exists()

    def test_pre_commit_config_exists(self, project_root: Path) -> None:
        """Test that pre-commit config exists."""
        assert (project_root / ".pre-commit-config.yaml").exists()

    def test_github_workflows_exist(self, project_root: Path) -> None:
        """Test that GitHub workflows exist."""
        workflows_dir = project_root / ".github" / "workflows"
        assert workflows_dir.exists()
        assert (workflows_dir / "ci.yml").exists()


class TestPackageImports:
    """Test that package imports work correctly."""

    def test_import_iqfmp(self) -> None:
        """Test that iqfmp package can be imported."""
        import iqfmp

        assert hasattr(iqfmp, "__version__")
        assert iqfmp.__version__ == "0.1.0"

    def test_import_models(self) -> None:
        """Test that models can be imported."""
        from iqfmp.models.factor import Factor, FactorMetrics, FactorStatus

        assert Factor is not None
        assert FactorMetrics is not None
        assert FactorStatus is not None

    def test_import_api(self) -> None:
        """Test that API can be imported."""
        from iqfmp.api.main import app, create_app

        assert app is not None
        assert create_app is not None


class TestApiEndpoints:
    """Test API endpoints."""

    def test_health_endpoint(self, client) -> None:
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestFactorModels:
    """Test Factor data models."""

    def test_factor_status_enum(self) -> None:
        """Test FactorStatus enum values."""
        from iqfmp.models.factor import FactorStatus

        assert FactorStatus.CANDIDATE == "candidate"
        assert FactorStatus.REJECTED == "rejected"
        assert FactorStatus.CORE == "core"
        assert FactorStatus.REDUNDANT == "redundant"

    def test_factor_metrics_creation(self) -> None:
        """Test FactorMetrics model creation."""
        from iqfmp.models.factor import FactorMetrics

        metrics = FactorMetrics(
            ic_mean=0.05,
            ic_std=0.02,
            ir=2.5,
            sharpe=1.8,
            max_drawdown=0.15,
            turnover=0.3,
        )
        assert metrics.ic_mean == 0.05
        assert metrics.ir == 2.5

    def test_factor_creation(self, sample_factor_data: dict) -> None:
        """Test Factor model creation."""
        from iqfmp.models.factor import Factor

        factor = Factor(**sample_factor_data)
        assert factor.id == sample_factor_data["id"]
        assert factor.name == sample_factor_data["name"]
        assert factor.family == sample_factor_data["family"]
        assert factor.status == "candidate"  # default value
