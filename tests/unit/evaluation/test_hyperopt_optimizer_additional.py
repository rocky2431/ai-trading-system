"""Additional tests for hyperparameter optimizers."""

from __future__ import annotations

import pytest

from iqfmp.evaluation import hyperopt_optimizer as ho

from iqfmp.evaluation.hyperopt_optimizer import (
    SearchSpace,
    InvalidSearchSpaceError,
    RandomSearchOptimizer,
    GeneticOptimizer,
    GridSearchOptimizer,
    BaseOptimizer,
    OptimizationResult,
    BayesianOptimizer,
    create_optimizer,
)


def test_search_space_validate_and_sample() -> None:
    space = SearchSpace(name="alpha", param_type="float", bounds=(0.1, 1.0))
    space.validate()
    sample = space.sample()
    assert 0.1 <= sample <= 1.0

    log_space = SearchSpace(name="log_alpha", param_type="float", bounds=(0.01, 1.0), log_scale=True)
    log_space.validate()
    log_sample = log_space.sample()
    assert 0.01 <= log_sample <= 1.0

    int_space = SearchSpace(name="steps", param_type="int", bounds=(1, 5), step=2)
    int_space.validate()
    int_sample = int_space.sample()
    assert int_sample in {1, 3, 5}

    cat_space = SearchSpace(name="mode", param_type="categorical", bounds=["a", "b", "c"])
    cat_space.validate()
    assert cat_space.sample() in {"a", "b", "c"}

    bad_space = SearchSpace(name="beta", param_type="float", bounds=(1.0, 0.1))
    with pytest.raises(InvalidSearchSpaceError):
        bad_space.validate()

    bad_type = SearchSpace(name="gamma", param_type="bad", bounds=(0.0, 1.0))
    with pytest.raises(InvalidSearchSpaceError):
        bad_type.validate()


def test_random_search_optimizer() -> None:
    space = SearchSpace(name="alpha", param_type="float", bounds=(0.1, 1.0))
    optimizer = RandomSearchOptimizer(
        search_space=[space],
        objective_fn=lambda p: p["alpha"],
        n_trials=5,
        seed=42,
    )

    result = optimizer.optimize()
    assert result.n_trials == 5
    assert "alpha" in result.best_params


def test_random_search_optimizer_minimize() -> None:
    space = SearchSpace(name="alpha", param_type="float", bounds=(0.1, 1.0))
    optimizer = RandomSearchOptimizer(
        search_space=[space],
        objective_fn=lambda p: p["alpha"],
        n_trials=6,
        maximize=False,
        seed=3,
    )

    result = optimizer.optimize()
    trial_values = [trial.objective_value for trial in result.trials]
    assert result.best_value == min(trial_values)


def test_grid_search_optimizer() -> None:
    space = SearchSpace(name="alpha", param_type="int", bounds=(1, 5))
    optimizer = GridSearchOptimizer(
        search_space=[space],
        objective_fn=lambda p: p["alpha"],
        n_trials=4,
        n_points_per_dim=3,
        seed=1,
    )

    result = optimizer.optimize()
    assert result.n_trials <= 4
    assert result.best_params


def test_grid_search_generate_grid_log_scale() -> None:
    space = SearchSpace(name="alpha", param_type="float", bounds=(1e-3, 1e-1), log_scale=True)
    optimizer = GridSearchOptimizer(
        search_space=[space],
        objective_fn=lambda p: p["alpha"],
        n_trials=10,
        n_points_per_dim=4,
    )

    grid = optimizer._generate_grid()
    assert len(grid) == 4
    assert all("alpha" in params for params in grid)


def test_genetic_optimizer() -> None:
    space = SearchSpace(name="alpha", param_type="int", bounds=(1, 5))
    optimizer = GeneticOptimizer(
        search_space=[space],
        objective_fn=lambda p: p["alpha"],
        n_trials=6,
        population_size=4,
        seed=7,
    )

    result = optimizer.optimize()
    assert result.trials


def test_bayesian_optimizer_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    search_space = [
        SearchSpace(name="alpha", param_type="float", bounds=(0.1, 1.0)),
        SearchSpace(name="beta", param_type="int", bounds=(1, 5)),
        SearchSpace(name="mode", param_type="categorical", bounds=["x", "y"]),
    ]
    optimizer = BayesianOptimizer(
        search_space=search_space,
        objective_fn=lambda p: p["alpha"] + p["beta"],
        n_trials=6,
        n_startup_trials=2,
        seed=11,
    )
    optimizer._use_optuna = False

    monkeypatch.setattr(ho.random, "random", lambda: 0.1)
    result = optimizer.optimize()

    assert result.optimization_method == "bayesian_fallback"
    assert result.n_trials == 6
    assert result.convergence_history


def test_create_optimizer_factory() -> None:
    space = SearchSpace(name="alpha", param_type="float", bounds=(0.1, 1.0))
    objective = lambda p: p["alpha"]

    assert isinstance(
        create_optimizer("random", [space], objective, n_trials=2),
        RandomSearchOptimizer,
    )
    assert isinstance(
        create_optimizer("grid", [space], objective, n_trials=2, n_points_per_dim=2),
        GridSearchOptimizer,
    )
    assert isinstance(
        create_optimizer("genetic", [space], objective, n_trials=2, population_size=2),
        GeneticOptimizer,
    )

    with pytest.raises(ValueError):
        create_optimizer("unknown", [space], objective)


def test_evaluate_params_failure_path() -> None:
    class _DummyOptimizer(BaseOptimizer):
        def optimize(self) -> OptimizationResult:  # pragma: no cover - not used
            return OptimizationResult(best_params={}, best_value=0.0)

    space = SearchSpace(name="alpha", param_type="float", bounds=(0.1, 1.0))
    optimizer = _DummyOptimizer(
        search_space=[space],
        objective_fn=lambda p: (_ for _ in ()).throw(ValueError("boom")),
        n_trials=1,
    )

    result = optimizer._evaluate_params({"alpha": 0.5}, trial_id=1)
    assert result.status == "failed"
