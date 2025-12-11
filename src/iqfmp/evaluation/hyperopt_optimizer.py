"""Hyperparameter Optimization Framework for Factor Mining.

This module implements multiple optimization algorithms for factor parameter tuning:
- Bayesian Optimization (using optuna)
- Genetic Algorithm (custom implementation)
- Random Search (baseline)
- Grid Search (exhaustive)
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional
import numpy as np


class InvalidSearchSpaceError(Exception):
    """Raised when the search space configuration is invalid."""

    pass


class OptimizationTimeoutError(Exception):
    """Raised when optimization exceeds time limit."""

    pass


@dataclass
class SearchSpace:
    """Definition of hyperparameter search space."""

    # Parameter name
    name: str

    # Type: 'float', 'int', 'categorical'
    param_type: str

    # For float/int: (low, high)
    # For categorical: list of choices
    bounds: tuple[float, float] | list[Any] = field(default_factory=list)

    # For float/int: log scale search
    log_scale: bool = False

    # Step size for discrete search
    step: Optional[float] = None

    def validate(self) -> None:
        """Validate search space configuration."""
        if self.param_type in ("float", "int"):
            if not isinstance(self.bounds, tuple) or len(self.bounds) != 2:
                raise InvalidSearchSpaceError(
                    f"Parameter {self.name}: bounds must be (low, high) tuple"
                )
            if self.bounds[0] >= self.bounds[1]:
                raise InvalidSearchSpaceError(
                    f"Parameter {self.name}: low must be less than high"
                )
        elif self.param_type == "categorical":
            if not isinstance(self.bounds, list) or len(self.bounds) < 2:
                raise InvalidSearchSpaceError(
                    f"Parameter {self.name}: categorical needs at least 2 choices"
                )
        else:
            raise InvalidSearchSpaceError(
                f"Unknown param_type: {self.param_type}"
            )

    def sample(self) -> Any:
        """Sample a random value from this search space."""
        if self.param_type == "float":
            low, high = self.bounds
            if self.log_scale:
                return np.exp(random.uniform(np.log(low), np.log(high)))
            return random.uniform(low, high)
        elif self.param_type == "int":
            low, high = self.bounds
            if self.step:
                choices = list(range(int(low), int(high) + 1, int(self.step)))
                return random.choice(choices)
            return random.randint(int(low), int(high))
        else:  # categorical
            return random.choice(self.bounds)


@dataclass
class TrialResult:
    """Result of a single optimization trial."""

    trial_id: int
    params: dict[str, Any]
    objective_value: float
    duration_seconds: float
    status: str = "completed"  # completed, failed, pruned
    user_attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Result of the full optimization process."""

    # Best parameters found
    best_params: dict[str, Any]
    best_value: float

    # All trials
    trials: list[TrialResult] = field(default_factory=list)

    # Metadata
    n_trials: int = 0
    optimization_method: str = "unknown"
    total_duration_seconds: float = 0.0
    search_space: list[SearchSpace] = field(default_factory=list)

    # Convergence info
    convergence_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for API response."""
        return {
            "best_params": self.best_params,
            "best_value": round(self.best_value, 6),
            "n_trials": self.n_trials,
            "optimization_method": self.optimization_method,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "convergence_history": [round(v, 6) for v in self.convergence_history],
            "trials_summary": {
                "total": len(self.trials),
                "completed": sum(1 for t in self.trials if t.status == "completed"),
                "failed": sum(1 for t in self.trials if t.status == "failed"),
                "pruned": sum(1 for t in self.trials if t.status == "pruned"),
            },
        }


class BaseOptimizer(ABC):
    """Base class for all optimizers."""

    def __init__(
        self,
        search_space: list[SearchSpace],
        objective_fn: Callable[[dict[str, Any]], float],
        n_trials: int = 100,
        maximize: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize optimizer.

        Args:
            search_space: List of SearchSpace definitions
            objective_fn: Function that takes params dict and returns objective value
            n_trials: Number of trials to run
            maximize: If True, maximize objective; else minimize
            seed: Random seed for reproducibility
        """
        self.search_space = search_space
        self.objective_fn = objective_fn
        self.n_trials = n_trials
        self.maximize = maximize
        self.seed = seed

        # Validate search space
        for space in search_space:
            space.validate()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    @abstractmethod
    def optimize(self) -> OptimizationResult:
        """Run optimization and return results."""
        pass

    def _evaluate_params(self, params: dict[str, Any], trial_id: int) -> TrialResult:
        """Evaluate a single set of parameters."""
        start_time = datetime.now()
        try:
            value = self.objective_fn(params)
            duration = (datetime.now() - start_time).total_seconds()
            return TrialResult(
                trial_id=trial_id,
                params=params,
                objective_value=value,
                duration_seconds=duration,
                status="completed",
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return TrialResult(
                trial_id=trial_id,
                params=params,
                objective_value=float("-inf") if self.maximize else float("inf"),
                duration_seconds=duration,
                status="failed",
                user_attrs={"error": str(e)},
            )


class RandomSearchOptimizer(BaseOptimizer):
    """Random search optimizer (baseline)."""

    def optimize(self) -> OptimizationResult:
        """Run random search optimization."""
        start_time = datetime.now()
        trials: list[TrialResult] = []
        best_params: dict[str, Any] = {}
        best_value = float("-inf") if self.maximize else float("inf")
        convergence_history: list[float] = []

        for i in range(self.n_trials):
            # Sample random parameters
            params = {space.name: space.sample() for space in self.search_space}

            # Evaluate
            result = self._evaluate_params(params, i)
            trials.append(result)

            # Update best
            if result.status == "completed":
                is_better = (
                    result.objective_value > best_value
                    if self.maximize
                    else result.objective_value < best_value
                )
                if is_better:
                    best_value = result.objective_value
                    best_params = params.copy()

            convergence_history.append(best_value)

        total_duration = (datetime.now() - start_time).total_seconds()

        return OptimizationResult(
            best_params=best_params,
            best_value=best_value,
            trials=trials,
            n_trials=self.n_trials,
            optimization_method="random_search",
            total_duration_seconds=total_duration,
            search_space=self.search_space,
            convergence_history=convergence_history,
        )


class GeneticOptimizer(BaseOptimizer):
    """Genetic Algorithm optimizer."""

    def __init__(
        self,
        search_space: list[SearchSpace],
        objective_fn: Callable[[dict[str, Any]], float],
        n_trials: int = 100,
        maximize: bool = True,
        seed: Optional[int] = None,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_count: int = 2,
    ) -> None:
        """Initialize genetic optimizer.

        Args:
            population_size: Number of individuals per generation
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_count: Number of best individuals to preserve
        """
        super().__init__(search_space, objective_fn, n_trials, maximize, seed)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = elite_count

    def _create_individual(self) -> dict[str, Any]:
        """Create a random individual."""
        return {space.name: space.sample() for space in self.search_space}

    def _mutate(self, individual: dict[str, Any]) -> dict[str, Any]:
        """Mutate an individual."""
        mutated = individual.copy()
        for space in self.search_space:
            if random.random() < self.mutation_rate:
                mutated[space.name] = space.sample()
        return mutated

    def _crossover(
        self, parent1: dict[str, Any], parent2: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Perform crossover between two parents."""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        child1, child2 = {}, {}
        for space in self.search_space:
            if random.random() < 0.5:
                child1[space.name] = parent1[space.name]
                child2[space.name] = parent2[space.name]
            else:
                child1[space.name] = parent2[space.name]
                child2[space.name] = parent1[space.name]

        return child1, child2

    def _select_parents(
        self, population: list[dict[str, Any]], fitness: list[float]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Tournament selection."""
        tournament_size = 3

        def tournament() -> dict[str, Any]:
            indices = random.sample(range(len(population)), tournament_size)
            if self.maximize:
                winner_idx = max(indices, key=lambda i: fitness[i])
            else:
                winner_idx = min(indices, key=lambda i: fitness[i])
            return population[winner_idx]

        return tournament(), tournament()

    def optimize(self) -> OptimizationResult:
        """Run genetic algorithm optimization."""
        start_time = datetime.now()
        trials: list[TrialResult] = []
        trial_id = 0

        # Initialize population
        population = [self._create_individual() for _ in range(self.population_size)]
        fitness = []

        # Evaluate initial population
        for individual in population:
            result = self._evaluate_params(individual, trial_id)
            trials.append(result)
            fitness.append(result.objective_value)
            trial_id += 1

        best_params = population[
            max(range(len(fitness)), key=lambda i: fitness[i])
            if self.maximize
            else min(range(len(fitness)), key=lambda i: fitness[i])
        ].copy()
        best_value = max(fitness) if self.maximize else min(fitness)
        convergence_history: list[float] = [best_value]

        # Evolution loop
        remaining_trials = self.n_trials - self.population_size
        generations = remaining_trials // self.population_size

        for _ in range(generations):
            # Sort by fitness
            sorted_indices = sorted(
                range(len(fitness)),
                key=lambda i: fitness[i],
                reverse=self.maximize,
            )

            # Keep elite
            new_population = [population[i] for i in sorted_indices[: self.elite_count]]
            new_fitness = [fitness[i] for i in sorted_indices[: self.elite_count]]

            # Generate offspring
            while len(new_population) < self.population_size:
                parent1, parent2 = self._select_parents(population, fitness)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                # Evaluate children
                for child in [child1, child2]:
                    if len(new_population) >= self.population_size:
                        break
                    result = self._evaluate_params(child, trial_id)
                    trials.append(result)
                    new_population.append(child)
                    new_fitness.append(result.objective_value)
                    trial_id += 1

            population = new_population
            fitness = new_fitness

            # Update best
            gen_best_idx = (
                max(range(len(fitness)), key=lambda i: fitness[i])
                if self.maximize
                else min(range(len(fitness)), key=lambda i: fitness[i])
            )
            gen_best_value = fitness[gen_best_idx]

            is_better = (
                gen_best_value > best_value
                if self.maximize
                else gen_best_value < best_value
            )
            if is_better:
                best_value = gen_best_value
                best_params = population[gen_best_idx].copy()

            convergence_history.append(best_value)

        total_duration = (datetime.now() - start_time).total_seconds()

        return OptimizationResult(
            best_params=best_params,
            best_value=best_value,
            trials=trials,
            n_trials=len(trials),
            optimization_method="genetic_algorithm",
            total_duration_seconds=total_duration,
            search_space=self.search_space,
            convergence_history=convergence_history,
        )


class BayesianOptimizer(BaseOptimizer):
    """Bayesian Optimization using Optuna (if available) or fallback to TPE-like."""

    def __init__(
        self,
        search_space: list[SearchSpace],
        objective_fn: Callable[[dict[str, Any]], float],
        n_trials: int = 100,
        maximize: bool = True,
        seed: Optional[int] = None,
        n_startup_trials: int = 10,
    ) -> None:
        """Initialize Bayesian optimizer.

        Args:
            n_startup_trials: Number of random trials before Bayesian optimization
        """
        super().__init__(search_space, objective_fn, n_trials, maximize, seed)
        self.n_startup_trials = n_startup_trials

        # Try to import optuna
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self._use_optuna = True
        except ImportError:
            self._use_optuna = False

    def optimize(self) -> OptimizationResult:
        """Run Bayesian optimization."""
        if self._use_optuna:
            return self._optimize_with_optuna()
        else:
            return self._optimize_fallback()

    def _optimize_with_optuna(self) -> OptimizationResult:
        """Optimize using Optuna."""
        import optuna

        start_time = datetime.now()
        trials: list[TrialResult] = []
        trial_id = 0

        def objective(trial: optuna.Trial) -> float:
            nonlocal trial_id, trials

            params: dict[str, Any] = {}
            for space in self.search_space:
                if space.param_type == "float":
                    low, high = space.bounds
                    params[space.name] = trial.suggest_float(
                        space.name, low, high, log=space.log_scale
                    )
                elif space.param_type == "int":
                    low, high = space.bounds
                    step = int(space.step) if space.step else 1
                    params[space.name] = trial.suggest_int(
                        space.name, int(low), int(high), step=step
                    )
                else:  # categorical
                    params[space.name] = trial.suggest_categorical(
                        space.name, space.bounds
                    )

            result = self._evaluate_params(params, trial_id)
            trials.append(result)
            trial_id += 1

            if result.status == "failed":
                raise optuna.TrialPruned()

            return result.objective_value

        # Create study
        direction = "maximize" if self.maximize else "minimize"
        sampler = optuna.samplers.TPESampler(
            seed=self.seed, n_startup_trials=self.n_startup_trials
        )
        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        total_duration = (datetime.now() - start_time).total_seconds()

        # Build convergence history
        convergence_history: list[float] = []
        best_so_far = float("-inf") if self.maximize else float("inf")
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                is_better = (
                    trial.value > best_so_far
                    if self.maximize
                    else trial.value < best_so_far
                )
                if is_better:
                    best_so_far = trial.value
            convergence_history.append(best_so_far)

        return OptimizationResult(
            best_params=study.best_params,
            best_value=study.best_value,
            trials=trials,
            n_trials=len(trials),
            optimization_method="bayesian_tpe",
            total_duration_seconds=total_duration,
            search_space=self.search_space,
            convergence_history=convergence_history,
        )

    def _optimize_fallback(self) -> OptimizationResult:
        """Fallback optimization without Optuna (simple TPE-like)."""
        start_time = datetime.now()
        trials: list[TrialResult] = []

        # Start with random search for startup trials
        for i in range(self.n_startup_trials):
            params = {space.name: space.sample() for space in self.search_space}
            result = self._evaluate_params(params, i)
            trials.append(result)

        # Simple exploitation: sample around best observed values
        best_idx = (
            max(range(len(trials)), key=lambda i: trials[i].objective_value)
            if self.maximize
            else min(range(len(trials)), key=lambda i: trials[i].objective_value)
        )
        best_params = trials[best_idx].params

        for i in range(self.n_startup_trials, self.n_trials):
            # 70% exploitation, 30% exploration
            if random.random() < 0.7:
                params = self._perturb_params(best_params)
            else:
                params = {space.name: space.sample() for space in self.search_space}

            result = self._evaluate_params(params, i)
            trials.append(result)

            # Update best
            is_better = (
                result.objective_value > trials[best_idx].objective_value
                if self.maximize
                else result.objective_value < trials[best_idx].objective_value
            )
            if result.status == "completed" and is_better:
                best_idx = i
                best_params = result.params.copy()

        total_duration = (datetime.now() - start_time).total_seconds()

        # Build convergence history
        convergence_history: list[float] = []
        best_so_far = float("-inf") if self.maximize else float("inf")
        for trial in trials:
            if trial.status == "completed":
                is_better = (
                    trial.objective_value > best_so_far
                    if self.maximize
                    else trial.objective_value < best_so_far
                )
                if is_better:
                    best_so_far = trial.objective_value
            convergence_history.append(best_so_far)

        return OptimizationResult(
            best_params=best_params,
            best_value=trials[best_idx].objective_value,
            trials=trials,
            n_trials=len(trials),
            optimization_method="bayesian_fallback",
            total_duration_seconds=total_duration,
            search_space=self.search_space,
            convergence_history=convergence_history,
        )

    def _perturb_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Perturb parameters around current best."""
        perturbed = {}
        for space in self.search_space:
            current = params.get(space.name)

            if space.param_type == "float":
                low, high = space.bounds
                # Gaussian perturbation
                std = (high - low) * 0.1
                new_val = current + random.gauss(0, std)
                new_val = max(low, min(high, new_val))
                perturbed[space.name] = new_val

            elif space.param_type == "int":
                low, high = space.bounds
                # Small integer perturbation
                delta = random.choice([-2, -1, 0, 1, 2])
                new_val = current + delta
                new_val = max(int(low), min(int(high), new_val))
                perturbed[space.name] = new_val

            else:  # categorical - small chance to change
                if random.random() < 0.2:
                    perturbed[space.name] = random.choice(space.bounds)
                else:
                    perturbed[space.name] = current

        return perturbed


class GridSearchOptimizer(BaseOptimizer):
    """Grid search optimizer (exhaustive)."""

    def __init__(
        self,
        search_space: list[SearchSpace],
        objective_fn: Callable[[dict[str, Any]], float],
        n_trials: int = 100,
        maximize: bool = True,
        seed: Optional[int] = None,
        n_points_per_dim: int = 5,
    ) -> None:
        """Initialize grid search optimizer.

        Args:
            n_points_per_dim: Number of grid points per dimension
        """
        super().__init__(search_space, objective_fn, n_trials, maximize, seed)
        self.n_points_per_dim = n_points_per_dim

    def _generate_grid(self) -> list[dict[str, Any]]:
        """Generate all grid points."""
        import itertools

        param_grids: dict[str, list[Any]] = {}

        for space in self.search_space:
            if space.param_type == "float":
                low, high = space.bounds
                if space.log_scale:
                    points = np.geomspace(low, high, self.n_points_per_dim).tolist()
                else:
                    points = np.linspace(low, high, self.n_points_per_dim).tolist()
                param_grids[space.name] = points

            elif space.param_type == "int":
                low, high = space.bounds
                step = int(space.step) if space.step else 1
                points = list(range(int(low), int(high) + 1, step))
                # Subsample if too many
                if len(points) > self.n_points_per_dim:
                    indices = np.linspace(0, len(points) - 1, self.n_points_per_dim)
                    points = [points[int(i)] for i in indices]
                param_grids[space.name] = points

            else:  # categorical
                param_grids[space.name] = list(space.bounds)

        # Generate all combinations
        keys = list(param_grids.keys())
        values = [param_grids[k] for k in keys]
        grid = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

        return grid

    def optimize(self) -> OptimizationResult:
        """Run grid search optimization."""
        start_time = datetime.now()
        trials: list[TrialResult] = []

        grid = self._generate_grid()

        # Limit to n_trials
        if len(grid) > self.n_trials:
            random.shuffle(grid)
            grid = grid[: self.n_trials]

        best_params: dict[str, Any] = {}
        best_value = float("-inf") if self.maximize else float("inf")
        convergence_history: list[float] = []

        for i, params in enumerate(grid):
            result = self._evaluate_params(params, i)
            trials.append(result)

            if result.status == "completed":
                is_better = (
                    result.objective_value > best_value
                    if self.maximize
                    else result.objective_value < best_value
                )
                if is_better:
                    best_value = result.objective_value
                    best_params = params.copy()

            convergence_history.append(best_value)

        total_duration = (datetime.now() - start_time).total_seconds()

        return OptimizationResult(
            best_params=best_params,
            best_value=best_value,
            trials=trials,
            n_trials=len(trials),
            optimization_method="grid_search",
            total_duration_seconds=total_duration,
            search_space=self.search_space,
            convergence_history=convergence_history,
        )


def create_optimizer(
    method: str,
    search_space: list[SearchSpace],
    objective_fn: Callable[[dict[str, Any]], float],
    n_trials: int = 100,
    maximize: bool = True,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> BaseOptimizer:
    """Factory function to create optimizer by method name.

    Args:
        method: One of 'bayesian', 'genetic', 'random', 'grid'
        search_space: List of SearchSpace definitions
        objective_fn: Objective function to optimize
        n_trials: Number of trials
        maximize: Whether to maximize (True) or minimize (False)
        seed: Random seed
        **kwargs: Additional method-specific parameters

    Returns:
        Configured optimizer instance
    """
    method = method.lower()

    if method == "bayesian":
        return BayesianOptimizer(
            search_space,
            objective_fn,
            n_trials,
            maximize,
            seed,
            n_startup_trials=kwargs.get("n_startup_trials", 10),
        )
    elif method == "genetic":
        return GeneticOptimizer(
            search_space,
            objective_fn,
            n_trials,
            maximize,
            seed,
            population_size=kwargs.get("population_size", 20),
            mutation_rate=kwargs.get("mutation_rate", 0.1),
            crossover_rate=kwargs.get("crossover_rate", 0.8),
            elite_count=kwargs.get("elite_count", 2),
        )
    elif method == "random":
        return RandomSearchOptimizer(
            search_space, objective_fn, n_trials, maximize, seed
        )
    elif method == "grid":
        return GridSearchOptimizer(
            search_space,
            objective_fn,
            n_trials,
            maximize,
            seed,
            n_points_per_dim=kwargs.get("n_points_per_dim", 5),
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")


# Factor-specific search spaces
FACTOR_SEARCH_SPACES = {
    "momentum": [
        SearchSpace("lookback_period", "int", (5, 60), step=5),
        SearchSpace("smoothing_window", "int", (3, 20)),
        SearchSpace("use_volume_weight", "categorical", [True, False]),
    ],
    "volatility": [
        SearchSpace("window_size", "int", (10, 100), step=10),
        SearchSpace("decay_factor", "float", (0.9, 0.99)),
        SearchSpace("vol_type", "categorical", ["close", "parkinson", "garman_klass"]),
    ],
    "value": [
        SearchSpace("pe_threshold", "float", (5.0, 30.0)),
        SearchSpace("pb_threshold", "float", (0.5, 3.0)),
        SearchSpace("lookback_days", "int", (30, 365), step=30),
    ],
    "liquidity": [
        SearchSpace("volume_window", "int", (5, 30)),
        SearchSpace("spread_percentile", "float", (0.5, 0.99)),
        SearchSpace("min_turnover", "float", (0.001, 0.1), log_scale=True),
    ],
}
