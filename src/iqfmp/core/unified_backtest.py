"""Unified Backtest Framework for IQFMP.

This module provides:
1. Nested Execution Framework - Multi-level backtesting (day → 30min → 5min)
2. Unified Parameters - Single source of truth for backtest configuration
3. Unified Entry Point - Consistent interface across all backtest modes

Architecture:
=============
```
                UnifiedBacktestRunner
                        │
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
    Standard        Nested          Crypto
    (daily)      (multi-freq)    (perpetual)
        │               │               │
  BacktestEngine  NestedExecutor  CryptoQlibBacktest
                   ├── day         ├── Funding Rate
                   ├── 30min       ├── Liquidation
                   └── 5min        └── Margin
```

References:
- Qlib NestedExecutor: vendor/qlib/qlib/backtest/executor.py
- Qlib nested_decision_execution: vendor/qlib/examples/nested_decision_execution/
- De Prado AFML: Chapter 7 (Purged CV), Chapter 14 (Deflated Sharpe)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# Qlib imports with fallback
# =============================================================================

if TYPE_CHECKING:
    from qlib.backtest.executor import BaseExecutor
    from qlib.contrib.strategy import TopkDropoutStrategy

try:
    from qlib.backtest.executor import BaseExecutor
    from qlib.contrib.strategy import TopkDropoutStrategy
    QLIB_NESTED_AVAILABLE = True
except ImportError:
    QLIB_NESTED_AVAILABLE = False
    BaseExecutor = None
    TopkDropoutStrategy = None
    logger.warning("Qlib nested execution not available. Using fallback.")

# Internal imports (after try block to avoid circular imports)
from iqfmp.core.crypto_backtest import (  # noqa: E402
    CryptoBacktestConfig,
    CryptoBacktestResult,
    CryptoExchange,
    CryptoQlibBacktest,
)
from iqfmp.core.qlib_backtest_adapter import (  # noqa: E402
    QlibBacktestConfig,
    QlibBacktestRunner,
)
from iqfmp.exchange.margin import (  # noqa: E402
    MarginCalculator,
    MarginConfig,
    MarginMode,
)

# =============================================================================
# Enhancement 1: Nested Execution Configuration
# =============================================================================

class ExecutionFrequency(Enum):
    """Supported execution frequencies."""
    DAY = "day"
    HOUR_4 = "4h"
    HOUR_1 = "1h"
    MIN_30 = "30min"
    MIN_15 = "15min"
    MIN_5 = "5min"
    MIN_1 = "1min"

    def to_qlib_freq(self) -> str:
        """Convert to Qlib frequency string."""
        mapping = {
            ExecutionFrequency.DAY: "day",
            ExecutionFrequency.HOUR_4: "240min",
            ExecutionFrequency.HOUR_1: "60min",
            ExecutionFrequency.MIN_30: "30min",
            ExecutionFrequency.MIN_15: "15min",
            ExecutionFrequency.MIN_5: "5min",
            ExecutionFrequency.MIN_1: "1min",
        }
        return mapping[self]


class InnerStrategyType(Enum):
    """Types of inner execution strategies."""
    TWAP = "twap"           # Time-weighted average price
    VWAP = "vwap"           # Volume-weighted average price
    SBB_EMA = "sbb_ema"     # Sell-Buy-Buy with EMA
    PASSIVE = "passive"     # Wait for price, minimal impact


@dataclass
class NestedExecutionLevel:
    """Configuration for a single execution level in nested backtesting.

    Example:
        # Day-level portfolio decisions
        level_day = NestedExecutionLevel(
            frequency=ExecutionFrequency.DAY,
            strategy_type=InnerStrategyType.SBB_EMA,
            strategy_kwargs={"hold_thresh": 1.0},
        )

        # 30-minute execution
        level_30min = NestedExecutionLevel(
            frequency=ExecutionFrequency.MIN_30,
            strategy_type=InnerStrategyType.TWAP,
        )
    """
    frequency: ExecutionFrequency
    strategy_type: InnerStrategyType = InnerStrategyType.TWAP
    strategy_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_qlib_config(self) -> dict[str, Any]:
        """Convert to Qlib-compatible configuration dict."""
        strategy_class_map = {
            InnerStrategyType.TWAP: "TWAPStrategy",
            InnerStrategyType.VWAP: "VWAPStrategy",
            InnerStrategyType.SBB_EMA: "SBBStrategyEMA",
            InnerStrategyType.PASSIVE: "SBBStrategyBase",
        }

        return {
            "time_per_step": self.frequency.to_qlib_freq(),
            "strategy_class": strategy_class_map[self.strategy_type],
            "strategy_kwargs": self.strategy_kwargs,
        }


@dataclass
class NestedExecutionConfig:
    """Configuration for nested execution (multi-level backtesting).

    This enables intraday + daily combined backtesting:
    - Outer level: Daily portfolio rebalancing decisions
    - Middle level: 30-minute execution timing
    - Inner level: 5-minute order execution with TWAP

    Example:
        config = NestedExecutionConfig(
            levels=[
                NestedExecutionLevel(ExecutionFrequency.DAY, InnerStrategyType.SBB_EMA),
                NestedExecutionLevel(ExecutionFrequency.MIN_30, InnerStrategyType.SBB_EMA),
                NestedExecutionLevel(ExecutionFrequency.MIN_5, InnerStrategyType.TWAP),
            ],
            track_data=False,
            verbose=False,
        )

    Raises:
        ValueError: If configuration is invalid (e.g., < 2 levels, wrong order)
    """
    levels: list[NestedExecutionLevel] = field(default_factory=lambda: [
        NestedExecutionLevel(ExecutionFrequency.DAY, InnerStrategyType.SBB_EMA),
        NestedExecutionLevel(ExecutionFrequency.MIN_30, InnerStrategyType.SBB_EMA),
        NestedExecutionLevel(ExecutionFrequency.MIN_5, InnerStrategyType.TWAP),
    ])
    track_data: bool = False      # Track data for RL training
    verbose: bool = False         # Print execution details
    generate_portfolio_metrics: bool = True

    def __post_init__(self) -> None:
        """Validate configuration at construction time."""
        self.validate()

    def validate(self) -> None:
        """Validate nested execution configuration."""
        if len(self.levels) < 2:
            raise ValueError("Nested execution requires at least 2 levels")

        # Verify frequencies are in decreasing order (day > 30min > 5min)
        freq_order = [
            ExecutionFrequency.DAY,
            ExecutionFrequency.HOUR_4,
            ExecutionFrequency.HOUR_1,
            ExecutionFrequency.MIN_30,
            ExecutionFrequency.MIN_15,
            ExecutionFrequency.MIN_5,
            ExecutionFrequency.MIN_1,
        ]

        for i in range(len(self.levels) - 1):
            curr_idx = freq_order.index(self.levels[i].frequency)
            next_idx = freq_order.index(self.levels[i + 1].frequency)
            if curr_idx >= next_idx:
                raise ValueError(
                    f"Nested levels must be in decreasing frequency order. "
                    f"Level {i} ({self.levels[i].frequency}) should be > "
                    f"Level {i+1} ({self.levels[i+1].frequency})"
                )

    def build_qlib_executor_config(self) -> dict[str, Any]:
        """Build Qlib executor configuration for nested execution.

        Returns a nested dict structure like:
        {
            "class": "NestedExecutor",
            "kwargs": {
                "time_per_step": "day",
                "inner_executor": {
                    "class": "NestedExecutor",
                    "kwargs": {
                        "time_per_step": "30min",
                        "inner_executor": {
                            "class": "SimulatorExecutor",
                            "kwargs": {"time_per_step": "5min"}
                        }
                    }
                }
            }
        }
        """
        self.validate()

        # Build from innermost to outermost
        def build_level(level_idx: int) -> dict[str, Any]:
            level = self.levels[level_idx]
            level_config = level.to_qlib_config()

            if level_idx == len(self.levels) - 1:
                # Innermost level: SimulatorExecutor
                return {
                    "class": "SimulatorExecutor",
                    "kwargs": {
                        "time_per_step": level_config["time_per_step"],
                        "generate_portfolio_metrics": self.generate_portfolio_metrics,
                        "verbose": self.verbose,
                        "track_data": self.track_data,
                    },
                }
            else:
                # Outer levels: NestedExecutor
                return {
                    "class": "NestedExecutor",
                    "kwargs": {
                        "time_per_step": level_config["time_per_step"],
                        "inner_executor": build_level(level_idx + 1),
                        "inner_strategy": {
                            "class": level_config["strategy_class"],
                            "kwargs": level_config.get("strategy_kwargs", {}),
                        },
                        "generate_portfolio_metrics": self.generate_portfolio_metrics,
                        "verbose": self.verbose,
                        "track_data": self.track_data,
                    },
                }

        return build_level(0)


# =============================================================================
# Enhancement 2: Unified Parameters
# =============================================================================

class BacktestMode(Enum):
    """Backtest execution modes."""
    STANDARD = "standard"   # Simple daily backtest (BacktestEngine)
    QLIB = "qlib"           # Qlib-based with TopkDropoutStrategy
    NESTED = "nested"       # Multi-level nested execution
    CRYPTO = "crypto"       # Crypto perpetual futures


@dataclass
class UnifiedBacktestParams:
    """Unified parameters for all backtest modes.

    This is the single source of truth for backtest configuration.
    Provides adapters to convert to mode-specific configs.

    Example:
        params = UnifiedBacktestParams(
            start_time="2023-01-01",
            end_time="2024-01-01",
            initial_capital=100000.0,
            mode=BacktestMode.NESTED,
            nested_config=NestedExecutionConfig(...),
        )

        # Auto-convert to appropriate engine config
        qlib_config = params.to_qlib_config()
        crypto_config = params.to_crypto_config()

    Raises:
        ValueError: If any financial parameter is invalid
    """
    # Time range
    start_time: str | datetime = "2023-01-01"
    end_time: str | datetime = "2024-01-01"

    # Capital
    initial_capital: float = 100_000.0

    # Trading costs (unified)
    commission_rate: float = 0.0004      # 0.04% per trade
    slippage_rate: float = 0.0001        # 0.01% slippage

    # Position sizing
    max_position_pct: float = 0.95       # Max 95% of capital per position
    leverage: int = 1                     # 1x = no leverage

    # Mode-specific
    mode: BacktestMode = BacktestMode.STANDARD

    # Nested execution config (for NESTED mode)
    nested_config: NestedExecutionConfig | None = None

    # Crypto-specific (for CRYPTO mode)
    funding_enabled: bool = True
    funding_hours: list[int] = field(default_factory=lambda: [0, 8, 16])
    liquidation_enabled: bool = True
    margin_mode: MarginMode = MarginMode.ISOLATED

    # Qlib-specific (for QLIB/NESTED modes)
    benchmark: str = "SH000300"          # CSI 300 for A-shares
    topk: int = 50                       # Top-k stocks to hold
    n_drop: int = 5                      # Stocks to drop per rebalance

    # Anti-overfitting
    strict_cv_mode: bool = True          # Fail on insufficient data

    def __post_init__(self) -> None:
        """Validate financial parameters at construction time."""
        if self.initial_capital <= 0:
            raise ValueError(
                f"initial_capital must be positive, got {self.initial_capital}"
            )
        if self.leverage < 1:
            raise ValueError(f"leverage must be >= 1, got {self.leverage}")
        if not (0 < self.max_position_pct <= 1):
            raise ValueError(
                f"max_position_pct must be in (0, 1], got {self.max_position_pct}"
            )
        if self.commission_rate < 0:
            raise ValueError(
                f"commission_rate must be non-negative, got {self.commission_rate}"
            )
        if self.slippage_rate < 0:
            raise ValueError(
                f"slippage_rate must be non-negative, got {self.slippage_rate}"
            )
        if self.topk <= 0:
            raise ValueError(f"topk must be positive, got {self.topk}")
        if self.n_drop < 0:
            raise ValueError(f"n_drop must be non-negative, got {self.n_drop}")

    def to_qlib_config(self) -> QlibBacktestConfig:
        """Convert to Qlib backtest configuration."""
        return QlibBacktestConfig(
            start_time=self.start_time,
            end_time=self.end_time,
            benchmark=self.benchmark,
            account=self.initial_capital,
            exchange_kwargs={
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": self.commission_rate,
                "close_cost": self.commission_rate + 0.001,  # + stamp duty
                "min_cost": 5,
            },
            freq="day",
        )

    def to_crypto_config(self) -> CryptoBacktestConfig:
        """Convert to crypto backtest configuration."""
        return CryptoBacktestConfig(
            initial_capital=self.initial_capital,
            leverage=self.leverage,
            margin_mode=self.margin_mode,  # Already MarginMode enum
            commission_rate=self.commission_rate,
            slippage_rate=self.slippage_rate,
            funding_enabled=self.funding_enabled,
            funding_hours=self.funding_hours,
            liquidation_enabled=self.liquidation_enabled,
            max_position_pct=self.max_position_pct,
            strict_cv_mode=self.strict_cv_mode,
        )

    def to_nested_executor_config(self) -> dict[str, Any]:
        """Get Qlib nested executor configuration."""
        if self.nested_config is None:
            # Default 3-level nested execution
            self.nested_config = NestedExecutionConfig()

        return self.nested_config.build_qlib_executor_config()


# =============================================================================
# Enhancement 1: Crypto Nested Backtest (combines Qlib Nested + Crypto features)
# =============================================================================

class CryptoNestedBacktest:
    """Nested execution backtest with crypto-specific features.

    Combines:
    - Qlib's NestedExecutor for multi-level backtesting
    - CryptoExchange for funding rate + trading costs
    - MarginCalculator for liquidation logic

    Architecture:
        NestedExecutor (day)
        └── inner_strategy: PortfolioStrategy
        └── inner_executor: NestedExecutor (30min)
             └── inner_strategy: SBBStrategyEMA
             └── inner_executor: SimulatorExecutor (5min)
                  └── inner_strategy: TWAPStrategy
        + CryptoExchange (Funding Rate 8h cycles)
        + MarginCalculator (Liquidation)

    Example:
        >>> params = UnifiedBacktestParams(
        ...     mode=BacktestMode.NESTED,
        ...     leverage=10,
        ...     funding_enabled=True,
        ... )
        >>> engine = CryptoNestedBacktest(params)
        >>> result = engine.run(signals_df, price_data)
    """

    def __init__(self, params: UnifiedBacktestParams) -> None:
        """Initialize nested crypto backtest.

        Args:
            params: Unified backtest parameters
        """
        self.params = params
        self.crypto_config = params.to_crypto_config()

        # Initialize crypto components
        margin_config = MarginConfig.from_leverage(params.leverage)
        self.margin_calc = MarginCalculator(margin_config)
        self.exchange = CryptoExchange(self.crypto_config, self.margin_calc)

        # Build nested executor config
        self.executor_config: dict[str, Any] | None = None
        if QLIB_NESTED_AVAILABLE and params.nested_config:
            self.executor_config = params.to_nested_executor_config()
            logger.info(
                f"CryptoNestedBacktest initialized with "
                f"{len(params.nested_config.levels)} execution levels"
            )
        else:
            logger.warning(
                "Qlib nested execution not available, "
                "falling back to single-level crypto backtest"
            )

    def run(
        self,
        signals: pd.DataFrame,
        price_data: pd.DataFrame,
        funding_rates: pd.DataFrame | None = None,
        symbols: list[str] | None = None,
        allow_fallback: bool = False,
    ) -> CryptoBacktestResult:
        """Run nested crypto backtest.

        Args:
            signals: Factor signals DataFrame (datetime index, stock columns)
            price_data: OHLCV price data
            funding_rates: Optional funding rate data
            symbols: List of symbols to trade
            allow_fallback: If True, fallback to single-level on error.
                            If False (default), propagate errors.

        Returns:
            CryptoBacktestResult with nested execution metrics

        Raises:
            RuntimeError: If nested execution fails and allow_fallback is False
        """
        if not QLIB_NESTED_AVAILABLE or self.executor_config is None:
            # Fallback to single-level crypto backtest
            return self._run_fallback(signals, price_data, funding_rates)

        try:
            return self._run_nested(signals, price_data, funding_rates, symbols)
        except (ValueError, KeyError, TypeError) as e:
            # Data-related errors that indicate recoverable issues
            if allow_fallback:
                logger.warning(
                    f"Nested execution failed with data error: {e}, "
                    f"falling back to single-level"
                )
                return self._run_fallback(signals, price_data, funding_rates)
            raise RuntimeError(f"Nested execution failed: {e}") from e
        except ImportError as e:
            # Qlib import issues at runtime
            logger.warning(f"Qlib import error: {e}, falling back to single-level")
            return self._run_fallback(signals, price_data, funding_rates)

    def _run_nested(
        self,
        signals: pd.DataFrame,
        price_data: pd.DataFrame,
        funding_rates: pd.DataFrame | None,
        _symbols: list[str] | None,  # Reserved for future symbol filtering
    ) -> CryptoBacktestResult:
        """Run with Qlib nested execution."""
        # Import Qlib components
        from qlib.backtest import backtest_loop
        from qlib.utils import init_instance_by_config

        # Create outer strategy (portfolio-level)
        outer_strategy = TopkDropoutStrategy(
            signal=signals,
            topk=self.params.topk,
            n_drop=self.params.n_drop,
            only_tradable=True,
            forbid_all_trade_at_limit=True,
        )

        # Build executor from config
        executor = init_instance_by_config(
            self.executor_config,
            accept_types=BaseExecutor,
        )

        # Run Qlib backtest loop
        portfolio_dict, indicator_dict = backtest_loop(
            start_time=self.params.start_time,
            end_time=self.params.end_time,
            trade_strategy=outer_strategy,
            trade_executor=executor,
        )

        # Extract results and apply crypto adjustments
        result = self._process_qlib_results(
            portfolio_dict,
            indicator_dict,
            price_data,
            funding_rates,
        )

        return result

    def _process_qlib_results(
        self,
        portfolio_dict: dict[str, Any],
        _indicator_dict: dict[str, Any],  # Reserved for future indicator analysis
        _price_data: pd.DataFrame,  # Reserved for slippage analysis
        funding_rates: pd.DataFrame | None,
    ) -> CryptoBacktestResult:
        """Process Qlib results and apply crypto adjustments."""
        # Extract portfolio metrics
        equity_curve: pd.Series | None = None
        for key, (df, _metrics) in portfolio_dict.items():
            if "account" in key.lower():
                equity_curve = df.get("total_value", df.iloc[:, 0])
                break

        if equity_curve is None:
            raise ValueError("Failed to extract equity curve from Qlib results")

        # Apply funding rate adjustments
        funding_paid = 0.0
        funding_received = 0.0
        settlements: list[Any] = []

        if self.params.funding_enabled and funding_rates is not None:
            # Calculate cumulative funding impact using proper alignment
            # Funding rates apply to position notional value at funding time
            aligned_rates = funding_rates.reindex(
                equity_curve.index,
                method="ffill"  # Forward-fill to align funding rate windows
            ).fillna(0)

            # Funding cost = position_value * funding_rate (at each funding time)
            position_values: pd.Series[float] = equity_curve * self.params.leverage
            funding_costs: pd.Series[float] = position_values * aligned_rates

            # Sum only at funding hours (0, 8, 16 UTC typically)
            total_funding = funding_costs.sum()
            if isinstance(total_funding, pd.Series):
                total_funding = total_funding.sum()

            if total_funding < 0:
                funding_paid = abs(float(total_funding))
            else:
                funding_received = float(total_funding)

        # Build result
        from iqfmp.core.crypto_backtest import CryptoBacktestResult

        returns = equity_curve.pct_change().dropna()

        # Division by zero protection
        initial_value = equity_curve.iloc[0]
        if initial_value == 0 or pd.isna(initial_value):
            total_return = 0.0
        else:
            total_return = float((equity_curve.iloc[-1] / initial_value) - 1)

        annual_factor = 365 / max(1, len(equity_curve))
        annual_return = float(((1 + total_return) ** annual_factor) - 1)

        if returns.std() > 0:
            sharpe = float(returns.mean() / returns.std() * np.sqrt(365))
        else:
            sharpe = 0.0

        # Drawdown calculation
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = float(abs(drawdown.min()))

        return CryptoBacktestResult(
            total_return=total_return,
            annualized_return=annual_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            equity_curve=equity_curve,
            drawdown_curve=drawdown,
            total_funding_paid=funding_paid,
            total_funding_received=funding_received,
            net_funding=funding_received - funding_paid,
            settlements=settlements,
        )

    def _run_fallback(
        self,
        signals: pd.DataFrame,
        price_data: pd.DataFrame,
        funding_rates: pd.DataFrame | None,  # noqa: ARG002 (reserved for future)
    ) -> CryptoBacktestResult:
        """Fallback to single-level crypto backtest."""
        # Use existing CryptoQlibBacktest
        symbols = (
            list(signals.columns) if hasattr(signals, 'columns') else ["ETHUSDT"]
        )
        engine = CryptoQlibBacktest(config=self.crypto_config, symbols=symbols)

        # Convert signals DataFrame to Series (single asset)
        if isinstance(signals, pd.DataFrame):
            signal_series = signals.iloc[:, 0]
        else:
            signal_series = signals

        return engine.run(price_data, signal_series)


# =============================================================================
# Enhancement 3: Unified Entry Point
# =============================================================================

class UnifiedBacktestRunner:
    """Unified entry point for all backtest modes.

    Automatically selects the appropriate engine based on configuration.

    Example:
        >>> runner = UnifiedBacktestRunner()
        >>>
        >>> # Simple daily backtest
        >>> result = runner.run(
        ...     signals=signals_df,
        ...     data=price_data,
        ...     params=UnifiedBacktestParams(mode=BacktestMode.STANDARD),
        ... )
        >>>
        >>> # Nested execution backtest
        >>> result = runner.run(
        ...     signals=signals_df,
        ...     data=price_data,
        ...     params=UnifiedBacktestParams(
        ...         mode=BacktestMode.NESTED,
        ...         nested_config=NestedExecutionConfig(...),
        ...     ),
        ... )
        >>>
        >>> # Crypto perpetual futures backtest
        >>> result = runner.run(
        ...     signals=signals_df,
        ...     data=price_data,
        ...     params=UnifiedBacktestParams(
        ...         mode=BacktestMode.CRYPTO,
        ...         leverage=10,
        ...         funding_enabled=True,
        ...     ),
        ... )
    """

    def __init__(self) -> None:
        """Initialize unified backtest runner."""
        pass  # Stateless runner - engines created per-run

    def run(
        self,
        signals: pd.DataFrame | pd.Series,
        data: pd.DataFrame,
        params: UnifiedBacktestParams | None = None,
        funding_rates: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> CryptoBacktestResult | dict[str, Any]:
        """Run backtest with unified interface.

        Args:
            signals: Factor signals (DataFrame or Series)
            data: OHLCV price data
            params: Unified backtest parameters
            funding_rates: Optional funding rate data (for crypto mode)
            **kwargs: Additional mode-specific arguments

        Returns:
            Backtest result (type depends on mode)
        """
        params = params or UnifiedBacktestParams()

        logger.info(f"Running backtest in {params.mode.value} mode")

        if params.mode == BacktestMode.STANDARD:
            return self._run_standard(signals, data, params, **kwargs)
        elif params.mode == BacktestMode.QLIB:
            return self._run_qlib(signals, data, params, **kwargs)
        elif params.mode == BacktestMode.NESTED:
            return self._run_nested(signals, data, params, funding_rates, **kwargs)
        elif params.mode == BacktestMode.CRYPTO:
            return self._run_crypto(signals, data, params, funding_rates, **kwargs)
        else:
            raise ValueError(f"Unknown backtest mode: {params.mode}")

    def _run_standard(
        self,
        signals: pd.DataFrame | pd.Series,
        data: pd.DataFrame,
        params: UnifiedBacktestParams,
        **_kwargs: Any,  # Reserved for future options
    ) -> dict[str, Any]:
        """Run standard daily backtest."""
        from iqfmp.core.backtest_engine import BacktestEngine

        # Ensure data has timestamp column (BacktestEngine expects this)
        data_prepared = data.copy()
        if "timestamp" not in data_prepared.columns:
            if isinstance(data_prepared.index, pd.DatetimeIndex):
                data_prepared["timestamp"] = data_prepared.index
            else:
                data_prepared["timestamp"] = pd.to_datetime(data_prepared.index)

        engine = BacktestEngine(df=data_prepared)

        # Convert signals to series if needed
        if isinstance(signals, pd.DataFrame):
            signal_series = signals.iloc[:, 0]
        else:
            signal_series = signals

        result = engine.run_factor_backtest(
            factor_code="unified_signal",
            factor_values=signal_series,
            initial_capital=params.initial_capital,
            start_date=str(params.start_time),
            end_date=str(params.end_time),
        )

        result_dict: dict[str, Any] = result.to_dict()
        return result_dict

    def _run_qlib(
        self,
        signals: pd.DataFrame | pd.Series,
        _data: pd.DataFrame,  # Qlib uses its own data loader
        params: UnifiedBacktestParams,
        **_kwargs: Any,  # Reserved for future options
    ) -> dict[str, Any]:
        """Run Qlib-based backtest."""
        qlib_config = params.to_qlib_config()
        runner = QlibBacktestRunner(qlib_config)

        # Ensure signals is DataFrame
        signals_df = signals.to_frame() if isinstance(signals, pd.Series) else signals

        result: dict[str, Any] = runner.run(
            signals=signals_df,
            strategy_type="topk",
            topk=params.topk,
            n_drop=params.n_drop,
        )
        return result

    def _run_nested(
        self,
        signals: pd.DataFrame | pd.Series,
        data: pd.DataFrame,
        params: UnifiedBacktestParams,
        funding_rates: pd.DataFrame | None,
        **_kwargs: Any,  # Reserved for future options
    ) -> CryptoBacktestResult:
        """Run nested execution backtest."""
        # Ensure signals is DataFrame
        signals_df = signals.to_frame() if isinstance(signals, pd.Series) else signals

        engine = CryptoNestedBacktest(params)
        return engine.run(
            signals=signals_df,
            price_data=data,
            funding_rates=funding_rates,
        )

    def _run_crypto(
        self,
        signals: pd.DataFrame | pd.Series,
        data: pd.DataFrame,
        params: UnifiedBacktestParams,
        _funding_rates: pd.DataFrame | None,  # CryptoQlibBacktest handles internally
        **kwargs: Any,
    ) -> CryptoBacktestResult:
        """Run crypto perpetual futures backtest."""
        crypto_config = params.to_crypto_config()
        engine = CryptoQlibBacktest(crypto_config)

        # Convert signals to series if needed
        signal_series = signals.iloc[:, 0] if isinstance(signals, pd.DataFrame) else signals

        return engine.run(
            data=data,
            signals=signal_series,
            symbol=kwargs.get("symbol", "ETHUSDT"),
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_backtest_runner() -> UnifiedBacktestRunner:
    """Create a unified backtest runner instance."""
    return UnifiedBacktestRunner()


def run_backtest(
    signals: pd.DataFrame | pd.Series,
    data: pd.DataFrame,
    mode: str = "standard",
    **kwargs: Any,
) -> CryptoBacktestResult | dict[str, Any]:
    """Convenience function to run backtest with minimal configuration.

    Args:
        signals: Factor signals
        data: OHLCV price data
        mode: "standard", "qlib", "nested", or "crypto"
        **kwargs: Additional parameters passed to UnifiedBacktestParams

    Returns:
        Backtest result

    Example:
        >>> result = run_backtest(
        ...     signals=my_signals,
        ...     data=price_data,
        ...     mode="nested",
        ...     leverage=10,
        ...     funding_enabled=True,
        ... )
    """
    mode_enum = BacktestMode(mode)

    # Extract known params
    params_kwargs: dict[str, Any] = {
        "mode": mode_enum,
    }

    # Map kwargs to UnifiedBacktestParams fields
    param_fields = {
        "start_time", "end_time", "initial_capital", "commission_rate",
        "slippage_rate", "max_position_pct", "leverage", "funding_enabled",
        "funding_hours", "liquidation_enabled", "margin_mode", "benchmark",
        "topk", "n_drop", "strict_cv_mode", "nested_config",
    }

    for key, value in kwargs.items():
        if key in param_fields:
            # Convert margin_mode string to enum if needed
            if key == "margin_mode" and isinstance(value, str):
                value = MarginMode.ISOLATED if value == "isolated" else MarginMode.CROSS
            params_kwargs[key] = value

    params = UnifiedBacktestParams(**params_kwargs)

    # Run remaining kwargs to runner
    runner_kwargs = {k: v for k, v in kwargs.items() if k not in param_fields}

    runner = UnifiedBacktestRunner()
    return runner.run(
        signals=signals,
        data=data,
        params=params,
        **runner_kwargs,
    )


# =============================================================================
# Prebuilt Nested Configurations
# =============================================================================

def create_standard_nested_config() -> NestedExecutionConfig:
    """Create standard 3-level nested execution configuration.

    Day → 30min → 5min with TWAP execution.
    """
    return NestedExecutionConfig(
        levels=[
            NestedExecutionLevel(ExecutionFrequency.DAY, InnerStrategyType.SBB_EMA),
            NestedExecutionLevel(
                ExecutionFrequency.MIN_30, InnerStrategyType.SBB_EMA
            ),
            NestedExecutionLevel(ExecutionFrequency.MIN_5, InnerStrategyType.TWAP),
        ],
    )


def create_crypto_nested_config() -> NestedExecutionConfig:
    """Create crypto-optimized nested execution configuration.

    4h → 1h → 5min for 24/7 crypto markets.
    """
    return NestedExecutionConfig(
        levels=[
            NestedExecutionLevel(ExecutionFrequency.HOUR_4, InnerStrategyType.SBB_EMA),
            NestedExecutionLevel(ExecutionFrequency.HOUR_1, InnerStrategyType.SBB_EMA),
            NestedExecutionLevel(ExecutionFrequency.MIN_5, InnerStrategyType.TWAP),
        ],
    )


def create_hft_nested_config() -> NestedExecutionConfig:
    """Create high-frequency nested execution configuration.

    1h → 15min → 1min for active trading.
    """
    return NestedExecutionConfig(
        levels=[
            NestedExecutionLevel(ExecutionFrequency.HOUR_1, InnerStrategyType.SBB_EMA),
            NestedExecutionLevel(ExecutionFrequency.MIN_15, InnerStrategyType.SBB_EMA),
            NestedExecutionLevel(ExecutionFrequency.MIN_1, InnerStrategyType.TWAP),
        ],
    )
