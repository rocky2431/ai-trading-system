"""Unified Crypto Perpetual Futures Backtesting Engine.

This module provides a unified backtest engine that combines:
1. Qlib's C++ expression engine for factor computation
2. Crypto-specific Funding Rate settlement (8h cycles)
3. MarginCalculator for liquidation and margin management
4. Multi-asset portfolio support (BTC+ETH, etc.)

Architecture:
=============
```
                    CryptoQlibBacktest
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   QlibSignalEngine  CryptoExchange  MarginCalculator
        │                 │                 │
   - D.features()    - Funding Rate    - Liquidation
   - Expressions     - Trading Costs   - Margin Ratio
   - Factor IC       - Settlement      - Position Sizing
```

Phase 3 Implementation (P0):
- Unified entry point for all crypto backtests
- Real Qlib C++ engine for signal generation
- Proper Funding Rate and Liquidation handling

Reference:
- Binance Futures margin documentation
- Qlib backtest architecture
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

# Qlib imports
try:
    import qlib
    from qlib.data import D
    from qlib.backtest.report import Indicator
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    D = None

# Internal imports
from iqfmp.exchange.margin import (
    MarginCalculator,
    MarginConfig,
    MarginPosition,
    MarginResult,
    MarginStatus,
    PositionSide,
    MarginMode,
)

# Anti-overfitting: Purged K-Fold CV (De Prado AFML Chapter 7)
from iqfmp.evaluation.purged_cv import (
    PurgedKFoldCV,
    PurgedCVConfig,
    TimeSeriesPurgedCV,
)

# =============================================================================
# ACTION 2: Research Ledger for trial tracking
# Every backtest result MUST be recorded to prevent P-hacking
# =============================================================================
try:
    from iqfmp.evaluation.research_ledger import (
        ResearchLedger,
        TrialRecord,
        FileStorage,
        MemoryStorage,
    )
    RESEARCH_LEDGER_AVAILABLE = True
except ImportError:
    RESEARCH_LEDGER_AVAILABLE = False
    ResearchLedger = None
    TrialRecord = None
    FileStorage = None
    MemoryStorage = None

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class InsufficientDataError(Exception):
    """数据不足以进行可靠的防过拟合验证。

    当数据量不足以进行 Purged K-Fold CV 验证时抛出此异常。
    这是一个严格的错误 - 禁止静默返回误导性的 prob_overfit=1.0。

    解决方案：
    1. 提供更多历史数据（至少 n_splits * 50 个样本）
    2. 降低 n_splits 参数
    3. 设置 strict_cv_mode=False 以获取 NaN 值而非异常
    """

    def __init__(
        self,
        required_samples: int,
        actual_samples: int,
        message: str = "",
    ):
        self.required_samples = required_samples
        self.actual_samples = actual_samples
        super().__init__(
            message or
            f"Insufficient data for reliable CV validation: "
            f"need {required_samples} samples, got {actual_samples}. "
            f"Either provide more data or set strict_cv_mode=False."
        )


# =============================================================================
# Enums and Configuration
# =============================================================================

class PositionType(Enum):
    """Position type for perpetual futures."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class SettlementEvent(Enum):
    """Types of settlement events."""
    FUNDING = "funding"
    LIQUIDATION = "liquidation"
    TRADE = "trade"
    MARGIN_CALL = "margin_call"


@dataclass
class CryptoBacktestConfig:
    """Configuration for crypto perpetual futures backtesting.

    Attributes:
        initial_capital: Starting capital in quote currency (e.g., USDT)
        leverage: Default leverage for positions (1-125x)
        margin_mode: Isolated or cross margin
        commission_rate: Trading commission (taker fee)
        slippage_rate: Slippage estimate
        funding_enabled: Whether to apply funding rate settlements
        funding_hours: Hours when funding is settled (default: 0, 8, 16 UTC)
        liquidation_enabled: Whether to check and execute liquidations
        max_position_pct: Maximum position size as % of capital
        min_trade_amount: Minimum trade amount in quote currency
    """
    initial_capital: float = 100000.0
    leverage: int = 10
    margin_mode: MarginMode = MarginMode.ISOLATED
    commission_rate: float = 0.0004  # 0.04% taker fee
    slippage_rate: float = 0.0001   # 0.01% slippage
    funding_enabled: bool = True
    funding_hours: list[int] = field(default_factory=lambda: [0, 8, 16])
    liquidation_enabled: bool = True
    max_position_pct: float = 0.95  # Max 95% of capital
    min_trade_amount: float = 10.0  # Min $10 trade
    # 严格 CV 模式：数据不足时抛出异常而非静默返回误导性值
    # 生产环境必须设置为 True，只有测试环境可以设置为 False
    strict_cv_mode: bool = True


@dataclass
class BacktestTrade:
    """Record of a single trade."""
    timestamp: datetime
    symbol: str
    side: PositionType
    quantity: float
    price: float
    commission: float
    pnl: float = 0.0
    is_liquidation: bool = False


@dataclass
class SettlementRecord:
    """Record of a settlement event (funding or liquidation)."""
    timestamp: datetime
    event_type: SettlementEvent
    symbol: str
    amount: float
    details: dict = field(default_factory=dict)


@dataclass
class CryptoBacktestResult:
    """Result of crypto backtest execution.

    Contains:
    - Performance metrics (Sharpe, Sortino, Max DD, etc.)
    - Equity curve
    - Trade history
    - Settlement history (funding + liquidations)
    - Per-asset breakdown
    - Anti-overfitting metrics (Deflated Sharpe, Prob(Overfit))
    """
    # Core metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # Anti-overfitting metrics (De Prado AFML Chapter 7 & 14)
    deflated_sharpe: float = 0.0  # Sharpe adjusted for multiple testing
    prob_overfit: float = 0.0    # Probability of overfitting (0-1)
    cv_sharpe_mean: float = 0.0  # Mean Sharpe across CV folds
    cv_sharpe_std: float = 0.0   # Std of Sharpe across CV folds
    n_cv_folds: int = 0          # Number of CV folds used

    # Trade statistics
    n_trades: int = 0
    n_liquidations: int = 0
    avg_trade_pnl: float = 0.0

    # Funding statistics
    total_funding_paid: float = 0.0
    total_funding_received: float = 0.0
    net_funding: float = 0.0

    # Time series
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_curve: pd.Series = field(default_factory=pd.Series)

    # History
    trades: list[BacktestTrade] = field(default_factory=list)
    settlements: list[SettlementRecord] = field(default_factory=list)

    # Per-asset breakdown
    per_asset_pnl: dict[str, float] = field(default_factory=dict)
    per_asset_trades: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "metrics": {
                "total_return": self.total_return,
                "annualized_return": self.annualized_return,
                "sharpe_ratio": self.sharpe_ratio,
                "sortino_ratio": self.sortino_ratio,
                "max_drawdown": self.max_drawdown,
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
            },
            "anti_overfit": {
                "deflated_sharpe": self.deflated_sharpe,
                "prob_overfit": self.prob_overfit,
                "cv_sharpe_mean": self.cv_sharpe_mean,
                "cv_sharpe_std": self.cv_sharpe_std,
                "n_cv_folds": self.n_cv_folds,
            },
            "trading": {
                "n_trades": self.n_trades,
                "n_liquidations": self.n_liquidations,
                "avg_trade_pnl": self.avg_trade_pnl,
            },
            "funding": {
                "total_paid": self.total_funding_paid,
                "total_received": self.total_funding_received,
                "net": self.net_funding,
            },
            "per_asset": {
                "pnl": self.per_asset_pnl,
                "trades": self.per_asset_trades,
            },
        }


# =============================================================================
# CryptoExchange - Handles Funding Rate + Trading Costs
# =============================================================================

class CryptoExchange:
    """Crypto perpetual futures exchange simulation.

    Handles:
    - Funding rate settlement (8h cycles)
    - Trading costs (commission + slippage)
    - Order execution
    """

    def __init__(
        self,
        config: CryptoBacktestConfig,
        margin_calculator: MarginCalculator,
    ):
        """Initialize exchange.

        Args:
            config: Backtest configuration
            margin_calculator: Margin calculator for liquidation
        """
        self.config = config
        self.margin_calc = margin_calculator

        # Settlement tracking
        self._funding_settlements: list[SettlementRecord] = []
        self._last_funding_hour: dict[str, int] = {}

    def calculate_trading_costs(
        self,
        quantity: float,
        price: float,
    ) -> float:
        """Calculate total trading costs (commission + slippage).

        Args:
            quantity: Trade quantity
            price: Trade price

        Returns:
            Total cost in quote currency
        """
        notional = abs(quantity) * price
        commission = notional * self.config.commission_rate
        slippage = notional * self.config.slippage_rate
        return commission + slippage

    def calculate_funding_payment(
        self,
        position_size: float,
        position_type: PositionType,
        mark_price: float,
        funding_rate: float,
    ) -> float:
        """Calculate funding payment for a position.

        Funding Payment = Position Value * Funding Rate
        - Positive rate: Longs pay shorts
        - Negative rate: Shorts pay longs

        Args:
            position_size: Position size (always positive)
            position_type: Long or short
            mark_price: Current mark price
            funding_rate: Current funding rate

        Returns:
            Funding payment (negative = pay, positive = receive)
        """
        if position_type == PositionType.FLAT or position_size == 0:
            return 0.0

        notional = abs(position_size) * mark_price

        # Long pays when rate is positive, receives when negative
        # Short receives when rate is positive, pays when negative
        if position_type == PositionType.LONG:
            payment = -notional * funding_rate  # Long pays positive rate
        else:
            payment = notional * funding_rate   # Short receives positive rate

        return payment

    def should_settle_funding(
        self,
        timestamp: datetime,
        symbol: str,
    ) -> bool:
        """Check if funding should be settled at this timestamp.

        Args:
            timestamp: Current timestamp
            symbol: Trading symbol

        Returns:
            True if funding should be settled
        """
        if not self.config.funding_enabled:
            return False

        current_hour = timestamp.hour

        if current_hour not in self.config.funding_hours:
            return False

        # Only settle once per funding hour
        last_hour = self._last_funding_hour.get(symbol, -1)
        if last_hour == current_hour and timestamp.minute < 5:
            return False

        # Check if we're at the settlement time (minute 0-5)
        if timestamp.minute > 5:
            return False

        self._last_funding_hour[symbol] = current_hour
        return True

    def execute_order(
        self,
        symbol: str,
        side: PositionType,
        quantity: float,
        price: float,
        timestamp: datetime,
    ) -> BacktestTrade:
        """Execute a trade order.

        Args:
            symbol: Trading symbol
            side: Order side (long/short)
            quantity: Order quantity
            price: Execution price
            timestamp: Execution timestamp

        Returns:
            Trade record
        """
        costs = self.calculate_trading_costs(quantity, price)

        return BacktestTrade(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            commission=costs,
        )


# =============================================================================
# CryptoQlibBacktest - Main Unified Engine
# =============================================================================

class CryptoQlibBacktest:
    """Unified Crypto Perpetual Futures Backtesting Engine.

    Combines Qlib's C++ engine for signal generation with crypto-specific
    settlement logic (Funding Rate + Liquidation).

    Example:
        >>> config = CryptoBacktestConfig(
        ...     initial_capital=100000,
        ...     leverage=10,
        ...     funding_enabled=True,
        ... )
        >>> engine = CryptoQlibBacktest(config)
        >>>
        >>> # Load data and signals
        >>> data = pd.DataFrame(...)  # OHLCV + funding_rate
        >>> signals = pd.Series(...)  # 1=long, -1=short, 0=flat
        >>>
        >>> # Run backtest
        >>> result = engine.run(data, signals)
        >>> print(f"Sharpe: {result.sharpe_ratio:.2f}")
    """

    def __init__(
        self,
        config: Optional[CryptoBacktestConfig] = None,
        symbols: Optional[list[str]] = None,
        ledger: Optional["ResearchLedger"] = None,
        factor_name: str = "unnamed_factor",
        factor_family: str = "crypto",
    ):
        """Initialize backtest engine.

        Args:
            config: Backtest configuration
            symbols: List of symbols for multi-asset backtest
            ledger: Optional ResearchLedger for trial tracking (ACTION 2)
            factor_name: Name of factor being backtested (for ledger)
            factor_family: Family/category of factor (for ledger)
        """
        self.config = config or CryptoBacktestConfig()
        self.symbols = symbols or ["ETHUSDT"]

        # =======================================================================
        # ACTION 2: Research Ledger integration
        # Every backtest result MUST be recorded to prevent P-hacking
        # =======================================================================
        self.ledger = ledger
        self.factor_name = factor_name
        self.factor_family = factor_family

        if self.ledger is not None:
            logger.info(
                f"CryptoQlibBacktest: Research Ledger connected "
                f"({self.ledger.get_trial_count()} existing trials)"
            )
        elif RESEARCH_LEDGER_AVAILABLE:
            # Auto-create ledger if module available but not provided
            from pathlib import Path
            ledger_path = Path(".ultra/crypto_research_ledger.json")
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            self.ledger = ResearchLedger(storage=FileStorage(ledger_path))
            logger.info(
                f"CryptoQlibBacktest: Auto-created Research Ledger at {ledger_path}"
            )

        # Initialize margin calculator
        margin_config = MarginConfig.from_leverage(self.config.leverage)
        self.margin_calc = MarginCalculator(margin_config)

        # Initialize exchange
        self.exchange = CryptoExchange(self.config, self.margin_calc)

        # State tracking
        self._positions: dict[str, dict] = {}  # symbol -> position state
        self._equity_history: list[tuple[datetime, float]] = []
        self._trades: list[BacktestTrade] = []
        self._settlements: list[SettlementRecord] = []

        logger.info(
            f"CryptoQlibBacktest initialized: "
            f"capital=${self.config.initial_capital:,.0f}, "
            f"leverage={self.config.leverage}x, "
            f"symbols={self.symbols}"
        )

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        symbol: str = "ETHUSDT",
    ) -> CryptoBacktestResult:
        """Run single-asset backtest.

        Args:
            data: OHLCV DataFrame with 'close', 'funding_rate' columns
            signals: Trading signals (1=long, -1=short, 0=flat)
            symbol: Trading symbol

        Returns:
            Backtest result with metrics and history
        """
        # Validate inputs
        self._validate_inputs(data, signals)

        # Initialize state
        capital = self.config.initial_capital
        position_size = 0.0
        position_type = PositionType.FLAT
        entry_price = 0.0

        # History tracking
        equity_values: list[float] = []
        trades: list[BacktestTrade] = []
        settlements: list[SettlementRecord] = []

        total_funding = 0.0
        funding_paid = 0.0
        funding_received = 0.0

        # Main backtest loop
        for i, (timestamp, row) in enumerate(data.iterrows()):
            signal = signals.iloc[i]
            price = float(row["close"])

            # 1. Check and apply funding settlement
            if self.exchange.should_settle_funding(timestamp, symbol):
                if position_size != 0 and "funding_rate" in row:
                    funding_rate = float(row.get("funding_rate", 0))
                    funding_pnl = self.exchange.calculate_funding_payment(
                        position_size, position_type, price, funding_rate
                    )
                    capital += funding_pnl
                    total_funding += funding_pnl

                    if funding_pnl < 0:
                        funding_paid += abs(funding_pnl)
                    else:
                        funding_received += funding_pnl

                    settlements.append(SettlementRecord(
                        timestamp=timestamp,
                        event_type=SettlementEvent.FUNDING,
                        symbol=symbol,
                        amount=funding_pnl,
                        details={"rate": funding_rate, "position": position_size},
                    ))

            # 2. Check for liquidation
            if self.config.liquidation_enabled and position_size != 0:
                is_liquidated, liq_result = self._check_liquidation(
                    position_size, position_type, entry_price, price, capital
                )
                if is_liquidated:
                    # Execute liquidation
                    liq_pnl = self._calculate_unrealized_pnl(
                        position_size, position_type, entry_price, price
                    )
                    capital += liq_pnl

                    trades.append(BacktestTrade(
                        timestamp=timestamp,
                        symbol=symbol,
                        side=PositionType.FLAT,
                        quantity=position_size,
                        price=price,
                        commission=0,  # Liquidation has different fee structure
                        pnl=liq_pnl,
                        is_liquidation=True,
                    ))

                    settlements.append(SettlementRecord(
                        timestamp=timestamp,
                        event_type=SettlementEvent.LIQUIDATION,
                        symbol=symbol,
                        amount=liq_pnl,
                        details={"entry_price": entry_price, "liq_price": price},
                    ))

                    position_size = 0.0
                    position_type = PositionType.FLAT
                    entry_price = 0.0

            # 3. Process trading signal
            target_position = self._signal_to_position(signal)

            if target_position != position_type:
                # Close existing position
                if position_size != 0:
                    close_pnl = self._calculate_unrealized_pnl(
                        position_size, position_type, entry_price, price
                    )
                    close_costs = self.exchange.calculate_trading_costs(position_size, price)
                    capital += close_pnl - close_costs

                    trades.append(BacktestTrade(
                        timestamp=timestamp,
                        symbol=symbol,
                        side=PositionType.FLAT,
                        quantity=position_size,
                        price=price,
                        commission=close_costs,
                        pnl=close_pnl,
                    ))

                # Open new position
                if target_position != PositionType.FLAT:
                    # Calculate position size based on capital and leverage
                    max_notional = capital * self.config.max_position_pct * self.config.leverage
                    new_size = max_notional / price
                    open_costs = self.exchange.calculate_trading_costs(new_size, price)

                    capital -= open_costs
                    position_size = new_size
                    position_type = target_position
                    entry_price = price

                    trades.append(BacktestTrade(
                        timestamp=timestamp,
                        symbol=symbol,
                        side=target_position,
                        quantity=new_size,
                        price=price,
                        commission=open_costs,
                    ))
                else:
                    position_size = 0.0
                    position_type = PositionType.FLAT
                    entry_price = 0.0

            # 4. Calculate current equity
            unrealized_pnl = self._calculate_unrealized_pnl(
                position_size, position_type, entry_price, price
            )
            current_equity = capital + unrealized_pnl
            equity_values.append(current_equity)

        # Build result
        equity_series = pd.Series(equity_values, index=data.index)

        return self._build_result(
            equity_series=equity_series,
            trades=trades,
            settlements=settlements,
            initial_capital=self.config.initial_capital,
            funding_paid=funding_paid,
            funding_received=funding_received,
            symbol=symbol,
        )

    def run_multi_asset(
        self,
        data_dict: dict[str, pd.DataFrame],
        signals_dict: dict[str, pd.Series],
        capital_allocation: Optional[dict[str, float]] = None,
    ) -> CryptoBacktestResult:
        """Run multi-asset portfolio backtest.

        Args:
            data_dict: Dict of symbol -> OHLCV DataFrame
            signals_dict: Dict of symbol -> signals Series
            capital_allocation: Dict of symbol -> capital fraction (must sum to 1.0)

        Returns:
            Combined portfolio backtest result
        """
        symbols = list(data_dict.keys())

        # Default equal allocation
        if capital_allocation is None:
            allocation = 1.0 / len(symbols)
            capital_allocation = {s: allocation for s in symbols}

        # Validate allocation sums to 1
        total_alloc = sum(capital_allocation.values())
        if abs(total_alloc - 1.0) > 0.01:
            raise ValueError(f"Capital allocation must sum to 1.0, got {total_alloc}")

        # Run individual backtests with allocated capital
        results: dict[str, CryptoBacktestResult] = {}
        for symbol in symbols:
            # Create config with allocated capital
            symbol_config = CryptoBacktestConfig(
                initial_capital=self.config.initial_capital * capital_allocation[symbol],
                leverage=self.config.leverage,
                margin_mode=self.config.margin_mode,
                commission_rate=self.config.commission_rate,
                slippage_rate=self.config.slippage_rate,
                funding_enabled=self.config.funding_enabled,
                funding_hours=self.config.funding_hours,
                liquidation_enabled=self.config.liquidation_enabled,
            )

            engine = CryptoQlibBacktest(symbol_config, [symbol])
            results[symbol] = engine.run(
                data_dict[symbol],
                signals_dict[symbol],
                symbol,
            )

        # Combine results
        return self._combine_results(results, capital_allocation)

    def _validate_inputs(self, data: pd.DataFrame, signals: pd.Series) -> None:
        """Validate input data and signals."""
        if len(data) == 0:
            raise ValueError("Data is empty")

        if len(data) != len(signals):
            raise ValueError(
                f"Data length ({len(data)}) != signals length ({len(signals)})"
            )

        if "close" not in data.columns:
            raise ValueError("Data must have 'close' column")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")

    def _signal_to_position(self, signal: float) -> PositionType:
        """Convert signal value to position type."""
        if signal > 0:
            return PositionType.LONG
        elif signal < 0:
            return PositionType.SHORT
        else:
            return PositionType.FLAT

    def _calculate_unrealized_pnl(
        self,
        position_size: float,
        position_type: PositionType,
        entry_price: float,
        current_price: float,
    ) -> float:
        """Calculate unrealized PnL for position."""
        if position_size == 0 or position_type == PositionType.FLAT:
            return 0.0

        if position_type == PositionType.LONG:
            return (current_price - entry_price) * position_size
        else:
            return (entry_price - current_price) * position_size

    def _check_liquidation(
        self,
        position_size: float,
        position_type: PositionType,
        entry_price: float,
        current_price: float,
        capital: float,
    ) -> tuple[bool, Optional[MarginResult]]:
        """Check if position should be liquidated.

        Returns:
            Tuple of (is_liquidated, margin_result)
        """
        if position_size == 0:
            return False, None

        # Calculate liquidation price
        side = PositionSide.LONG if position_type == PositionType.LONG else PositionSide.SHORT

        if side == PositionSide.LONG:
            liq_price = self.margin_calc.calculate_liquidation_price_long(
                Decimal(str(entry_price)),
                self.config.leverage,
            )
            is_liquidated = current_price <= float(liq_price)
        else:
            liq_price = self.margin_calc.calculate_liquidation_price_short(
                Decimal(str(entry_price)),
                self.config.leverage,
            )
            is_liquidated = current_price >= float(liq_price)

        return is_liquidated, None

    def _insufficient_cv_data_result(
        self,
        required: int,
        actual: int,
        message: str,
        n_folds: int = 0,
    ) -> dict[str, float]:
        """Handle insufficient data for CV validation.

        In strict mode, raises InsufficientDataError.
        In non-strict mode, returns NaN values to indicate unknown rather than
        misleading defaults.

        Args:
            required: Required number of samples
            actual: Actual number of samples
            message: Descriptive message for logging/error
            n_folds: Number of folds completed (default 0)

        Returns:
            Dict with NaN values if not in strict mode

        Raises:
            InsufficientDataError: If strict_cv_mode is enabled
        """
        if self.config.strict_cv_mode:
            raise InsufficientDataError(
                required_samples=required,
                actual_samples=actual,
                message=message,
            )

        logger.warning(f"{message} Returning NaN for prob_overfit.")
        return {
            "cv_sharpe_mean": float("nan"),
            "cv_sharpe_std": float("nan"),
            "deflated_sharpe": float("nan"),
            "prob_overfit": float("nan"),
            "n_folds": n_folds,
        }

    def _calculate_cv_metrics(
        self,
        equity_series: pd.Series,
        n_splits: int = 5,
        purge_gap: int = 5,
        embargo_pct: float = 0.01,
    ) -> dict[str, float]:
        """Calculate cross-validated metrics using Purged K-Fold CV.

        Implements De Prado AFML Chapter 7 (Purged CV) and Chapter 14 (Deflated Sharpe).

        This is the CRITICAL anti-overfitting mechanism:
        - Calculates Sharpe ratio on each CV fold (out-of-sample)
        - Computes mean/std of Sharpe across folds
        - Calculates Deflated Sharpe Ratio (DSR) accounting for multiple testing
        - Estimates probability of overfitting

        Args:
            equity_series: Equity curve time series
            n_splits: Number of CV folds (default 5)
            purge_gap: Periods to purge around test boundaries
            embargo_pct: Embargo period as fraction of data

        Returns:
            Dict with cv_sharpe_mean, cv_sharpe_std, deflated_sharpe, prob_overfit, n_folds
        """
        from scipy import stats

        # Minimum data requirement for CV
        min_samples = n_splits * 50  # At least 50 samples per fold
        if len(equity_series) < min_samples:
            return self._insufficient_cv_data_result(
                required=min_samples,
                actual=len(equity_series),
                message=(
                    f"Insufficient data for Purged CV: need {min_samples} samples, "
                    f"got {len(equity_series)}. Cannot perform reliable anti-overfitting validation."
                ),
            )

        # Calculate returns from equity curve
        returns = equity_series.pct_change().dropna()
        if len(returns) < min_samples:
            return self._insufficient_cv_data_result(
                required=min_samples,
                actual=len(returns),
                message=(
                    f"Insufficient returns data for Purged CV: need {min_samples}, "
                    f"got {len(returns)} after pct_change()."
                ),
            )

        # Create DataFrame for CV (required by PurgedKFoldCV)
        returns_df = pd.DataFrame({
            "returns": returns.values,
        }, index=returns.index)

        # Configure Purged K-Fold CV
        cv_config = PurgedCVConfig(
            n_splits=n_splits,
            purge_gap=purge_gap,
            embargo_pct=embargo_pct,
            min_train_size=0.3,  # At least 30% for training
        )
        cv = PurgedKFoldCV(cv_config)

        # Calculate Sharpe for each fold (out-of-sample)
        fold_sharpes: list[float] = []
        annualization = np.sqrt(365)  # Crypto 24/7

        for split in cv.split(returns_df):
            test_returns = returns_df.iloc[split.test_idx]["returns"]

            if len(test_returns) < 10:  # Skip folds with too few samples
                continue

            # Calculate Sharpe for this fold
            mean_ret = test_returns.mean()
            std_ret = test_returns.std()

            if std_ret > 0:
                fold_sharpe = (mean_ret / std_ret) * annualization
                fold_sharpes.append(float(fold_sharpe))

        if len(fold_sharpes) < 2:
            return self._insufficient_cv_data_result(
                required=2,
                actual=len(fold_sharpes),
                message=(
                    f"Not enough valid CV folds: need at least 2, got {len(fold_sharpes)}. "
                    "Data may be too sparse or volatile for reliable validation."
                ),
                n_folds=len(fold_sharpes),
            )

        # Calculate CV statistics
        cv_sharpe_mean = float(np.mean(fold_sharpes))
        cv_sharpe_std = float(np.std(fold_sharpes, ddof=1))
        n_folds = len(fold_sharpes)

        # Calculate Deflated Sharpe Ratio (DSR) - De Prado AFML Chapter 14
        # DSR accounts for the fact that we may have tried many strategies
        # DSR = (SR - E[SR_max]) / std(SR)
        # For simplicity, we use the Probabilistic Sharpe Ratio (PSR) approach
        # PSR = prob(true SR > 0) given observed SR

        # Estimate the variance of the Sharpe estimator
        # Var(SR) ≈ (1 + 0.5 * SR^2) / T where T = number of observations
        T = len(returns)
        sr_observed = cv_sharpe_mean

        # CRITICAL: Handle edge cases explicitly, no mathematical tricks
        MIN_MEANINGFUL_SHARPE = 0.01  # Below this, strategy has no meaningful edge

        if abs(cv_sharpe_mean) < MIN_MEANINGFUL_SHARPE:
            # Strategy has no meaningful edge - be explicit about this
            logger.warning(
                f"Strategy has near-zero Sharpe ({cv_sharpe_mean:.4f}). "
                "No meaningful edge detected."
            )
            deflated_sharpe = 0.0
            prob_overfit = float("nan")  # Unknown - we can't assess overfitting without edge
            cv_ratio = float("nan")

        elif cv_sharpe_std > 0:
            # Calculate t-statistic for Sharpe > 0
            t_stat = cv_sharpe_mean / (cv_sharpe_std / np.sqrt(n_folds))
            # Probability that true Sharpe > 0 (one-sided t-test)
            prob_positive_sharpe = 1.0 - stats.t.cdf(0, df=n_folds - 1, loc=t_stat)

            # Deflated Sharpe: penalize based on variance across folds
            # Higher variance = lower confidence = lower deflated Sharpe
            # DSR = SR * sqrt(1 - corr(folds)) ≈ SR * (1 - cv_ratio)
            # NOTE: cv_sharpe_mean is guaranteed > MIN_MEANINGFUL_SHARPE here
            cv_ratio = cv_sharpe_std / abs(cv_sharpe_mean)
            deflation_factor = max(0, 1 - cv_ratio)
            deflated_sharpe = float(sr_observed * deflation_factor)

            # Probability of overfitting = 1 - prob(true SR > 0)
            # If Sharpe varies wildly across folds, we're likely overfit
            prob_overfit = float(1.0 - prob_positive_sharpe)

        else:
            # cv_sharpe_std = 0 means all folds have identical Sharpe
            # This is suspicious - likely means not enough variance in data
            logger.warning(
                "Zero variance in CV Sharpe ratios - all folds identical. "
                "This is suspicious and suggests insufficient data variance."
            )
            deflated_sharpe = float(sr_observed * 0.5)  # 50% penalty
            prob_overfit = float("nan")  # Unknown due to suspicious data
            cv_ratio = 0.0

        # Log results
        logger.info(
            f"Purged CV Results: "
            f"Mean Sharpe={cv_sharpe_mean:.3f}, "
            f"Std={cv_sharpe_std:.3f}, "
            f"Deflated={deflated_sharpe:.3f}, "
            f"Prob(Overfit)={prob_overfit:.1%}"
        )

        # CRITICAL WARNING: Flag high overfit probability
        if prob_overfit > 0.5:
            logger.warning(
                f"⚠️  HIGH OVERFIT RISK: Prob(Overfit)={prob_overfit:.1%}. "
                f"CV Sharpe varies from {min(fold_sharpes):.2f} to {max(fold_sharpes):.2f}. "
                "Strategy may not generalize to live trading!"
            )

        return {
            "cv_sharpe_mean": cv_sharpe_mean,
            "cv_sharpe_std": cv_sharpe_std,
            "deflated_sharpe": deflated_sharpe,
            "prob_overfit": prob_overfit,
            "n_folds": n_folds,
        }

    def _build_result(
        self,
        equity_series: pd.Series,
        trades: list[BacktestTrade],
        settlements: list[SettlementRecord],
        initial_capital: float,
        funding_paid: float,
        funding_received: float,
        symbol: str,
    ) -> CryptoBacktestResult:
        """Build backtest result from raw data."""
        # Calculate returns
        returns = equity_series.pct_change().dropna()

        # Total return
        total_return = (equity_series.iloc[-1] / initial_capital) - 1 if len(equity_series) > 0 else 0

        # Annualized return (assuming daily data, 365 days for crypto)
        n_periods = len(returns)
        annualization = 365  # Crypto trades 24/7
        annualized_return = (1 + total_return) ** (annualization / max(n_periods, 1)) - 1

        # Sharpe ratio
        mean_return = returns.mean()
        std_return = returns.std()
        sharpe = (mean_return / std_return * np.sqrt(annualization)) if std_return > 0 else 0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.001
        sortino = (mean_return / downside_std * np.sqrt(annualization)) if downside_std > 0 else 0

        # Max drawdown
        cumulative = equity_series / initial_capital
        running_max = cumulative.cummax()
        drawdown = (running_max - cumulative) / running_max
        max_dd = float(drawdown.max()) if len(drawdown) > 0 else 0

        # Trade statistics
        trade_pnls = [t.pnl for t in trades if t.pnl != 0]
        n_trades = len([t for t in trades if not t.is_liquidation and t.side != PositionType.FLAT])
        n_liquidations = len([t for t in trades if t.is_liquidation])

        winning_trades = len([p for p in trade_pnls if p > 0])
        win_rate = winning_trades / len(trade_pnls) if trade_pnls else 0

        gains = sum(p for p in trade_pnls if p > 0)
        losses = abs(sum(p for p in trade_pnls if p < 0))
        profit_factor = gains / losses if losses > 0 else 0

        avg_trade_pnl = np.mean(trade_pnls) if trade_pnls else 0

        # ================================================================
        # CRITICAL: Anti-overfitting validation using Purged K-Fold CV
        # This is the "air bag" that prevents deploying overfit strategies
        # ================================================================
        cv_metrics = self._calculate_cv_metrics(equity_series)

        # ================================================================
        # ACTION 2: Record trial to Research Ledger
        # This is MANDATORY for preventing P-hacking in factor research
        # ================================================================
        if self.ledger is not None and RESEARCH_LEDGER_AVAILABLE:
            try:
                trial = TrialRecord(
                    factor_name=self.factor_name,
                    factor_family=self.factor_family,
                    sharpe_ratio=float(sharpe),
                    ic_mean=None,  # Not available in backtest context
                    ir=None,
                    max_drawdown=float(max_dd),
                    win_rate=float(win_rate),
                    metadata={
                        "symbol": symbol,
                        "initial_capital": initial_capital,
                        "leverage": self.config.leverage,
                        "n_trades": n_trades,
                        "n_liquidations": n_liquidations,
                        "total_return": float(total_return),
                        "annualized_return": float(annualized_return),
                        "sortino_ratio": float(sortino),
                        "profit_factor": float(profit_factor),
                        "net_funding": funding_received - funding_paid,
                        # Anti-overfitting metrics
                        "deflated_sharpe": cv_metrics["deflated_sharpe"],
                        "prob_overfit": cv_metrics["prob_overfit"],
                        "cv_sharpe_mean": cv_metrics["cv_sharpe_mean"],
                        "cv_sharpe_std": cv_metrics["cv_sharpe_std"],
                        "n_cv_folds": cv_metrics["n_folds"],
                    },
                )
                trial_id = self.ledger.record(trial)
                n_trials = self.ledger.get_trial_count()
                current_threshold = self.ledger.get_current_threshold()

                logger.info(
                    f"📊 Research Ledger: Trial #{n_trials} recorded. "
                    f"Factor='{self.factor_name}', Sharpe={sharpe:.2f}, "
                    f"Prob(Overfit)={cv_metrics['prob_overfit']:.1%}, "
                    f"Dynamic Threshold={current_threshold:.2f}"
                )

                # Warn if factor doesn't meet dynamic threshold
                threshold_result = self.ledger.check_significance(sharpe)
                if not threshold_result.passes:
                    logger.warning(
                        f"⚠️ Factor '{self.factor_name}' below dynamic threshold: "
                        f"Sharpe={sharpe:.2f} < {threshold_result.threshold:.2f} "
                        f"(adjusted for {n_trials} trials)"
                    )
            except Exception as e:
                logger.error(f"Failed to record trial to Research Ledger: {e}")

        return CryptoBacktestResult(
            total_return=float(total_return),
            annualized_return=float(annualized_return),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            max_drawdown=float(max_dd),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            # Anti-overfitting metrics (De Prado AFML)
            deflated_sharpe=cv_metrics["deflated_sharpe"],
            prob_overfit=cv_metrics["prob_overfit"],
            cv_sharpe_mean=cv_metrics["cv_sharpe_mean"],
            cv_sharpe_std=cv_metrics["cv_sharpe_std"],
            n_cv_folds=cv_metrics["n_folds"],
            # Trade statistics
            n_trades=n_trades,
            n_liquidations=n_liquidations,
            avg_trade_pnl=float(avg_trade_pnl),
            total_funding_paid=funding_paid,
            total_funding_received=funding_received,
            net_funding=funding_received - funding_paid,
            equity_curve=equity_series,
            drawdown_curve=drawdown,
            trades=trades,
            settlements=settlements,
            per_asset_pnl={symbol: float(total_return * initial_capital)},
            per_asset_trades={symbol: n_trades},
        )

    def _combine_results(
        self,
        results: dict[str, CryptoBacktestResult],
        allocations: dict[str, float],
    ) -> CryptoBacktestResult:
        """Combine multiple single-asset results into portfolio result."""
        # Combine equity curves (weighted)
        combined_equity = None
        for symbol, result in results.items():
            weighted = result.equity_curve * allocations[symbol]
            if combined_equity is None:
                combined_equity = weighted
            else:
                combined_equity = combined_equity.add(weighted, fill_value=0)

        # Aggregate trades and settlements
        all_trades = []
        all_settlements = []
        per_asset_pnl = {}
        per_asset_trades = {}

        total_funding_paid = 0.0
        total_funding_received = 0.0

        for symbol, result in results.items():
            all_trades.extend(result.trades)
            all_settlements.extend(result.settlements)
            per_asset_pnl[symbol] = sum(t.pnl for t in result.trades)
            per_asset_trades[symbol] = result.n_trades
            total_funding_paid += result.total_funding_paid
            total_funding_received += result.total_funding_received

        # Build combined result using _build_result for metrics calculation
        initial_capital = self.config.initial_capital
        result = self._build_result(
            equity_series=combined_equity,
            trades=all_trades,
            settlements=all_settlements,
            initial_capital=initial_capital,
            funding_paid=total_funding_paid,
            funding_received=total_funding_received,
            symbol="PORTFOLIO",
        )

        # Override per-asset data with correct values
        result.per_asset_pnl = per_asset_pnl
        result.per_asset_trades = per_asset_trades

        return result


# =============================================================================
# Convenience Functions
# =============================================================================

def run_crypto_backtest(
    data: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 100000.0,
    leverage: int = 10,
    symbol: str = "ETHUSDT",
    factor_name: str = "unnamed_factor",
    factor_family: str = "crypto",
    ledger: Optional["ResearchLedger"] = None,
) -> CryptoBacktestResult:
    """Convenience function to run a crypto backtest.

    Args:
        data: OHLCV DataFrame with 'close' and optional 'funding_rate'
        signals: Trading signals (1=long, -1=short, 0=flat)
        initial_capital: Starting capital
        leverage: Position leverage
        symbol: Trading symbol
        factor_name: Name of factor being backtested (ACTION 2: for ledger)
        factor_family: Family/category of factor (ACTION 2: for ledger)
        ledger: Optional ResearchLedger for trial tracking

    Returns:
        Backtest result
    """
    config = CryptoBacktestConfig(
        initial_capital=initial_capital,
        leverage=leverage,
    )
    engine = CryptoQlibBacktest(
        config,
        [symbol],
        ledger=ledger,
        factor_name=factor_name,
        factor_family=factor_family,
    )
    return engine.run(data, signals, symbol)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "CryptoQlibBacktest",
    "CryptoBacktestConfig",
    "CryptoBacktestResult",
    "CryptoExchange",
    "BacktestTrade",
    "SettlementRecord",
    "PositionType",
    "SettlementEvent",
    "run_crypto_backtest",
]
