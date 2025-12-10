"""Backtest Engine for strategy evaluation.

This module provides:
- BacktestEngine: Core backtesting engine
- PerformanceMetrics: Performance calculations
- BacktestResult: Backtest results container
- BacktestReport: Report generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd


class BacktestError(Exception):
    """Base exception for backtest errors."""

    pass


class InsufficientDataError(BacktestError):
    """Raised when there is insufficient data for backtesting."""

    pass


class TradeType(Enum):
    """Trade type (long or short)."""

    LONG = "long"
    SHORT = "short"


class TradeStatus(Enum):
    """Trade status."""

    OPEN = "open"
    CLOSED = "closed"


@dataclass
class Trade:
    """Represents a completed trade."""

    symbol: str
    trade_type: TradeType
    entry_price: float
    quantity: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    status: TradeStatus = TradeStatus.CLOSED
    commission: float = 0.0
    slippage: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pnl(self) -> float:
        """Calculate trade P&L."""
        if self.exit_price is None:
            return 0.0

        if self.trade_type == TradeType.LONG:
            return (self.exit_price - self.entry_price) * self.quantity
        else:  # SHORT
            return (self.entry_price - self.exit_price) * self.quantity

    @property
    def return_pct(self) -> float:
        """Calculate return percentage."""
        if self.exit_price is None:
            return 0.0

        if self.trade_type == TradeType.LONG:
            return (self.exit_price - self.entry_price) / self.entry_price
        else:  # SHORT
            return (self.entry_price - self.exit_price) / self.entry_price

    def is_winner(self) -> bool:
        """Check if trade is profitable."""
        return self.pnl > 0


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    avg_trade_return: float = 0.0
    trade_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "avg_trade_return": self.avg_trade_return,
            "trade_count": self.trade_count,
        }


class PerformanceCalculator:
    """Calculator for performance metrics."""

    def __init__(self, risk_free_rate: float = 0.0) -> None:
        """Initialize calculator.

        Args:
            risk_free_rate: Annual risk-free rate
        """
        self.risk_free_rate = risk_free_rate

    def calculate_sharpe(
        self,
        equity_curve: pd.Series,
        periods_per_year: int = 252,
    ) -> float:
        """Calculate Sharpe ratio.

        Args:
            equity_curve: Equity curve series
            periods_per_year: Trading periods per year

        Returns:
            Annualized Sharpe ratio
        """
        returns = equity_curve.pct_change().dropna()
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - self.risk_free_rate / periods_per_year
        sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()
        return float(sharpe)

    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown.

        Args:
            equity_curve: Equity curve series

        Returns:
            Maximum drawdown as decimal (0-1)
        """
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        return float(abs(drawdown.min()))

    def calculate_win_rate(self, trades: list[Trade]) -> float:
        """Calculate win rate.

        Args:
            trades: List of completed trades

        Returns:
            Win rate as decimal (0-1)
        """
        if len(trades) == 0:
            return 0.0

        winners = sum(1 for t in trades if t.is_winner())
        return winners / len(trades)

    def calculate_profit_factor(self, trades: list[Trade]) -> float:
        """Calculate profit factor.

        Args:
            trades: List of completed trades

        Returns:
            Profit factor (gross profit / gross loss)
        """
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def calculate_sortino(
        self,
        equity_curve: pd.Series,
        periods_per_year: int = 252,
    ) -> float:
        """Calculate Sortino ratio.

        Args:
            equity_curve: Equity curve series
            periods_per_year: Trading periods per year

        Returns:
            Annualized Sortino ratio
        """
        returns = equity_curve.pct_change().dropna()
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - self.risk_free_rate / periods_per_year
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float("inf") if excess_returns.mean() > 0 else 0.0

        sortino = np.sqrt(periods_per_year) * excess_returns.mean() / downside_returns.std()
        return float(sortino)

    def calculate_calmar(
        self,
        equity_curve: pd.Series,
        periods_per_year: int = 252,
    ) -> float:
        """Calculate Calmar ratio.

        Args:
            equity_curve: Equity curve series
            periods_per_year: Trading periods per year

        Returns:
            Calmar ratio (annualized return / max drawdown)
        """
        returns = equity_curve.pct_change().dropna()
        if len(returns) == 0:
            return 0.0

        annual_return = returns.mean() * periods_per_year
        max_dd = self.calculate_max_drawdown(equity_curve)

        if max_dd == 0:
            return float("inf") if annual_return > 0 else 0.0

        return float(annual_return / max_dd)

    def calculate_all(
        self,
        equity_curve: pd.Series,
        trades: list[Trade],
    ) -> PerformanceMetrics:
        """Calculate all metrics.

        Args:
            equity_curve: Equity curve series
            trades: List of completed trades

        Returns:
            PerformanceMetrics with all calculated values
        """
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

        return PerformanceMetrics(
            total_return=float(total_return),
            sharpe_ratio=self.calculate_sharpe(equity_curve),
            max_drawdown=self.calculate_max_drawdown(equity_curve),
            win_rate=self.calculate_win_rate(trades),
            profit_factor=self.calculate_profit_factor(trades),
            sortino_ratio=self.calculate_sortino(equity_curve),
            calmar_ratio=self.calculate_calmar(equity_curve),
            avg_trade_return=sum(t.return_pct for t in trades) / len(trades) if trades else 0.0,
            trade_count=len(trades),
        )


@dataclass
class BacktestConfig:
    """Configuration for backtest engine."""

    initial_capital: float = 100000.0
    commission: float = 0.001
    slippage: float = 0.0005
    position_size: float = 0.1  # 10% of capital per trade
    min_data_points: int = 5

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.initial_capital <= 0:
            raise BacktestError("Initial capital must be positive")
        if self.commission < 0:
            raise BacktestError("Commission cannot be negative")
        if self.slippage < 0:
            raise BacktestError("Slippage cannot be negative")


@dataclass
class BacktestResult:
    """Result of a backtest run."""

    equity_curve: pd.Series
    trades: list[Trade]
    initial_capital: float
    config: Optional[BacktestConfig] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _metrics: Optional[PerformanceMetrics] = field(default=None, repr=False)

    @property
    def final_equity(self) -> float:
        """Get final equity value."""
        return float(self.equity_curve.iloc[-1])

    @property
    def total_return(self) -> float:
        """Get total return percentage."""
        return (self.final_equity / self.initial_capital) - 1

    @property
    def trade_count(self) -> int:
        """Get number of trades."""
        return len(self.trades)

    def get_metrics(self) -> PerformanceMetrics:
        """Get performance metrics."""
        if self._metrics is None:
            calculator = PerformanceCalculator()
            self._metrics = calculator.calculate_all(self.equity_curve, self.trades)
        return self._metrics

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "final_equity": self.final_equity,
            "total_return": self.total_return,
            "trade_count": self.trade_count,
            "initial_capital": self.initial_capital,
            "metrics": self.get_metrics().to_dict(),
        }


class BacktestEngine:
    """Core backtesting engine."""

    def __init__(self, config: BacktestConfig) -> None:
        """Initialize backtest engine.

        Args:
            config: Backtest configuration
        """
        self.config = config
        self.calculator = PerformanceCalculator()

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
    ) -> BacktestResult:
        """Run backtest.

        Args:
            data: OHLCV data with DatetimeIndex
            signals: Trading signals (1=long, -1=short, 0=flat)

        Returns:
            BacktestResult with equity curve and trades

        Raises:
            BacktestError: If data or signals are invalid
            InsufficientDataError: If not enough data points
        """
        # Validate inputs
        if len(data) < self.config.min_data_points:
            raise InsufficientDataError(
                f"Need at least {self.config.min_data_points} data points"
            )

        if not data.index.equals(signals.index):
            raise BacktestError("Data and signals must have matching index")

        # Initialize state
        capital = self.config.initial_capital
        position = 0.0  # Current position size
        position_type: Optional[TradeType] = None
        entry_price = 0.0
        entry_time: Optional[datetime] = None

        trades: list[Trade] = []
        equity_values: list[float] = []

        for i, (timestamp, row) in enumerate(data.iterrows()):
            signal = signals.iloc[i]
            price = row["close"]

            # Calculate current equity
            if position != 0 and position_type is not None:
                if position_type == TradeType.LONG:
                    unrealized_pnl = (price - entry_price) * position
                else:
                    unrealized_pnl = (entry_price - price) * position
                current_equity = capital + unrealized_pnl
            else:
                current_equity = capital

            equity_values.append(current_equity)

            # Process signals
            if signal > 0 and position <= 0:
                # Close short if any
                if position < 0:
                    close_pnl = self._close_position(
                        entry_price, price, abs(position), TradeType.SHORT
                    )
                    capital += close_pnl
                    trades.append(Trade(
                        symbol="BACKTEST",
                        trade_type=TradeType.SHORT,
                        entry_price=entry_price,
                        exit_price=self._apply_slippage(price, TradeType.SHORT, False),
                        quantity=abs(position),
                        entry_time=entry_time or timestamp,
                        exit_time=timestamp,
                    ))
                    position = 0.0

                # Open long
                trade_value = capital * self.config.position_size
                entry_price = self._apply_slippage(price, TradeType.LONG, True)
                position = trade_value / entry_price
                position_type = TradeType.LONG
                entry_time = timestamp
                capital -= self._apply_commission(trade_value)

            elif signal < 0 and position >= 0:
                # Close long if any
                if position > 0:
                    close_pnl = self._close_position(
                        entry_price, price, position, TradeType.LONG
                    )
                    capital += close_pnl
                    trades.append(Trade(
                        symbol="BACKTEST",
                        trade_type=TradeType.LONG,
                        entry_price=entry_price,
                        exit_price=self._apply_slippage(price, TradeType.LONG, False),
                        quantity=position,
                        entry_time=entry_time or timestamp,
                        exit_time=timestamp,
                    ))
                    position = 0.0

                # Open short
                trade_value = capital * self.config.position_size
                entry_price = self._apply_slippage(price, TradeType.SHORT, True)
                position = trade_value / entry_price
                position_type = TradeType.SHORT
                entry_time = timestamp
                capital -= self._apply_commission(trade_value)

            elif signal == 0 and position != 0:
                # Close position
                if position_type == TradeType.LONG:
                    close_pnl = self._close_position(
                        entry_price, price, position, TradeType.LONG
                    )
                else:
                    close_pnl = self._close_position(
                        entry_price, price, abs(position), TradeType.SHORT
                    )

                capital += close_pnl
                trades.append(Trade(
                    symbol="BACKTEST",
                    trade_type=position_type or TradeType.LONG,
                    entry_price=entry_price,
                    exit_price=self._apply_slippage(
                        price, position_type or TradeType.LONG, False
                    ),
                    quantity=abs(position),
                    entry_time=entry_time or timestamp,
                    exit_time=timestamp,
                ))
                position = 0.0
                position_type = None

        # Close any remaining position at end of backtest
        if position != 0 and position_type is not None:
            final_price = data["close"].iloc[-1]
            final_timestamp = data.index[-1]

            if position_type == TradeType.LONG:
                close_pnl = self._close_position(
                    entry_price, final_price, position, TradeType.LONG
                )
            else:
                close_pnl = self._close_position(
                    entry_price, final_price, abs(position), TradeType.SHORT
                )

            capital += close_pnl
            trades.append(Trade(
                symbol="BACKTEST",
                trade_type=position_type,
                entry_price=entry_price,
                exit_price=self._apply_slippage(
                    final_price, position_type, False
                ),
                quantity=abs(position),
                entry_time=entry_time or final_timestamp,
                exit_time=final_timestamp,
            ))

            # Update final equity
            equity_values[-1] = capital

        # Create equity curve
        equity_curve = pd.Series(equity_values, index=data.index)

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            initial_capital=self.config.initial_capital,
            config=self.config,
        )

    def _apply_slippage(
        self,
        price: float,
        trade_type: TradeType,
        is_entry: bool,
    ) -> float:
        """Apply slippage to price.

        Args:
            price: Original price
            trade_type: Long or short
            is_entry: True if entering, False if exiting

        Returns:
            Adjusted price
        """
        slippage = price * self.config.slippage

        if trade_type == TradeType.LONG:
            return price + slippage if is_entry else price - slippage
        else:  # SHORT
            return price - slippage if is_entry else price + slippage

    def _apply_commission(self, trade_value: float) -> float:
        """Calculate commission.

        Args:
            trade_value: Trade value

        Returns:
            Commission amount
        """
        return trade_value * self.config.commission

    def _close_position(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
        trade_type: TradeType,
    ) -> float:
        """Calculate P&L for closing a position.

        Args:
            entry_price: Entry price
            exit_price: Exit price
            quantity: Position quantity
            trade_type: Long or short

        Returns:
            Net P&L after costs
        """
        if trade_type == TradeType.LONG:
            gross_pnl = (exit_price - entry_price) * quantity
        else:
            gross_pnl = (entry_price - exit_price) * quantity

        # Subtract commission
        trade_value = exit_price * quantity
        commission = self._apply_commission(trade_value)

        return gross_pnl - commission


@dataclass
class ReportConfig:
    """Configuration for backtest report."""

    include_trades: bool = True
    include_monthly: bool = True
    include_charts: bool = False


class BacktestReport:
    """Report generator for backtest results."""

    def __init__(
        self,
        result: BacktestResult,
        config: Optional[ReportConfig] = None,
    ) -> None:
        """Initialize report.

        Args:
            result: Backtest result
            config: Report configuration
        """
        self.result = result
        self.config = config or ReportConfig()
        self.calculator = PerformanceCalculator()

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics.

        Returns:
            Dictionary of summary statistics
        """
        metrics = self.result.get_metrics()

        return {
            "initial_capital": self.result.initial_capital,
            "final_equity": self.result.final_equity,
            "total_return": self.result.total_return,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "trade_count": self.result.trade_count,
        }

    def to_markdown(self) -> str:
        """Generate markdown report.

        Returns:
            Markdown string
        """
        summary = self.get_summary()
        metrics = self.result.get_metrics()

        md = "# Backtest Report\n\n"
        md += "## Summary\n\n"
        md += f"- Initial Capital: ${summary['initial_capital']:,.2f}\n"
        md += f"- Final Equity: ${summary['final_equity']:,.2f}\n"
        md += f"- Total Return: {summary['total_return']:.2%}\n\n"

        md += "## Performance Metrics\n\n"
        md += f"- Sharpe Ratio: {metrics.sharpe_ratio:.2f}\n"
        md += f"- Max Drawdown: {metrics.max_drawdown:.2%}\n"
        md += f"- Win Rate: {metrics.win_rate:.2%}\n"
        md += f"- Profit Factor: {metrics.profit_factor:.2f}\n"
        md += f"- Sortino Ratio: {metrics.sortino_ratio:.2f}\n"
        md += f"- Calmar Ratio: {metrics.calmar_ratio:.2f}\n\n"

        md += "## Trade Statistics\n\n"
        md += f"- Total Trades: {self.result.trade_count}\n"

        return md

    def get_monthly_returns(self) -> pd.Series:
        """Get monthly returns.

        Returns:
            Series of monthly returns
        """
        monthly = self.result.equity_curve.resample("M").last()
        returns = monthly.pct_change().dropna()
        return returns

    def get_trade_statistics(self) -> dict[str, Any]:
        """Get trade statistics.

        Returns:
            Dictionary of trade statistics
        """
        trades = self.result.trades

        if len(trades) == 0:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
            }

        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl < 0]

        return {
            "total_trades": len(trades),
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "avg_win": sum(t.pnl for t in winners) / len(winners) if winners else 0.0,
            "avg_loss": sum(t.pnl for t in losers) / len(losers) if losers else 0.0,
        }
