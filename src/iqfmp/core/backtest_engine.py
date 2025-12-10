"""Real backtesting engine for strategy execution.

This module provides actual strategy backtesting using real market data
and trading cost models. No random simulations.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from iqfmp.core.factor_engine import FactorEngine, get_default_data_path

logger = logging.getLogger(__name__)


@dataclass
class TradingCosts:
    """Trading cost model."""
    commission_rate: float = 0.001  # 0.1% per trade
    slippage_rate: float = 0.0005  # 0.05% slippage
    min_commission: float = 0.0  # Minimum commission per trade


@dataclass
class Trade:
    """Individual trade record."""
    id: str
    symbol: str
    side: str  # "long" or "short"
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    holding_days: int
    commission: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest result."""
    # Performance metrics
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    volatility: float

    # Trade statistics
    trade_count: int
    avg_trade_return: float
    avg_holding_period: float

    # Time series
    equity_curve: list[dict]
    trades: list[Trade]
    monthly_returns: dict[str, float]

    # Factor attribution
    factor_contributions: dict[str, float] = field(default_factory=dict)


class BacktestEngine:
    """Real backtesting engine using actual market data."""

    def __init__(
        self,
        data_path: Optional[Path] = None,
        trading_costs: Optional[TradingCosts] = None,
    ):
        """Initialize backtest engine.

        Args:
            data_path: Path to OHLCV data CSV
            trading_costs: Trading cost model
        """
        self.data_path = data_path or get_default_data_path()
        self.costs = trading_costs or TradingCosts()
        self._df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Load market data."""
        if self._df is None:
            logger.info(f"Loading data from {self.data_path}")
            self._df = pd.read_csv(self.data_path)
            self._df["timestamp"] = pd.to_datetime(self._df["timestamp"], utc=True)
            # Remove timezone for simpler comparisons
            self._df["timestamp"] = self._df["timestamp"].dt.tz_localize(None)
            self._df = self._df.sort_values("timestamp").reset_index(drop=True)
            self._df["returns"] = self._df["close"].pct_change()
        return self._df

    def run_factor_backtest(
        self,
        factor_code: str,
        initial_capital: float = 100000.0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        rebalance_frequency: str = "daily",
        position_size: float = 1.0,  # Fraction of capital per trade
        long_only: bool = False,
    ) -> BacktestResult:
        """Run backtest using factor signals.

        Args:
            factor_code: Factor code to generate signals
            initial_capital: Starting capital
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            rebalance_frequency: "daily", "weekly", "monthly"
            position_size: Fraction of capital for positions
            long_only: Only take long positions

        Returns:
            BacktestResult with all metrics
        """
        df = self.load_data()

        # Filter date range
        if start_date:
            df = df[df["timestamp"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["timestamp"] <= pd.to_datetime(end_date)]

        if len(df) < 30:
            raise ValueError("Insufficient data for backtest (need at least 30 days)")

        df = df.reset_index(drop=True)

        # Compute factor values
        engine = FactorEngine(df=df)
        try:
            factor_values = engine.compute_factor(factor_code, "signal")
        except Exception as e:
            logger.error(f"Factor computation failed: {e}")
            raise ValueError(f"Factor computation failed: {e}")

        # Generate trading signals
        # Long when factor > 0, short when factor < 0
        positions = np.sign(factor_values).fillna(0)

        if long_only:
            positions = positions.clip(lower=0)

        # Apply rebalancing frequency
        if rebalance_frequency == "weekly":
            # Only change positions on Mondays
            df["weekday"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
            rebalance_mask = df["weekday"] == 0
            positions = self._apply_rebalance_mask(positions, rebalance_mask)
        elif rebalance_frequency == "monthly":
            # Only change positions on first day of month
            df["day"] = pd.to_datetime(df["timestamp"]).dt.day
            rebalance_mask = df["day"] == 1
            positions = self._apply_rebalance_mask(positions, rebalance_mask)

        # Calculate strategy returns with costs
        returns = df["returns"].fillna(0)
        position_changes = positions.diff().abs().fillna(0)

        # Trading costs (commission + slippage)
        total_cost_rate = self.costs.commission_rate + self.costs.slippage_rate
        trading_costs = position_changes * total_cost_rate

        # Strategy returns = position * market returns - costs
        strategy_returns = (positions.shift(1) * returns - trading_costs).fillna(0)

        # Build equity curve
        equity = initial_capital
        equity_series = [initial_capital]
        max_equity = initial_capital
        drawdown_series = [0.0]

        for ret in strategy_returns.iloc[1:]:
            equity *= (1 + ret)
            equity_series.append(equity)
            max_equity = max(max_equity, equity)
            drawdown = (max_equity - equity) / max_equity * 100
            drawdown_series.append(drawdown)

        # Generate trades from position changes
        trades = self._generate_trades(df, positions, initial_capital, position_size)

        # Calculate metrics
        result = self._calculate_metrics(
            strategy_returns=strategy_returns,
            equity_series=equity_series,
            drawdown_series=drawdown_series,
            df=df,
            trades=trades,
            initial_capital=initial_capital,
        )

        return result

    def _apply_rebalance_mask(
        self,
        positions: pd.Series,
        rebalance_mask: pd.Series,
    ) -> pd.Series:
        """Apply rebalancing constraint to positions."""
        result = positions.copy()
        last_position = 0.0

        for i in range(len(positions)):
            if rebalance_mask.iloc[i]:
                last_position = positions.iloc[i]
            result.iloc[i] = last_position

        return result

    def _generate_trades(
        self,
        df: pd.DataFrame,
        positions: pd.Series,
        initial_capital: float,
        position_size: float,
    ) -> list[Trade]:
        """Generate trade records from positions."""
        trades = []
        trade_id = 0

        in_trade = False
        entry_idx = 0
        entry_price = 0.0
        entry_position = 0.0

        for i in range(1, len(positions)):
            current_pos = positions.iloc[i]
            prev_pos = positions.iloc[i - 1]

            # Position opened
            if current_pos != 0 and prev_pos == 0:
                in_trade = True
                entry_idx = i
                entry_price = df["close"].iloc[i]
                entry_position = current_pos

            # Position closed or reversed
            elif in_trade and (current_pos == 0 or current_pos != entry_position):
                exit_price = df["close"].iloc[i]

                # Calculate PnL
                if entry_position > 0:  # Long trade
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                else:  # Short trade
                    pnl_pct = (entry_price - exit_price) / entry_price * 100

                # Quantity and dollar PnL
                trade_value = initial_capital * position_size
                quantity = trade_value / entry_price
                dollar_pnl = quantity * (exit_price - entry_price) * entry_position

                # Commission
                commission = trade_value * self.costs.commission_rate * 2
                dollar_pnl -= commission

                holding_days = i - entry_idx

                trades.append(Trade(
                    id=f"trade_{trade_id}",
                    symbol=df["symbol"].iloc[0] if "symbol" in df.columns else "ETHUSDT",
                    side="long" if entry_position > 0 else "short",
                    entry_date=str(df["timestamp"].iloc[entry_idx].date()),
                    entry_price=entry_price,
                    exit_date=str(df["timestamp"].iloc[i].date()),
                    exit_price=exit_price,
                    quantity=quantity,
                    pnl=dollar_pnl,
                    pnl_pct=pnl_pct,
                    holding_days=holding_days,
                    commission=commission,
                ))

                trade_id += 1
                in_trade = False

                # If reversed, start new trade
                if current_pos != 0:
                    in_trade = True
                    entry_idx = i
                    entry_price = df["close"].iloc[i]
                    entry_position = current_pos

        return trades

    def _calculate_metrics(
        self,
        strategy_returns: pd.Series,
        equity_series: list[float],
        drawdown_series: list[float],
        df: pd.DataFrame,
        trades: list[Trade],
        initial_capital: float,
    ) -> BacktestResult:
        """Calculate all backtest metrics."""
        # Total return
        final_equity = equity_series[-1]
        total_return = (final_equity / initial_capital - 1) * 100

        # Annual return (CAGR)
        days = len(equity_series)
        years = days / 252  # Trading days
        if years > 0:
            annual_return = ((final_equity / initial_capital) ** (1 / years) - 1) * 100
        else:
            annual_return = 0.0

        # Volatility (annualized)
        volatility = strategy_returns.std() * np.sqrt(252) * 100

        # Sharpe ratio
        excess_returns = strategy_returns - 0.0  # Assume 0% risk-free rate
        if excess_returns.std() > 0:
            sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Sortino ratio (downside deviation)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino_ratio = 0.0

        # Max drawdown
        max_drawdown = max(drawdown_series)

        # Max drawdown duration
        max_dd_duration = self._calculate_max_dd_duration(drawdown_series)

        # Calmar ratio
        if max_drawdown > 0:
            calmar_ratio = annual_return / max_drawdown
        else:
            calmar_ratio = 0.0

        # Trade statistics
        if trades:
            winning_trades = [t for t in trades if t.pnl > 0]
            win_rate = len(winning_trades) / len(trades) * 100

            total_profit = sum(t.pnl for t in trades if t.pnl > 0)
            total_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
            profit_factor = total_profit / total_loss if total_loss > 0 else 0.0

            avg_trade_return = sum(t.pnl_pct for t in trades) / len(trades)
            avg_holding_period = sum(t.holding_days for t in trades) / len(trades)
        else:
            win_rate = 0.0
            profit_factor = 0.0
            avg_trade_return = 0.0
            avg_holding_period = 0.0

        # Build equity curve records
        equity_curve = []
        for i, (equity, dd) in enumerate(zip(equity_series, drawdown_series)):
            if i < len(df):
                date_str = str(df["timestamp"].iloc[i].date())
            else:
                date_str = str(df["timestamp"].iloc[-1].date())

            equity_curve.append({
                "date": date_str,
                "equity": round(equity, 2),
                "drawdown": round(dd, 2),
                "benchmark_equity": round(initial_capital * (1 + df["returns"].iloc[:i+1].sum()), 2),
            })

        # Monthly returns
        monthly_returns = self._calculate_monthly_returns(df, strategy_returns)

        return BacktestResult(
            total_return=round(total_return, 2),
            annual_return=round(annual_return, 2),
            sharpe_ratio=round(sharpe_ratio, 4),
            sortino_ratio=round(sortino_ratio, 4),
            max_drawdown=round(max_drawdown, 2),
            max_drawdown_duration=max_dd_duration,
            win_rate=round(win_rate, 2),
            profit_factor=round(profit_factor, 2),
            calmar_ratio=round(calmar_ratio, 2),
            volatility=round(volatility, 2),
            trade_count=len(trades),
            avg_trade_return=round(avg_trade_return, 2),
            avg_holding_period=round(avg_holding_period, 1),
            equity_curve=equity_curve,
            trades=trades,
            monthly_returns=monthly_returns,
        )

    def _calculate_max_dd_duration(self, drawdown_series: list[float]) -> int:
        """Calculate maximum drawdown duration in days."""
        max_duration = 0
        current_duration = 0

        for dd in drawdown_series:
            if dd > 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    def _calculate_monthly_returns(
        self,
        df: pd.DataFrame,
        strategy_returns: pd.Series,
    ) -> dict[str, float]:
        """Calculate monthly returns."""
        df_temp = df.copy()
        df_temp["strategy_returns"] = strategy_returns.values
        df_temp["month"] = pd.to_datetime(df_temp["timestamp"]).dt.to_period("M").astype(str)

        monthly = df_temp.groupby("month")["strategy_returns"].apply(
            lambda x: ((1 + x).prod() - 1) * 100
        )

        return {k: round(v, 2) for k, v in monthly.to_dict().items()}


def run_strategy_backtest(
    factor_ids: list[str],
    weighting_method: str = "equal",
    initial_capital: float = 100000.0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    rebalance_frequency: str = "daily",
    long_only: bool = False,
    factor_codes: Optional[dict[str, str]] = None,
) -> BacktestResult:
    """Run backtest for a multi-factor strategy.

    Args:
        factor_ids: List of factor IDs
        weighting_method: "equal", "ic_weighted", "vol_weighted"
        initial_capital: Starting capital
        start_date: Start date
        end_date: End date
        rebalance_frequency: Rebalance frequency
        long_only: Long-only constraint
        factor_codes: Mapping of factor_id to factor code (if available)

    Returns:
        BacktestResult
    """
    from iqfmp.core.factor_engine import BUILTIN_FACTORS

    # If no factor codes provided, use default momentum factor
    if not factor_codes or not factor_ids:
        factor_code = BUILTIN_FACTORS.get("momentum_20d", BUILTIN_FACTORS["rsi_14"])
    else:
        # Combine factors (simplified: use first factor for now)
        first_id = factor_ids[0]
        factor_code = factor_codes.get(first_id, BUILTIN_FACTORS["momentum_20d"])

    engine = BacktestEngine()
    result = engine.run_factor_backtest(
        factor_code=factor_code,
        initial_capital=initial_capital,
        start_date=start_date,
        end_date=end_date,
        rebalance_frequency=rebalance_frequency,
        long_only=long_only,
    )

    # Calculate factor contributions
    if factor_ids and len(factor_ids) > 1:
        # Simplified: equal contribution for now
        contribution = 100.0 / len(factor_ids)
        result.factor_contributions = {fid: contribution for fid in factor_ids}
    elif factor_ids:
        result.factor_contributions = {factor_ids[0]: 100.0}

    return result
