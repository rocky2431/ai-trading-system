"""Position Management for trading strategies.

This module provides:
- Position: Represents a trading position
- PositionManager: Manages open/close positions
- PositionSizer: Kelly/Fixed/RiskParity sizing
- StopLoss: Price/Percent/Trailing/Time stops
- TakeProfit: Fixed/Trailing take profits
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import pandas as pd


class PositionSide(Enum):
    """Position side (long or short)."""

    LONG = "long"
    SHORT = "short"


class PositionStatus(Enum):
    """Position status."""

    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"


class InvalidPositionError(Exception):
    """Raised when position configuration is invalid."""

    pass


class InsufficientFundsError(Exception):
    """Raised when there are insufficient funds."""

    pass


@dataclass
class Position:
    """Represents a trading position."""

    symbol: str
    side: PositionSide
    entry_price: float
    quantity: float
    entry_time: Optional[datetime] = None
    status: PositionStatus = PositionStatus.OPEN
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set default entry time if not provided."""
        if self.entry_time is None:
            self.entry_time = datetime.now()

    def calculate_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L.

        Args:
            current_price: Current market price

        Returns:
            Unrealized profit/loss
        """
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            return (self.entry_price - current_price) * self.quantity

    def calculate_return_pct(self, current_price: float) -> float:
        """Calculate return percentage.

        Args:
            current_price: Current market price

        Returns:
            Return as decimal (0.1 = 10%)
        """
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) / self.entry_price
        else:  # SHORT
            return (self.entry_price - current_price) / self.entry_price

    def close(self, exit_price: float) -> Position:
        """Close the position.

        Args:
            exit_price: Exit price

        Returns:
            Self with updated status and P&L
        """
        self.exit_price = exit_price
        self.exit_time = datetime.now()
        self.pnl = self.calculate_pnl(exit_price)
        self.status = PositionStatus.CLOSED
        return self


# ============ Position Sizing ============


class PositionSizer(ABC):
    """Base class for position sizing strategies."""

    @abstractmethod
    def calculate(self, capital: float, **kwargs: Any) -> float:
        """Calculate position size.

        Args:
            capital: Available capital
            **kwargs: Strategy-specific parameters

        Returns:
            Position size in capital units
        """
        pass


class KellySizer(PositionSizer):
    """Kelly Criterion position sizing.

    Kelly formula: f = (p * b - q) / b
    where:
        p = win rate
        b = win/loss ratio
        q = 1 - p (loss rate)
    """

    def __init__(
        self,
        fraction: float = 1.0,
        max_position_pct: float = 1.0,
    ) -> None:
        """Initialize Kelly sizer.

        Args:
            fraction: Fraction of Kelly to use (e.g., 0.5 for half-Kelly)
            max_position_pct: Maximum position as fraction of capital
        """
        self.fraction = fraction
        self.max_position_pct = max_position_pct

    def calculate(
        self,
        capital: float,
        win_rate: float = 0.5,
        win_loss_ratio: float = 1.0,
        **kwargs: Any,
    ) -> float:
        """Calculate position size using Kelly criterion.

        Args:
            capital: Available capital
            win_rate: Historical win rate (0-1)
            win_loss_ratio: Average win / average loss

        Returns:
            Position size in capital units
        """
        # Kelly formula
        q = 1 - win_rate
        kelly = (win_rate * win_loss_ratio - q) / win_loss_ratio

        # Apply fraction
        kelly *= self.fraction

        # No negative positions
        if kelly <= 0:
            return 0.0

        # Apply max position limit
        kelly = min(kelly, self.max_position_pct)

        return capital * kelly


class FixedSizer(PositionSizer):
    """Fixed position sizing (percentage or amount)."""

    def __init__(
        self,
        position_pct: Optional[float] = None,
        fixed_amount: Optional[float] = None,
    ) -> None:
        """Initialize fixed sizer.

        Args:
            position_pct: Position as percentage of capital
            fixed_amount: Fixed position amount
        """
        self.position_pct = position_pct
        self.fixed_amount = fixed_amount

    def calculate(self, capital: float, **kwargs: Any) -> float:
        """Calculate position size.

        Args:
            capital: Available capital

        Returns:
            Position size in capital units
        """
        if self.fixed_amount is not None:
            return min(self.fixed_amount, capital)
        elif self.position_pct is not None:
            return capital * self.position_pct
        else:
            return capital * 0.1  # Default 10%


class RiskParitySizer(PositionSizer):
    """Risk parity position sizing.

    Size = target_risk / asset_volatility * capital
    """

    def __init__(self, target_risk: float = 0.02) -> None:
        """Initialize risk parity sizer.

        Args:
            target_risk: Target daily risk (e.g., 0.02 = 2%)
        """
        self.target_risk = target_risk

    def calculate(
        self,
        capital: float,
        asset_volatility: float = 0.2,
        **kwargs: Any,
    ) -> float:
        """Calculate position size for target risk.

        Args:
            capital: Available capital
            asset_volatility: Asset's volatility (annualized)

        Returns:
            Position size in capital units
        """
        if asset_volatility <= 0:
            return 0.0

        size_pct = self.target_risk / asset_volatility
        return capital * size_pct


# ============ Stop Loss ============


class StopLoss(ABC):
    """Base class for stop loss strategies."""

    @abstractmethod
    def should_trigger(
        self,
        position: Position,
        current_price: Optional[float] = None,
        current_time: Optional[datetime] = None,
    ) -> bool:
        """Check if stop loss should trigger.

        Args:
            position: The position to check
            current_price: Current market price
            current_time: Current time

        Returns:
            True if stop should trigger
        """
        pass


class PriceStopLoss(StopLoss):
    """Fixed price stop loss."""

    def __init__(self, stop_price: float) -> None:
        """Initialize with stop price.

        Args:
            stop_price: Price at which to stop out
        """
        self.stop_price = stop_price

    def should_trigger(
        self,
        position: Position,
        current_price: Optional[float] = None,
        current_time: Optional[datetime] = None,
    ) -> bool:
        """Check if price stop triggered."""
        if current_price is None:
            return False

        if position.side == PositionSide.LONG:
            return current_price <= self.stop_price
        else:  # SHORT
            return current_price >= self.stop_price


class PercentStopLoss(StopLoss):
    """Percentage-based stop loss."""

    def __init__(self, stop_pct: float) -> None:
        """Initialize with stop percentage.

        Args:
            stop_pct: Stop loss percentage (e.g., 0.05 = 5%)
        """
        self.stop_pct = stop_pct

    def should_trigger(
        self,
        position: Position,
        current_price: Optional[float] = None,
        current_time: Optional[datetime] = None,
    ) -> bool:
        """Check if percentage stop triggered."""
        if current_price is None:
            return False

        if position.side == PositionSide.LONG:
            stop_price = position.entry_price * (1 - self.stop_pct)
            return current_price <= stop_price
        else:  # SHORT
            stop_price = position.entry_price * (1 + self.stop_pct)
            return current_price >= stop_price


class TrailingStopLoss(StopLoss):
    """Trailing stop loss."""

    def __init__(self, trail_pct: float) -> None:
        """Initialize with trail percentage.

        Args:
            trail_pct: Trail percentage (e.g., 0.05 = 5%)
        """
        self.trail_pct = trail_pct
        self._high_water_mark: Optional[float] = None

    def update(self, position: Position, high_price: float) -> None:
        """Update high water mark.

        Args:
            position: The position
            high_price: New high price
        """
        if position.side == PositionSide.LONG:
            if self._high_water_mark is None:
                self._high_water_mark = high_price
            else:
                self._high_water_mark = max(self._high_water_mark, high_price)
        else:  # SHORT - track low water mark
            if self._high_water_mark is None:
                self._high_water_mark = high_price
            else:
                self._high_water_mark = min(self._high_water_mark, high_price)

    def should_trigger(
        self,
        position: Position,
        current_price: Optional[float] = None,
        current_time: Optional[datetime] = None,
    ) -> bool:
        """Check if trailing stop triggered."""
        if current_price is None or self._high_water_mark is None:
            return False

        if position.side == PositionSide.LONG:
            stop_price = self._high_water_mark * (1 - self.trail_pct)
            return current_price <= stop_price
        else:  # SHORT
            stop_price = self._high_water_mark * (1 + self.trail_pct)
            return current_price >= stop_price


class TimeStopLoss(StopLoss):
    """Time-based stop loss."""

    def __init__(self, max_hold_hours: float) -> None:
        """Initialize with maximum hold time.

        Args:
            max_hold_hours: Maximum hours to hold position
        """
        self.max_hold_hours = max_hold_hours

    def should_trigger(
        self,
        position: Position,
        current_price: Optional[float] = None,
        current_time: Optional[datetime] = None,
    ) -> bool:
        """Check if time stop triggered."""
        if current_time is None or position.entry_time is None:
            return False

        hold_duration = current_time - position.entry_time
        max_duration = timedelta(hours=self.max_hold_hours)
        return hold_duration >= max_duration


# ============ Take Profit ============


class TakeProfit(ABC):
    """Base class for take profit strategies."""

    @abstractmethod
    def should_trigger(
        self,
        position: Position,
        current_price: float,
    ) -> bool:
        """Check if take profit should trigger."""
        pass


class FixedTakeProfit(TakeProfit):
    """Fixed percentage take profit."""

    def __init__(self, profit_pct: float) -> None:
        """Initialize with profit percentage.

        Args:
            profit_pct: Take profit percentage (e.g., 0.1 = 10%)
        """
        self.profit_pct = profit_pct

    def should_trigger(
        self,
        position: Position,
        current_price: float,
    ) -> bool:
        """Check if fixed take profit triggered."""
        if position.side == PositionSide.LONG:
            target_price = position.entry_price * (1 + self.profit_pct)
            return current_price >= target_price
        else:  # SHORT
            target_price = position.entry_price * (1 - self.profit_pct)
            return current_price <= target_price


class TrailingTakeProfit(TakeProfit):
    """Trailing take profit.

    Activates at activation_pct profit, then trails by trail_pct.
    """

    def __init__(
        self,
        activation_pct: float,
        trail_pct: float,
    ) -> None:
        """Initialize trailing take profit.

        Args:
            activation_pct: Profit percentage to activate (e.g., 0.05 = 5%)
            trail_pct: Trail percentage after activation (e.g., 0.02 = 2%)
        """
        self.activation_pct = activation_pct
        self.trail_pct = trail_pct
        self.is_activated = False
        self._high_water_mark: Optional[float] = None

    def update(self, position: Position, high_price: float) -> None:
        """Update state with new high price.

        Args:
            position: The position
            high_price: New high price
        """
        # Check activation
        if position.side == PositionSide.LONG:
            activation_price = position.entry_price * (1 + self.activation_pct)
            if high_price >= activation_price:
                self.is_activated = True
        else:  # SHORT
            activation_price = position.entry_price * (1 - self.activation_pct)
            if high_price <= activation_price:
                self.is_activated = True

        # Update high water mark if activated
        if self.is_activated:
            if position.side == PositionSide.LONG:
                if self._high_water_mark is None:
                    self._high_water_mark = high_price
                else:
                    self._high_water_mark = max(self._high_water_mark, high_price)
            else:  # SHORT
                if self._high_water_mark is None:
                    self._high_water_mark = high_price
                else:
                    self._high_water_mark = min(self._high_water_mark, high_price)

    def should_trigger(
        self,
        position: Position,
        current_price: float,
    ) -> bool:
        """Check if trailing take profit triggered."""
        if not self.is_activated or self._high_water_mark is None:
            return False

        if position.side == PositionSide.LONG:
            trigger_price = self._high_water_mark * (1 - self.trail_pct)
            return current_price <= trigger_price
        else:  # SHORT
            trigger_price = self._high_water_mark * (1 + self.trail_pct)
            return current_price >= trigger_price


# ============ Position Manager ============


@dataclass
class PositionConfig:
    """Configuration for position manager."""

    initial_capital: float = 100000.0
    max_positions: int = 10
    max_position_size: float = 0.1  # Max 10% per position
    commission_rate: float = 0.001  # 0.1% commission


@dataclass
class CloseResult:
    """Result of closing a position."""

    symbol: str
    pnl: float
    status: PositionStatus
    position: Position


class PositionManager:
    """Manages trading positions."""

    def __init__(self, config: PositionConfig) -> None:
        """Initialize position manager.

        Args:
            config: Position manager configuration
        """
        self.config = config
        self.capital = config.initial_capital
        self.positions: dict[str, Position] = {}
        self.closed_positions: list[Position] = []
        self.stop_losses: list[StopLoss] = []
        self.take_profits: list[TakeProfit] = []

    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        price: float,
        quantity: float,
        **metadata: Any,
    ) -> Position:
        """Open a new position.

        Args:
            symbol: Trading symbol
            side: Long or short
            price: Entry price
            quantity: Position quantity
            **metadata: Additional metadata

        Returns:
            The opened position

        Raises:
            InvalidPositionError: If position is invalid
            InsufficientFundsError: If insufficient funds
        """
        # Validate inputs
        if quantity <= 0:
            raise InvalidPositionError("Quantity must be positive")
        if price <= 0:
            raise InvalidPositionError("Price must be positive")

        # Check max positions
        if len(self.positions) >= self.config.max_positions:
            raise InvalidPositionError(
                f"Maximum positions ({self.config.max_positions}) reached"
            )

        # Check funds
        cost = price * quantity
        if cost > self.capital:
            raise InsufficientFundsError(
                f"Insufficient funds: need {cost}, have {self.capital}"
            )

        # Check max position size
        max_cost = self.config.initial_capital * self.config.max_position_size
        if cost > max_cost:
            raise InvalidPositionError(
                f"Position size {cost} exceeds max {max_cost}"
            )

        # Create position
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=price,
            quantity=quantity,
            metadata=metadata,
        )

        # Deduct capital
        self.capital -= cost

        # Store position
        self.positions[symbol] = position

        return position

    def close_position(
        self,
        symbol: str,
        price: float,
    ) -> CloseResult:
        """Close an existing position.

        Args:
            symbol: Symbol to close
            price: Exit price

        Returns:
            Close result with P&L

        Raises:
            InvalidPositionError: If position not found
        """
        if symbol not in self.positions:
            raise InvalidPositionError(f"Position not found: {symbol}")

        position = self.positions[symbol]
        position.close(price)

        # Add back capital plus P&L
        cost = position.entry_price * position.quantity
        self.capital += cost + position.pnl

        # Move to closed
        self.closed_positions.append(position)
        del self.positions[symbol]

        return CloseResult(
            symbol=symbol,
            pnl=position.pnl,
            status=position.status,
            position=position,
        )

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position by symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position or None if not found
        """
        return self.positions.get(symbol)

    def add_stop_loss(self, stop_loss: StopLoss) -> None:
        """Add a stop loss strategy.

        Args:
            stop_loss: Stop loss to add
        """
        self.stop_losses.append(stop_loss)

    def add_take_profit(self, take_profit: TakeProfit) -> None:
        """Add a take profit strategy.

        Args:
            take_profit: Take profit to add
        """
        self.take_profits.append(take_profit)

    def check_stops(
        self,
        prices: dict[str, float],
        current_time: Optional[datetime] = None,
    ) -> list[CloseResult]:
        """Check and trigger stop losses.

        Args:
            prices: Current prices by symbol
            current_time: Current time (for time stops)

        Returns:
            List of closed positions
        """
        closed = []
        symbols_to_close = []

        for symbol, position in self.positions.items():
            price = prices.get(symbol)
            if price is None:
                continue

            for stop in self.stop_losses:
                if stop.should_trigger(position, price, current_time):
                    symbols_to_close.append((symbol, price))
                    break

        for symbol, price in symbols_to_close:
            result = self.close_position(symbol, price)
            closed.append(result)

        return closed

    def process_signals(
        self,
        signals: pd.DataFrame,
    ) -> list[Position]:
        """Process trading signals.

        Args:
            signals: DataFrame with columns: symbol, signal, price
                signal > 0: long, signal < 0: short, signal = 0: no action

        Returns:
            List of opened positions
        """
        opened = []

        for _, row in signals.iterrows():
            symbol = row["symbol"]
            signal = row["signal"]
            price = row["price"]

            if signal == 0:
                continue

            side = PositionSide.LONG if signal > 0 else PositionSide.SHORT

            # Calculate quantity (simple fixed fraction)
            position_value = self.capital * 0.01  # 1% per position
            quantity = position_value / price

            try:
                position = self.open_position(
                    symbol=symbol,
                    side=side,
                    price=price,
                    quantity=quantity,
                )
                opened.append(position)
            except (InvalidPositionError, InsufficientFundsError):
                continue

        return opened

    def get_portfolio_value(
        self,
        prices: dict[str, float],
    ) -> float:
        """Calculate total portfolio value.

        Args:
            prices: Current prices by symbol

        Returns:
            Total portfolio value
        """
        total = self.capital

        for symbol, position in self.positions.items():
            price = prices.get(symbol, position.entry_price)
            # Add position value
            position_value = position.entry_price * position.quantity
            unrealized_pnl = position.calculate_pnl(price)
            total += position_value + unrealized_pnl

        return total
