"""Position Management for trading strategies.

This module provides:
- Position: Represents a trading position
- PositionManager: Manages open/close positions with Redis persistence
- PositionSizer: Kelly/Fixed/RiskParity sizing
- StopLoss: Price/Percent/Trailing/Time stops
- TakeProfit: Fixed/Trailing take profits
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


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


class PositionStorageError(Exception):
    """Raised when position storage (Redis) is unavailable."""

    pass


def _to_decimal(value: float | Decimal | str | None) -> Decimal | None:
    """Convert value to Decimal for financial precision."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


@dataclass
class Position:
    """Represents a trading position.

    Per CLAUDE.md: All financial data uses Decimal, not float.
    """

    symbol: str
    side: PositionSide
    entry_price: Decimal
    quantity: Decimal
    entry_time: Optional[datetime] = None
    status: PositionStatus = PositionStatus.OPEN
    exit_price: Optional[Decimal] = None
    exit_time: Optional[datetime] = None
    pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Convert float inputs to Decimal and set default entry time."""
        # Convert float inputs to Decimal for financial precision
        self.entry_price = _to_decimal(self.entry_price) or Decimal("0")
        self.quantity = _to_decimal(self.quantity) or Decimal("0")
        self.exit_price = _to_decimal(self.exit_price)
        if not isinstance(self.pnl, Decimal):
            self.pnl = _to_decimal(self.pnl) or Decimal("0")

        if self.entry_time is None:
            self.entry_time = datetime.now()

    def calculate_pnl(self, current_price: float | Decimal) -> Decimal:
        """Calculate unrealized P&L.

        Args:
            current_price: Current market price

        Returns:
            Unrealized profit/loss as Decimal
        """
        price = _to_decimal(current_price) or Decimal("0")
        if self.side == PositionSide.LONG:
            return (price - self.entry_price) * self.quantity
        else:  # SHORT
            return (self.entry_price - price) * self.quantity

    def calculate_return_pct(self, current_price: float | Decimal) -> Decimal:
        """Calculate return percentage.

        Args:
            current_price: Current market price

        Returns:
            Return as Decimal (0.1 = 10%)
        """
        price = _to_decimal(current_price) or Decimal("0")
        if self.entry_price == Decimal("0"):
            return Decimal("0")
        if self.side == PositionSide.LONG:
            return (price - self.entry_price) / self.entry_price
        else:  # SHORT
            return (self.entry_price - price) / self.entry_price

    def close(self, exit_price: float | Decimal) -> Position:
        """Close the position.

        Args:
            exit_price: Exit price

        Returns:
            Self with updated status and P&L
        """
        self.exit_price = _to_decimal(exit_price)
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

    initial_capital: Decimal = field(default_factory=lambda: Decimal("100000"))
    max_positions: int = 10
    max_position_size: Decimal = field(default_factory=lambda: Decimal("0.1"))  # Max 10% per position
    commission_rate: Decimal = field(default_factory=lambda: Decimal("0.001"))  # 0.1% commission

    def __post_init__(self) -> None:
        """Convert float inputs to Decimal."""
        if not isinstance(self.initial_capital, Decimal):
            self.initial_capital = _to_decimal(self.initial_capital) or Decimal("100000")
        if not isinstance(self.max_position_size, Decimal):
            self.max_position_size = _to_decimal(self.max_position_size) or Decimal("0.1")
        if not isinstance(self.commission_rate, Decimal):
            self.commission_rate = _to_decimal(self.commission_rate) or Decimal("0.001")


@dataclass
class CloseResult:
    """Result of closing a position."""

    symbol: str
    pnl: Decimal
    status: PositionStatus
    position: Position


class PositionManagerError(Exception):
    """Raised when position manager operations fail."""

    pass


class PositionManager:
    """Manages trading positions with Redis persistence.

    Critical state per CLAUDE.md: Position data must be persistent to survive
    service restarts. Uses Redis for persistence.
    """

    REDIS_KEY_PREFIX = "iqfmp:positions:"
    REDIS_CLOSED_KEY = "iqfmp:closed_positions"
    REDIS_CAPITAL_KEY = "iqfmp:position_capital"

    def __init__(self, config: PositionConfig, redis_client: Any = None) -> None:
        """Initialize position manager with Redis persistence.

        Args:
            config: Position manager configuration
            redis_client: Optional Redis client for dependency injection (testing)
        """
        self.config = config
        self._redis = redis_client if redis_client is not None else self._get_redis_client()
        self.stop_losses: list[StopLoss] = []
        self.take_profits: list[TakeProfit] = []

        # Initialize capital from Redis or use config default
        self._init_capital()

    def _get_redis_client(self) -> Any:
        """Get Redis client. Raises PositionStorageError if unavailable.

        Per CLAUDE.md: Critical state must be persisted to PostgreSQL/Redis.
        Position data is critical state and requires persistent storage.
        """
        try:
            from iqfmp.db import get_redis_client
            client = get_redis_client()
            if client is None:
                raise PositionStorageError(
                    "Redis unavailable. Position data requires persistent storage "
                    "per CLAUDE.md critical state rules."
                )
            return client
        except PositionStorageError:
            raise
        except Exception as e:
            raise PositionStorageError(
                f"Failed to connect to Redis for position storage: {e}"
            ) from e

    def _init_capital(self) -> None:
        """Initialize capital from Redis or config default."""
        if self._redis:
            try:
                stored_capital = self._redis.get(self.REDIS_CAPITAL_KEY)
                if stored_capital:
                    self.capital = Decimal(stored_capital)
                    return
            except Exception as e:
                logger.warning(f"Failed to load capital from Redis: {e}")
        self.capital: Decimal = self.config.initial_capital

    def _save_capital(self) -> None:
        """Save capital to Redis."""
        if self._redis:
            try:
                self._redis.set(self.REDIS_CAPITAL_KEY, str(self.capital))
            except Exception as e:
                logger.warning(f"Failed to save capital to Redis: {e}")

    def _serialize_position(self, position: Position) -> str:
        """Serialize Position to JSON for Redis storage."""
        return json.dumps({
            "symbol": position.symbol,
            "side": position.side.value,
            "entry_price": str(position.entry_price),
            "quantity": str(position.quantity),
            "entry_time": position.entry_time.isoformat() if position.entry_time else None,
            "status": position.status.value,
            "exit_price": str(position.exit_price) if position.exit_price else None,
            "exit_time": position.exit_time.isoformat() if position.exit_time else None,
            "pnl": str(position.pnl),
            "metadata": position.metadata,
        })

    def _deserialize_position(self, data: str) -> Position:
        """Deserialize JSON to Position."""
        obj = json.loads(data)
        return Position(
            symbol=obj["symbol"],
            side=PositionSide(obj["side"]),
            entry_price=obj["entry_price"],
            quantity=obj["quantity"],
            entry_time=datetime.fromisoformat(obj["entry_time"]) if obj.get("entry_time") else None,
            status=PositionStatus(obj["status"]),
            exit_price=obj.get("exit_price"),
            exit_time=datetime.fromisoformat(obj["exit_time"]) if obj.get("exit_time") else None,
            pnl=obj.get("pnl", 0.0),
            metadata=obj.get("metadata", {}),
        )

    @property
    def positions(self) -> dict[str, Position]:
        """Get all open positions from Redis."""
        positions = {}
        if self._redis:
            try:
                cursor = 0
                while True:
                    cursor, keys = self._redis.scan(cursor, match=f"{self.REDIS_KEY_PREFIX}*", count=100)
                    for key in keys:
                        data = self._redis.get(key)
                        if data:
                            position = self._deserialize_position(data)
                            if position.status == PositionStatus.OPEN:
                                positions[position.symbol] = position
                    if cursor == 0:
                        break
            except Exception as e:
                logger.warning(f"Failed to load positions from Redis: {e}")
        return positions

    @property
    def closed_positions(self) -> list[Position]:
        """Get all closed positions from Redis."""
        closed = []
        if self._redis:
            try:
                data_list = self._redis.lrange(self.REDIS_CLOSED_KEY, 0, -1)
                for data in data_list:
                    closed.append(self._deserialize_position(data))
            except Exception as e:
                logger.warning(f"Failed to load closed positions from Redis: {e}")
        return closed

    def _save_position(self, position: Position) -> None:
        """Save position to Redis."""
        if self._redis:
            try:
                key = f"{self.REDIS_KEY_PREFIX}{position.symbol}"
                self._redis.set(key, self._serialize_position(position))
            except Exception as e:
                logger.warning(f"Failed to save position {position.symbol}: {e}")

    def _delete_position(self, symbol: str) -> None:
        """Delete position from Redis."""
        if self._redis:
            try:
                key = f"{self.REDIS_KEY_PREFIX}{symbol}"
                self._redis.delete(key)
            except Exception as e:
                logger.warning(f"Failed to delete position {symbol}: {e}")

    def _add_closed_position(self, position: Position) -> None:
        """Add closed position to Redis list."""
        if self._redis:
            try:
                self._redis.lpush(self.REDIS_CLOSED_KEY, self._serialize_position(position))
            except Exception as e:
                logger.warning(f"Failed to save closed position {position.symbol}: {e}")

    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        price: float | Decimal,
        quantity: float | Decimal,
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
        # Convert to Decimal
        price_dec = _to_decimal(price) or Decimal("0")
        quantity_dec = _to_decimal(quantity) or Decimal("0")

        # Validate inputs
        if quantity_dec <= Decimal("0"):
            raise InvalidPositionError("Quantity must be positive")
        if price_dec <= Decimal("0"):
            raise InvalidPositionError("Price must be positive")

        # Check max positions
        if len(self.positions) >= self.config.max_positions:
            raise InvalidPositionError(
                f"Maximum positions ({self.config.max_positions}) reached"
            )

        # Check funds
        cost = price_dec * quantity_dec
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
            entry_price=price_dec,
            quantity=quantity_dec,
            metadata=metadata,
        )

        # Deduct capital
        self.capital -= cost
        self._save_capital()

        # Store position to Redis
        self._save_position(position)
        logger.info(f"Opened {side.value} position for {symbol}: qty={quantity}, price={price}")

        return position

    def close_position(
        self,
        symbol: str,
        price: float | Decimal,
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
        self._save_capital()

        # Move to closed in Redis
        self._add_closed_position(position)
        self._delete_position(symbol)
        logger.info(f"Closed position for {symbol}: pnl={position.pnl}")

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
        prices: dict[str, float | Decimal],
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
            position_value = self.capital * Decimal("0.01")  # 1% per position
            price_dec = _to_decimal(price) or Decimal("1")
            quantity = position_value / price_dec

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
        prices: dict[str, float | Decimal],
    ) -> Decimal:
        """Calculate total portfolio value.

        Args:
            prices: Current prices by symbol

        Returns:
            Total portfolio value as Decimal
        """
        total = self.capital

        for symbol, position in self.positions.items():
            price = prices.get(symbol) or position.entry_price
            price_dec = _to_decimal(price) if not isinstance(price, Decimal) else price
            # Add position value
            position_value = position.entry_price * position.quantity
            unrealized_pnl = position.calculate_pnl(price_dec)
            total += position_value + unrealized_pnl

        return total
