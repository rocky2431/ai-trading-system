"""Position and PnL monitoring module for IQFMP.

Provides real-time position and PnL monitoring:
- PositionTracker: Track positions across exchanges
- PnLCalculator: Calculate unrealized/realized PnL
- MarginMonitor: Monitor margin usage and alerts
- RealtimeUpdater: WebSocket-based real-time updates
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Optional
import uuid

from iqfmp.exchange.adapter import ExchangeAdapter


# ==================== Enums ====================


class PositionSide(Enum):
    """Position side."""

    LONG = "long"
    SHORT = "short"


class PnLType(Enum):
    """PnL type."""

    REALIZED = "realized"
    UNREALIZED = "unrealized"


class MarginLevel(Enum):
    """Margin level status."""

    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"


class UpdateType(Enum):
    """Update event type."""

    POSITION = "position"
    PNL = "pnl"
    MARGIN = "margin"
    BALANCE = "balance"


# ==================== Data Models ====================


@dataclass
class PositionData:
    """Position data model."""

    symbol: str
    side: PositionSide
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    leverage: int = 1
    maintenance_margin_rate: Decimal = Decimal("0.005")
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value."""
        return self.quantity * self.current_price

    @property
    def margin_used(self) -> Decimal:
        """Calculate margin used."""
        return self.notional_value / Decimal(self.leverage)

    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate unrealized PnL."""
        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity

    @property
    def unrealized_pnl_percent(self) -> Decimal:
        """Calculate unrealized PnL percentage."""
        if self.margin_used == Decimal("0"):
            return Decimal("0")
        return self.unrealized_pnl / self.margin_used

    @property
    def liquidation_price(self) -> Decimal:
        """Calculate liquidation price."""
        if self.side == PositionSide.LONG:
            # Long: liq_price = entry * (1 - 1/leverage + mmr)
            factor = Decimal("1") - Decimal("1") / Decimal(
                self.leverage
            ) + self.maintenance_margin_rate
            return self.entry_price * factor
        else:
            # Short: liq_price = entry * (1 + 1/leverage - mmr)
            factor = Decimal("1") + Decimal("1") / Decimal(
                self.leverage
            ) - self.maintenance_margin_rate
            return self.entry_price * factor


@dataclass
class PnLRecord:
    """PnL record."""

    symbol: str
    pnl_type: PnLType
    amount: Decimal
    timestamp: datetime
    entry_price: Optional[Decimal] = None
    exit_price: Optional[Decimal] = None
    quantity: Optional[Decimal] = None
    commission: Optional[Decimal] = None


@dataclass
class MarginAlert:
    """Margin alert."""

    level: MarginLevel
    message: str
    usage: Decimal
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class UpdateEvent:
    """Update event for subscribers."""

    type: UpdateType
    symbol: str
    data: dict[str, Any]
    timestamp: datetime


# ==================== PositionTracker ====================


class PositionTracker:
    """Track positions across exchanges."""

    def __init__(self, adapter: ExchangeAdapter) -> None:
        """Initialize position tracker.

        Args:
            adapter: Exchange adapter
        """
        self._adapter = adapter
        self._positions: dict[str, PositionData] = {}

    async def sync(self) -> None:
        """Sync positions from exchange."""
        raw_positions = await self._adapter.fetch_positions()

        # Clear existing positions
        self._positions.clear()

        # Parse and store positions
        for raw in raw_positions:
            position = self._parse_position(raw)
            if position.quantity > Decimal("0"):
                self._positions[position.symbol] = position

    def _parse_position(self, raw: dict[str, Any]) -> PositionData:
        """Parse raw position data.

        Args:
            raw: Raw position data from exchange

        Returns:
            Parsed position data
        """
        side_str = raw.get("side", "long").lower()
        side = PositionSide.LONG if side_str == "long" else PositionSide.SHORT

        return PositionData(
            symbol=raw.get("symbol", ""),
            side=side,
            quantity=Decimal(str(raw.get("contracts", 0))),
            entry_price=Decimal(str(raw.get("entryPrice", 0))),
            current_price=Decimal(str(raw.get("markPrice", 0))),
            leverage=int(raw.get("leverage", 1)),
        )

    def get_all_positions(self) -> list[PositionData]:
        """Get all positions.

        Returns:
            List of positions
        """
        return list(self._positions.values())

    def get_position(self, symbol: str) -> Optional[PositionData]:
        """Get position by symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position or None
        """
        return self._positions.get(symbol)

    def update_position(self, position: PositionData) -> None:
        """Update or add position.

        Args:
            position: Position to update
        """
        self._positions[position.symbol] = position

    def update_price(self, symbol: str, price: Decimal) -> None:
        """Update position current price.

        Args:
            symbol: Trading symbol
            price: New price
        """
        position = self._positions.get(symbol)
        if position:
            self._positions[symbol] = PositionData(
                symbol=position.symbol,
                side=position.side,
                quantity=position.quantity,
                entry_price=position.entry_price,
                current_price=price,
                leverage=position.leverage,
                maintenance_margin_rate=position.maintenance_margin_rate,
            )

    def close_position(self, symbol: str) -> None:
        """Close/remove position.

        Args:
            symbol: Trading symbol
        """
        self._positions.pop(symbol, None)

    @property
    def total_unrealized_pnl(self) -> Decimal:
        """Get total unrealized PnL."""
        return sum(
            (p.unrealized_pnl for p in self._positions.values()),
            Decimal("0"),
        )

    @property
    def total_margin_used(self) -> Decimal:
        """Get total margin used."""
        return sum(
            (p.margin_used for p in self._positions.values()),
            Decimal("0"),
        )


# ==================== PnLCalculator ====================


class PnLCalculator:
    """Calculate and track PnL."""

    def __init__(self) -> None:
        """Initialize PnL calculator."""
        self._history: list[PnLRecord] = []

    def calculate_unrealized(
        self,
        side: PositionSide,
        quantity: Decimal,
        entry_price: Decimal,
        current_price: Decimal,
    ) -> Decimal:
        """Calculate unrealized PnL.

        Args:
            side: Position side
            quantity: Position quantity
            entry_price: Entry price
            current_price: Current price

        Returns:
            Unrealized PnL
        """
        if side == PositionSide.LONG:
            return (current_price - entry_price) * quantity
        else:
            return (entry_price - current_price) * quantity

    def calculate_realized(
        self,
        side: PositionSide,
        quantity: Decimal,
        entry_price: Decimal,
        exit_price: Decimal,
        commission: Decimal = Decimal("0"),
    ) -> Decimal:
        """Calculate realized PnL.

        Args:
            side: Position side
            quantity: Position quantity
            entry_price: Entry price
            exit_price: Exit price
            commission: Commission paid

        Returns:
            Realized PnL
        """
        if side == PositionSide.LONG:
            gross_pnl = (exit_price - entry_price) * quantity
        else:
            gross_pnl = (entry_price - exit_price) * quantity

        return gross_pnl - commission

    def record(self, record: PnLRecord) -> None:
        """Record PnL.

        Args:
            record: PnL record
        """
        self._history.append(record)

    def get_history(
        self, symbol: Optional[str] = None, limit: int = 100
    ) -> list[PnLRecord]:
        """Get PnL history.

        Args:
            symbol: Filter by symbol (optional)
            limit: Maximum records

        Returns:
            List of PnL records
        """
        if symbol:
            filtered = [r for r in self._history if r.symbol == symbol]
        else:
            filtered = self._history

        return sorted(filtered, key=lambda r: r.timestamp, reverse=True)[
            :limit
        ]

    @property
    def total_realized_pnl(self) -> Decimal:
        """Get total realized PnL."""
        return sum(
            (
                r.amount
                for r in self._history
                if r.pnl_type == PnLType.REALIZED
            ),
            Decimal("0"),
        )

    def get_symbol_pnl(self, symbol: str) -> Decimal:
        """Get total PnL for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Total PnL for symbol
        """
        return sum(
            (
                r.amount
                for r in self._history
                if r.symbol == symbol and r.pnl_type == PnLType.REALIZED
            ),
            Decimal("0"),
        )


# ==================== MarginMonitor ====================


class MarginMonitor:
    """Monitor margin usage and generate alerts."""

    def __init__(
        self,
        adapter: ExchangeAdapter,
        currency: str = "USDT",
    ) -> None:
        """Initialize margin monitor.

        Args:
            adapter: Exchange adapter
            currency: Base currency for margin
        """
        self._adapter = adapter
        self._currency = currency
        self._total_balance = Decimal("0")
        self._available_balance = Decimal("0")
        self._used_margin = Decimal("0")

        # Default thresholds
        self._warning_threshold = Decimal("0.5")  # 50%
        self._danger_threshold = Decimal("0.7")  # 70%
        self._critical_threshold = Decimal("0.85")  # 85%

    async def update_balance(self) -> None:
        """Update balance from exchange."""
        balance = await self._adapter.fetch_balance()

        self._total_balance = Decimal(str(balance.total))
        self._available_balance = Decimal(str(balance.free))
        self._used_margin = Decimal(str(balance.used))

    @property
    def total_balance(self) -> Decimal:
        """Get total balance."""
        return self._total_balance

    @property
    def available_balance(self) -> Decimal:
        """Get available balance."""
        return self._available_balance

    @property
    def margin_usage(self) -> Decimal:
        """Calculate margin usage ratio."""
        if self._total_balance == Decimal("0"):
            return Decimal("0")
        return self._used_margin / self._total_balance

    @property
    def margin_level(self) -> MarginLevel:
        """Get current margin level."""
        usage = self.margin_usage

        if usage >= self._critical_threshold:
            return MarginLevel.CRITICAL
        elif usage >= self._danger_threshold:
            return MarginLevel.DANGER
        elif usage >= self._warning_threshold:
            return MarginLevel.WARNING
        else:
            return MarginLevel.SAFE

    def set_thresholds(
        self,
        warning: Decimal,
        danger: Decimal,
        critical: Decimal,
    ) -> None:
        """Set custom margin thresholds.

        Args:
            warning: Warning threshold (0-1)
            danger: Danger threshold (0-1)
            critical: Critical threshold (0-1)
        """
        self._warning_threshold = warning
        self._danger_threshold = danger
        self._critical_threshold = critical

    def check_alerts(self) -> list[MarginAlert]:
        """Check for margin alerts.

        Returns:
            List of margin alerts
        """
        alerts: list[MarginAlert] = []
        level = self.margin_level

        if level != MarginLevel.SAFE:
            message = self._get_alert_message(level)
            alerts.append(
                MarginAlert(
                    level=level,
                    message=message,
                    usage=self.margin_usage,
                )
            )

        return alerts

    def _get_alert_message(self, level: MarginLevel) -> str:
        """Get alert message for level.

        Args:
            level: Margin level

        Returns:
            Alert message
        """
        usage_pct = self.margin_usage * 100
        messages = {
            MarginLevel.WARNING: f"Margin usage at {usage_pct:.1f}% - consider reducing positions",
            MarginLevel.DANGER: f"Margin usage at {usage_pct:.1f}% - high liquidation risk",
            MarginLevel.CRITICAL: f"Margin usage at {usage_pct:.1f}% - imminent liquidation risk!",
        }
        return messages.get(level, "")


# ==================== RealtimeUpdater ====================


class RealtimeUpdater:
    """Handle real-time updates via WebSocket."""

    def __init__(self, adapter: ExchangeAdapter) -> None:
        """Initialize realtime updater.

        Args:
            adapter: Exchange adapter
        """
        self._adapter = adapter
        self._subscribers: dict[str, tuple[UpdateType, Callable]] = {}
        self._running = False

    def subscribe(
        self,
        update_type: UpdateType,
        callback: Callable[[UpdateEvent], None],
    ) -> str:
        """Subscribe to updates.

        Args:
            update_type: Type of updates to receive
            callback: Callback function

        Returns:
            Subscription ID
        """
        sub_id = str(uuid.uuid4())
        self._subscribers[sub_id] = (update_type, callback)
        return sub_id

    def unsubscribe(self, sub_id: str) -> None:
        """Unsubscribe from updates.

        Args:
            sub_id: Subscription ID
        """
        self._subscribers.pop(sub_id, None)

    def has_subscriber(self, update_type: UpdateType) -> bool:
        """Check if there are subscribers for update type.

        Args:
            update_type: Update type

        Returns:
            True if subscribers exist
        """
        return any(
            ut == update_type for ut, _ in self._subscribers.values()
        )

    def emit(self, event: UpdateEvent) -> None:
        """Emit update event to subscribers.

        Args:
            event: Update event
        """
        for sub_id, (update_type, callback) in self._subscribers.items():
            if update_type == event.type:
                try:
                    callback(event)
                except Exception:
                    pass  # Don't let subscriber errors crash the updater

    @property
    def is_running(self) -> bool:
        """Check if updater is running."""
        return self._running

    async def start(self) -> None:
        """Start WebSocket stream."""
        self._running = True
        # In production, this would start the WebSocket connection
        # and call _handle_message for each incoming message

    async def stop(self) -> None:
        """Stop WebSocket stream."""
        self._running = False

    async def _handle_message(self, message: dict[str, Any]) -> None:
        """Handle incoming WebSocket message.

        Args:
            message: Raw message
        """
        # Parse message and emit appropriate event
        event_type = message.get("type", "")

        if event_type == "position":
            self.emit(
                UpdateEvent(
                    type=UpdateType.POSITION,
                    symbol=message.get("symbol", ""),
                    data=message.get("data", {}),
                    timestamp=datetime.now(),
                )
            )
        elif event_type == "balance":
            self.emit(
                UpdateEvent(
                    type=UpdateType.BALANCE,
                    symbol="",
                    data=message.get("data", {}),
                    timestamp=datetime.now(),
                )
            )


# ==================== Module Exports ====================


__all__ = [
    # Enums
    "MarginLevel",
    "PnLType",
    "PositionSide",
    "UpdateType",
    # Models
    "MarginAlert",
    "PnLRecord",
    "PositionData",
    "UpdateEvent",
    # Classes
    "MarginMonitor",
    "PnLCalculator",
    "PositionTracker",
    "RealtimeUpdater",
]
