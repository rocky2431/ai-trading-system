"""Margin Calculation Engine for Perpetual Futures.

Implements proper margin calculations for:
1. Initial margin - Required to open a position
2. Maintenance margin - Required to keep a position open
3. Liquidation price - Price at which position is forcibly closed
4. Margin ratio - Current margin health indicator

This is a P1 fix for IQFMP - addressing the lack of proper margin
calculation which could lead to unexpected liquidations.

Reference:
- Binance Futures margin documentation
- BitMEX margin mechanics
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from enum import Enum, auto
from typing import Optional


class PositionSide(Enum):
    """Position side for perpetual futures."""
    LONG = "long"
    SHORT = "short"


class MarginMode(Enum):
    """Margin mode."""
    ISOLATED = "isolated"
    CROSS = "cross"


class MarginStatus(Enum):
    """Margin health status."""
    HEALTHY = "healthy"  # Margin ratio < 50%
    WARNING = "warning"  # Margin ratio 50-70%
    DANGER = "danger"    # Margin ratio 70-90%
    CRITICAL = "critical"  # Margin ratio > 90%
    LIQUIDATION = "liquidation"  # At liquidation


@dataclass
class MarginConfig:
    """Configuration for margin calculations.

    Attributes:
        initial_margin_rate: Rate for initial margin (1/leverage)
        maintenance_margin_rate: Rate for maintenance margin
        taker_fee_rate: Taker fee rate for liquidation calculation
        maker_fee_rate: Maker fee rate
        funding_fee_buffer: Buffer for funding fees
    """
    initial_margin_rate: Decimal = Decimal("0.10")  # 10x leverage default
    maintenance_margin_rate: Decimal = Decimal("0.04")  # 4% maintenance
    taker_fee_rate: Decimal = Decimal("0.0004")  # 0.04% taker fee
    maker_fee_rate: Decimal = Decimal("0.0002")  # 0.02% maker fee
    funding_fee_buffer: Decimal = Decimal("0.001")  # 0.1% buffer

    @classmethod
    def from_leverage(cls, leverage: int) -> "MarginConfig":
        """Create config from leverage.

        Args:
            leverage: Leverage multiplier (e.g., 10 for 10x)

        Returns:
            MarginConfig with appropriate rates
        """
        if leverage <= 0:
            raise ValueError("Leverage must be positive")
        if leverage > 125:
            raise ValueError("Maximum leverage is 125x")

        initial_rate = Decimal("1") / Decimal(str(leverage))

        # Maintenance margin rate scales with leverage
        # Higher leverage = higher maintenance margin rate
        if leverage <= 5:
            maintenance_rate = Decimal("0.01")  # 1%
        elif leverage <= 10:
            maintenance_rate = Decimal("0.025")  # 2.5%
        elif leverage <= 20:
            maintenance_rate = Decimal("0.04")  # 4%
        elif leverage <= 50:
            maintenance_rate = Decimal("0.05")  # 5%
        else:
            maintenance_rate = Decimal("0.06")  # 6%

        return cls(
            initial_margin_rate=initial_rate,
            maintenance_margin_rate=maintenance_rate,
        )


@dataclass
class MarginPosition:
    """Position data for margin calculations.

    Attributes:
        symbol: Trading symbol
        side: Position side (LONG/SHORT)
        quantity: Position quantity (contracts)
        entry_price: Average entry price
        mark_price: Current mark price
        leverage: Position leverage
        mode: Margin mode (isolated/cross)
    """
    symbol: str
    side: PositionSide
    quantity: Decimal
    entry_price: Decimal
    mark_price: Decimal
    leverage: int = 10
    mode: MarginMode = MarginMode.ISOLATED

    @property
    def position_value(self) -> Decimal:
        """Calculate position value at mark price."""
        return self.quantity * self.mark_price

    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value at entry price."""
        return self.quantity * self.entry_price

    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate unrealized PnL."""
        price_diff = self.mark_price - self.entry_price
        if self.side == PositionSide.SHORT:
            price_diff = -price_diff
        return self.quantity * price_diff


@dataclass
class MarginResult:
    """Result of margin calculations.

    Attributes:
        initial_margin: Required initial margin
        maintenance_margin: Required maintenance margin
        margin_balance: Current margin balance
        margin_ratio: Current margin ratio (maintenance_margin / margin_balance)
        liquidation_price: Estimated liquidation price
        available_margin: Margin available for new positions
        status: Current margin health status
        timestamp: Calculation timestamp
    """
    initial_margin: Decimal
    maintenance_margin: Decimal
    margin_balance: Decimal
    margin_ratio: Decimal
    liquidation_price: Decimal
    available_margin: Decimal
    status: MarginStatus
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_safe(self) -> bool:
        """Check if margin is in safe zone."""
        return self.status in (MarginStatus.HEALTHY, MarginStatus.WARNING)

    @property
    def distance_to_liquidation(self) -> Decimal:
        """Calculate distance to liquidation as percentage."""
        return Decimal("1") - self.margin_ratio


class MarginCalculator:
    """Margin calculation engine for perpetual futures.

    Implements Binance-style margin calculations with proper handling of:
    - Initial margin (required to open position)
    - Maintenance margin (required to keep position)
    - Liquidation price (forced closure price)
    - Cross vs isolated margin modes

    Example:
        >>> config = MarginConfig.from_leverage(10)
        >>> calculator = MarginCalculator(config)
        >>>
        >>> position = MarginPosition(
        ...     symbol="ETHUSDT",
        ...     side=PositionSide.LONG,
        ...     quantity=Decimal("1.0"),
        ...     entry_price=Decimal("2000"),
        ...     mark_price=Decimal("2050"),
        ...     leverage=10,
        ... )
        >>>
        >>> result = calculator.calculate_margin(position, Decimal("250"))
        >>> print(f"Liquidation price: {result.liquidation_price}")
    """

    # Margin ratio thresholds
    WARNING_RATIO = Decimal("0.50")   # 50%
    DANGER_RATIO = Decimal("0.70")    # 70%
    CRITICAL_RATIO = Decimal("0.90")  # 90%

    def __init__(self, config: Optional[MarginConfig] = None):
        """Initialize margin calculator.

        Args:
            config: Margin configuration. Uses defaults if None.
        """
        self.config = config or MarginConfig()

    def calculate_initial_margin(
        self,
        position_size: Decimal,
        entry_price: Decimal,
        leverage: int,
    ) -> Decimal:
        """Calculate initial margin required to open position.

        Initial Margin = Notional Value / Leverage

        Args:
            position_size: Position size in contracts
            entry_price: Entry price
            leverage: Leverage multiplier

        Returns:
            Initial margin required
        """
        if leverage <= 0:
            raise ValueError("Leverage must be positive")

        notional = position_size * entry_price
        margin = notional / Decimal(str(leverage))

        return margin.quantize(Decimal("0.00000001"), rounding=ROUND_UP)

    def calculate_maintenance_margin(
        self,
        position_size: Decimal,
        mark_price: Decimal,
    ) -> Decimal:
        """Calculate maintenance margin required to keep position.

        Maintenance Margin = Notional Value * Maintenance Margin Rate

        Args:
            position_size: Position size in contracts
            mark_price: Current mark price

        Returns:
            Maintenance margin required
        """
        notional = position_size * mark_price
        margin = notional * self.config.maintenance_margin_rate

        return margin.quantize(Decimal("0.00000001"), rounding=ROUND_UP)

    def calculate_margin_ratio(
        self,
        margin_balance: Decimal,
        maintenance_margin: Decimal,
    ) -> Decimal:
        """Calculate margin ratio.

        Margin Ratio = Maintenance Margin / Margin Balance

        When margin ratio >= 100%, position is liquidated.

        Args:
            margin_balance: Current margin balance
            maintenance_margin: Required maintenance margin

        Returns:
            Margin ratio (0-1+, where 1 = liquidation)
        """
        if margin_balance <= Decimal("0"):
            return Decimal("1")  # At liquidation

        ratio = maintenance_margin / margin_balance
        return ratio.quantize(Decimal("0.0001"), rounding=ROUND_UP)

    def calculate_liquidation_price_long(
        self,
        entry_price: Decimal,
        leverage: int,
        maintenance_margin_rate: Optional[Decimal] = None,
    ) -> Decimal:
        """Calculate liquidation price for long position.

        For LONG positions (isolated margin):
        Liquidation Price = Entry Price * (1 - 1/Leverage + MMR)

        Where MMR = Maintenance Margin Rate

        Args:
            entry_price: Position entry price
            leverage: Position leverage
            maintenance_margin_rate: Override maintenance rate

        Returns:
            Liquidation price for long position
        """
        if leverage <= 0:
            raise ValueError("Leverage must be positive")

        mmr = maintenance_margin_rate or self.config.maintenance_margin_rate
        leverage_decimal = Decimal(str(leverage))

        # Liquidation Price = Entry * (1 - (1/Leverage) + MMR)
        # Simplified: when margin ratio hits 100%
        liq_factor = Decimal("1") - (Decimal("1") / leverage_decimal) + mmr
        liq_price = entry_price * liq_factor

        return liq_price.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

    def calculate_liquidation_price_short(
        self,
        entry_price: Decimal,
        leverage: int,
        maintenance_margin_rate: Optional[Decimal] = None,
    ) -> Decimal:
        """Calculate liquidation price for short position.

        For SHORT positions (isolated margin):
        Liquidation Price = Entry Price * (1 + 1/Leverage - MMR)

        Args:
            entry_price: Position entry price
            leverage: Position leverage
            maintenance_margin_rate: Override maintenance rate

        Returns:
            Liquidation price for short position
        """
        if leverage <= 0:
            raise ValueError("Leverage must be positive")

        mmr = maintenance_margin_rate or self.config.maintenance_margin_rate
        leverage_decimal = Decimal(str(leverage))

        # Liquidation Price = Entry * (1 + (1/Leverage) - MMR)
        liq_factor = Decimal("1") + (Decimal("1") / leverage_decimal) - mmr
        liq_price = entry_price * liq_factor

        return liq_price.quantize(Decimal("0.01"), rounding=ROUND_UP)

    def get_margin_status(
        self,
        margin_ratio: Decimal,
    ) -> MarginStatus:
        """Determine margin health status from ratio.

        Args:
            margin_ratio: Current margin ratio

        Returns:
            MarginStatus enum value
        """
        if margin_ratio >= Decimal("1"):
            return MarginStatus.LIQUIDATION
        elif margin_ratio >= self.CRITICAL_RATIO:
            return MarginStatus.CRITICAL
        elif margin_ratio >= self.DANGER_RATIO:
            return MarginStatus.DANGER
        elif margin_ratio >= self.WARNING_RATIO:
            return MarginStatus.WARNING
        else:
            return MarginStatus.HEALTHY

    def calculate_margin(
        self,
        position: MarginPosition,
        margin_balance: Decimal,
    ) -> MarginResult:
        """Calculate complete margin status for a position.

        Args:
            position: Position data
            margin_balance: Current margin balance

        Returns:
            MarginResult with all calculations
        """
        # Calculate margins
        initial_margin = self.calculate_initial_margin(
            position.quantity,
            position.entry_price,
            position.leverage,
        )

        maintenance_margin = self.calculate_maintenance_margin(
            position.quantity,
            position.mark_price,
        )

        # Calculate margin ratio
        margin_ratio = self.calculate_margin_ratio(
            margin_balance,
            maintenance_margin,
        )

        # Calculate liquidation price
        if position.side == PositionSide.LONG:
            liquidation_price = self.calculate_liquidation_price_long(
                position.entry_price,
                position.leverage,
            )
        else:
            liquidation_price = self.calculate_liquidation_price_short(
                position.entry_price,
                position.leverage,
            )

        # Calculate available margin
        available_margin = max(
            Decimal("0"),
            margin_balance - maintenance_margin,
        )

        # Get status
        status = self.get_margin_status(margin_ratio)

        return MarginResult(
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            margin_balance=margin_balance,
            margin_ratio=margin_ratio,
            liquidation_price=liquidation_price,
            available_margin=available_margin,
            status=status,
        )

    def check_margin_call(
        self,
        position: MarginPosition,
        margin_balance: Decimal,
    ) -> tuple[bool, str]:
        """Check if position is subject to margin call.

        Args:
            position: Position data
            margin_balance: Current margin balance

        Returns:
            (is_margin_call, message)
        """
        result = self.calculate_margin(position, margin_balance)

        if result.status == MarginStatus.LIQUIDATION:
            return True, f"Position at liquidation. Margin ratio: {result.margin_ratio * 100:.2f}%"
        elif result.status == MarginStatus.CRITICAL:
            return True, f"CRITICAL: Margin ratio {result.margin_ratio * 100:.2f}%. Add margin immediately."
        elif result.status == MarginStatus.DANGER:
            return True, f"DANGER: Margin ratio {result.margin_ratio * 100:.2f}%. Consider reducing position."
        elif result.status == MarginStatus.WARNING:
            return False, f"WARNING: Margin ratio {result.margin_ratio * 100:.2f}%. Monitor closely."
        else:
            return False, f"Healthy: Margin ratio {result.margin_ratio * 100:.2f}%"

    def calculate_max_position_size(
        self,
        available_margin: Decimal,
        entry_price: Decimal,
        leverage: int,
    ) -> Decimal:
        """Calculate maximum position size given available margin.

        Max Size = (Available Margin * Leverage) / Entry Price

        Args:
            available_margin: Available margin for new position
            entry_price: Entry price
            leverage: Desired leverage

        Returns:
            Maximum position size in contracts
        """
        if entry_price <= Decimal("0"):
            raise ValueError("Entry price must be positive")
        if leverage <= 0:
            raise ValueError("Leverage must be positive")

        max_notional = available_margin * Decimal(str(leverage))
        max_size = max_notional / entry_price

        return max_size.quantize(Decimal("0.001"), rounding=ROUND_DOWN)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "MarginMode",
    "MarginStatus",
    "PositionSide",
    # Data Classes
    "MarginConfig",
    "MarginPosition",
    "MarginResult",
    # Main Class
    "MarginCalculator",
]
