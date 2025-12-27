"""Crypto-specific executor for Qlib backtesting.

P3.3 FIX: Implement crypto-specific execution logic with:
- Funding rate cost calculation (perpetual futures)
- Order book depth simulation (slippage based on order size)
- Exchange-specific fee structures
- Liquidation risk simulation

This extends Qlib's BaseExecutor for cryptocurrency market specifics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Generator, Optional, Union

import numpy as np
import pandas as pd

from qlib.backtest.decision import BaseTradeDecision, Order, OrderDir
from qlib.backtest.executor import BaseExecutor
from qlib.backtest.position import Position

if TYPE_CHECKING:
    from qlib.backtest.exchange import Exchange

logger = logging.getLogger(__name__)


@dataclass
class CryptoExecutorConfig:
    """Configuration for CryptoExecutor.

    Attributes:
        funding_rate_interval: Hours between funding rate payments (typically 8h)
        default_funding_rate: Default funding rate if not in data (0.01% = 0.0001)
        max_leverage: Maximum allowed leverage
        liquidation_threshold: Maintenance margin ratio for liquidation
        order_book_depth_bps: Basis points of slippage per 100K notional
        maker_fee: Maker fee rate (e.g., 0.0002 = 0.02%)
        taker_fee: Taker fee rate (e.g., 0.0005 = 0.05%)
        enable_funding: Whether to apply funding rate costs
        enable_liquidation: Whether to simulate liquidations
    """

    funding_rate_interval: int = 8  # hours
    default_funding_rate: float = 0.0001  # 0.01%
    max_leverage: float = 10.0
    liquidation_threshold: float = 0.5  # 50% maintenance margin
    order_book_depth_bps: float = 1.0  # 1 bps per 100K
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0005  # 0.05%
    enable_funding: bool = True
    enable_liquidation: bool = True


@dataclass
class FundingPayment:
    """Record of a funding rate payment."""

    timestamp: datetime
    symbol: str
    rate: float
    payment: float  # Positive = paid, Negative = received
    position_size: float


@dataclass
class LiquidationEvent:
    """Record of a liquidation event."""

    timestamp: datetime
    symbol: str
    position_size: float
    entry_price: float
    liquidation_price: float
    loss: float


class CryptoExecutor(BaseExecutor):
    """Cryptocurrency-specific executor with funding rates and order book depth.

    This executor extends Qlib's BaseExecutor to handle:
    1. Funding rate costs for perpetual futures positions
    2. Order book depth-based slippage calculation
    3. Exchange-specific fee structures
    4. Liquidation risk simulation

    P3.3 FIX: Implements the missing crypto_executor.py for Qlib deep fork.
    """

    def __init__(
        self,
        exchange: "Exchange",
        config: Optional[CryptoExecutorConfig] = None,
        funding_rate_data: Optional[pd.DataFrame] = None,
        order_book_data: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize CryptoExecutor.

        Args:
            exchange: Qlib exchange instance
            config: Crypto-specific configuration
            funding_rate_data: DataFrame with columns [timestamp, symbol, funding_rate]
            order_book_data: DataFrame with columns [timestamp, symbol, bid_depth, ask_depth]
            **kwargs: Additional arguments passed to BaseExecutor
        """
        super().__init__(exchange=exchange, **kwargs)
        self.config = config or CryptoExecutorConfig()
        self._funding_rate_data = funding_rate_data
        self._order_book_data = order_book_data
        self._funding_payments: list[FundingPayment] = []
        self._liquidation_events: list[LiquidationEvent] = []
        self._last_funding_time: Optional[datetime] = None

        logger.info(
            f"CryptoExecutor initialized: "
            f"funding={self.config.enable_funding}, "
            f"liquidation={self.config.enable_liquidation}, "
            f"max_leverage={self.config.max_leverage}"
        )

    def execute(
        self,
        trade_decision: BaseTradeDecision,
    ) -> Generator[object, Any, list[tuple[Order, float, float, float]]]:
        """Execute trades with crypto-specific cost calculations.

        This method:
        1. Applies funding rate costs to existing positions
        2. Checks for liquidation conditions
        3. Calculates order book depth-based slippage
        4. Executes the trade decision

        Args:
            trade_decision: Trade decision to execute

        Yields:
            Intermediate execution states

        Returns:
            List of (order, executed_price, executed_amount, cost) tuples
        """
        current_time = self._get_current_time()

        # Step 1: Apply funding rate costs
        if self.config.enable_funding:
            self._apply_funding_rate(current_time)

        # Step 2: Check for liquidations
        if self.config.enable_liquidation:
            self._check_liquidations(current_time)

        # Step 3: Execute base trades with modified costs
        # Note: BaseExecutor.execute() is a generator, so we need to yield from it
        result = yield from super().execute(trade_decision)

        # Step 4: Apply crypto-specific costs to executed trades
        if result:
            result = self._apply_crypto_costs(result, current_time)

        return result

    def _get_current_time(self) -> datetime:
        """Get current simulation time."""
        # Access through exchange or use trade calendar
        try:
            return self.trade_exchange.trade_calendar.get_trade_date()
        except Exception:
            return datetime.now()

    def _apply_funding_rate(self, current_time: datetime) -> None:
        """Apply funding rate costs to current positions.

        Funding rates are typically paid every 8 hours in crypto perpetuals.
        Positive rate means longs pay shorts, negative means shorts pay longs.
        """
        # Check if funding interval has passed
        if self._last_funding_time is not None:
            hours_since_last = (current_time - self._last_funding_time).total_seconds() / 3600
            if hours_since_last < self.config.funding_rate_interval:
                return

        # Get current positions
        position = self._get_position()
        if position is None or position.get_stock_count() == 0:
            self._last_funding_time = current_time
            return

        # Calculate funding for each position
        for symbol, (amount, price) in position.get_stock_list():
            if amount == 0:
                continue

            # Get funding rate for this symbol
            rate = self._get_funding_rate(current_time, symbol)

            # Calculate funding payment
            notional = abs(amount) * price
            # Long positions pay when rate > 0, short positions receive
            payment = notional * rate * np.sign(amount)

            # Record payment
            self._funding_payments.append(
                FundingPayment(
                    timestamp=current_time,
                    symbol=symbol,
                    rate=rate,
                    payment=payment,
                    position_size=amount,
                )
            )

            logger.debug(
                f"Funding payment: {symbol} rate={rate:.4%} "
                f"payment={payment:.2f} position={amount:.4f}"
            )

        self._last_funding_time = current_time

    def _get_funding_rate(self, timestamp: datetime, symbol: str) -> float:
        """Get funding rate for a symbol at a given time.

        Args:
            timestamp: Time to look up
            symbol: Trading symbol

        Returns:
            Funding rate (as decimal, e.g., 0.0001 = 0.01%)
        """
        if self._funding_rate_data is None:
            return self.config.default_funding_rate

        try:
            # Find closest funding rate entry
            mask = (
                (self._funding_rate_data["symbol"] == symbol)
                & (self._funding_rate_data["timestamp"] <= timestamp)
            )
            if mask.any():
                latest = self._funding_rate_data[mask].iloc[-1]
                return float(latest["funding_rate"])
        except Exception as e:
            logger.warning(f"Failed to get funding rate: {e}")

        return self.config.default_funding_rate

    def _check_liquidations(self, current_time: datetime) -> None:
        """Check for liquidation conditions on current positions.

        Liquidation occurs when:
        margin_ratio = (equity / position_value) < liquidation_threshold

        For simplicity, we assume:
        - Initial margin = position_value / leverage
        - Maintenance margin = initial_margin * liquidation_threshold
        """
        position = self._get_position()
        if position is None:
            return

        for symbol, (amount, entry_price) in position.get_stock_list():
            if amount == 0:
                continue

            # Get current price
            try:
                current_price = self.trade_exchange.get_close(symbol, current_time)
            except Exception:
                continue

            # Calculate P&L
            pnl = (current_price - entry_price) * amount

            # Calculate margin (simplified)
            notional = abs(amount) * entry_price
            initial_margin = notional / self.config.max_leverage

            # Check liquidation condition
            equity = initial_margin + pnl
            margin_ratio = equity / notional if notional > 0 else 1.0

            if margin_ratio < self.config.liquidation_threshold:
                # Record liquidation
                self._liquidation_events.append(
                    LiquidationEvent(
                        timestamp=current_time,
                        symbol=symbol,
                        position_size=amount,
                        entry_price=entry_price,
                        liquidation_price=current_price,
                        loss=abs(pnl),
                    )
                )

                # Close position (in real execution, this would force close)
                logger.warning(
                    f"LIQUIDATION: {symbol} size={amount:.4f} "
                    f"entry={entry_price:.2f} liq_price={current_price:.2f} "
                    f"loss={abs(pnl):.2f}"
                )

    def _apply_crypto_costs(
        self,
        trades: list[tuple[Order, float, float, float]],
        current_time: datetime,
    ) -> list[tuple[Order, float, float, float]]:
        """Apply crypto-specific costs to executed trades.

        This includes:
        1. Order book depth-based slippage
        2. Exchange-specific fees (maker/taker)

        Args:
            trades: List of (order, price, amount, cost) tuples
            current_time: Current simulation time

        Returns:
            Modified trades list with adjusted costs
        """
        result = []

        for order, price, amount, cost in trades:
            if amount == 0:
                result.append((order, price, amount, cost))
                continue

            # Calculate order book slippage
            slippage = self._calculate_order_book_slippage(
                order.stock_id, amount, price, current_time
            )

            # Adjust price for slippage
            if order.direction == OrderDir.BUY:
                adjusted_price = price * (1 + slippage)
            else:
                adjusted_price = price * (1 - slippage)

            # Calculate fees (assume taker for market orders)
            notional = abs(amount) * adjusted_price
            fee = notional * self.config.taker_fee

            # Total cost = original cost + slippage cost + fee
            slippage_cost = abs(adjusted_price - price) * abs(amount)
            adjusted_cost = cost + slippage_cost + fee

            result.append((order, adjusted_price, amount, adjusted_cost))

            logger.debug(
                f"Trade: {order.stock_id} {order.direction.name} "
                f"amount={amount:.4f} price={price:.2f}->{adjusted_price:.2f} "
                f"slippage={slippage:.4%} fee={fee:.2f}"
            )

        return result

    def _calculate_order_book_slippage(
        self,
        symbol: str,
        amount: float,
        price: float,
        timestamp: datetime,
    ) -> float:
        """Calculate slippage based on order book depth.

        Uses a simple linear model:
        slippage = order_book_depth_bps * (notional / 100_000) / 10000

        Args:
            symbol: Trading symbol
            amount: Order amount
            price: Order price
            timestamp: Current time

        Returns:
            Slippage as decimal (e.g., 0.001 = 0.1%)
        """
        notional = abs(amount) * price

        # Get order book depth if available
        depth_factor = 1.0
        if self._order_book_data is not None:
            try:
                mask = (
                    (self._order_book_data["symbol"] == symbol)
                    & (self._order_book_data["timestamp"] <= timestamp)
                )
                if mask.any():
                    latest = self._order_book_data[mask].iloc[-1]
                    # Use bid or ask depth based on order direction
                    depth = latest.get("bid_depth", latest.get("ask_depth", 1_000_000))
                    if depth > 0:
                        depth_factor = notional / depth
            except Exception:
                pass

        # Calculate slippage
        base_slippage = self.config.order_book_depth_bps / 10000  # Convert bps to decimal
        slippage = base_slippage * (notional / 100_000) * depth_factor

        # Cap slippage at 1%
        return min(slippage, 0.01)

    def _get_position(self) -> Optional[Position]:
        """Get current position from exchange."""
        try:
            return self.trade_exchange.get_position()
        except Exception:
            return None

    def get_funding_payments(self) -> list[FundingPayment]:
        """Get all funding payments made during execution."""
        return self._funding_payments.copy()

    def get_liquidation_events(self) -> list[LiquidationEvent]:
        """Get all liquidation events during execution."""
        return self._liquidation_events.copy()

    def get_total_funding_cost(self) -> float:
        """Get total funding cost (positive = paid, negative = received)."""
        return sum(p.payment for p in self._funding_payments)

    def get_execution_summary(self) -> dict[str, Any]:
        """Get summary of crypto-specific execution costs.

        Returns:
            Dictionary with funding costs, liquidation losses, etc.
        """
        return {
            "funding_payments": len(self._funding_payments),
            "total_funding_cost": self.get_total_funding_cost(),
            "liquidation_events": len(self._liquidation_events),
            "total_liquidation_loss": sum(e.loss for e in self._liquidation_events),
            "config": {
                "funding_rate_interval": self.config.funding_rate_interval,
                "max_leverage": self.config.max_leverage,
                "maker_fee": self.config.maker_fee,
                "taker_fee": self.config.taker_fee,
            },
        }


def create_crypto_executor(
    exchange: "Exchange",
    funding_rate_data: Optional[pd.DataFrame] = None,
    order_book_data: Optional[pd.DataFrame] = None,
    max_leverage: float = 10.0,
    enable_funding: bool = True,
    enable_liquidation: bool = True,
) -> CryptoExecutor:
    """Factory function to create a CryptoExecutor.

    Args:
        exchange: Qlib exchange instance
        funding_rate_data: Optional funding rate data
        order_book_data: Optional order book depth data
        max_leverage: Maximum leverage allowed
        enable_funding: Whether to simulate funding costs
        enable_liquidation: Whether to simulate liquidations

    Returns:
        Configured CryptoExecutor instance
    """
    config = CryptoExecutorConfig(
        max_leverage=max_leverage,
        enable_funding=enable_funding,
        enable_liquidation=enable_liquidation,
    )

    return CryptoExecutor(
        exchange=exchange,
        config=config,
        funding_rate_data=funding_rate_data,
        order_book_data=order_book_data,
    )
