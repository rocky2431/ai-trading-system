"""Emergency close module for IQFMP.

Provides emergency position closing functionality:
- EmergencyCloseManager: Manage emergency close operations
- ConfirmationGate: Human confirmation for safety
- TelegramNotifier: Send notifications via Telegram
- CloseResult: Close operation results
"""

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Optional

from iqfmp.exchange.adapter import OrderSide, OrderStatus, OrderType
from iqfmp.exchange.execution import (
    ExecutionResult,
    OrderAction,
    OrderExecutor,
    OrderRequest,
)
from iqfmp.exchange.monitoring import PositionData, PositionSide, PositionTracker


# ==================== Enums ====================


class CloseStatus(Enum):
    """Close operation status."""

    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    PENDING = "pending"


class RequestStatus(Enum):
    """Close request status."""

    PENDING_CONFIRMATION = "pending_confirmation"
    CONFIRMED = "confirmed"
    EXECUTING = "executing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


# ==================== Data Models ====================


@dataclass
class CloseResult:
    """Result of a single position close operation."""

    symbol: str
    side: PositionSide
    quantity: Decimal
    entry_price: Decimal
    close_price: Optional[Decimal] = None
    expected_price: Optional[Decimal] = None
    status: CloseStatus = CloseStatus.PENDING
    pnl: Optional[Decimal] = None
    error: Optional[str] = None
    order_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def slippage_percent(self) -> Optional[float]:
        """Calculate slippage percentage."""
        if self.expected_price is None or self.close_price is None:
            return None
        if self.expected_price == Decimal("0"):
            return None
        slippage = (self.expected_price - self.close_price) / self.expected_price
        return float(slippage)


@dataclass
class ConfirmationToken:
    """Token for confirming emergency close request."""

    request_id: str
    token: str = field(default_factory=lambda: secrets.token_hex(16))
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    confirmed: bool = False
    cancelled: bool = False

    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    @property
    def is_valid_for_confirmation(self) -> bool:
        """Check if token can be used for confirmation."""
        return not self.confirmed and not self.cancelled and not self.is_expired


@dataclass
class CloseRequest:
    """Emergency close request."""

    request_id: str
    positions: list[PositionData]
    confirmation_token: str
    status: RequestStatus = RequestStatus.PENDING_CONFIRMATION
    created_at: datetime = field(default_factory=datetime.now)
    results: list[CloseResult] = field(default_factory=list)

    @property
    def total_notional_value(self) -> Decimal:
        """Calculate total notional value of all positions."""
        return sum(p.notional_value for p in self.positions)


# ==================== ConfirmationGate ====================


class ConfirmationGate:
    """Gate for human confirmation of emergency close."""

    def __init__(self, timeout_seconds: int = 300) -> None:
        """Initialize confirmation gate.

        Args:
            timeout_seconds: Token expiry time in seconds
        """
        self.timeout_seconds = timeout_seconds
        self._pending_confirmations: dict[str, ConfirmationToken] = {}
        self._tokens_by_value: dict[str, str] = {}  # token_value -> request_id

    def create_confirmation(self, request_id: str) -> ConfirmationToken:
        """Create a new confirmation token.

        Args:
            request_id: Request ID to associate with token

        Returns:
            New confirmation token
        """
        expires_at = datetime.now() + timedelta(seconds=self.timeout_seconds)
        token = ConfirmationToken(
            request_id=request_id,
            expires_at=expires_at,
        )
        self._pending_confirmations[request_id] = token
        self._tokens_by_value[token.token] = request_id
        return token

    def confirm(self, token_value: str) -> bool:
        """Confirm a close request using token.

        Args:
            token_value: Token value to confirm

        Returns:
            True if confirmation successful, False otherwise
        """
        request_id = self._tokens_by_value.get(token_value)
        if request_id is None:
            return False

        token = self._pending_confirmations.get(request_id)
        if token is None:
            return False

        if not token.is_valid_for_confirmation:
            return False

        token.confirmed = True
        return True

    def cancel(self, token_value: str) -> bool:
        """Cancel a close request.

        Args:
            token_value: Token value to cancel

        Returns:
            True if cancellation successful, False otherwise
        """
        request_id = self._tokens_by_value.get(token_value)
        if request_id is None:
            return False

        token = self._pending_confirmations.get(request_id)
        if token is None:
            return False

        token.cancelled = True
        return True

    def get_pending(self) -> list[ConfirmationToken]:
        """Get all pending confirmations.

        Returns:
            List of pending confirmation tokens
        """
        return [
            t for t in self._pending_confirmations.values()
            if t.is_valid_for_confirmation
        ]

    def cleanup_expired(self) -> int:
        """Remove expired confirmations.

        Returns:
            Number of expired confirmations removed
        """
        expired_ids = [
            rid for rid, token in self._pending_confirmations.items()
            if token.is_expired
        ]
        for rid in expired_ids:
            token = self._pending_confirmations.pop(rid, None)
            if token:
                self._tokens_by_value.pop(token.token, None)
        return len(expired_ids)


# ==================== TelegramNotifier ====================


class TelegramNotifier:
    """Send notifications via Telegram."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        """Initialize Telegram notifier.

        Args:
            bot_token: Telegram bot token
            chat_id: Chat ID to send messages to
            enabled: Whether notifications are enabled
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled and bot_token is not None

    async def _send_message(self, message: str) -> bool:
        """Send message to Telegram.

        Args:
            message: Message to send

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return True

        if not self.bot_token or not self.chat_id:
            return False

        try:
            import httpx

            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML",
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=data)
                if response.status_code == 200:
                    return True
                else:
                    # Log error but don't raise
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Telegram API error: {response.status_code} - {response.text}"
                    )
                    return False
        except ImportError:
            # httpx not installed, try with standard library
            import logging
            logging.getLogger(__name__).warning("httpx not installed for Telegram")
            return False
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to send Telegram message: {e}")
            return False

    def _format_close_request(
        self,
        request_id: str,
        positions: list[str],
        confirm_token: str,
    ) -> str:
        """Format close request notification message.

        Args:
            request_id: Request ID
            positions: List of position descriptions
            confirm_token: Confirmation token

        Returns:
            Formatted message
        """
        lines = [
            "🚨 <b>紧急平仓请求</b>",
            "",
            f"<b>请求ID</b>: {request_id}",
            "",
            "<b>待平仓持仓:</b>",
        ]
        for pos in positions:
            lines.append(f"  • {pos}")
        lines.extend([
            "",
            f"<b>确认令牌</b>: <code>{confirm_token}</code>",
            "",
            "回复 CONFIRM 确认平仓",
            "回复 CANCEL 取消请求",
        ])
        return "\n".join(lines)

    async def send_close_request(
        self,
        request_id: str,
        positions: list[str],
        confirm_token: str,
    ) -> bool:
        """Send close request notification.

        Args:
            request_id: Request ID
            positions: List of position descriptions
            confirm_token: Confirmation token

        Returns:
            True if sent successfully
        """
        message = self._format_close_request(request_id, positions, confirm_token)
        return await self._send_message(message)

    async def send_close_results(self, results: list[CloseResult]) -> bool:
        """Send close results notification.

        Args:
            results: List of close results

        Returns:
            True if sent successfully
        """
        success_count = sum(1 for r in results if r.status == CloseStatus.SUCCESS)
        failed_count = len(results) - success_count

        lines = ["📊 <b>平仓结果</b>", ""]

        if success_count > 0:
            lines.append(f"✅ 成功: {success_count}")
        if failed_count > 0:
            lines.append(f"❌ 失败: {failed_count}")

        lines.append("")

        total_pnl = Decimal("0")
        for r in results:
            status_icon = "✅" if r.status == CloseStatus.SUCCESS else "❌"
            lines.append(f"{status_icon} {r.symbol} {r.side.value}")
            if r.pnl is not None:
                total_pnl += r.pnl
                lines.append(f"   PnL: {r.pnl}")
            if r.error:
                lines.append(f"   错误: {r.error}")

        lines.extend(["", f"<b>总 PnL</b>: {total_pnl}"])

        return await self._send_message("\n".join(lines))

    async def send_error_alert(self, error: str, context: str) -> bool:
        """Send error alert notification.

        Args:
            error: Error message
            context: Error context

        Returns:
            True if sent successfully
        """
        message = f"⚠️ <b>错误告警</b>\n\n<b>上下文</b>: {context}\n<b>错误</b>: {error}"
        return await self._send_message(message)


# ==================== EmergencyCloseManager ====================


class EmergencyCloseManager:
    """Manage emergency close operations."""

    def __init__(
        self,
        executor: OrderExecutor,
        position_tracker: PositionTracker,
        require_confirmation: bool = True,
        confirmation_timeout: int = 300,
        notifier: Optional[TelegramNotifier] = None,
        max_retries: int = 3,
    ) -> None:
        """Initialize emergency close manager.

        Args:
            executor: Order executor
            position_tracker: Position tracker
            require_confirmation: Whether to require human confirmation
            confirmation_timeout: Confirmation timeout in seconds
            notifier: Optional Telegram notifier
            max_retries: Maximum retries for failed closes
        """
        self.executor = executor
        self.position_tracker = position_tracker
        self.require_confirmation = require_confirmation
        self.notifier = notifier
        self.max_retries = max_retries

        self._confirmation_gate = ConfirmationGate(timeout_seconds=confirmation_timeout)
        self._pending_requests: dict[str, CloseRequest] = {}
        self._request_counter = 0

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        self._request_counter += 1
        return f"close-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._request_counter}"

    async def request_close_all(self) -> Optional[CloseRequest]:
        """Request to close all positions.

        Returns:
            Close request if positions exist, None otherwise
        """
        positions = self.position_tracker.get_all_positions()
        if not positions:
            return None

        request_id = self._generate_request_id()
        token = self._confirmation_gate.create_confirmation(request_id)

        request = CloseRequest(
            request_id=request_id,
            positions=positions,
            confirmation_token=token.token,
            status=RequestStatus.PENDING_CONFIRMATION,
        )
        self._pending_requests[token.token] = request

        # Send notification if notifier is available
        if self.notifier and self.notifier.enabled:
            position_strs = [
                f"{p.symbol} {p.side.value.upper()} {p.quantity}"
                for p in positions
            ]
            await self.notifier.send_close_request(
                request_id=request_id,
                positions=position_strs,
                confirm_token=token.token,
            )

        return request

    async def confirm_and_execute(
        self,
        token_value: str,
    ) -> Optional[list[CloseResult]]:
        """Confirm and execute close request.

        Args:
            token_value: Confirmation token

        Returns:
            List of close results if confirmed, None otherwise
        """
        if not self._confirmation_gate.confirm(token_value):
            return None

        request = self._pending_requests.get(token_value)
        if request is None:
            return None

        request.status = RequestStatus.CONFIRMED
        results = await self.execute_close(request.positions)
        request.results = results
        request.status = RequestStatus.COMPLETED

        # Send results notification
        if self.notifier and self.notifier.enabled:
            await self.notifier.send_close_results(results)

        return results

    def cancel_request(self, token_value: str) -> bool:
        """Cancel a pending close request.

        Args:
            token_value: Confirmation token

        Returns:
            True if cancelled successfully
        """
        if not self._confirmation_gate.cancel(token_value):
            return False

        request = self._pending_requests.get(token_value)
        if request:
            request.status = RequestStatus.CANCELLED
        return True

    async def execute_close(
        self,
        positions: list[PositionData],
    ) -> list[CloseResult]:
        """Execute close for given positions.

        Args:
            positions: Positions to close

        Returns:
            List of close results
        """
        results = []

        for position in positions:
            result = await self._close_position_with_retry(position)
            results.append(result)

        return results

    async def _close_position_with_retry(
        self,
        position: PositionData,
    ) -> CloseResult:
        """Close a single position with retry logic.

        Args:
            position: Position to close

        Returns:
            Close result
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                result = await self._close_position(position)
                if result.status == CloseStatus.SUCCESS:
                    return result
                last_error = result.error
            except Exception as e:
                last_error = str(e)

        # All retries failed
        return CloseResult(
            symbol=position.symbol,
            side=position.side,
            quantity=position.quantity,
            entry_price=position.entry_price,
            expected_price=position.current_price,
            status=CloseStatus.FAILED,
            error=last_error,
        )

    async def _close_position(self, position: PositionData) -> CloseResult:
        """Close a single position.

        Args:
            position: Position to close

        Returns:
            Close result
        """
        # Determine order side based on position
        if position.side == PositionSide.LONG:
            order_side = OrderSide.SELL
            action = OrderAction.CLOSE_LONG
        else:
            order_side = OrderSide.BUY
            action = OrderAction.CLOSE_SHORT

        # Create order request
        order_request = OrderRequest(
            symbol=position.symbol,
            side=order_side,
            order_type=OrderType.MARKET,
            quantity=position.quantity,
            action=action,
            reduce_only=True,
        )

        try:
            execution_result: ExecutionResult = await self.executor.execute(
                order_request
            )

            if execution_result.success:
                # Calculate PnL
                pnl = None
                if execution_result.average_price is not None:
                    if position.side == PositionSide.LONG:
                        pnl = (
                            execution_result.average_price - position.entry_price
                        ) * position.quantity
                    else:
                        pnl = (
                            position.entry_price - execution_result.average_price
                        ) * position.quantity

                return CloseResult(
                    symbol=position.symbol,
                    side=position.side,
                    quantity=position.quantity,
                    entry_price=position.entry_price,
                    close_price=execution_result.average_price,
                    expected_price=position.current_price,
                    status=CloseStatus.SUCCESS,
                    pnl=pnl,
                    order_id=execution_result.order_id,
                )
            else:
                return CloseResult(
                    symbol=position.symbol,
                    side=position.side,
                    quantity=position.quantity,
                    entry_price=position.entry_price,
                    expected_price=position.current_price,
                    status=CloseStatus.FAILED,
                    error=execution_result.error,
                )
        except Exception as e:
            return CloseResult(
                symbol=position.symbol,
                side=position.side,
                quantity=position.quantity,
                entry_price=position.entry_price,
                expected_price=position.current_price,
                status=CloseStatus.FAILED,
                error=str(e),
            )

    async def close_all_immediately(self) -> list[CloseResult]:
        """Close all positions immediately without confirmation.

        Returns:
            List of close results
        """
        positions = self.position_tracker.get_all_positions()
        if not positions:
            return []

        results = await self.execute_close(positions)

        # Send results notification
        if self.notifier and self.notifier.enabled:
            await self.notifier.send_close_results(results)

        return results

    async def close_symbol(self, symbol: str) -> Optional[CloseResult]:
        """Close position for specific symbol.

        Args:
            symbol: Symbol to close

        Returns:
            Close result if position exists, None otherwise
        """
        position = self.position_tracker.get_position(symbol)
        if position is None:
            return None

        return await self._close_position_with_retry(position)
