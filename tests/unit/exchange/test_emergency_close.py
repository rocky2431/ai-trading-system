"""Tests for emergency close module.

Tests cover:
- CloseResult model
- ConfirmationToken model
- ConfirmationGate
- TelegramNotifier (mocked)
- EmergencyCloseManager
- Integration scenarios
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iqfmp.exchange.adapter import (
    ExchangeAdapter,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from iqfmp.exchange.execution import (
    ExecutionResult,
    OrderAction,
    OrderExecutor,
    OrderRequest,
)
from iqfmp.exchange.monitoring import PositionData, PositionSide


# ==================== Model Tests ====================


class TestCloseResult:
    """Tests for CloseResult model."""

    def test_close_result_success(self):
        """Test successful close result creation."""
        from iqfmp.exchange.emergency import CloseResult, CloseStatus

        result = CloseResult(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            close_price=Decimal("51000"),
            status=CloseStatus.SUCCESS,
            pnl=Decimal("100"),
        )
        assert result.symbol == "BTC/USDT"
        assert result.status == CloseStatus.SUCCESS
        assert result.pnl == Decimal("100")
        assert result.error is None

    def test_close_result_failure(self):
        """Test failed close result."""
        from iqfmp.exchange.emergency import CloseResult, CloseStatus

        result = CloseResult(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            close_price=None,
            status=CloseStatus.FAILED,
            error="Insufficient liquidity",
        )
        assert result.status == CloseStatus.FAILED
        assert result.error == "Insufficient liquidity"
        assert result.close_price is None

    def test_close_result_slippage_calculation(self):
        """Test slippage calculation."""
        from iqfmp.exchange.emergency import CloseResult, CloseStatus

        result = CloseResult(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            close_price=Decimal("49900"),
            expected_price=Decimal("50000"),
            status=CloseStatus.SUCCESS,
        )
        # Slippage = (expected - actual) / expected = (50000 - 49900) / 50000 = 0.002
        assert result.slippage_percent == pytest.approx(0.002, rel=0.01)

    def test_close_result_no_slippage_when_no_expected(self):
        """Test no slippage when expected price not set."""
        from iqfmp.exchange.emergency import CloseResult, CloseStatus

        result = CloseResult(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            close_price=Decimal("49900"),
            status=CloseStatus.SUCCESS,
        )
        assert result.slippage_percent is None


class TestConfirmationToken:
    """Tests for ConfirmationToken model."""

    def test_token_creation(self):
        """Test token creation with default values."""
        from iqfmp.exchange.emergency import ConfirmationToken

        token = ConfirmationToken(request_id="req-123")
        assert token.request_id == "req-123"
        assert token.token is not None
        assert len(token.token) == 32
        assert token.confirmed is False
        assert token.cancelled is False

    def test_token_expiry(self):
        """Test token expiry check."""
        from iqfmp.exchange.emergency import ConfirmationToken

        # Non-expired token
        token = ConfirmationToken(
            request_id="req-123",
            expires_at=datetime.now() + timedelta(minutes=5),
        )
        assert token.is_expired is False

        # Expired token
        expired = ConfirmationToken(
            request_id="req-456",
            expires_at=datetime.now() - timedelta(minutes=1),
        )
        assert expired.is_expired is True

    def test_token_valid_for_confirmation(self):
        """Test token validity for confirmation."""
        from iqfmp.exchange.emergency import ConfirmationToken

        token = ConfirmationToken(
            request_id="req-123",
            expires_at=datetime.now() + timedelta(minutes=5),
        )
        assert token.is_valid_for_confirmation is True

        # Already confirmed
        token.confirmed = True
        assert token.is_valid_for_confirmation is False


# ==================== ConfirmationGate Tests ====================


class TestConfirmationGate:
    """Tests for ConfirmationGate."""

    def test_create_confirmation(self):
        """Test creating confirmation request."""
        from iqfmp.exchange.emergency import ConfirmationGate

        gate = ConfirmationGate(timeout_seconds=300)
        token = gate.create_confirmation(request_id="req-123")

        assert token.request_id == "req-123"
        assert token.token is not None
        assert "req-123" in gate._pending_confirmations

    def test_confirm_valid_token(self):
        """Test confirming with valid token."""
        from iqfmp.exchange.emergency import ConfirmationGate

        gate = ConfirmationGate(timeout_seconds=300)
        token = gate.create_confirmation(request_id="req-123")

        result = gate.confirm(token.token)
        assert result is True
        assert token.confirmed is True

    def test_confirm_invalid_token(self):
        """Test confirming with invalid token."""
        from iqfmp.exchange.emergency import ConfirmationGate

        gate = ConfirmationGate(timeout_seconds=300)
        result = gate.confirm("invalid-token")
        assert result is False

    def test_confirm_expired_token(self):
        """Test confirming expired token."""
        from iqfmp.exchange.emergency import ConfirmationGate

        gate = ConfirmationGate(timeout_seconds=0)  # Immediate expiry
        token = gate.create_confirmation(request_id="req-123")
        # Token should be expired immediately

        import time
        time.sleep(0.1)  # Wait for expiry

        result = gate.confirm(token.token)
        assert result is False

    def test_cancel_confirmation(self):
        """Test cancelling confirmation."""
        from iqfmp.exchange.emergency import ConfirmationGate

        gate = ConfirmationGate(timeout_seconds=300)
        token = gate.create_confirmation(request_id="req-123")

        result = gate.cancel(token.token)
        assert result is True
        assert token.cancelled is True

        # Cannot confirm after cancel
        confirm_result = gate.confirm(token.token)
        assert confirm_result is False

    def test_get_pending_confirmations(self):
        """Test getting pending confirmations."""
        from iqfmp.exchange.emergency import ConfirmationGate

        gate = ConfirmationGate(timeout_seconds=300)
        gate.create_confirmation(request_id="req-1")
        gate.create_confirmation(request_id="req-2")

        pending = gate.get_pending()
        assert len(pending) == 2

    def test_cleanup_expired(self):
        """Test cleanup of expired confirmations."""
        from iqfmp.exchange.emergency import ConfirmationGate

        gate = ConfirmationGate(timeout_seconds=0)
        gate.create_confirmation(request_id="req-1")
        gate.create_confirmation(request_id="req-2")

        import time
        time.sleep(0.1)

        gate.cleanup_expired()
        pending = gate.get_pending()
        assert len(pending) == 0


# ==================== TelegramNotifier Tests ====================


class TestTelegramNotifier:
    """Tests for TelegramNotifier."""

    def test_notifier_initialization(self):
        """Test notifier initialization."""
        from iqfmp.exchange.emergency import TelegramNotifier

        notifier = TelegramNotifier(
            bot_token="test-token",
            chat_id="123456",
        )
        assert notifier.bot_token == "test-token"
        assert notifier.chat_id == "123456"

    @pytest.mark.asyncio
    async def test_send_close_request_notification(self):
        """Test sending close request notification."""
        from iqfmp.exchange.emergency import TelegramNotifier

        notifier = TelegramNotifier(
            bot_token="test-token",
            chat_id="123456",
        )

        with patch.object(notifier, "_send_message", new_callable=AsyncMock) as mock:
            mock.return_value = True
            result = await notifier.send_close_request(
                request_id="req-123",
                positions=["BTC/USDT LONG 0.1", "ETH/USDT SHORT 1.0"],
                confirm_token="abc123",
            )
            assert result is True
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_close_result_notification(self):
        """Test sending close result notification."""
        from iqfmp.exchange.emergency import CloseResult, CloseStatus, TelegramNotifier

        notifier = TelegramNotifier(
            bot_token="test-token",
            chat_id="123456",
        )

        results = [
            CloseResult(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.1"),
                entry_price=Decimal("50000"),
                close_price=Decimal("51000"),
                status=CloseStatus.SUCCESS,
                pnl=Decimal("100"),
            )
        ]

        with patch.object(notifier, "_send_message", new_callable=AsyncMock) as mock:
            mock.return_value = True
            result = await notifier.send_close_results(results)
            assert result is True
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_error_alert(self):
        """Test sending error alert."""
        from iqfmp.exchange.emergency import TelegramNotifier

        notifier = TelegramNotifier(
            bot_token="test-token",
            chat_id="123456",
        )

        with patch.object(notifier, "_send_message", new_callable=AsyncMock) as mock:
            mock.return_value = True
            result = await notifier.send_error_alert(
                error="Connection timeout",
                context="Emergency close",
            )
            assert result is True
            mock.assert_called_once()

    def test_format_close_request_message(self):
        """Test formatting close request message."""
        from iqfmp.exchange.emergency import TelegramNotifier

        notifier = TelegramNotifier(
            bot_token="test-token",
            chat_id="123456",
        )

        message = notifier._format_close_request(
            request_id="req-123",
            positions=["BTC/USDT LONG 0.1"],
            confirm_token="abc123",
        )

        assert "紧急平仓请求" in message
        assert "req-123" in message
        assert "BTC/USDT" in message
        assert "abc123" in message

    def test_disabled_notifier(self):
        """Test notifier when disabled."""
        from iqfmp.exchange.emergency import TelegramNotifier

        notifier = TelegramNotifier(enabled=False)
        assert notifier.enabled is False


# ==================== EmergencyCloseManager Tests ====================


class TestEmergencyCloseManager:
    """Tests for EmergencyCloseManager."""

    @pytest.fixture
    def mock_executor(self):
        """Create mock order executor."""
        executor = MagicMock(spec=OrderExecutor)
        executor.execute = AsyncMock()
        return executor

    @pytest.fixture
    def mock_position_tracker(self):
        """Create mock position tracker."""
        from iqfmp.exchange.monitoring import PositionTracker

        tracker = MagicMock(spec=PositionTracker)
        return tracker

    def test_manager_initialization(self, mock_executor, mock_position_tracker):
        """Test manager initialization."""
        from iqfmp.exchange.emergency import EmergencyCloseManager

        manager = EmergencyCloseManager(
            executor=mock_executor,
            position_tracker=mock_position_tracker,
        )
        assert manager.executor == mock_executor
        assert manager.require_confirmation is True

    def test_manager_without_confirmation(self, mock_executor, mock_position_tracker):
        """Test manager without confirmation requirement."""
        from iqfmp.exchange.emergency import EmergencyCloseManager

        manager = EmergencyCloseManager(
            executor=mock_executor,
            position_tracker=mock_position_tracker,
            require_confirmation=False,
        )
        assert manager.require_confirmation is False

    @pytest.mark.asyncio
    async def test_request_close_all(self, mock_executor, mock_position_tracker):
        """Test requesting close all positions."""
        from iqfmp.exchange.emergency import CloseRequest, EmergencyCloseManager

        positions = [
            PositionData(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.1"),
                entry_price=Decimal("50000"),
                current_price=Decimal("51000"),
            ),
            PositionData(
                symbol="ETH/USDT",
                side=PositionSide.SHORT,
                quantity=Decimal("1.0"),
                entry_price=Decimal("3000"),
                current_price=Decimal("2900"),
            ),
        ]
        mock_position_tracker.get_all_positions = MagicMock(return_value=positions)

        manager = EmergencyCloseManager(
            executor=mock_executor,
            position_tracker=mock_position_tracker,
        )

        request = await manager.request_close_all()
        assert isinstance(request, CloseRequest)
        assert len(request.positions) == 2
        assert request.status.value == "pending_confirmation"

    @pytest.mark.asyncio
    async def test_request_close_empty(self, mock_executor, mock_position_tracker):
        """Test requesting close with no positions."""
        from iqfmp.exchange.emergency import EmergencyCloseManager

        mock_position_tracker.get_all_positions = MagicMock(return_value=[])

        manager = EmergencyCloseManager(
            executor=mock_executor,
            position_tracker=mock_position_tracker,
        )

        request = await manager.request_close_all()
        assert request is None

    @pytest.mark.asyncio
    async def test_execute_close(self, mock_executor, mock_position_tracker):
        """Test executing emergency close."""
        from iqfmp.exchange.emergency import CloseStatus, EmergencyCloseManager

        positions = [
            PositionData(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.1"),
                entry_price=Decimal("50000"),
                current_price=Decimal("51000"),
            ),
        ]

        mock_executor.execute.return_value = ExecutionResult(
            success=True,
            status=OrderStatus.CLOSED,
            order_id="order-123",
            filled_quantity=Decimal("0.1"),
            average_price=Decimal("50900"),
        )

        manager = EmergencyCloseManager(
            executor=mock_executor,
            position_tracker=mock_position_tracker,
            require_confirmation=False,
        )

        results = await manager.execute_close(positions)
        assert len(results) == 1
        assert results[0].status == CloseStatus.SUCCESS
        assert results[0].close_price == Decimal("50900")

    @pytest.mark.asyncio
    async def test_execute_close_with_failure(self, mock_executor, mock_position_tracker):
        """Test executing close with partial failure."""
        from iqfmp.exchange.emergency import CloseStatus, EmergencyCloseManager

        positions = [
            PositionData(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.1"),
                entry_price=Decimal("50000"),
                current_price=Decimal("51000"),
            ),
            PositionData(
                symbol="ETH/USDT",
                side=PositionSide.SHORT,
                quantity=Decimal("1.0"),
                entry_price=Decimal("3000"),
                current_price=Decimal("2900"),
            ),
        ]

        # First succeeds, second fails (provide enough values for retries)
        mock_executor.execute.side_effect = [
            ExecutionResult(
                success=True,
                status=OrderStatus.CLOSED,
                order_id="order-1",
                filled_quantity=Decimal("0.1"),
                average_price=Decimal("50900"),
            ),
            # ETH fails 3 times (max_retries=3)
            ExecutionResult(
                success=False,
                status=OrderStatus.REJECTED,
                error="Insufficient margin",
            ),
            ExecutionResult(
                success=False,
                status=OrderStatus.REJECTED,
                error="Insufficient margin",
            ),
            ExecutionResult(
                success=False,
                status=OrderStatus.REJECTED,
                error="Insufficient margin",
            ),
        ]

        manager = EmergencyCloseManager(
            executor=mock_executor,
            position_tracker=mock_position_tracker,
            require_confirmation=False,
        )

        results = await manager.execute_close(positions)
        assert len(results) == 2
        assert results[0].status == CloseStatus.SUCCESS
        assert results[1].status == CloseStatus.FAILED
        assert "Insufficient margin" in results[1].error

    @pytest.mark.asyncio
    async def test_execute_close_with_retry(self, mock_executor, mock_position_tracker):
        """Test close execution with retry on failure."""
        from iqfmp.exchange.emergency import CloseStatus, EmergencyCloseManager

        positions = [
            PositionData(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.1"),
                entry_price=Decimal("50000"),
                current_price=Decimal("51000"),
            ),
        ]

        # Fail first, succeed on retry
        mock_executor.execute.side_effect = [
            ExecutionResult(
                success=False,
                status=OrderStatus.REJECTED,
                error="Temporary error",
            ),
            ExecutionResult(
                success=True,
                status=OrderStatus.CLOSED,
                order_id="order-1",
                filled_quantity=Decimal("0.1"),
                average_price=Decimal("50900"),
            ),
        ]

        manager = EmergencyCloseManager(
            executor=mock_executor,
            position_tracker=mock_position_tracker,
            require_confirmation=False,
            max_retries=3,
        )

        results = await manager.execute_close(positions)
        assert len(results) == 1
        assert results[0].status == CloseStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_confirm_and_execute(self, mock_executor, mock_position_tracker):
        """Test confirming and executing close."""
        from iqfmp.exchange.emergency import CloseStatus, EmergencyCloseManager

        positions = [
            PositionData(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.1"),
                entry_price=Decimal("50000"),
                current_price=Decimal("51000"),
            ),
        ]
        mock_position_tracker.get_all_positions = MagicMock(return_value=positions)

        mock_executor.execute.return_value = ExecutionResult(
            success=True,
            status=OrderStatus.CLOSED,
            order_id="order-123",
            filled_quantity=Decimal("0.1"),
            average_price=Decimal("50900"),
        )

        manager = EmergencyCloseManager(
            executor=mock_executor,
            position_tracker=mock_position_tracker,
            require_confirmation=True,
        )

        # Create request
        request = await manager.request_close_all()

        # Confirm and execute
        results = await manager.confirm_and_execute(request.confirmation_token)
        assert len(results) == 1
        assert results[0].status == CloseStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_confirm_invalid_token(self, mock_executor, mock_position_tracker):
        """Test confirming with invalid token."""
        from iqfmp.exchange.emergency import EmergencyCloseManager

        manager = EmergencyCloseManager(
            executor=mock_executor,
            position_tracker=mock_position_tracker,
        )

        results = await manager.confirm_and_execute("invalid-token")
        assert results is None

    @pytest.mark.asyncio
    async def test_cancel_close_request(self, mock_executor, mock_position_tracker):
        """Test cancelling close request."""
        from iqfmp.exchange.emergency import EmergencyCloseManager, RequestStatus

        positions = [
            PositionData(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.1"),
                entry_price=Decimal("50000"),
                current_price=Decimal("51000"),
            ),
        ]
        mock_position_tracker.get_all_positions = MagicMock(return_value=positions)

        manager = EmergencyCloseManager(
            executor=mock_executor,
            position_tracker=mock_position_tracker,
        )

        request = await manager.request_close_all()
        cancelled = manager.cancel_request(request.confirmation_token)
        assert cancelled is True
        assert request.status == RequestStatus.CANCELLED


# ==================== CloseRequest Tests ====================


class TestCloseRequest:
    """Tests for CloseRequest model."""

    def test_close_request_creation(self):
        """Test close request creation."""
        from iqfmp.exchange.emergency import CloseRequest, RequestStatus

        positions = [
            PositionData(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.1"),
                entry_price=Decimal("50000"),
                current_price=Decimal("51000"),
            ),
        ]

        request = CloseRequest(
            request_id="req-123",
            positions=positions,
            confirmation_token="token-abc",
        )

        assert request.request_id == "req-123"
        assert len(request.positions) == 1
        assert request.status == RequestStatus.PENDING_CONFIRMATION

    def test_close_request_total_value(self):
        """Test calculating total position value."""
        from iqfmp.exchange.emergency import CloseRequest

        positions = [
            PositionData(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.1"),
                entry_price=Decimal("50000"),
                current_price=Decimal("50000"),
            ),
            PositionData(
                symbol="ETH/USDT",
                side=PositionSide.SHORT,
                quantity=Decimal("1.0"),
                entry_price=Decimal("3000"),
                current_price=Decimal("3000"),
            ),
        ]

        request = CloseRequest(
            request_id="req-123",
            positions=positions,
            confirmation_token="token-abc",
        )

        # Total = 0.1 * 50000 + 1.0 * 3000 = 5000 + 3000 = 8000
        assert request.total_notional_value == Decimal("8000")


# ==================== Integration Tests ====================


class TestEmergencyCloseIntegration:
    """Integration tests for emergency close system."""

    @pytest.mark.asyncio
    async def test_full_close_flow_without_confirmation(self):
        """Test full close flow without confirmation."""
        from iqfmp.exchange.emergency import CloseStatus, EmergencyCloseManager

        # Setup mocks
        executor = MagicMock(spec=OrderExecutor)
        executor.execute = AsyncMock(return_value=ExecutionResult(
            success=True,
            status=OrderStatus.CLOSED,
            order_id="order-123",
            filled_quantity=Decimal("0.1"),
            average_price=Decimal("50900"),
        ))

        from iqfmp.exchange.monitoring import PositionTracker
        tracker = MagicMock(spec=PositionTracker)
        tracker.get_all_positions = MagicMock(return_value=[
            PositionData(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.1"),
                entry_price=Decimal("50000"),
                current_price=Decimal("51000"),
            ),
        ])

        manager = EmergencyCloseManager(
            executor=executor,
            position_tracker=tracker,
            require_confirmation=False,
        )

        # Execute close directly
        results = await manager.close_all_immediately()
        assert len(results) == 1
        assert results[0].status == CloseStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_full_close_flow_with_confirmation(self):
        """Test full close flow with confirmation."""
        from iqfmp.exchange.emergency import CloseStatus, EmergencyCloseManager

        # Setup mocks
        executor = MagicMock(spec=OrderExecutor)
        executor.execute = AsyncMock(return_value=ExecutionResult(
            success=True,
            status=OrderStatus.CLOSED,
            order_id="order-123",
            filled_quantity=Decimal("0.1"),
            average_price=Decimal("50900"),
        ))

        from iqfmp.exchange.monitoring import PositionTracker
        tracker = MagicMock(spec=PositionTracker)
        tracker.get_all_positions = MagicMock(return_value=[
            PositionData(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.1"),
                entry_price=Decimal("50000"),
                current_price=Decimal("51000"),
            ),
        ])

        manager = EmergencyCloseManager(
            executor=executor,
            position_tracker=tracker,
            require_confirmation=True,
        )

        # Step 1: Request close
        request = await manager.request_close_all()
        assert request is not None

        # Step 2: Confirm and execute
        results = await manager.confirm_and_execute(request.confirmation_token)
        assert len(results) == 1
        assert results[0].status == CloseStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_close_with_notification(self):
        """Test close with Telegram notification."""
        from iqfmp.exchange.emergency import (
            CloseStatus,
            EmergencyCloseManager,
            TelegramNotifier,
        )

        # Setup mocks
        executor = MagicMock(spec=OrderExecutor)
        executor.execute = AsyncMock(return_value=ExecutionResult(
            success=True,
            status=OrderStatus.CLOSED,
            order_id="order-123",
            filled_quantity=Decimal("0.1"),
            average_price=Decimal("50900"),
        ))

        from iqfmp.exchange.monitoring import PositionTracker
        tracker = MagicMock(spec=PositionTracker)
        tracker.get_all_positions = MagicMock(return_value=[
            PositionData(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.1"),
                entry_price=Decimal("50000"),
                current_price=Decimal("51000"),
            ),
        ])

        notifier = MagicMock(spec=TelegramNotifier)
        notifier.send_close_request = AsyncMock(return_value=True)
        notifier.send_close_results = AsyncMock(return_value=True)
        notifier.enabled = True

        manager = EmergencyCloseManager(
            executor=executor,
            position_tracker=tracker,
            require_confirmation=False,
            notifier=notifier,
        )

        results = await manager.close_all_immediately()
        assert len(results) == 1
        notifier.send_close_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_specific_symbol(self):
        """Test closing specific symbol only."""
        from iqfmp.exchange.emergency import CloseStatus, EmergencyCloseManager

        executor = MagicMock(spec=OrderExecutor)
        executor.execute = AsyncMock(return_value=ExecutionResult(
            success=True,
            status=OrderStatus.CLOSED,
            order_id="order-123",
            filled_quantity=Decimal("0.1"),
            average_price=Decimal("50900"),
        ))

        from iqfmp.exchange.monitoring import PositionTracker
        tracker = MagicMock(spec=PositionTracker)
        tracker.get_position = MagicMock(return_value=PositionData(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
        ))

        manager = EmergencyCloseManager(
            executor=executor,
            position_tracker=tracker,
            require_confirmation=False,
        )

        result = await manager.close_symbol("BTC/USDT")
        assert result is not None
        assert result.status == CloseStatus.SUCCESS


# ==================== Boundary Tests ====================


class TestEmergencyCloseBoundary:
    """Boundary tests for emergency close."""

    @pytest.mark.asyncio
    async def test_close_very_small_position(self):
        """Test closing very small position."""
        from iqfmp.exchange.emergency import CloseStatus, EmergencyCloseManager

        executor = MagicMock(spec=OrderExecutor)
        executor.execute = AsyncMock(return_value=ExecutionResult(
            success=True,
            status=OrderStatus.CLOSED,
            order_id="order-123",
            filled_quantity=Decimal("0.00001"),
            average_price=Decimal("50900"),
        ))

        from iqfmp.exchange.monitoring import PositionTracker
        tracker = MagicMock(spec=PositionTracker)
        tracker.get_all_positions = MagicMock(return_value=[
            PositionData(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.00001"),  # Minimum quantity
                entry_price=Decimal("50000"),
                current_price=Decimal("51000"),
            ),
        ])

        manager = EmergencyCloseManager(
            executor=executor,
            position_tracker=tracker,
            require_confirmation=False,
        )

        results = await manager.close_all_immediately()
        assert len(results) == 1
        assert results[0].status == CloseStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_close_many_positions(self):
        """Test closing many positions at once."""
        from iqfmp.exchange.emergency import CloseStatus, EmergencyCloseManager

        executor = MagicMock(spec=OrderExecutor)
        executor.execute = AsyncMock(return_value=ExecutionResult(
            success=True,
            status=OrderStatus.CLOSED,
            order_id="order-123",
            filled_quantity=Decimal("0.1"),
            average_price=Decimal("50900"),
        ))

        from iqfmp.exchange.monitoring import PositionTracker
        tracker = MagicMock(spec=PositionTracker)

        # Create 20 positions
        positions = [
            PositionData(
                symbol=f"COIN{i}/USDT",
                side=PositionSide.LONG if i % 2 == 0 else PositionSide.SHORT,
                quantity=Decimal("0.1"),
                entry_price=Decimal("100"),
                current_price=Decimal("101"),
            )
            for i in range(20)
        ]
        tracker.get_all_positions = MagicMock(return_value=positions)

        manager = EmergencyCloseManager(
            executor=executor,
            position_tracker=tracker,
            require_confirmation=False,
        )

        results = await manager.close_all_immediately()
        assert len(results) == 20
        assert all(r.status == CloseStatus.SUCCESS for r in results)


# ==================== Exception Tests ====================


class TestEmergencyCloseExceptions:
    """Exception handling tests for emergency close."""

    @pytest.mark.asyncio
    async def test_executor_exception(self):
        """Test handling executor exception."""
        from iqfmp.exchange.emergency import CloseStatus, EmergencyCloseManager

        executor = MagicMock(spec=OrderExecutor)
        executor.execute = AsyncMock(side_effect=Exception("Network error"))

        from iqfmp.exchange.monitoring import PositionTracker
        tracker = MagicMock(spec=PositionTracker)
        tracker.get_all_positions = MagicMock(return_value=[
            PositionData(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.1"),
                entry_price=Decimal("50000"),
                current_price=Decimal("51000"),
            ),
        ])

        manager = EmergencyCloseManager(
            executor=executor,
            position_tracker=tracker,
            require_confirmation=False,
        )

        results = await manager.close_all_immediately()
        assert len(results) == 1
        assert results[0].status == CloseStatus.FAILED
        assert "Network error" in results[0].error

    @pytest.mark.asyncio
    async def test_position_tracker_exception(self):
        """Test handling position tracker exception."""
        from iqfmp.exchange.emergency import EmergencyCloseManager

        executor = MagicMock(spec=OrderExecutor)

        from iqfmp.exchange.monitoring import PositionTracker
        tracker = MagicMock(spec=PositionTracker)
        tracker.get_all_positions = MagicMock(side_effect=Exception("DB error"))

        manager = EmergencyCloseManager(
            executor=executor,
            position_tracker=tracker,
            require_confirmation=False,
        )

        with pytest.raises(Exception) as exc_info:
            await manager.close_all_immediately()
        assert "DB error" in str(exc_info.value)
