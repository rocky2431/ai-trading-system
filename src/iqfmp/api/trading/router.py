"""Trading API router.

REST endpoints for real-time trading operations.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from iqfmp.api.trading.schemas import (
    CancelAllOrdersResponse,
    CancelOrderResponse,
    CloseAllPositionsResponse,
    ClosePositionRequest,
    ClosePositionResponse,
    CreateOrderRequest,
    CreateOrderResponse,
    OrderResponse,
    PositionResponse,
    TradingConfigResponse,
    TradingState,
    UpdateTradingConfigRequest,
)
from iqfmp.api.trading.service import get_trading_service

router = APIRouter(tags=["Trading"])


# ==================== Trading State ====================


@router.get("/state", response_model=TradingState)
async def get_trading_state() -> TradingState:
    """Get complete trading state including positions, orders, account, and risk."""
    service = get_trading_service()
    return await service.get_trading_state()


@router.post("/connect")
async def connect_exchange() -> dict:
    """Connect to configured exchange."""
    service = get_trading_service()
    success = await service.initialize()
    if success:
        return {"success": True, "message": "Connected to exchange"}
    status = service.get_exchange_status()
    raise HTTPException(
        status_code=503,
        detail=f"Failed to connect: {status.error}",
    )


@router.post("/disconnect")
async def disconnect_exchange() -> dict:
    """Disconnect from exchange."""
    service = get_trading_service()
    await service.disconnect()
    return {"success": True, "message": "Disconnected from exchange"}


# ==================== Positions ====================


@router.get("/positions", response_model=PositionResponse)
async def get_positions() -> PositionResponse:
    """Get all open positions."""
    service = get_trading_service()
    positions = await service.get_positions()
    return PositionResponse(positions=positions, total=len(positions))


@router.post("/positions/{position_id}/close", response_model=ClosePositionResponse)
async def close_position(
    position_id: str,
    request: Optional[ClosePositionRequest] = None,
) -> ClosePositionResponse:
    """Close a specific position."""
    service = get_trading_service()
    price = request.price if request else None
    return await service.close_position(position_id, price=price)


@router.post("/positions/close-all", response_model=CloseAllPositionsResponse)
async def close_all_positions() -> CloseAllPositionsResponse:
    """Close all open positions (emergency close)."""
    service = get_trading_service()
    return await service.close_all_positions()


# ==================== Orders ====================


@router.get("/orders", response_model=OrderResponse)
async def get_orders(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
) -> OrderResponse:
    """Get open orders."""
    service = get_trading_service()
    orders = await service.get_orders(symbol)
    return OrderResponse(orders=orders, total=len(orders))


@router.post("/orders", response_model=CreateOrderResponse)
async def create_order(request: CreateOrderRequest) -> CreateOrderResponse:
    """Create a new order."""
    service = get_trading_service()
    return await service.create_order(
        symbol=request.symbol,
        side=request.side,
        order_type=request.type,
        size=request.size,
        price=request.price,
        stop_price=request.stop_price,
        leverage=request.leverage,
        reduce_only=request.reduce_only,
    )


@router.delete("/orders/{order_id}", response_model=CancelOrderResponse)
async def cancel_order(
    order_id: str,
    symbol: str = Query(..., description="Trading symbol"),
) -> CancelOrderResponse:
    """Cancel an order."""
    service = get_trading_service()
    result = await service.cancel_order(order_id, symbol)
    return CancelOrderResponse(
        success=result["success"],
        message=result["message"],
    )


@router.post("/orders/cancel-all", response_model=CancelAllOrdersResponse)
async def cancel_all_orders(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
) -> CancelAllOrdersResponse:
    """Cancel all open orders."""
    service = get_trading_service()
    result = await service.cancel_all_orders(symbol)
    return CancelAllOrdersResponse(
        success=result["success"],
        message=result["message"],
        canceled_count=result["canceled_count"],
    )


# ==================== Configuration ====================


@router.get("/config", response_model=TradingConfigResponse)
async def get_trading_config() -> TradingConfigResponse:
    """Get trading configuration and exchange status."""
    service = get_trading_service()
    return TradingConfigResponse(
        config=service.get_config(),
        exchange_status=service.get_exchange_status(),
    )


@router.put("/config", response_model=TradingConfigResponse)
async def update_trading_config(
    request: UpdateTradingConfigRequest,
) -> TradingConfigResponse:
    """Update trading configuration."""
    service = get_trading_service()
    config = service.update_config(
        enabled=request.enabled,
        symbols=request.symbols,
        max_leverage=request.max_leverage,
        default_leverage=request.default_leverage,
        risk_controls=request.risk_controls,
    )
    return TradingConfigResponse(
        config=config,
        exchange_status=service.get_exchange_status(),
    )
