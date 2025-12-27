/**
 * Trading API - 实盘交易 API
 */

import { api } from './client'

// ============== Enums ==============

export type PositionSide = 'long' | 'short'
export type OrderSide = 'buy' | 'sell'
export type OrderType = 'limit' | 'market' | 'stop' | 'stop_limit'
export type OrderStatus = 'open' | 'closed' | 'canceled' | 'expired' | 'rejected' | 'partially_filled'
export type RiskLevel = 'normal' | 'warning' | 'danger' | 'critical'

// ============== Position Types ==============

export interface Position {
  id: string
  symbol: string
  side: PositionSide
  size: number
  entryPrice: number
  markPrice: number
  leverage: number
  unrealizedPnl: number
  unrealizedPnlPercent: number
  marginUsed: number
  liquidationPrice: number | null
  createdAt: string
}

export interface PositionResponse {
  positions: Position[]
  total: number
}

export interface ClosePositionRequest {
  reduce_only?: boolean
  price?: number
}

export interface ClosePositionResponse {
  success: boolean
  message: string
  order_id?: string
  realized_pnl?: number
}

export interface CloseAllPositionsResponse {
  success: boolean
  message: string
  closed_count: number
  total_realized_pnl: number
}

// ============== Order Types ==============

export interface Order {
  id: string
  symbol: string
  side: OrderSide
  type: OrderType
  price: number | null
  size: number
  filled: number
  remaining: number
  status: OrderStatus
  createdAt: string
  updatedAt?: string
}

export interface OrderResponse {
  orders: Order[]
  total: number
}

export interface CreateOrderRequest {
  symbol: string
  side: OrderSide
  type: OrderType
  size: number
  price?: number
  stop_price?: number
  leverage?: number
  reduce_only?: boolean
  post_only?: boolean
  client_order_id?: string
}

export interface CreateOrderResponse {
  success: boolean
  message: string
  order?: Order
}

export interface CancelOrderResponse {
  success: boolean
  message: string
}

export interface CancelAllOrdersResponse {
  success: boolean
  message: string
  canceled_count: number
}

// ============== Account Types ==============

export interface AccountInfo {
  totalEquity: number
  availableBalance: number
  marginUsed: number
  unrealizedPnl: number
  realizedPnl: number
  todayPnl: number
  todayPnlPercent: number
}

export interface PnLDataPoint {
  timestamp: string
  realizedPnl: number
  unrealizedPnl: number
  totalPnl: number
  equity: number
}

// ============== Risk Types ==============

export interface RiskAlert {
  id: string
  type: string
  message: string
  severity: RiskLevel
  timestamp: string
}

export interface RiskMetrics {
  level: RiskLevel
  marginUsagePercent: number
  maxDrawdownPercent: number
  currentDrawdownPercent: number
  dailyLossPercent: number
  positionConcentration: number
  alerts: RiskAlert[]
}

// ============== Trading State ==============

export interface TradingState {
  account: AccountInfo
  positions: Position[]
  openOrders: Order[]
  pnlHistory: PnLDataPoint[]
  risk: RiskMetrics
  isConnected: boolean
  lastUpdated: string
}

// ============== Config Types ==============

export interface ExchangeStatus {
  exchange_id: string
  connected: boolean
  last_heartbeat: string | null
  error: string | null
}

export interface TradingConfig {
  enabled: boolean
  exchange_id: string | null
  symbols: string[]
  max_leverage: number
  default_leverage: number
  risk_controls: Record<string, unknown>
}

export interface UpdateTradingConfigRequest {
  enabled?: boolean
  symbols?: string[]
  max_leverage?: number
  default_leverage?: number
  risk_controls?: Record<string, unknown>
}

export interface TradingConfigResponse {
  config: TradingConfig
  exchange_status: ExchangeStatus | null
}

// ============== API ==============

export const tradingApi = {
  // Trading State
  getState: () => api.get<TradingState>('/trading/state'),

  // Connection
  connect: () => api.post<{ success: boolean; message: string }>('/trading/connect'),
  disconnect: () => api.post<{ success: boolean; message: string }>('/trading/disconnect'),

  // Positions
  getPositions: () => api.get<PositionResponse>('/trading/positions'),
  closePosition: (positionId: string, request?: ClosePositionRequest) =>
    api.post<ClosePositionResponse>(`/trading/positions/${positionId}/close`, request),
  closeAllPositions: () => api.post<CloseAllPositionsResponse>('/trading/positions/close-all'),

  // Orders
  getOrders: (symbol?: string) =>
    api.get<OrderResponse>('/trading/orders', symbol ? { symbol } : undefined),
  createOrder: (request: CreateOrderRequest) =>
    api.post<CreateOrderResponse>('/trading/orders', request),
  cancelOrder: (orderId: string, symbol: string) =>
    api.delete<CancelOrderResponse>(`/trading/orders/${orderId}?symbol=${encodeURIComponent(symbol)}`),
  cancelAllOrders: (symbol?: string) =>
    api.post<CancelAllOrdersResponse>('/trading/orders/cancel-all', symbol ? { symbol } : undefined),

  // Config
  getConfig: () => api.get<TradingConfigResponse>('/trading/config'),
  updateConfig: (request: UpdateTradingConfigRequest) =>
    api.put<TradingConfigResponse>('/trading/config', request),
}
