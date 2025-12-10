/**
 * 交易相关类型定义
 * 实盘监控面板所需的数据类型
 */

export type PositionSide = 'long' | 'short'
export type OrderStatus = 'open' | 'filled' | 'cancelled' | 'partial'
export type RiskLevel = 'normal' | 'warning' | 'danger' | 'critical'

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
  liquidationPrice: number
  createdAt: string
}

export interface Order {
  id: string
  symbol: string
  side: PositionSide
  type: 'market' | 'limit' | 'stop'
  price: number
  size: number
  filled: number
  status: OrderStatus
  createdAt: string
}

export interface PnLDataPoint {
  timestamp: string
  realizedPnl: number
  unrealizedPnl: number
  totalPnl: number
  equity: number
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

export interface RiskAlert {
  id: string
  type: 'margin' | 'drawdown' | 'loss' | 'concentration'
  message: string
  severity: 'warning' | 'danger' | 'critical'
  timestamp: string
}

export interface AccountSummary {
  totalEquity: number
  availableBalance: number
  marginUsed: number
  unrealizedPnl: number
  realizedPnl: number
  todayPnl: number
  todayPnlPercent: number
}

export interface TradingState {
  account: AccountSummary
  positions: Position[]
  openOrders: Order[]
  pnlHistory: PnLDataPoint[]
  risk: RiskMetrics
  isConnected: boolean
  lastUpdated: string
}
