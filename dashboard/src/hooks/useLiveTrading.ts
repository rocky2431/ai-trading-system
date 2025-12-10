/**
 * 实盘交易数据 Hook
 * 模拟 WebSocket 实时数据更新
 */

import { useState, useEffect, useCallback } from 'react'
import type {
  TradingState,
  Position,
  Order,
  PnLDataPoint,
  RiskMetrics,
  RiskAlert,
} from '@/types/trading'

// 模拟初始持仓
const initialPositions: Position[] = [
  {
    id: 'pos-1',
    symbol: 'BTC-USDT',
    side: 'long',
    size: 0.5,
    entryPrice: 42150,
    markPrice: 43280,
    leverage: 10,
    unrealizedPnl: 565,
    unrealizedPnlPercent: 2.68,
    marginUsed: 2107.5,
    liquidationPrice: 38000,
    createdAt: '2025-12-10T08:30:00',
  },
  {
    id: 'pos-2',
    symbol: 'ETH-USDT',
    side: 'long',
    size: 5,
    entryPrice: 2280,
    markPrice: 2345,
    leverage: 5,
    unrealizedPnl: 325,
    unrealizedPnlPercent: 2.85,
    marginUsed: 2280,
    liquidationPrice: 1900,
    createdAt: '2025-12-10T10:15:00',
  },
  {
    id: 'pos-3',
    symbol: 'SOL-USDT',
    side: 'short',
    size: 50,
    entryPrice: 115,
    markPrice: 112.5,
    leverage: 10,
    unrealizedPnl: 125,
    unrealizedPnlPercent: 2.17,
    marginUsed: 575,
    liquidationPrice: 125,
    createdAt: '2025-12-10T14:00:00',
  },
]

// 模拟挂单
const initialOrders: Order[] = [
  {
    id: 'ord-1',
    symbol: 'BTC-USDT',
    side: 'long',
    type: 'limit',
    price: 41500,
    size: 0.3,
    filled: 0,
    status: 'open',
    createdAt: '2025-12-10T15:30:00',
  },
  {
    id: 'ord-2',
    symbol: 'ETH-USDT',
    side: 'short',
    type: 'stop',
    price: 2400,
    size: 3,
    filled: 0,
    status: 'open',
    createdAt: '2025-12-10T16:00:00',
  },
]

// 生成 PnL 历史数据
function generatePnLHistory(): PnLDataPoint[] {
  const history: PnLDataPoint[] = []
  const now = new Date()
  let equity = 50000
  let realizedPnl = 0

  for (let i = 24; i >= 0; i--) {
    const timestamp = new Date(now.getTime() - i * 60 * 60 * 1000)
    const change = (Math.random() - 0.48) * 500
    equity += change
    realizedPnl += change > 0 ? change * 0.3 : 0
    const unrealizedPnl = (Math.random() - 0.45) * 1500

    history.push({
      timestamp: timestamp.toISOString(),
      realizedPnl,
      unrealizedPnl,
      totalPnl: realizedPnl + unrealizedPnl,
      equity: Math.max(equity, 45000),
    })
  }

  return history
}

// 生成风险告警
function generateAlerts(): RiskAlert[] {
  return [
    {
      id: 'alert-1',
      type: 'margin',
      message: 'Margin usage approaching 60% threshold',
      severity: 'warning',
      timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
    },
  ]
}

// 计算风险等级
function calculateRiskLevel(marginUsage: number, drawdown: number): RiskMetrics['level'] {
  if (marginUsage > 80 || drawdown > 15) return 'critical'
  if (marginUsage > 60 || drawdown > 10) return 'danger'
  if (marginUsage > 40 || drawdown > 5) return 'warning'
  return 'normal'
}

export function useLiveTrading() {
  const [state, setState] = useState<TradingState>(() => {
    const pnlHistory = generatePnLHistory()
    const marginUsage = 45.2
    const drawdown = 3.5

    return {
      account: {
        totalEquity: 52350.5,
        availableBalance: 28680.25,
        marginUsed: 4962.5,
        unrealizedPnl: 1015,
        realizedPnl: 1335.5,
        todayPnl: 850.25,
        todayPnlPercent: 1.65,
      },
      positions: initialPositions,
      openOrders: initialOrders,
      pnlHistory,
      risk: {
        level: calculateRiskLevel(marginUsage, drawdown),
        marginUsagePercent: marginUsage,
        maxDrawdownPercent: 8.5,
        currentDrawdownPercent: drawdown,
        dailyLossPercent: 0,
        positionConcentration: 42.5,
        alerts: generateAlerts(),
      },
      isConnected: true,
      lastUpdated: new Date().toISOString(),
    }
  })

  const [isClosingAll, setIsClosingAll] = useState(false)
  const [closeConfirmOpen, setCloseConfirmOpen] = useState(false)

  // 模拟实时数据更新
  useEffect(() => {
    const interval = setInterval(() => {
      setState((prev) => {
        // 更新持仓价格
        const updatedPositions = prev.positions.map((pos) => {
          const priceChange = (Math.random() - 0.5) * pos.markPrice * 0.002
          const newMarkPrice = pos.markPrice + priceChange
          const priceDiff = newMarkPrice - pos.entryPrice
          const unrealizedPnl = pos.side === 'long'
            ? priceDiff * pos.size
            : -priceDiff * pos.size
          const unrealizedPnlPercent = (unrealizedPnl / pos.marginUsed) * 100

          return {
            ...pos,
            markPrice: Number(newMarkPrice.toFixed(2)),
            unrealizedPnl: Number(unrealizedPnl.toFixed(2)),
            unrealizedPnlPercent: Number(unrealizedPnlPercent.toFixed(2)),
          }
        })

        // 计算账户总计
        const totalUnrealized = updatedPositions.reduce((sum, p) => sum + p.unrealizedPnl, 0)
        const totalMargin = updatedPositions.reduce((sum, p) => sum + p.marginUsed, 0)

        // 更新 PnL 历史
        const lastPnl = prev.pnlHistory[prev.pnlHistory.length - 1]
        const newPnlPoint: PnLDataPoint = {
          timestamp: new Date().toISOString(),
          realizedPnl: lastPnl.realizedPnl,
          unrealizedPnl: totalUnrealized,
          totalPnl: lastPnl.realizedPnl + totalUnrealized,
          equity: prev.account.totalEquity + (totalUnrealized - prev.account.unrealizedPnl),
        }

        const newHistory = [...prev.pnlHistory.slice(1), newPnlPoint]

        // 更新风险指标
        const marginUsage = (totalMargin / prev.account.totalEquity) * 100
        const maxEquity = Math.max(...newHistory.map(h => h.equity))
        const currentDrawdown = ((maxEquity - newPnlPoint.equity) / maxEquity) * 100

        return {
          ...prev,
          positions: updatedPositions,
          account: {
            ...prev.account,
            unrealizedPnl: Number(totalUnrealized.toFixed(2)),
            marginUsed: totalMargin,
            todayPnl: Number((prev.account.todayPnl + (Math.random() - 0.5) * 10).toFixed(2)),
            todayPnlPercent: Number(((prev.account.todayPnl / prev.account.totalEquity) * 100).toFixed(2)),
          },
          pnlHistory: newHistory,
          risk: {
            ...prev.risk,
            level: calculateRiskLevel(marginUsage, currentDrawdown),
            marginUsagePercent: Number(marginUsage.toFixed(1)),
            currentDrawdownPercent: Number(Math.max(0, currentDrawdown).toFixed(2)),
          },
          lastUpdated: new Date().toISOString(),
        }
      })
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  // 紧急平仓处理
  const closeAllPositions = useCallback(async () => {
    setIsClosingAll(true)

    // 模拟平仓过程
    await new Promise(resolve => setTimeout(resolve, 2000))

    setState((prev) => ({
      ...prev,
      positions: [],
      openOrders: [],
      account: {
        ...prev.account,
        realizedPnl: prev.account.realizedPnl + prev.account.unrealizedPnl,
        unrealizedPnl: 0,
        marginUsed: 0,
        availableBalance: prev.account.totalEquity,
      },
      risk: {
        ...prev.risk,
        level: 'normal',
        marginUsagePercent: 0,
        positionConcentration: 0,
        alerts: [],
      },
    }))

    setIsClosingAll(false)
    setCloseConfirmOpen(false)
  }, [])

  // 关闭单个持仓
  const closePosition = useCallback(async (positionId: string) => {
    await new Promise(resolve => setTimeout(resolve, 500))

    setState((prev) => {
      const position = prev.positions.find(p => p.id === positionId)
      if (!position) return prev

      const newPositions = prev.positions.filter(p => p.id !== positionId)
      const totalUnrealized = newPositions.reduce((sum, p) => sum + p.unrealizedPnl, 0)
      const totalMargin = newPositions.reduce((sum, p) => sum + p.marginUsed, 0)

      return {
        ...prev,
        positions: newPositions,
        account: {
          ...prev.account,
          realizedPnl: prev.account.realizedPnl + position.unrealizedPnl,
          unrealizedPnl: totalUnrealized,
          marginUsed: totalMargin,
          availableBalance: prev.account.availableBalance + position.marginUsed,
        },
      }
    })
  }, [])

  return {
    ...state,
    isClosingAll,
    closeConfirmOpen,
    setCloseConfirmOpen,
    closeAllPositions,
    closePosition,
  }
}
