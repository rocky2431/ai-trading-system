/**
 * 实盘交易数据 Hook
 *
 * 连接真实 Trading API，支持 WebSocket 实时数据更新
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import {
  tradingApi,
  TradingState,
  Position,
  Order,
  AccountInfo,
  PnLDataPoint,
  RiskMetrics,
  RiskLevel,
  RiskAlert,
} from '@/api/trading'

// Re-export types for compatibility
export type {
  Position,
  Order,
  AccountInfo as Account,
  PnLDataPoint,
  RiskMetrics,
  RiskAlert,
}

export type { RiskLevel }

// WebSocket message types
interface TradingWebSocketMessage {
  type:
    | 'trading_state_update'
    | 'position_update'
    | 'order_update'
    | 'account_update'
    | 'risk_alert'
  data: Record<string, unknown>
  timestamp: string
}

// Default empty state
const defaultState: TradingState = {
  account: {
    totalEquity: 0,
    availableBalance: 0,
    marginUsed: 0,
    unrealizedPnl: 0,
    realizedPnl: 0,
    todayPnl: 0,
    todayPnlPercent: 0,
  },
  positions: [],
  openOrders: [],
  pnlHistory: [],
  risk: {
    level: 'normal' as RiskLevel,
    marginUsagePercent: 0,
    maxDrawdownPercent: 0,
    currentDrawdownPercent: 0,
    dailyLossPercent: 0,
    positionConcentration: 0,
    alerts: [],
  },
  isConnected: false,
  lastUpdated: new Date().toISOString(),
}

export function useLiveTrading() {
  const [state, setState] = useState<TradingState>(defaultState)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  const [isClosingAll, setIsClosingAll] = useState(false)
  const [closeConfirmOpen, setCloseConfirmOpen] = useState(false)

  // WebSocket ref
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  // Fetch trading state from API
  const fetchState = useCallback(async () => {
    try {
      const data = await tradingApi.getState()
      setState(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to fetch trading state'))
      // Keep previous state on error
    }
  }, [])

  // Initial load
  useEffect(() => {
    const load = async () => {
      setLoading(true)
      await fetchState()
      setLoading(false)
    }
    load()
  }, [fetchState])

  // Polling fallback when WebSocket is not connected
  useEffect(() => {
    if (!state.isConnected) {
      // Poll every 5 seconds when not connected via WebSocket
      const interval = setInterval(fetchState, 5000)
      return () => clearInterval(interval)
    }
    return undefined
  }, [state.isConnected, fetchState])

  // WebSocket connection with exponential backoff
  useEffect(() => {
    let retryCount = 0
    const MAX_RETRIES = 5
    const BASE_DELAY = 1000 // 1 second

    const connectWebSocket = () => {
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsHost = window.location.host
      const wsUrl = `${wsProtocol}//${wsHost}/api/v1/system/ws`

      try {
        wsRef.current = new WebSocket(wsUrl)

        wsRef.current.onopen = () => {
          // Reset retry count on successful connection
          retryCount = 0
        }

        wsRef.current.onclose = () => {
          // Exponential backoff with max retries
          if (retryCount < MAX_RETRIES) {
            const delay = BASE_DELAY * Math.pow(2, retryCount)
            retryCount++
            reconnectTimeoutRef.current = setTimeout(connectWebSocket, delay)
          }
        }

        wsRef.current.onerror = () => {
          // Error handled in onclose
        }

        wsRef.current.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data) as TradingWebSocketMessage

            // Handle trading-related messages
            switch (message.type) {
              case 'trading_state_update':
                // Full state update
                if (message.data) {
                  setState((prev) => ({
                    ...prev,
                    ...(message.data as Partial<TradingState>),
                    lastUpdated: message.timestamp,
                  }))
                }
                break

              case 'position_update':
                // Position update
                fetchState()
                break

              case 'order_update':
                // Order update
                fetchState()
                break

              case 'account_update':
                // Account update
                if (message.data?.account) {
                  setState((prev) => ({
                    ...prev,
                    account: message.data.account as AccountInfo,
                    lastUpdated: message.timestamp,
                  }))
                }
                break

              case 'risk_alert':
                // Risk alert
                if (message.data?.alert) {
                  setState((prev) => ({
                    ...prev,
                    risk: {
                      ...prev.risk,
                      alerts: [
                        message.data.alert as RiskAlert,
                        ...prev.risk.alerts.slice(0, 9),
                      ],
                    },
                    lastUpdated: message.timestamp,
                  }))
                }
                break
            }
          } catch (err) {
            console.error('Failed to parse WebSocket message:', err)
          }
        }
      } catch (err) {
        console.error('Failed to connect WebSocket:', err)
      }
    }

    connectWebSocket()

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [fetchState])

  // Connect to exchange
  const connect = useCallback(async () => {
    try {
      await tradingApi.connect()
      await fetchState()
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to connect to exchange'))
    }
  }, [fetchState])

  // Disconnect from exchange
  const disconnect = useCallback(async () => {
    try {
      await tradingApi.disconnect()
      setState((prev) => ({ ...prev, isConnected: false }))
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to disconnect'))
    }
  }, [])

  // Close single position
  const closePosition = useCallback(
    async (positionId: string) => {
      try {
        const result = await tradingApi.closePosition(positionId)
        if (result.success) {
          // Refresh state
          await fetchState()
        } else {
          throw new Error(result.message)
        }
      } catch (err) {
        setError(err instanceof Error ? err : new Error('Failed to close position'))
        throw err
      }
    },
    [fetchState]
  )

  // Close all positions (emergency)
  const closeAllPositions = useCallback(async () => {
    setIsClosingAll(true)

    try {
      const result = await tradingApi.closeAllPositions()
      if (result.success) {
        await fetchState()
      } else {
        throw new Error(result.message)
      }
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to close all positions'))
      throw err
    } finally {
      setIsClosingAll(false)
      setCloseConfirmOpen(false)
    }
  }, [fetchState])

  // Cancel order
  const cancelOrder = useCallback(
    async (orderId: string, symbol: string) => {
      try {
        const result = await tradingApi.cancelOrder(orderId, symbol)
        if (result.success) {
          await fetchState()
        } else {
          throw new Error(result.message)
        }
      } catch (err) {
        setError(err instanceof Error ? err : new Error('Failed to cancel order'))
        throw err
      }
    },
    [fetchState]
  )

  // Cancel all orders
  const cancelAllOrders = useCallback(
    async (symbol?: string) => {
      try {
        const result = await tradingApi.cancelAllOrders(symbol)
        if (result.success) {
          await fetchState()
        } else {
          throw new Error(result.message)
        }
      } catch (err) {
        setError(err instanceof Error ? err : new Error('Failed to cancel all orders'))
        throw err
      }
    },
    [fetchState]
  )

  return {
    // State
    account: state.account,
    positions: state.positions,
    openOrders: state.openOrders,
    pnlHistory: state.pnlHistory,
    risk: state.risk,
    isConnected: state.isConnected,
    lastUpdated: state.lastUpdated,

    // Loading/Error
    loading,
    error,

    // Close all UI state
    isClosingAll,
    closeConfirmOpen,
    setCloseConfirmOpen,

    // Actions
    connect,
    disconnect,
    closePosition,
    closeAllPositions,
    cancelOrder,
    cancelAllOrders,
    refresh: fetchState,
  }
}
