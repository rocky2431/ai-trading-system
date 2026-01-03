/**
 * useReviewQueue - 审核队列 Hook
 *
 * 提供审核队列数据和操作方法，支持 WebSocket 实时更新
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import {
  reviewApi,
  ReviewRequest,
  ReviewDecision,
  ReviewQueueStats,
  ReviewConfig,
} from '@/api/review'
import { tokenStorage } from '@/api/auth'

// WebSocket 消息类型
interface ReviewWebSocketMessage {
  type: 'review_request_created' | 'review_decision' | 'review_timeout' | 'review_stats_update'
  data: {
    request_id?: string
    factor_name?: string
    code_summary?: string
    priority?: number
    approved?: boolean
    reviewer?: string
    reason?: string
    status?: string
    pending_count?: number
    approved_count?: number
    rejected_count?: number
    timeout_count?: number
  }
  timestamp: string
}

interface UseReviewQueueResult {
  // 数据
  pendingRequests: ReviewRequest[]
  decisionHistory: ReviewDecision[]
  stats: ReviewQueueStats | null
  config: ReviewConfig | null

  // 分页
  pendingPage: number
  setPendingPage: (page: number) => void
  pendingTotal: number
  historyPage: number
  setHistoryPage: (page: number) => void
  historyTotal: number

  // 状态
  loading: boolean
  error: Error | null

  // WebSocket 状态
  wsConnected: boolean
  lastEvent: ReviewWebSocketMessage | null

  // 操作
  refresh: () => Promise<void>
  approve: (requestId: string, reason?: string) => Promise<void>
  reject: (requestId: string, reason: string) => Promise<void>
  updateConfig: (config: Partial<ReviewConfig>) => Promise<void>
  processTimeouts: () => Promise<number>
}

export function useReviewQueue(): UseReviewQueueResult {
  // 数据状态
  const [pendingRequests, setPendingRequests] = useState<ReviewRequest[]>([])
  const [decisionHistory, setDecisionHistory] = useState<ReviewDecision[]>([])
  const [stats, setStats] = useState<ReviewQueueStats | null>(null)
  const [config, setConfig] = useState<ReviewConfig | null>(null)

  // 分页状态
  const [pendingPage, setPendingPage] = useState(1)
  const [pendingTotal, setPendingTotal] = useState(0)
  const [historyPage, setHistoryPage] = useState(1)
  const [historyTotal, setHistoryTotal] = useState(0)

  // 加载状态
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  // WebSocket 状态
  const [wsConnected, setWsConnected] = useState(false)
  const [lastEvent, setLastEvent] = useState<ReviewWebSocketMessage | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  // 刷新所有数据
  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)

    try {
      const [pendingRes, historyRes, statsRes, configRes] = await Promise.all([
        reviewApi.getPendingRequests(pendingPage, 10),
        reviewApi.getDecisionHistory(historyPage, 10),
        reviewApi.getStats(),
        reviewApi.getConfig(),
      ])

      setPendingRequests(pendingRes.items)
      setPendingTotal(pendingRes.total)
      setDecisionHistory(historyRes.items)
      setHistoryTotal(historyRes.total)
      setStats(statsRes)
      setConfig(configRes)
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to load review data'))
    } finally {
      setLoading(false)
    }
  }, [pendingPage, historyPage])

  // 初始加载和分页变化时刷新
  useEffect(() => {
    refresh()
  }, [refresh])

  // 批准请求
  const approve = useCallback(async (requestId: string, reason?: string) => {
    await reviewApi.decide(requestId, { approved: true, reason })
    await refresh()
  }, [refresh])

  // 拒绝请求
  const reject = useCallback(async (requestId: string, reason: string) => {
    await reviewApi.decide(requestId, { approved: false, reason })
    await refresh()
  }, [refresh])

  // 更新配置
  const updateConfig = useCallback(async (newConfig: Partial<ReviewConfig>) => {
    const updated = await reviewApi.updateConfig(newConfig)
    setConfig(updated)
  }, [])

  // 处理超时
  const processTimeouts = useCallback(async () => {
    const result = await reviewApi.processTimeouts()
    if (result.timed_out_count > 0) {
      await refresh()
    }
    return result.timed_out_count
  }, [refresh])

  // WebSocket 消息处理
  const handleWebSocketMessage = useCallback((event: MessageEvent) => {
    try {
      const message = JSON.parse(event.data) as ReviewWebSocketMessage

      // 只处理审核相关消息
      if (!message.type.startsWith('review_')) {
        return
      }

      setLastEvent(message)

      switch (message.type) {
        case 'review_request_created':
          // 新审核请求 - 刷新待审核列表
          refresh()
          break

        case 'review_decision':
          // 审核决策 - 刷新所有数据
          refresh()
          break

        case 'review_timeout':
          // 超时 - 刷新数据
          refresh()
          break

        case 'review_stats_update':
          // 统计更新 - 直接更新 stats 状态
          if (message.data) {
            setStats((prev) => ({
              pending_count: message.data.pending_count ?? prev?.pending_count ?? 0,
              approved_count: message.data.approved_count ?? prev?.approved_count ?? 0,
              rejected_count: message.data.rejected_count ?? prev?.rejected_count ?? 0,
              timeout_count: message.data.timeout_count ?? prev?.timeout_count ?? 0,
              average_review_time_seconds: prev?.average_review_time_seconds ?? null,
              oldest_pending_age_seconds: prev?.oldest_pending_age_seconds ?? null,
            }))
          }
          break
      }
    } catch (err) {
      console.error('Failed to parse WebSocket message:', err)
    }
  }, [refresh])

  // WebSocket 连接管理 with exponential backoff
  useEffect(() => {
    let retryCount = 0
    const MAX_RETRIES = 5
    const BASE_DELAY = 1000 // 1 second

    const connectWebSocket = () => {
      // Get authentication token
      const token = tokenStorage.getAccessToken()
      if (!token) {
        console.warn('No auth token available for WebSocket connection')
        return
      }

      // 使用系统 WebSocket 端点
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsHost = window.location.host
      const wsUrl = `${wsProtocol}//${wsHost}/api/v1/system/ws?token=${encodeURIComponent(token)}`

      try {
        wsRef.current = new WebSocket(wsUrl)

        wsRef.current.onopen = () => {
          setWsConnected(true)
          retryCount = 0 // Reset on successful connection
        }

        wsRef.current.onclose = () => {
          setWsConnected(false)

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

        wsRef.current.onmessage = handleWebSocketMessage
      } catch {
        // Connection error - will be retried via onclose
      }
    }

    connectWebSocket()

    // 清理
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [handleWebSocketMessage])

  return {
    pendingRequests,
    decisionHistory,
    stats,
    config,
    pendingPage,
    setPendingPage,
    pendingTotal,
    historyPage,
    setHistoryPage,
    historyTotal,
    loading,
    error,
    wsConnected,
    lastEvent,
    refresh,
    approve,
    reject,
    updateConfig,
    processTimeouts,
  }
}
