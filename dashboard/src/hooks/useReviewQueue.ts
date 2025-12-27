/**
 * useReviewQueue - 审核队列 Hook
 *
 * 提供审核队列数据和操作方法
 */

import { useState, useEffect, useCallback } from 'react'
import {
  reviewApi,
  ReviewRequest,
  ReviewDecision,
  ReviewQueueStats,
  ReviewConfig,
} from '@/api/review'

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
    refresh,
    approve,
    reject,
    updateConfig,
    processTimeouts,
  }
}
