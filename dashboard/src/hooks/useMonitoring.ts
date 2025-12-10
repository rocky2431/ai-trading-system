/**
 * Monitoring Hook - IQFMP 监控指标
 * 聚合系统状态、因子、RD Loop、风险等指标
 */

import { useState, useEffect, useCallback } from 'react'
import { systemApi } from '@/api'
import type { SystemStatusResponse } from '@/api'

// IQFMP 监控指标类型
export interface IQFMPMetrics {
  // LLM 指标
  llm: {
    totalRequests: number
    successRate: number
    avgLatency: number
    tokensUsed: number
    costEstimate: number
    requestsPerMinute: number
  }
  // 因子指标
  factors: {
    totalGenerated: number
    totalEvaluated: number
    passRate: number
    avgIC: number
    avgSharpe: number
    pendingEvaluation: number
  }
  // RD Loop 指标
  rdLoop: {
    isRunning: boolean
    currentIteration: number
    totalIterations: number
    coreFactorsCount: number
    hypothesesTested: number
    currentPhase: string
  }
  // 风险指标
  risk: {
    currentDrawdown: number
    maxDrawdownThreshold: number
    currentLeverage: number
    maxLeverage: number
    positionConcentration: number
    violationsCount: number
    riskLevel: 'safe' | 'warning' | 'danger'
  }
  // 向量存储指标
  vectorStore: {
    totalVectors: number
    searchLatency: number
    similarityChecks: number
    duplicatesFound: number
  }
  // 系统资源
  resources: {
    cpuUsage: number
    memoryUsage: number
    memoryUsed: number
    memoryTotal: number
    diskUsage: number
  }
  // 系统健康
  systemHealth: 'healthy' | 'degraded' | 'unhealthy'
  uptime: number
}

// 转换 API 响应到监控指标
function apiToMonitoringMetrics(response: SystemStatusResponse): IQFMPMetrics {
  return {
    llm: {
      totalRequests: response.llm_metrics.total_requests,
      successRate: response.llm_metrics.success_rate,
      avgLatency: response.llm_metrics.avg_latency,
      tokensUsed: response.llm_metrics.tokens_used,
      costEstimate: response.llm_metrics.cost_estimate,
      requestsPerMinute: response.llm_metrics.last_hour_requests.slice(-1)[0] || 0,
    },
    factors: {
      totalGenerated: 0, // 从其他 API 获取
      totalEvaluated: 0,
      passRate: 0,
      avgIC: 0,
      avgSharpe: 0,
      pendingEvaluation: 0,
    },
    rdLoop: {
      isRunning: false,
      currentIteration: 0,
      totalIterations: 100,
      coreFactorsCount: 0,
      hypothesesTested: 0,
      currentPhase: 'idle',
    },
    risk: {
      currentDrawdown: 0,
      maxDrawdownThreshold: 0.15,
      currentLeverage: 1.0,
      maxLeverage: 3.0,
      positionConcentration: 0,
      violationsCount: 0,
      riskLevel: 'safe',
    },
    vectorStore: {
      totalVectors: 0,
      searchLatency: 0,
      similarityChecks: 0,
      duplicatesFound: 0,
    },
    resources: {
      cpuUsage: response.resources.cpu.usage,
      memoryUsage: response.resources.memory.percentage,
      memoryUsed: response.resources.memory.used,
      memoryTotal: response.resources.memory.total,
      diskUsage: response.resources.disk.percentage,
    },
    systemHealth: response.system_health,
    uptime: response.uptime,
  }
}

export function useMonitoring(refreshInterval = 5000) {
  const [metrics, setMetrics] = useState<IQFMPMetrics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  const loadMetrics = useCallback(async () => {
    try {
      const response = await systemApi.getStatus()
      const baseMetrics = apiToMonitoringMetrics(response)

      // 尝试获取额外的因子统计
      try {
        const factorsResponse = await fetch(
          `${import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'}/factors/stats`
        )
        if (factorsResponse.ok) {
          const factorStats = await factorsResponse.json()
          baseMetrics.factors = {
            totalGenerated: factorStats.total || 0,
            totalEvaluated: factorStats.evaluated || 0,
            passRate: factorStats.pass_rate || 0,
            avgIC: factorStats.avg_ic || 0,
            avgSharpe: factorStats.avg_sharpe || 0,
            pendingEvaluation: factorStats.pending || 0,
          }
        }
      } catch {
        // 使用默认值
      }

      // 尝试获取 RD Loop 状态
      try {
        const rdLoopResponse = await fetch(
          `${import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'}/pipeline/rdloop/state`
        )
        if (rdLoopResponse.ok) {
          const rdLoopState = await rdLoopResponse.json()
          baseMetrics.rdLoop = {
            isRunning: rdLoopState.is_running || false,
            currentIteration: rdLoopState.iteration || 0,
            totalIterations: 100,
            coreFactorsCount: rdLoopState.core_factors_count || 0,
            hypothesesTested: rdLoopState.total_hypotheses_tested || 0,
            currentPhase: rdLoopState.phase || 'idle',
          }
        }
      } catch {
        // 使用默认值
      }

      setMetrics(baseMetrics)
      setError(null)
    } catch (err) {
      console.error('Failed to load monitoring metrics:', err)
      setError(err instanceof Error ? err : new Error('Failed to load monitoring metrics'))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadMetrics()
    const interval = setInterval(loadMetrics, refreshInterval)
    return () => clearInterval(interval)
  }, [loadMetrics, refreshInterval])

  return { metrics, loading, error, refresh: loadMetrics }
}
