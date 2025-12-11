/**
 * Agent 状态 Hook - 使用真实 API 数据
 */

import { useState, useEffect, useCallback } from 'react'
import { systemApi } from '@/api'
import type { SystemStatusResponse } from '@/api'
import type { AgentSystemStatus } from '@/types/agent'

// 将 API 响应转换为前端类型
function apiToAgentStatus(response: SystemStatusResponse): AgentSystemStatus {
  return {
    agents: response.agents.map(agent => ({
      id: agent.id,
      name: agent.name,
      type: agent.type,
      status: agent.status,
      currentTask: agent.current_task,
      progress: agent.progress,
      startedAt: agent.started_at,
      lastActivity: agent.last_activity,
    })),
    taskQueue: response.task_queue.map(task => ({
      id: task.id,
      type: task.type,
      status: task.status,
      createdAt: task.created_at,
      startedAt: task.started_at,
      completedAt: task.completed_at,
      agentId: task.agent_id,
      priority: task.priority,
    })),
    llmMetrics: {
      provider: response.llm_metrics.provider,
      model: response.llm_metrics.model,
      totalRequests: response.llm_metrics.total_requests,
      successRate: response.llm_metrics.success_rate,
      avgLatency: response.llm_metrics.avg_latency,
      p95Latency: response.llm_metrics.p95_latency,
      p99Latency: response.llm_metrics.p99_latency,
      tokensUsed: response.llm_metrics.tokens_used,
      costEstimate: response.llm_metrics.cost_estimate,
      lastHourRequests: response.llm_metrics.last_hour_requests,
    },
    resources: {
      cpu: {
        usage: response.resources.cpu.usage,
        cores: response.resources.cpu.cores,
      },
      memory: {
        used: response.resources.memory.used,
        total: response.resources.memory.total,
        percentage: response.resources.memory.percentage,
      },
      disk: {
        used: response.resources.disk.used,
        total: response.resources.disk.total,
        percentage: response.resources.disk.percentage,
      },
    },
    systemHealth: response.system_health,
    uptime: response.uptime,
  }
}

export function useAgentStatus() {
  const [status, setStatus] = useState<AgentSystemStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  const loadStatus = useCallback(async () => {
    try {
      const response = await systemApi.getStatus()
      setStatus(apiToAgentStatus(response))
      setError(null)
    } catch (err) {
      console.error('Failed to load system status:', err)
      setError(err instanceof Error ? err : new Error('Failed to load system status'))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadStatus()

    // 定期刷新系统状态 (每 5 秒)
    const interval = setInterval(loadStatus, 5000)

    return () => clearInterval(interval)
  }, [loadStatus])

  return { status, loading, error, refresh: loadStatus }
}
