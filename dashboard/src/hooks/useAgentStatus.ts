/**
 * Agent 状态 Hook - 提供 Mock 数据，后续连接真实 API
 */

import { useState, useEffect } from 'react'
import type { AgentSystemStatus, Agent, TaskQueueItem, LLMMetrics, ResourceMetrics } from '@/types/agent'

function generateMockAgents(): Agent[] {
  return [
    {
      id: 'agent-1',
      name: 'Factor Generator',
      type: 'factor_generation',
      status: 'running',
      currentTask: 'Generating momentum factor',
      progress: 67,
      startedAt: new Date(Date.now() - 1000 * 60 * 5).toISOString(),
      lastActivity: new Date().toISOString(),
    },
    {
      id: 'agent-2',
      name: 'Factor Evaluator',
      type: 'evaluation',
      status: 'idle',
      currentTask: null,
      progress: 0,
      startedAt: null,
      lastActivity: new Date(Date.now() - 1000 * 60 * 10).toISOString(),
    },
    {
      id: 'agent-3',
      name: 'Strategy Builder',
      type: 'strategy',
      status: 'paused',
      currentTask: 'Building multi-factor strategy',
      progress: 45,
      startedAt: new Date(Date.now() - 1000 * 60 * 15).toISOString(),
      lastActivity: new Date(Date.now() - 1000 * 60 * 2).toISOString(),
    },
    {
      id: 'agent-4',
      name: 'Backtester',
      type: 'backtest',
      status: 'running',
      currentTask: 'Running 3-year backtest',
      progress: 23,
      startedAt: new Date(Date.now() - 1000 * 60 * 3).toISOString(),
      lastActivity: new Date().toISOString(),
    },
  ]
}

function generateMockTaskQueue(): TaskQueueItem[] {
  return [
    {
      id: 'task-1',
      type: 'factor_generation',
      status: 'running',
      createdAt: new Date(Date.now() - 1000 * 60 * 10).toISOString(),
      startedAt: new Date(Date.now() - 1000 * 60 * 5).toISOString(),
      completedAt: null,
      agentId: 'agent-1',
      priority: 'high',
    },
    {
      id: 'task-2',
      type: 'evaluation',
      status: 'pending',
      createdAt: new Date(Date.now() - 1000 * 60 * 8).toISOString(),
      startedAt: null,
      completedAt: null,
      agentId: null,
      priority: 'normal',
    },
    {
      id: 'task-3',
      type: 'backtest',
      status: 'running',
      createdAt: new Date(Date.now() - 1000 * 60 * 6).toISOString(),
      startedAt: new Date(Date.now() - 1000 * 60 * 3).toISOString(),
      completedAt: null,
      agentId: 'agent-4',
      priority: 'high',
    },
    {
      id: 'task-4',
      type: 'factor_generation',
      status: 'completed',
      createdAt: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
      startedAt: new Date(Date.now() - 1000 * 60 * 25).toISOString(),
      completedAt: new Date(Date.now() - 1000 * 60 * 15).toISOString(),
      agentId: 'agent-1',
      priority: 'normal',
    },
    {
      id: 'task-5',
      type: 'strategy',
      status: 'pending',
      createdAt: new Date(Date.now() - 1000 * 60 * 2).toISOString(),
      startedAt: null,
      completedAt: null,
      agentId: null,
      priority: 'low',
    },
  ]
}

function generateMockLLMMetrics(): LLMMetrics {
  return {
    provider: 'OpenRouter',
    model: 'deepseek-coder-v2',
    totalRequests: 1247,
    successRate: 98.5,
    avgLatency: 856,
    p95Latency: 1520,
    p99Latency: 2340,
    tokensUsed: 2450000,
    costEstimate: 12.35,
    lastHourRequests: [45, 52, 48, 61, 55, 42, 38, 67, 72, 58, 49, 53],
  }
}

function generateMockResourceMetrics(): ResourceMetrics {
  return {
    cpu: {
      usage: 45 + Math.random() * 20,
      cores: 8,
    },
    memory: {
      used: 6.2 + Math.random() * 0.5,
      total: 16,
      percentage: ((6.2 + Math.random() * 0.5) / 16) * 100,
    },
    disk: {
      used: 125,
      total: 500,
      percentage: 25,
    },
  }
}

export function useAgentStatus() {
  const [status, setStatus] = useState<AgentSystemStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    // 模拟初始加载
    const loadData = () => {
      try {
        const mockData: AgentSystemStatus = {
          agents: generateMockAgents(),
          taskQueue: generateMockTaskQueue(),
          llmMetrics: generateMockLLMMetrics(),
          resources: generateMockResourceMetrics(),
          systemHealth: 'healthy',
          uptime: 3600 * 24 * 3 + 3600 * 5 + 60 * 23,
        }
        setStatus(mockData)
        setLoading(false)
      } catch (err) {
        setError(err instanceof Error ? err : new Error('Unknown error'))
        setLoading(false)
      }
    }

    loadData()

    // 模拟实时更新
    const interval = setInterval(() => {
      setStatus((prev) => {
        if (!prev) return prev
        return {
          ...prev,
          resources: generateMockResourceMetrics(),
          agents: prev.agents.map((agent) => ({
            ...agent,
            progress: agent.status === 'running'
              ? Math.min(100, agent.progress + Math.random() * 2)
              : agent.progress,
            lastActivity: agent.status === 'running'
              ? new Date().toISOString()
              : agent.lastActivity,
          })),
        }
      })
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  return { status, loading, error }
}
