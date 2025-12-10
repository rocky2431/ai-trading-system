/**
 * Agent 相关类型定义
 */

export type AgentStatus = 'idle' | 'running' | 'paused' | 'error'

export interface Agent {
  id: string
  name: string
  type: 'factor_generation' | 'evaluation' | 'strategy' | 'backtest'
  status: AgentStatus
  currentTask: string | null
  progress: number
  startedAt: string | null
  lastActivity: string
}

export interface TaskQueueItem {
  id: string
  type: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  createdAt: string
  startedAt: string | null
  completedAt: string | null
  agentId: string | null
  priority: 'low' | 'normal' | 'high'
}

export interface LLMMetrics {
  provider: string
  model: string
  totalRequests: number
  successRate: number
  avgLatency: number
  p95Latency: number
  p99Latency: number
  tokensUsed: number
  costEstimate: number
  lastHourRequests: number[]
}

export interface ResourceMetrics {
  cpu: {
    usage: number
    cores: number
  }
  memory: {
    used: number
    total: number
    percentage: number
  }
  disk: {
    used: number
    total: number
    percentage: number
  }
}

export interface AgentSystemStatus {
  agents: Agent[]
  taskQueue: TaskQueueItem[]
  llmMetrics: LLMMetrics
  resources: ResourceMetrics
  systemHealth: 'healthy' | 'degraded' | 'unhealthy'
  uptime: number
}
