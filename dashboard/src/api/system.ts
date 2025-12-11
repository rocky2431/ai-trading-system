/**
 * System API - 系统状态 API
 */

import { api } from './client'

// API Response Types
export interface AgentResponse {
  id: string
  name: string
  type: 'factor_generation' | 'evaluation' | 'strategy' | 'backtest'
  status: 'idle' | 'running' | 'paused' | 'error'
  current_task: string | null
  progress: number
  started_at: string | null
  last_activity: string | null
}

export interface TaskQueueItemResponse {
  id: string
  type: 'factor_generation' | 'evaluation' | 'strategy' | 'backtest'
  status: 'pending' | 'running' | 'completed' | 'failed'
  created_at: string
  started_at: string | null
  completed_at: string | null
  agent_id: string | null
  priority: 'high' | 'normal' | 'low'
}

export interface LLMMetricsResponse {
  provider: string
  model: string
  total_requests: number
  success_rate: number
  avg_latency: number
  p95_latency: number
  p99_latency: number
  tokens_used: number
  cost_estimate: number
  last_hour_requests: number[]
}

export interface CPUMetrics {
  usage: number
  cores: number
}

export interface MemoryMetrics {
  used: number
  total: number
  percentage: number
}

export interface DiskMetrics {
  used: number
  total: number
  percentage: number
}

export interface ResourceMetricsResponse {
  cpu: CPUMetrics
  memory: MemoryMetrics
  disk: DiskMetrics
}

export interface SystemStatusResponse {
  agents: AgentResponse[]
  task_queue: TaskQueueItemResponse[]
  llm_metrics: LLMMetricsResponse
  resources: ResourceMetricsResponse
  system_health: 'healthy' | 'degraded' | 'unhealthy'
  uptime: number
}

// Agent Config Types
export type AgentType = 'factor_generation' | 'evaluation' | 'strategy' | 'backtest'

export interface AgentConfigResponse {
  id: string
  agent_type: AgentType
  name: string
  description: string | null
  system_prompt: string | null
  user_prompt_template: string | null
  examples: string | null
  config: Record<string, unknown> | null
  is_enabled: boolean
  created_at: string | null
  updated_at: string | null
}

export interface AgentConfigListResponse {
  configs: AgentConfigResponse[]
  total: number
}

export interface AgentConfigUpdateRequest {
  name?: string
  description?: string
  system_prompt?: string
  user_prompt_template?: string
  examples?: string
  config?: Record<string, unknown>
  is_enabled?: boolean
}

export interface AgentConfigOperationResponse {
  success: boolean
  message: string
  config?: AgentConfigResponse
}

// System API
export const systemApi = {
  // 获取完整系统状态
  getStatus: () => api.get<SystemStatusResponse>('/system/status'),

  // 获取资源指标
  getResources: () => api.get<ResourceMetricsResponse>('/system/resources'),

  // 获取 LLM 指标
  getLLMMetrics: () => api.get<LLMMetricsResponse>('/system/llm'),

  // Agent Config APIs
  getAgentConfigs: () => api.get<AgentConfigListResponse>('/system/agent-configs'),
  getAgentConfig: (agentType: AgentType) => api.get<AgentConfigResponse>(`/system/agent-configs/${agentType}`),
  updateAgentConfig: (agentType: AgentType, data: AgentConfigUpdateRequest) =>
    api.put<AgentConfigOperationResponse>(`/system/agent-configs/${agentType}`, data),
  initAgentConfigs: () => api.post<AgentConfigOperationResponse>('/system/agent-configs/init', {}),
}
