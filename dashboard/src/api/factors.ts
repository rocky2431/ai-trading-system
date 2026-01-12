/**
 * Factors API - 因子相关 API 调用
 */

import { api } from './client'

// API Response Types (匹配后端 schema)
export interface FactorMetricsResponse {
  ic_mean: number
  ic_std: number
  ir: number
  sharpe: number
  max_drawdown: number
  turnover: number
  win_rate: number | null  // Win rate from backtest (0.0-1.0), null if not available
  ic_by_split: Record<string, number>
  sharpe_by_split: Record<string, number>
}

export interface StabilityResponse {
  time_stability: Record<string, number>
  market_stability: Record<string, number>
  regime_stability: Record<string, number>
}

export interface FactorResponse {
  id: string
  name: string
  family: string[]
  code: string
  code_hash: string
  target_task: string
  status: string
  metrics: FactorMetricsResponse | null
  stability: StabilityResponse | null
  cluster_id: string | null
  experiment_number: number
  created_at: string
}

export interface FactorListResponse {
  factors: FactorResponse[]
  total: number
  page: number
  page_size: number
}

export interface FactorStatsResponse {
  total_factors: number
  by_status: Record<string, number>
  total_trials: number
  current_threshold: number
  // Extended fields for monitoring dashboard (aligned with backend FactorStatsResponse)
  evaluated_count: number
  pass_rate: number
  avg_ic: number
  avg_sharpe: number
  pending_count: number
}

export interface FactorEvaluateResponse {
  factor_id: string
  metrics: FactorMetricsResponse
  stability: StabilityResponse
  passed_threshold: boolean
  experiment_number: number
}

// ============== Mining Task Types ==============

export interface MiningTaskStatus {
  id: string
  name: string
  description: string
  factor_families: string[]
  target_count: number
  generated_count: number
  passed_count: number
  failed_count: number
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  error_message: string | null
  created_at: string
  started_at: string | null
  completed_at: string | null
}

export interface MiningTaskListResponse {
  tasks: MiningTaskStatus[]
  total: number
}

export interface MiningTaskCreateRequest {
  name: string
  description?: string
  factor_families?: string[]
  target_count?: number
  auto_evaluate?: boolean
}

export interface MiningTaskCreateResponse {
  success: boolean
  message: string
  task_id: string | null
}

export interface MiningTaskCancelResponse {
  success: boolean
  message: string
}

// ============== Factor Library Types ==============

export interface FactorLibraryStats {
  total_factors: number
  core_factors: number
  candidate_factors: number
  rejected_factors: number
  redundant_factors: number
  by_family: Record<string, number>
  avg_sharpe: number
  avg_ic: number
  best_factor_id: string | null
  best_sharpe: number
}

export interface FactorCompareResponse {
  factors: FactorResponse[]
  correlation_matrix: Record<string, Record<string, number>>
  ranking: string[]
}

// API Functions
export const factorsApi = {
  /**
   * 生成新因子 (使用 LLM)
   */
  generate: (data: {
    description: string
    family?: string[]
    target_task?: string
  }) => api.post<FactorResponse>('/factors/generate', data),

  /**
   * 创建因子 (直接提供代码)
   */
  create: (data: {
    name: string
    family?: string[]
    code: string
    target_task?: string
  }) => api.post<FactorResponse>('/factors', data),

  /**
   * 获取因子列表
   */
  list: (params?: {
    page?: number
    page_size?: number
    family?: string
    status?: string
  }) => api.get<FactorListResponse>('/factors', params),

  /**
   * 获取单个因子
   */
  get: (factorId: string) => api.get<FactorResponse>(`/factors/${factorId}`),

  /**
   * 获取统计信息
   */
  stats: () => api.get<FactorStatsResponse>('/factors/stats'),

  /**
   * 评估因子
   */
  evaluate: (factorId: string, data: {
    splits?: string[]
    market_splits?: string[]
    frequency_splits?: string[]
    symbol?: string
    timeframe?: string
  }) => api.post<FactorEvaluateResponse>(`/factors/${factorId}/evaluate`, data),

  /**
   * 更新因子状态
   */
  updateStatus: (factorId: string, status: string) =>
    api.put<FactorResponse>(`/factors/${factorId}/status`, { status }),

  /**
   * 删除因子
   */
  delete: (factorId: string) => api.delete<void>(`/factors/${factorId}`),

  // ============== Mining Task API ==============

  /**
   * 创建挖掘任务
   */
  createMiningTask: (data: MiningTaskCreateRequest) =>
    api.post<MiningTaskCreateResponse>('/factors/mining', data),

  /**
   * 获取挖掘任务列表
   */
  listMiningTasks: (params?: { status?: string; limit?: number }) =>
    api.get<MiningTaskListResponse>('/factors/mining', params),

  /**
   * 获取单个挖掘任务
   */
  getMiningTask: (taskId: string) =>
    api.get<MiningTaskStatus>(`/factors/mining/${taskId}`),

  /**
   * 取消挖掘任务
   */
  cancelMiningTask: (taskId: string) =>
    api.delete<MiningTaskCancelResponse>(`/factors/mining/${taskId}`),

  // ============== Factor Library API ==============

  /**
   * 获取因子库统计
   */
  getLibraryStats: () =>
    api.get<FactorLibraryStats>('/factors/library/stats'),

  /**
   * 比较多个因子
   */
  compare: (factorIds: string[]) =>
    api.post<FactorCompareResponse>('/factors/compare', { factor_ids: factorIds }),
}
