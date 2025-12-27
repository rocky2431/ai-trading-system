/**
 * RL Training API - RL 训练 API 调用
 *
 * 对应后端: src/iqfmp/api/rl/router.py
 */

import { api } from './client'

// ============== Types ==============

export type RLTaskStatus =
  | 'pending'
  | 'started'
  | 'running'
  | 'success'
  | 'failed'
  | 'revoked'

export interface RLTrainingConfig {
  total_timesteps: number
  learning_rate: number
  order_amount: number
  time_per_step: number
  save_model: boolean
  model_path?: string
}

export interface RLTrainingRequest {
  train_data_path: string
  test_data_path: string
  config?: Partial<RLTrainingConfig>
  name?: string
}

export interface RLBacktestRequest {
  model_path: string
  data_path: string
  config?: Record<string, unknown>
  name?: string
}

export interface RLTaskResponse {
  task_id: string
  celery_task_id: string
  status: RLTaskStatus
  task_type: 'training' | 'backtest'
  name: string | null
  created_at: string
  started_at: string | null
  completed_at: string | null
  result: Record<string, unknown> | null
  error: string | null
}

export interface RLTaskListResponse {
  items: RLTaskResponse[]
  total: number
  page: number
  page_size: number
  has_next: boolean
}

export interface RLModelInfo {
  model_id: string
  path: string
  task_id: string
  created_at: string
  metrics: Record<string, unknown> | null
  config: Record<string, unknown> | null
}

export interface RLModelListResponse {
  models: RLModelInfo[]
  total: number
}

export interface RLStatsResponse {
  total_training_jobs: number
  successful_jobs: number
  failed_jobs: number
  running_jobs: number
  total_models: number
  average_training_time_seconds: number | null
}

// ============== API ==============

export const rlApi = {
  /**
   * 提交 RL 训练任务
   */
  submitTraining: (data: RLTrainingRequest) =>
    api.post<RLTaskResponse>('/rl/training', data),

  /**
   * 提交 RL 回测任务
   */
  submitBacktest: (data: RLBacktestRequest) =>
    api.post<RLTaskResponse>('/rl/backtest', data),

  /**
   * 获取任务列表
   */
  getTasks: (params?: {
    page?: number
    page_size?: number
    task_type?: 'training' | 'backtest'
    status?: RLTaskStatus
  }) =>
    api.get<RLTaskListResponse>('/rl/tasks', params),

  /**
   * 获取任务详情
   */
  getTask: (taskId: string) =>
    api.get<RLTaskResponse>(`/rl/tasks/${taskId}`),

  /**
   * 取消任务
   */
  cancelTask: (taskId: string) =>
    api.post<{ message: string; success: boolean }>(`/rl/tasks/${taskId}/cancel`),

  /**
   * 获取模型列表
   */
  getModels: () =>
    api.get<RLModelListResponse>('/rl/models'),

  /**
   * 获取统计信息
   */
  getStats: () =>
    api.get<RLStatsResponse>('/rl/stats'),
}
