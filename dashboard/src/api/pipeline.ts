/**
 * Pipeline API - 流水线与 RD Loop API
 *
 * 对应后端: src/iqfmp/api/pipeline/router.py
 */

import { api } from './client'

// ============== Pipeline Types ==============

export type PipelineStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

export type PipelineType = 'factor_mining' | 'evaluation' | 'backtest' | 'full'

export interface PipelineConfig {
  max_iterations?: number
  target_count?: number
  auto_evaluate?: boolean
  [key: string]: unknown
}

export interface PipelineRunRequest {
  pipeline_type: PipelineType
  config?: PipelineConfig
}

export interface PipelineRunResponse {
  run_id: string
  status: PipelineStatus
  message: string
  created_at: string
}

export interface PipelineStatusResponse {
  run_id: string
  status: PipelineStatus
  progress: number
  current_step: string | null
  started_at: string | null
  completed_at: string | null
  error_message: string | null
  result: Record<string, unknown> | null
}

export interface PipelineListResponse {
  runs: PipelineStatusResponse[]
  total: number
}

// ============== RD Loop Types ==============

export type RDLoopPhase =
  | 'initialization'
  | 'hypothesis_generation'
  | 'factor_coding'
  | 'evaluation'
  | 'benchmark'
  | 'selection'
  | 'completed'
  | 'failed'
  | 'idle'

export type HypothesisFamily =
  | 'momentum'
  | 'mean_reversion'
  | 'volatility'
  | 'volume'
  | 'price_action'
  | 'trend'
  | 'oscillator'
  | 'custom'

export interface RDLoopConfigRequest {
  max_iterations?: number
  max_hypotheses_per_iteration?: number
  target_core_factors?: number
  ic_threshold?: number
  ir_threshold?: number
  novelty_threshold?: number
  run_benchmark?: boolean
  enable_combination?: boolean
  focus_families?: HypothesisFamily[]
}

export interface RDLoopRunRequest {
  config?: RDLoopConfigRequest
  data_source?: string
}

export interface RDLoopRunResponse {
  run_id: string
  status: string
  message: string
  created_at: string
}

export interface RDLoopStateResponse {
  run_id: string
  phase: RDLoopPhase
  iteration: number
  total_hypotheses_tested: number
  core_factors_count: number
  core_factors: string[]
  is_running: boolean
  stop_requested: boolean
}

export interface RDLoopIterationResult {
  iteration: number
  hypotheses_tested: number
  factors_passed: number
  best_ic: number
  best_sharpe: number
  duration_seconds: number
}

export interface RDLoopStatisticsResponse {
  run_id: string
  state: RDLoopStateResponse
  hypothesis_stats: Record<string, unknown>
  ledger_stats: Record<string, unknown>
  benchmark_top_factors: Array<Record<string, unknown>>
  iteration_results: RDLoopIterationResult[]
}

export interface RDLoopCoreFactorResponse {
  name: string
  family: string
  code: string
  metrics: Record<string, number>
  hypothesis: string
}

export interface RDLoopRunInfo {
  run_id: string
  status: string
  phase: RDLoopPhase
  iteration: number
  core_factors_count: number
  error?: string
  created_at?: string
}

export interface RDLoopRunsListResponse {
  runs: RDLoopRunInfo[]
  total: number
}

// ============== WebSocket Types ==============

export type PipelineWSMessageType =
  | 'status'
  | 'progress'
  | 'rd_loop_phase'
  | 'rd_loop_progress'
  | 'rd_loop_complete'
  | 'rd_loop_error'
  | 'pong'

export interface PipelineWSMessage {
  type: PipelineWSMessageType
  data: Record<string, unknown>
}

// ============== API ==============

export const pipelineApi = {
  // ============== Pipeline Operations ==============

  /**
   * 启动流水线
   */
  run: (data: PipelineRunRequest) =>
    api.post<PipelineRunResponse>('/pipeline/run', data),

  /**
   * 获取流水线状态
   */
  getStatus: (runId: string) =>
    api.get<PipelineStatusResponse>(`/pipeline/${runId}/status`),

  /**
   * 获取流水线列表
   */
  listRuns: (params?: { status?: PipelineStatus }) =>
    api.get<PipelineListResponse>('/pipeline/runs', params),

  /**
   * 取消流水线
   */
  cancel: (runId: string) =>
    api.post<{ message: string; run_id: string }>(`/pipeline/${runId}/cancel`, {}),

  // ============== RD Loop Operations ==============

  /**
   * 启动 RD Loop
   */
  runRDLoop: (data: RDLoopRunRequest) =>
    api.post<RDLoopRunResponse>('/pipeline/rd-loop/run', data),

  /**
   * 获取 RD Loop 状态
   */
  getRDLoopState: (runId: string) =>
    api.get<RDLoopStateResponse>(`/pipeline/rd-loop/${runId}/state`),

  /**
   * 获取 RD Loop 统计信息
   */
  getRDLoopStatistics: (runId: string) =>
    api.get<RDLoopStatisticsResponse>(`/pipeline/rd-loop/${runId}/statistics`),

  /**
   * 获取 RD Loop 发现的核心因子
   */
  getRDLoopFactors: (runId: string) =>
    api.get<RDLoopCoreFactorResponse[]>(`/pipeline/rd-loop/${runId}/factors`),

  /**
   * 停止 RD Loop
   */
  stopRDLoop: (runId: string) =>
    api.post<{ message: string; run_id: string }>(`/pipeline/rd-loop/${runId}/stop`, {}),

  /**
   * 获取当前 RD Loop 状态 (监控仪表板用)
   */
  getCurrentRDLoopState: () =>
    api.get<{
      run_id: string | null
      is_running: boolean
      phase: RDLoopPhase
      iteration: number
      total_hypotheses_tested: number
      core_factors_count: number
    }>('/pipeline/rdloop/state'),

  /**
   * 获取 RD Loop 运行历史
   */
  listRDLoopRuns: () =>
    api.get<RDLoopRunsListResponse>('/pipeline/rd-loop/runs'),

  // ============== WebSocket ==============

  /**
   * 创建流水线 WebSocket 连接
   */
  createWebSocket: (runId: string): WebSocket => {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsHost = window.location.host
    return new WebSocket(`${wsProtocol}//${wsHost}/api/v1/pipeline/${runId}/ws`)
  },

  /**
   * 流水线 WebSocket 消息处理工具
   */
  wsUtils: {
    /**
     * 发送 ping 消息
     */
    ping: (ws: WebSocket): void => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }))
      }
    },

    /**
     * 请求状态更新
     */
    requestStatus: (ws: WebSocket): void => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'get_status' }))
      }
    },

    /**
     * 解析 WebSocket 消息
     */
    parseMessage: (event: MessageEvent): PipelineWSMessage | null => {
      try {
        return JSON.parse(event.data) as PipelineWSMessage
      } catch {
        console.error('Failed to parse WebSocket message:', event.data)
        return null
      }
    },
  },
}
