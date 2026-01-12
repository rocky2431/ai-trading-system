/**
 * Backtest API - 回测相关 API 调用
 *
 * NOTE: Strategy 相关 API 已迁移至 strategies.ts
 * 请使用 import { strategiesApi } from './strategies'
 */

import { api } from './client'

// ============== Backtest Types ==============

export interface BacktestConfig {
  start_date: string
  end_date: string
  initial_capital?: number
  commission_rate?: number
  slippage?: number
  benchmark?: string
  risk_free_rate?: number
}

export interface BacktestMetrics {
  total_return: number
  annual_return: number
  sharpe_ratio: number
  sortino_ratio: number
  max_drawdown: number
  max_drawdown_duration: number
  win_rate: number
  profit_factor: number
  calmar_ratio: number
  volatility: number
  beta: number
  alpha: number
  information_ratio: number
  trade_count: number
  avg_trade_return: number
  avg_holding_period: number
}

export interface BacktestResponse {
  id: string
  strategy_id: string
  strategy_name: string
  name: string
  description: string
  config: BacktestConfig
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  metrics: BacktestMetrics | null
  error_message: string | null
  created_at: string
  started_at: string | null
  completed_at: string | null
}

export interface BacktestListResponse {
  backtests: BacktestResponse[]
  total: number
  page: number
  page_size: number
}

export interface BacktestEquityCurve {
  date: string
  equity: number
  drawdown: number
  benchmark_equity: number
}

export interface BacktestTrade {
  id: string
  symbol: string
  side: string
  entry_date: string
  entry_price: number
  exit_date: string | null
  exit_price: number | null
  quantity: number
  pnl: number
  pnl_pct: number
  holding_days: number
}

export interface BacktestDetailResponse {
  backtest: BacktestResponse
  equity_curve: BacktestEquityCurve[]
  trades: BacktestTrade[]
  monthly_returns: Record<string, number>
  factor_contributions: Record<string, number>
}

export interface BacktestStatsResponse {
  total_strategies: number
  total_backtests: number
  running_backtests: number
  completed_today: number
  avg_sharpe: number
  best_strategy_id: string | null
  best_sharpe: number
}

export interface GenericResponse {
  success: boolean
  message: string
}

// ============== Optimization Types ==============

// Type-safe union types for optimization configuration
export type OptimizationDirection = 'maximize' | 'minimize'
export type OptimizationMetricName = 'sharpe' | 'calmar' | 'total_return' | 'sortino' | 'ic'
export type SamplerType = 'tpe' | 'cmaes' | 'random' | 'grid'
export type PrunerType = 'median' | 'hyperband' | 'percentile' | 'none'

export interface OptimizationMetric {
  name: OptimizationMetricName
  direction: OptimizationDirection
}

export interface OptimizationConfig {
  n_trials: number
  n_jobs?: number
  timeout?: number | null
  metrics: OptimizationMetric[]
  sampler: SamplerType
  sampler_kwargs?: Record<string, unknown>
  pruner: PrunerType
  pruner_kwargs?: Record<string, unknown>
  custom_search_spaces?: Record<string, unknown>[]
  cross_validation_folds?: number
  walk_forward_enabled?: boolean
  lookahead_check?: boolean
  lookahead_mode?: string
}

export interface OptimizationTrialResult {
  trial_id: number
  params: Record<string, unknown>
  metrics: BacktestMetrics
  duration_seconds: number
  rank: number
  pruned: boolean
}

export interface OptimizationResponse {
  id: string
  strategy_id: string
  name: string
  description: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  current_trial: number
  total_trials: number
  best_trial_id: number | null
  best_params: Record<string, unknown> | null
  best_metrics: BacktestMetrics | null
  top_trials: OptimizationTrialResult[]
  error_message: string | null
  created_at: string
  started_at: string | null
  completed_at: string | null
}

export interface OptimizationDetailResponse {
  optimization: OptimizationResponse
  all_trials: OptimizationTrialResult[]
  param_importance: Record<string, number>
  convergence_history: { trial_id: number; value: number }[]
}

export interface OptimizationListResponse {
  optimizations: OptimizationResponse[]
  total: number
  page: number
  page_size: number
}

export interface OptimizationRequest {
  strategy_id: string
  backtest_config: BacktestConfig
  optimization_config: OptimizationConfig
  name?: string
  description?: string
}

// ============== API Functions ==============

export const backtestApi = {
  // ============== Backtest APIs ==============
  createBacktest: (data: {
    strategy_id: string
    config: BacktestConfig
    name?: string
    description?: string
  }) => api.post<GenericResponse>('/backtest/backtests', data),

  listBacktests: (params?: {
    strategy_id?: string
    status?: string
    page?: number
    page_size?: number
  }) => api.get<BacktestListResponse>('/backtest/backtests', params),

  getBacktest: (backtestId: string) =>
    api.get<BacktestResponse>(`/backtest/backtests/${backtestId}`),

  getBacktestDetail: (backtestId: string) =>
    api.get<BacktestDetailResponse>(`/backtest/backtests/${backtestId}/detail`),

  deleteBacktest: (backtestId: string) =>
    api.delete<GenericResponse>(`/backtest/backtests/${backtestId}`),

  // Stats API
  getStats: () => api.get<BacktestStatsResponse>('/backtest/stats'),

  // ============== Optimization APIs ==============

  /**
   * 创建参数优化任务
   */
  createOptimization: (data: OptimizationRequest) =>
    api.post<GenericResponse>('/backtest/optimizations', data),

  /**
   * 获取优化任务列表
   */
  listOptimizations: (params?: {
    strategy_id?: string
    status?: string
    page?: number
    page_size?: number
  }) => api.get<OptimizationListResponse>('/backtest/optimizations', params),

  /**
   * 获取单个优化任务
   */
  getOptimization: (optimizationId: string) =>
    api.get<OptimizationResponse>(`/backtest/optimizations/${optimizationId}`),

  /**
   * 获取优化任务详情（包含所有试验结果）
   */
  getOptimizationDetail: (optimizationId: string) =>
    api.get<OptimizationDetailResponse>(`/backtest/optimizations/${optimizationId}/detail`),

  /**
   * 取消运行中的优化任务
   */
  cancelOptimization: (optimizationId: string) =>
    api.post<GenericResponse>(`/backtest/optimizations/${optimizationId}/cancel`, {}),

  /**
   * 删除优化任务
   */
  deleteOptimization: (optimizationId: string) =>
    api.delete<GenericResponse>(`/backtest/optimizations/${optimizationId}`),
}
