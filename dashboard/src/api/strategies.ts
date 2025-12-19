/**
 * Strategies API - 策略管理 API
 *
 * 对应后端: src/iqfmp/api/strategies/router.py
 *
 * 注意: 这是独立的策略 API，使用 /strategies 路由
 * 而不是 /backtest/strategies (已废弃)
 */

import { api } from './client'

// ============== Types ==============

export interface StrategyCreateRequest {
  name: string
  description?: string
  factor_ids?: string[]
  factor_weights?: Record<string, number>
  code: string
  config?: Record<string, unknown>
}

export interface StrategyUpdateRequest {
  name?: string
  description?: string
  factor_ids?: string[]
  factor_weights?: Record<string, number>
  code?: string
  config?: Record<string, unknown>
  status?: string
}

export interface StrategyResponse {
  id: string
  name: string
  description: string | null
  factor_ids: string[]
  factor_weights: Record<string, number> | null
  code: string
  config: Record<string, unknown> | null
  status: string
  created_at: string | null
  updated_at: string | null
}

export interface StrategyListResponse {
  strategies: StrategyResponse[]
  total: number
  page: number
  page_size: number
}

export interface BacktestRequest {
  start_date: string
  end_date: string
  initial_capital?: number
  commission?: number
}

export interface BacktestResultResponse {
  id: string
  strategy_id: string
  start_date: string
  end_date: string
  total_return: number
  sharpe_ratio: number | null
  max_drawdown: number | null
  win_rate: number | null
  profit_factor: number | null
  trade_count: number | null
  created_at: string | null
}

export interface BacktestListResponse {
  results: BacktestResultResponse[]
  total: number
}

// ============== API ==============

export const strategiesApi = {
  /**
   * 创建策略
   */
  create: (data: StrategyCreateRequest) =>
    api.post<StrategyResponse>('/strategies', data),

  /**
   * 获取策略列表
   */
  list: (params?: { page?: number; page_size?: number; status?: string }) =>
    api.get<StrategyListResponse>('/strategies', params),

  /**
   * 获取单个策略
   */
  get: (strategyId: string) =>
    api.get<StrategyResponse>(`/strategies/${strategyId}`),

  /**
   * 更新策略
   */
  update: (strategyId: string, data: StrategyUpdateRequest) =>
    api.patch<StrategyResponse>(`/strategies/${strategyId}`, data),

  /**
   * 删除策略
   */
  delete: (strategyId: string) =>
    api.delete<void>(`/strategies/${strategyId}`),

  /**
   * 运行策略回测
   */
  runBacktest: (strategyId: string, data: BacktestRequest) =>
    api.post<BacktestResultResponse>(`/strategies/${strategyId}/backtest`, data),

  /**
   * 获取策略的回测结果列表
   */
  listBacktests: (strategyId: string) =>
    api.get<BacktestListResponse>(`/strategies/${strategyId}/backtests`),
}
