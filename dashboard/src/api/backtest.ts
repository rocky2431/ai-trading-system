/**
 * Backtest API - 策略回测相关 API 调用
 *
 * 注意: Strategy 相关 API 已迁移至独立的 strategies.ts 模块
 * 本文件中的 Strategy API 已标记为 @deprecated
 * 请使用 import { strategiesApi } from './strategies'
 */

import { api } from './client'

// ============== Strategy Types (已废弃，请使用 strategies.ts) ==============

export interface StrategyResponse {
  id: string
  name: string
  description: string
  factor_ids: string[]
  weighting_method: string
  rebalance_frequency: string
  universe: string
  custom_universe: string[]
  long_only: boolean
  max_positions: number
  status: string
  created_at: string
  updated_at: string
}

export interface StrategyListResponse {
  strategies: StrategyResponse[]
  total: number
  page: number
  page_size: number
}

export interface StrategyCreateRequest {
  name: string
  description?: string
  factor_ids?: string[]
  weighting_method?: string
  rebalance_frequency?: string
  universe?: string
  custom_universe?: string[]
  long_only?: boolean
  max_positions?: number
}

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

// ============== API Functions ==============

export const backtestApi = {
  // ============== Strategy APIs (已废弃) ==============
  // @deprecated 请使用 strategiesApi.create() 替代
  /**
   * @deprecated 已废弃，请使用 strategiesApi.create()
   * 路由已迁移至 /strategies
   */
  createStrategy: (data: StrategyCreateRequest) =>
    api.post<StrategyResponse>('/backtest/strategies', data),

  /**
   * @deprecated 已废弃，请使用 strategiesApi.list()
   * 路由已迁移至 /strategies
   */
  listStrategies: (params?: { page?: number; page_size?: number; status?: string }) =>
    api.get<StrategyListResponse>('/backtest/strategies', params),

  /**
   * @deprecated 已废弃，请使用 strategiesApi.get()
   * 路由已迁移至 /strategies
   */
  getStrategy: (strategyId: string) =>
    api.get<StrategyResponse>(`/backtest/strategies/${strategyId}`),

  /**
   * @deprecated 已废弃，请使用 strategiesApi.delete()
   * 路由已迁移至 /strategies
   */
  deleteStrategy: (strategyId: string) =>
    api.delete<GenericResponse>(`/backtest/strategies/${strategyId}`),

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
}
