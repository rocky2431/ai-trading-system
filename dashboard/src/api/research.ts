/**
 * Research API - 研究账本 API
 */

import { api } from './client'

// API Response Types
export interface TrialResponse {
  trial_id: string
  factor_name: string
  factor_family: string
  sharpe_ratio: number
  ic_mean: number
  ir: number
  max_drawdown: number
  win_rate: number
  created_at: string
  metadata: Record<string, unknown> | null
}

export interface LedgerListResponse {
  trials: TrialResponse[]
  total: number
  page: number
  page_size: number
}

export interface StatisticsResponse {
  total_trials: number
  mean_sharpe: number
  std_sharpe: number
  max_sharpe: number
  min_sharpe: number
  median_sharpe: number
}

export interface StatsResponse {
  overall: StatisticsResponse
  by_family: Record<string, StatisticsResponse> | null
}

export interface ThresholdHistoryItem {
  n_trials: number
  threshold: number
}

export interface ThresholdConfigResponse {
  base_sharpe_threshold: number
  confidence_level: number
  min_trials_for_adjustment: number
}

export interface ThresholdResponse {
  current_threshold: number
  n_trials: number
  config: ThresholdConfigResponse
  threshold_history: ThresholdHistoryItem[]
}

// Extended threshold details for UI
export interface ThresholdDetails {
  currentThreshold: number
  nTrials: number
  config: {
    baseSharpeThreshold: number
    confidenceLevel: number
    minTrialsForAdjustment: number
  }
  formula: {
    name: string
    description: string
    reference: string
    equation: string
  }
}

export interface LedgerListParams {
  page?: number
  page_size?: number
  family?: string
  min_sharpe?: number
  start_date?: string
  end_date?: string
}

// Research API
export const researchApi = {
  // 获取账本列表
  listLedger: (params: LedgerListParams = {}) =>
    api.get<LedgerListResponse>('/research/ledger', {
      page: params.page,
      page_size: params.page_size,
      family: params.family,
      min_sharpe: params.min_sharpe,
      start_date: params.start_date,
      end_date: params.end_date,
    }),

  // 获取统计信息
  getStats: (groupByFamily = false) =>
    api.get<StatsResponse>('/research/stats', {
      group_by_family: groupByFamily ? 1 : 0,
    }),

  // 获取阈值信息
  getThresholds: () => api.get<ThresholdResponse>('/research/thresholds'),
}
