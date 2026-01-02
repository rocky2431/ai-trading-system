/**
 * Factor 相关类型定义
 */

export type FactorFamily =
  | 'momentum'
  | 'value'
  | 'volatility'
  | 'liquidity'
  | 'sentiment'
  | 'fundamental'

// Status values aligned with backend: src/iqfmp/api/factors/schemas.py
export type FactorStatus =
  | 'candidate'   // Newly created, pending evaluation
  | 'rejected'    // Failed evaluation
  | 'core'        // Passed evaluation, included in strategy
  | 'redundant'   // Passed but redundant with existing factors

export interface FactorMetrics {
  ic: number                    // Information Coefficient
  icir: number                  // IC Information Ratio
  sharpe: number                // Sharpe Ratio
  maxDrawdown: number           // Maximum Drawdown %
  winRate: number               // Win Rate %
  turnover: number              // Turnover Rate %
  stability: number             // Stability Score (0-1)
}

export interface FactorEvaluation {
  id: string
  factorId: string
  evaluatedAt: string
  metrics: FactorMetrics
  timeSlice: string             // e.g., "2023-01 to 2024-12"
  marketSlice: string           // e.g., "BTC/ETH + Altcoins"
  notes: string | null
}

export interface Factor {
  id: string
  name: string
  family: FactorFamily
  status: FactorStatus
  description: string
  code: string                  // Factor calculation code
  createdAt: string
  updatedAt: string
  authorId: string
  authorName: string
  latestMetrics: FactorMetrics | null
  evaluationCount: number
  tags: string[]
}

export interface FactorFilter {
  family?: FactorFamily | 'all'
  status?: FactorStatus | 'all'
  search?: string
  sortBy?: 'name' | 'ic' | 'sharpe' | 'stability' | 'createdAt'
  sortOrder?: 'asc' | 'desc'
}

export interface FactorListResponse {
  factors: Factor[]
  total: number
  page: number
  pageSize: number
}
