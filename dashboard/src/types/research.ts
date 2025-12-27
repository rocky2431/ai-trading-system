/**
 * Research Ledger 相关类型定义
 * 用于追踪因子研究试验和防止过拟合
 */

export type TrialStatus = 'passed' | 'failed' | 'inconclusive'

export interface TrialMetrics {
  ic: number
  icir: number
  sharpe: number
  maxDrawdown: number
  stability: number
}

export interface Trial {
  id: string
  trialNumber: number           // 全局试验序号
  factorId: string
  factorName: string
  hypothesis: string            // 研究假设
  status: TrialStatus
  metrics: TrialMetrics
  adjustedSharpe: number        // Deflated Sharpe Ratio
  threshold: number             // 当时的动态阈值
  passedThreshold: boolean
  createdAt: string
  duration: number              // 试验时长（秒）
  notes: string | null
}

export interface ThresholdHistory {
  timestamp: string
  trialCount: number
  threshold: number
}

export interface ThresholdConfig {
  baseSharpeThreshold: number
  confidenceLevel: number
  minTrialsForAdjustment: number
}

export interface ThresholdFormula {
  name: string
  description: string
  reference: string
  equation: string
  components: {
    expectedMax: string
    confidenceMultiplier: string
    adjustment: string
  }
}

export interface ThresholdDetails {
  currentThreshold: number
  nTrials: number
  config: ThresholdConfig
  formula: ThresholdFormula
}

export interface ResearchStats {
  totalTrials: number
  passedTrials: number
  failedTrials: number
  inconclusiveTrials: number
  passRate: number
  currentThreshold: number
  averageIC: number
  averageSharpe: number
  bestTrial: Trial | null
  worstTrial: Trial | null
  trialsByMonth: { month: string; count: number; passed: number }[]
}

export interface OverfittingRisk {
  level: 'low' | 'medium' | 'high' | 'critical'
  score: number                 // 0-100
  factors: {
    multipleTestingPenalty: number
    dataSnopingRisk: number
    parameterSensitivity: number
    outOfSampleDegradation: number
  }
  recommendation: string
}

export interface ApprovalRecord {
  requestId: string
  factorName: string
  status: 'approved' | 'rejected' | 'pending' | 'timeout'
  reviewer: string | null
  reason: string | null
  decidedAt: string | null
  createdAt: string
}

export interface ResearchLedgerData {
  trials: Trial[]
  stats: ResearchStats
  thresholdHistory: ThresholdHistory[]
  thresholdDetails: ThresholdDetails
  overfittingRisk: OverfittingRisk
  approvalRecords: ApprovalRecord[]
}
