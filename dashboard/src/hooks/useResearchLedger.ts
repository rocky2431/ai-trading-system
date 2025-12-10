/**
 * Research Ledger Hook - 提供研究账本数据
 */

import { useState, useEffect, useMemo } from 'react'
import type { Trial, ResearchLedgerData, ThresholdHistory, ResearchStats, OverfittingRisk } from '@/types/research'

function generateMockTrials(): Trial[] {
  const factors = [
    { id: 'factor-001', name: 'RSI Momentum' },
    { id: 'factor-002', name: 'Funding Rate Carry' },
    { id: 'factor-003', name: 'Realized Volatility' },
    { id: 'factor-004', name: 'Volume Imbalance' },
    { id: 'factor-005', name: 'MACD Crossover' },
    { id: 'factor-006', name: 'Bollinger Width' },
    { id: 'factor-007', name: 'OI Change' },
    { id: 'factor-008', name: 'Price Momentum' },
  ]

  const hypotheses = [
    'Short-term momentum predicts returns',
    'Funding rate carries alpha in crypto',
    'Low volatility predicts higher returns',
    'Volume imbalance indicates buying pressure',
    'Technical crossovers capture trends',
    'Bollinger width captures regime changes',
    'Open interest changes predict moves',
    'Price momentum persists short-term',
  ]

  const trials: Trial[] = []
  const baseDate = new Date('2024-06-01')

  for (let i = 0; i < 45; i++) {
    const factor = factors[i % factors.length]
    const trialDate = new Date(baseDate)
    trialDate.setDate(baseDate.getDate() + Math.floor(i * 4.5))

    const ic = Math.random() * 0.08 - 0.01
    const icir = ic * (3 + Math.random() * 2)
    const sharpe = 0.3 + Math.random() * 2.2
    const threshold = 1.0 + (i * 0.015)
    const adjustedSharpe = sharpe * (1 - i * 0.008)

    let status: Trial['status']
    if (adjustedSharpe >= threshold) {
      status = 'passed'
    } else if (adjustedSharpe >= threshold * 0.8) {
      status = 'inconclusive'
    } else {
      status = 'failed'
    }

    trials.push({
      id: `trial-${String(i + 1).padStart(3, '0')}`,
      trialNumber: i + 1,
      factorId: factor.id,
      factorName: factor.name,
      hypothesis: hypotheses[i % hypotheses.length],
      status,
      metrics: {
        ic,
        icir,
        sharpe,
        maxDrawdown: 5 + Math.random() * 25,
        stability: 0.3 + Math.random() * 0.6,
      },
      adjustedSharpe,
      threshold,
      passedThreshold: adjustedSharpe >= threshold,
      createdAt: trialDate.toISOString(),
      duration: 300 + Math.floor(Math.random() * 1800),
      notes: i % 5 === 0 ? 'Promising results, needs more validation' : null,
    })
  }

  return trials.reverse()
}

function generateThresholdHistory(trials: Trial[]): ThresholdHistory[] {
  const history: ThresholdHistory[] = []
  const sortedTrials = [...trials].sort(
    (a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
  )

  sortedTrials.forEach((trial, index) => {
    if (index % 5 === 0 || index === sortedTrials.length - 1) {
      history.push({
        timestamp: trial.createdAt,
        trialCount: index + 1,
        threshold: trial.threshold,
      })
    }
  })

  return history
}

function calculateStats(trials: Trial[]): ResearchStats {
  const passed = trials.filter(t => t.status === 'passed')
  const failed = trials.filter(t => t.status === 'failed')
  const inconclusive = trials.filter(t => t.status === 'inconclusive')

  const avgIC = trials.reduce((sum, t) => sum + t.metrics.ic, 0) / trials.length
  const avgSharpe = trials.reduce((sum, t) => sum + t.metrics.sharpe, 0) / trials.length

  const sortedBySharpe = [...trials].sort((a, b) => b.adjustedSharpe - a.adjustedSharpe)

  // Group by month
  const byMonth: Record<string, { count: number; passed: number }> = {}
  trials.forEach(trial => {
    const month = trial.createdAt.substring(0, 7)
    if (!byMonth[month]) byMonth[month] = { count: 0, passed: 0 }
    byMonth[month].count++
    if (trial.status === 'passed') byMonth[month].passed++
  })

  const trialsByMonth = Object.entries(byMonth)
    .map(([month, data]) => ({ month, ...data }))
    .sort((a, b) => a.month.localeCompare(b.month))

  return {
    totalTrials: trials.length,
    passedTrials: passed.length,
    failedTrials: failed.length,
    inconclusiveTrials: inconclusive.length,
    passRate: (passed.length / trials.length) * 100,
    currentThreshold: trials[0]?.threshold || 1.0,
    averageIC: avgIC,
    averageSharpe: avgSharpe,
    bestTrial: sortedBySharpe[0] || null,
    worstTrial: sortedBySharpe[sortedBySharpe.length - 1] || null,
    trialsByMonth,
  }
}

function calculateOverfittingRisk(trials: Trial[], stats: ResearchStats): OverfittingRisk {
  const multipleTestingPenalty = Math.min(100, trials.length * 2)
  const dataSnopingRisk = Math.min(100, (1 - stats.passRate / 100) * 80 + 20)
  const parameterSensitivity = 30 + Math.random() * 40
  const outOfSampleDegradation = 20 + Math.random() * 50

  const score = (multipleTestingPenalty + dataSnopingRisk + parameterSensitivity + outOfSampleDegradation) / 4

  let level: OverfittingRisk['level']
  let recommendation: string

  if (score < 30) {
    level = 'low'
    recommendation = 'Research methodology appears sound. Continue with caution.'
  } else if (score < 50) {
    level = 'medium'
    recommendation = 'Consider reducing trial frequency. Focus on out-of-sample validation.'
  } else if (score < 70) {
    level = 'high'
    recommendation = 'High risk of overfitting. Implement stricter validation protocols.'
  } else {
    level = 'critical'
    recommendation = 'Critical overfitting risk. Pause new trials and review methodology.'
  }

  return {
    level,
    score,
    factors: {
      multipleTestingPenalty,
      dataSnopingRisk,
      parameterSensitivity,
      outOfSampleDegradation,
    },
    recommendation,
  }
}

export function useResearchLedger() {
  const [loading, setLoading] = useState(true)
  const [error] = useState<Error | null>(null)

  const [trials] = useState<Trial[]>(() => generateMockTrials())

  useEffect(() => {
    const timer = setTimeout(() => setLoading(false), 500)
    return () => clearTimeout(timer)
  }, [])

  const data = useMemo<ResearchLedgerData>(() => {
    const stats = calculateStats(trials)
    const thresholdHistory = generateThresholdHistory(trials)
    const overfittingRisk = calculateOverfittingRisk(trials, stats)

    return {
      trials,
      stats,
      thresholdHistory,
      overfittingRisk,
    }
  }, [trials])

  return { data, loading, error }
}
