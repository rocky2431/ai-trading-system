/**
 * Factors Hook - 提供因子列表和筛选功能
 */

import { useState, useEffect, useMemo } from 'react'
import type { Factor, FactorFilter, FactorFamily, FactorStatus } from '@/types/factor'

const mockFactors: Factor[] = [
  {
    id: 'factor-001',
    name: 'RSI Momentum',
    family: 'momentum',
    status: 'approved',
    description: 'Relative Strength Index based momentum factor with 14-day lookback',
    code: 'def rsi_momentum(close, period=14):\n    delta = close.diff()\n    gain = delta.where(delta > 0, 0)\n    loss = -delta.where(delta < 0, 0)\n    avg_gain = gain.rolling(period).mean()\n    avg_loss = loss.rolling(period).mean()\n    rs = avg_gain / avg_loss\n    return 100 - (100 / (1 + rs))',
    createdAt: '2024-11-15T10:30:00Z',
    updatedAt: '2024-12-05T14:20:00Z',
    authorId: 'user-001',
    authorName: 'Alice Chen',
    latestMetrics: {
      ic: 0.045,
      icir: 0.82,
      sharpe: 1.45,
      maxDrawdown: 12.3,
      winRate: 56.2,
      turnover: 25.5,
      stability: 0.78,
    },
    evaluationCount: 12,
    tags: ['momentum', 'technical', 'rsi'],
  },
  {
    id: 'factor-002',
    name: 'Funding Rate Carry',
    family: 'value',
    status: 'approved',
    description: 'Perpetual funding rate based carry factor for crypto futures',
    code: 'def funding_rate_carry(funding_rate, window=7):\n    return funding_rate.rolling(window).mean()',
    createdAt: '2024-10-20T08:15:00Z',
    updatedAt: '2024-12-01T09:45:00Z',
    authorId: 'user-002',
    authorName: 'Bob Wang',
    latestMetrics: {
      ic: 0.062,
      icir: 1.15,
      sharpe: 2.10,
      maxDrawdown: 8.5,
      winRate: 61.8,
      turnover: 15.2,
      stability: 0.85,
    },
    evaluationCount: 18,
    tags: ['carry', 'funding', 'crypto'],
  },
  {
    id: 'factor-003',
    name: 'Realized Volatility',
    family: 'volatility',
    status: 'evaluating',
    description: '20-day realized volatility using close-to-close returns',
    code: 'def realized_vol(close, window=20):\n    returns = close.pct_change()\n    return returns.rolling(window).std() * np.sqrt(252)',
    createdAt: '2024-12-01T11:00:00Z',
    updatedAt: '2024-12-08T16:30:00Z',
    authorId: 'user-001',
    authorName: 'Alice Chen',
    latestMetrics: {
      ic: 0.028,
      icir: 0.55,
      sharpe: 0.95,
      maxDrawdown: 18.7,
      winRate: 52.1,
      turnover: 32.8,
      stability: 0.62,
    },
    evaluationCount: 5,
    tags: ['volatility', 'risk'],
  },
  {
    id: 'factor-004',
    name: 'Volume Imbalance',
    family: 'liquidity',
    status: 'approved',
    description: 'Buy/sell volume imbalance indicator',
    code: 'def volume_imbalance(buy_vol, sell_vol, window=5):\n    imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol)\n    return imbalance.rolling(window).mean()',
    createdAt: '2024-09-10T14:00:00Z',
    updatedAt: '2024-11-28T10:15:00Z',
    authorId: 'user-003',
    authorName: 'Carol Li',
    latestMetrics: {
      ic: 0.038,
      icir: 0.72,
      sharpe: 1.28,
      maxDrawdown: 14.2,
      winRate: 54.8,
      turnover: 42.1,
      stability: 0.71,
    },
    evaluationCount: 22,
    tags: ['liquidity', 'volume', 'orderflow'],
  },
  {
    id: 'factor-005',
    name: 'Twitter Sentiment',
    family: 'sentiment',
    status: 'draft',
    description: 'Aggregated Twitter sentiment score for crypto assets',
    code: 'def twitter_sentiment(sentiment_scores, window=3):\n    return sentiment_scores.rolling(window).mean()',
    createdAt: '2024-12-05T09:30:00Z',
    updatedAt: '2024-12-05T09:30:00Z',
    authorId: 'user-002',
    authorName: 'Bob Wang',
    latestMetrics: null,
    evaluationCount: 0,
    tags: ['sentiment', 'social', 'twitter'],
  },
  {
    id: 'factor-006',
    name: 'MACD Crossover',
    family: 'momentum',
    status: 'rejected',
    description: 'MACD line crossover signal with histogram confirmation',
    code: 'def macd_crossover(close, fast=12, slow=26, signal=9):\n    ema_fast = close.ewm(span=fast).mean()\n    ema_slow = close.ewm(span=slow).mean()\n    macd = ema_fast - ema_slow\n    signal_line = macd.ewm(span=signal).mean()\n    return macd - signal_line',
    createdAt: '2024-08-15T16:00:00Z',
    updatedAt: '2024-10-20T11:45:00Z',
    authorId: 'user-001',
    authorName: 'Alice Chen',
    latestMetrics: {
      ic: 0.012,
      icir: 0.25,
      sharpe: 0.45,
      maxDrawdown: 28.5,
      winRate: 48.2,
      turnover: 55.3,
      stability: 0.35,
    },
    evaluationCount: 8,
    tags: ['momentum', 'macd', 'technical'],
  },
  {
    id: 'factor-007',
    name: 'Open Interest Change',
    family: 'sentiment',
    status: 'approved',
    description: 'Rate of change in open interest as sentiment indicator',
    code: 'def oi_change(open_interest, window=5):\n    return open_interest.pct_change(window)',
    createdAt: '2024-10-01T13:20:00Z',
    updatedAt: '2024-11-15T08:50:00Z',
    authorId: 'user-003',
    authorName: 'Carol Li',
    latestMetrics: {
      ic: 0.051,
      icir: 0.95,
      sharpe: 1.68,
      maxDrawdown: 10.8,
      winRate: 58.5,
      turnover: 28.4,
      stability: 0.81,
    },
    evaluationCount: 15,
    tags: ['sentiment', 'oi', 'derivatives'],
  },
  {
    id: 'factor-008',
    name: 'Bollinger Band Width',
    family: 'volatility',
    status: 'approved',
    description: 'Bollinger Band width as volatility measure',
    code: 'def bb_width(close, window=20, num_std=2):\n    sma = close.rolling(window).mean()\n    std = close.rolling(window).std()\n    upper = sma + num_std * std\n    lower = sma - num_std * std\n    return (upper - lower) / sma',
    createdAt: '2024-07-20T10:00:00Z',
    updatedAt: '2024-11-10T14:30:00Z',
    authorId: 'user-002',
    authorName: 'Bob Wang',
    latestMetrics: {
      ic: 0.033,
      icir: 0.68,
      sharpe: 1.15,
      maxDrawdown: 16.2,
      winRate: 53.5,
      turnover: 22.8,
      stability: 0.74,
    },
    evaluationCount: 20,
    tags: ['volatility', 'bollinger', 'technical'],
  },
]

export function useFactors(initialFilter?: FactorFilter) {
  const [factors] = useState<Factor[]>(mockFactors)
  const [filter, setFilter] = useState<FactorFilter>(initialFilter || {
    family: 'all',
    status: 'all',
    search: '',
    sortBy: 'createdAt',
    sortOrder: 'desc',
  })
  const [loading, setLoading] = useState(true)
  const [error] = useState<Error | null>(null)

  useEffect(() => {
    // Simulate loading
    const timer = setTimeout(() => setLoading(false), 500)
    return () => clearTimeout(timer)
  }, [])

  const filteredFactors = useMemo(() => {
    let result = [...factors]

    // Family filter
    if (filter.family && filter.family !== 'all') {
      result = result.filter(f => f.family === filter.family)
    }

    // Status filter
    if (filter.status && filter.status !== 'all') {
      result = result.filter(f => f.status === filter.status)
    }

    // Search filter
    if (filter.search) {
      const searchLower = filter.search.toLowerCase()
      result = result.filter(f =>
        f.name.toLowerCase().includes(searchLower) ||
        f.description.toLowerCase().includes(searchLower) ||
        f.tags.some(t => t.toLowerCase().includes(searchLower))
      )
    }

    // Sort
    if (filter.sortBy) {
      result.sort((a, b) => {
        let aVal: number | string
        let bVal: number | string

        switch (filter.sortBy) {
          case 'name':
            aVal = a.name
            bVal = b.name
            break
          case 'ic':
            aVal = a.latestMetrics?.ic ?? -999
            bVal = b.latestMetrics?.ic ?? -999
            break
          case 'sharpe':
            aVal = a.latestMetrics?.sharpe ?? -999
            bVal = b.latestMetrics?.sharpe ?? -999
            break
          case 'stability':
            aVal = a.latestMetrics?.stability ?? -999
            bVal = b.latestMetrics?.stability ?? -999
            break
          case 'createdAt':
          default:
            aVal = new Date(a.createdAt).getTime()
            bVal = new Date(b.createdAt).getTime()
        }

        if (typeof aVal === 'string') {
          return filter.sortOrder === 'asc'
            ? aVal.localeCompare(bVal as string)
            : (bVal as string).localeCompare(aVal)
        }
        return filter.sortOrder === 'asc' ? aVal - (bVal as number) : (bVal as number) - aVal
      })
    }

    return result
  }, [factors, filter])

  const stats = useMemo(() => {
    const byFamily: Record<FactorFamily, number> = {
      momentum: 0,
      value: 0,
      volatility: 0,
      liquidity: 0,
      sentiment: 0,
      fundamental: 0,
    }
    const byStatus: Record<FactorStatus, number> = {
      draft: 0,
      evaluating: 0,
      approved: 0,
      rejected: 0,
      archived: 0,
    }

    factors.forEach(f => {
      byFamily[f.family]++
      byStatus[f.status]++
    })

    return { byFamily, byStatus, total: factors.length }
  }, [factors])

  return {
    factors: filteredFactors,
    allFactors: factors,
    filter,
    setFilter,
    loading,
    error,
    stats,
  }
}
