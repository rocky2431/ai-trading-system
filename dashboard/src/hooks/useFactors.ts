/**
 * Factors Hook - Factor list and filtering functionality
 * Uses real API data from backend
 */

import { useState, useEffect, useMemo, useCallback } from 'react'
import { factorsApi } from '@/api'
import type { FactorResponse } from '@/api'
import type { Factor, FactorFilter, FactorFamily, FactorStatus } from '@/types/factor'

/**
 * Calculate split-to-mean IC ratio from IC values across data splits.
 * Ratio = (average IC across splits) / (overall IC mean)
 * Values close to 1.0 indicate that split ICs are consistent with overall mean.
 * NOTE: This is NOT a true stability metric (would require std/mean analysis).
 * Returns 0.7 as default when data is unavailable.
 */
function calculateStability(metrics: FactorResponse['metrics']): number {
  if (!metrics || !metrics.ic_by_split || metrics.ic_mean === 0) {
    return 0.7
  }
  const icValues = Object.values(metrics.ic_by_split)
  if (icValues.length === 0) {
    return 0.7
  }
  const avgIcAcrossSplits = icValues.reduce((a, b) => a + b, 0) / icValues.length
  return avgIcAcrossSplits / metrics.ic_mean || 0.7
}

// Transform API response to frontend Factor type
function apiToFactor(response: FactorResponse): Factor {
  return {
    id: response.id,
    name: response.name,
    family: (response.family?.[0] as FactorFamily) || 'momentum',
    status: mapApiStatus(response.status),
    description: `Factor for ${response.target_task}`,
    code: response.code,
    createdAt: response.created_at,
    updatedAt: response.created_at,
    authorId: 'system',
    authorName: 'System',
    latestMetrics: response.metrics ? {
      ic: response.metrics.ic_mean,
      icir: response.metrics.ir,
      sharpe: response.metrics.sharpe,
      maxDrawdown: response.metrics.max_drawdown * 100,
      // Use real win_rate from API, convert to percentage (0-100)
      // If null, display as null to indicate data unavailable
      winRate: response.metrics.win_rate !== null ? response.metrics.win_rate * 100 : null,
      turnover: response.metrics.turnover * 100,
      stability: calculateStability(response.metrics),
    } : null,
    evaluationCount: response.experiment_number,
    tags: response.family || [],
  }
}

// Validated status mapping - returns status if valid, defaults to 'candidate' for unknown values
function mapApiStatus(status: string): FactorStatus {
  const validStatuses: FactorStatus[] = ['candidate', 'rejected', 'core', 'redundant']
  if (validStatuses.includes(status as FactorStatus)) {
    return status as FactorStatus
  }
  return 'candidate'
}

export function useFactors(initialFilter?: FactorFilter) {
  const [factors, setFactors] = useState<Factor[]>([])
  const [filter, setFilter] = useState<FactorFilter>(initialFilter || {
    family: 'all',
    status: 'all',
    search: '',
    sortBy: 'createdAt',
    sortOrder: 'desc',
  })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  // Load factor data from API
  const loadFactors = useCallback(async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await factorsApi.list({
        page: 1,
        page_size: 100,
      })
      setFactors(response.factors.map(apiToFactor))
    } catch (err) {
      console.error('Failed to load factors:', err)
      setError(err instanceof Error ? err : new Error('Failed to load factors'))
      setFactors([])
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadFactors()
  }, [loadFactors])

  // Generate new factor via LLM
  const generateFactor = useCallback(async (description: string, family?: string[]) => {
    setLoading(true)
    try {
      const response = await factorsApi.generate({
        description,
        family,
      })
      const newFactor = apiToFactor(response)
      setFactors(prev => [newFactor, ...prev])
      return newFactor
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to generate factor'))
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  // Evaluate factor performance
  const evaluateFactor = useCallback(async (
    factorId: string,
    options?: {
      symbol?: string
      timeframe?: string
      splits?: string[]
    }
  ) => {
    try {
      const response = await factorsApi.evaluate(factorId, {
        splits: options?.splits ?? ['train', 'valid', 'test'],
        symbol: options?.symbol,
        timeframe: options?.timeframe,
      })
      // Reload factor list to get updated metrics
      await loadFactors()
      return response
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to evaluate factor'))
      throw err
    }
  }, [loadFactors])

  // Update factor status (candidate/core/rejected/redundant)
  const updateFactorStatus = useCallback(async (factorId: string, status: string) => {
    try {
      await factorsApi.updateStatus(factorId, status)
      await loadFactors()
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to update factor status'))
      throw err
    }
  }, [loadFactors])

  // Delete factor
  const deleteFactor = useCallback(async (factorId: string) => {
    try {
      await factorsApi.delete(factorId)
      setFactors(prev => prev.filter(f => f.id !== factorId))
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to delete factor'))
      throw err
    }
  }, [])

  // Filter and sort factors
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

  // Statistics by family and status
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
      candidate: 0,
      rejected: 0,
      core: 0,
      redundant: 0,
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
    // Actions
    generateFactor,
    evaluateFactor,
    updateFactorStatus,
    deleteFactor,
    refresh: loadFactors,
  }
}
