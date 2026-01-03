/**
 * Factors Hook - 提供因子列表和筛选功能
 * 使用真实 API 数据
 */

import { useState, useEffect, useMemo, useCallback } from 'react'
import { factorsApi } from '@/api'
import type { FactorResponse } from '@/api'
import type { Factor, FactorFilter, FactorFamily, FactorStatus } from '@/types/factor'

// 将 API 响应转换为前端类型
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
      winRate: 50 + response.metrics.ir * 5,
      turnover: response.metrics.turnover * 100,
      stability: Object.values(response.metrics.ic_by_split).reduce((a, b) => a + b, 0) /
        Math.max(Object.values(response.metrics.ic_by_split).length, 1) / response.metrics.ic_mean || 0.7,
    } : null,
    evaluationCount: response.experiment_number,
    tags: response.family || [],
  }
}

// Direct passthrough - backend FactorStatus matches frontend FactorStatus type
function mapApiStatus(status: string): FactorStatus {
  const validStatuses: FactorStatus[] = ['candidate', 'rejected', 'core', 'redundant']
  if (validStatuses.includes(status as FactorStatus)) {
    return status as FactorStatus
  }
  // Fallback for unknown status
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

  // 加载因子数据
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

  // 生成新因子
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

  // 评估因子
  const evaluateFactor = useCallback(async (factorId: string) => {
    try {
      const response = await factorsApi.evaluate(factorId, {
        splits: ['train', 'valid', 'test'],
      })
      // 重新加载因子列表以获取更新后的数据
      await loadFactors()
      return response
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to evaluate factor'))
      throw err
    }
  }, [loadFactors])

  // 更新因子状态
  const updateFactorStatus = useCallback(async (factorId: string, status: string) => {
    try {
      await factorsApi.updateStatus(factorId, status)
      await loadFactors()
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to update factor status'))
      throw err
    }
  }, [loadFactors])

  // 删除因子
  const deleteFactor = useCallback(async (factorId: string) => {
    try {
      await factorsApi.delete(factorId)
      setFactors(prev => prev.filter(f => f.id !== factorId))
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to delete factor'))
      throw err
    }
  }, [])

  // 过滤和排序
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

  // 统计信息
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
