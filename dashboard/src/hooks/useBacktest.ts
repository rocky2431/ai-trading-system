/**
 * Backtest Hooks - 策略和回测管理
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import {
  backtestApi,
  type BacktestResponse,
  type BacktestConfig,
  type BacktestDetailResponse,
  type BacktestStatsResponse,
  type OptimizationResponse,
  type OptimizationDetailResponse,
  type OptimizationRequest,
} from '@/api/backtest'
import {
  strategiesApi,
  type StrategyResponse,
  type StrategyCreateRequest,
  type StrategyTemplateResponse,
  type CreateFromTemplateRequest,
} from '@/api/strategies'

// ============== Strategies Hook ==============

export function useStrategies(params?: { page?: number; page_size?: number; status?: string }) {
  const [strategies, setStrategies] = useState<StrategyResponse[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)

  const fetchStrategies = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await strategiesApi.list(params)
      setStrategies(response.strategies)
      setTotal(response.total)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch strategies')
    } finally {
      setLoading(false)
    }
  }, [params?.page, params?.page_size, params?.status])

  const createStrategy = useCallback(async (request: StrategyCreateRequest) => {
    try {
      setCreating(true)
      setError(null)
      const response = await strategiesApi.create(request)
      await fetchStrategies()
      return { success: true, strategy: response }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to create strategy'
      setError(message)
      return { success: false, message }
    } finally {
      setCreating(false)
    }
  }, [fetchStrategies])

  const deleteStrategy = useCallback(async (strategyId: string) => {
    try {
      setError(null)
      await strategiesApi.delete(strategyId)
      await fetchStrategies()
      return { success: true, message: 'Strategy deleted' }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete strategy'
      setError(message)
      return { success: false, message }
    }
  }, [fetchStrategies])

  useEffect(() => {
    fetchStrategies()
  }, [fetchStrategies])

  return {
    strategies,
    total,
    loading,
    error,
    creating,
    createStrategy,
    deleteStrategy,
    refetch: fetchStrategies,
  }
}

// ============== Single Strategy Hook ==============

export function useStrategy(strategyId: string | null) {
  const [strategy, setStrategy] = useState<StrategyResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchStrategy = useCallback(async () => {
    if (!strategyId) return

    try {
      setLoading(true)
      setError(null)
      const response = await strategiesApi.get(strategyId)
      setStrategy(response)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch strategy')
    } finally {
      setLoading(false)
    }
  }, [strategyId])

  useEffect(() => {
    fetchStrategy()
  }, [fetchStrategy])

  return { strategy, loading, error, refetch: fetchStrategy }
}

// ============== Backtests Hook ==============

export function useBacktests(params?: {
  strategy_id?: string
  status?: string
  page?: number
  page_size?: number
}) {
  const [backtests, setBacktests] = useState<BacktestResponse[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)

  // Track request ID to ignore stale responses
  const requestIdRef = useRef(0)
  const isMountedRef = useRef(true)

  const fetchBacktests = useCallback(async () => {
    const currentRequestId = ++requestIdRef.current

    try {
      setLoading(true)
      setError(null)
      const response = await backtestApi.listBacktests(params)

      // Ignore response if component unmounted or newer request started
      if (!isMountedRef.current || currentRequestId !== requestIdRef.current) {
        return
      }

      setBacktests(response.backtests)
      setTotal(response.total)
    } catch (err) {
      if (!isMountedRef.current || currentRequestId !== requestIdRef.current) {
        return
      }
      setError(err instanceof Error ? err.message : 'Failed to fetch backtests')
    } finally {
      if (isMountedRef.current && currentRequestId === requestIdRef.current) {
        setLoading(false)
      }
    }
  }, [params?.strategy_id, params?.status, params?.page, params?.page_size])

  const createBacktest = useCallback(async (
    strategyId: string,
    config: BacktestConfig,
    name?: string,
    description?: string
  ) => {
    try {
      setCreating(true)
      setError(null)
      const response = await backtestApi.createBacktest({
        strategy_id: strategyId,
        config,
        name,
        description,
      })
      if (response.success) {
        await fetchBacktests()
      }
      return response
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to create backtest'
      setError(message)
      return { success: false, message }
    } finally {
      setCreating(false)
    }
  }, [fetchBacktests])

  const deleteBacktest = useCallback(async (backtestId: string) => {
    try {
      setError(null)
      const response = await backtestApi.deleteBacktest(backtestId)
      if (response.success) {
        await fetchBacktests()
      }
      return response
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete backtest'
      setError(message)
      return { success: false, message }
    }
  }, [fetchBacktests])

  // Reset mounted ref on mount/unmount
  useEffect(() => {
    isMountedRef.current = true
    return () => {
      isMountedRef.current = false
    }
  }, [])

  useEffect(() => {
    fetchBacktests()
  }, [fetchBacktests])

  // Auto-refresh for running backtests
  useEffect(() => {
    const hasRunning = backtests.some(b => b.status === 'running' || b.status === 'pending')
    if (!hasRunning) return

    const interval = setInterval(fetchBacktests, 3000)
    return () => clearInterval(interval)
  }, [backtests, fetchBacktests])

  return {
    backtests,
    total,
    loading,
    error,
    creating,
    createBacktest,
    deleteBacktest,
    refetch: fetchBacktests,
  }
}

// ============== Backtest Detail Hook ==============

export function useBacktestDetail(backtestId: string | null) {
  const [detail, setDetail] = useState<BacktestDetailResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchDetail = useCallback(async () => {
    if (!backtestId) return

    try {
      setLoading(true)
      setError(null)
      const response = await backtestApi.getBacktestDetail(backtestId)
      setDetail(response)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch backtest detail')
    } finally {
      setLoading(false)
    }
  }, [backtestId])

  useEffect(() => {
    fetchDetail()
  }, [fetchDetail])

  return { detail, loading, error, refetch: fetchDetail }
}

// ============== Backtest Stats Hook ==============

export function useBacktestStats() {
  const [stats, setStats] = useState<BacktestStatsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchStats = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await backtestApi.getStats()
      setStats(response)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch backtest stats')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchStats()
  }, [fetchStats])

  return { stats, loading, error, refetch: fetchStats }
}

// ============== Optimizations Hook ==============

export function useOptimizations(params?: {
  strategy_id?: string
  status?: string
  page?: number
  page_size?: number
}) {
  const [optimizations, setOptimizations] = useState<OptimizationResponse[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)

  const requestIdRef = useRef(0)
  const isMountedRef = useRef(true)

  const fetchOptimizations = useCallback(async () => {
    const currentRequestId = ++requestIdRef.current

    try {
      setLoading(true)
      setError(null)
      const response = await backtestApi.listOptimizations(params)

      if (!isMountedRef.current || currentRequestId !== requestIdRef.current) {
        return
      }

      setOptimizations(response.optimizations)
      setTotal(response.total)
    } catch (err) {
      if (!isMountedRef.current || currentRequestId !== requestIdRef.current) {
        return
      }
      setError(err instanceof Error ? err.message : 'Failed to fetch optimizations')
    } finally {
      if (isMountedRef.current && currentRequestId === requestIdRef.current) {
        setLoading(false)
      }
    }
  }, [params?.strategy_id, params?.status, params?.page, params?.page_size])

  const createOptimization = useCallback(async (request: OptimizationRequest) => {
    try {
      setCreating(true)
      setError(null)
      const response = await backtestApi.createOptimization(request)
      if (response.success) {
        await fetchOptimizations()
      }
      return response
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to create optimization'
      setError(message)
      return { success: false, message }
    } finally {
      setCreating(false)
    }
  }, [fetchOptimizations])

  const cancelOptimization = useCallback(async (optimizationId: string) => {
    try {
      setError(null)
      const response = await backtestApi.cancelOptimization(optimizationId)
      if (response.success) {
        await fetchOptimizations()
      }
      return response
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to cancel optimization'
      setError(message)
      return { success: false, message }
    }
  }, [fetchOptimizations])

  const deleteOptimization = useCallback(async (optimizationId: string) => {
    try {
      setError(null)
      const response = await backtestApi.deleteOptimization(optimizationId)
      if (response.success) {
        await fetchOptimizations()
      }
      return response
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete optimization'
      setError(message)
      return { success: false, message }
    }
  }, [fetchOptimizations])

  useEffect(() => {
    isMountedRef.current = true
    return () => {
      isMountedRef.current = false
    }
  }, [])

  useEffect(() => {
    fetchOptimizations()
  }, [fetchOptimizations])

  // Auto-refresh for running optimizations
  useEffect(() => {
    const hasRunning = optimizations.some(o => o.status === 'running' || o.status === 'pending')
    if (!hasRunning) return

    const interval = setInterval(fetchOptimizations, 5000)
    return () => clearInterval(interval)
  }, [optimizations, fetchOptimizations])

  return {
    optimizations,
    total,
    loading,
    error,
    creating,
    createOptimization,
    cancelOptimization,
    deleteOptimization,
    refetch: fetchOptimizations,
  }
}

// ============== Optimization Detail Hook ==============

export function useOptimizationDetail(optimizationId: string | null) {
  const [detail, setDetail] = useState<OptimizationDetailResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchDetail = useCallback(async () => {
    if (!optimizationId) return

    try {
      setLoading(true)
      setError(null)
      const response = await backtestApi.getOptimizationDetail(optimizationId)
      setDetail(response)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch optimization detail')
    } finally {
      setLoading(false)
    }
  }, [optimizationId])

  useEffect(() => {
    fetchDetail()
  }, [fetchDetail])

  return { detail, loading, error, refetch: fetchDetail }
}

// ============== Strategy Templates Hook ==============

export function useStrategyTemplates(params?: {
  category?: string
  risk_level?: string
  search?: string
}) {
  const [templates, setTemplates] = useState<StrategyTemplateResponse[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [creatingFromTemplate, setCreatingFromTemplate] = useState(false)

  const fetchTemplates = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await strategiesApi.listTemplates(params)
      setTemplates(response.templates)
      setTotal(response.total)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch templates')
    } finally {
      setLoading(false)
    }
  }, [params?.category, params?.risk_level, params?.search])

  const createFromTemplate = useCallback(async (request: CreateFromTemplateRequest) => {
    try {
      setCreatingFromTemplate(true)
      setError(null)
      const strategy = await strategiesApi.createFromTemplate(request)
      return { success: true, strategy }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to create strategy from template'
      setError(message)
      return { success: false, message }
    } finally {
      setCreatingFromTemplate(false)
    }
  }, [])

  useEffect(() => {
    fetchTemplates()
  }, [fetchTemplates])

  return {
    templates,
    total,
    loading,
    error,
    creatingFromTemplate,
    createFromTemplate,
    refetch: fetchTemplates,
  }
}
