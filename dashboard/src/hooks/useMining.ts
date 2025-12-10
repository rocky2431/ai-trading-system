/**
 * Mining Task Hooks - 因子挖掘任务管理
 */

import { useState, useEffect, useCallback } from 'react'
import {
  factorsApi,
  type MiningTaskStatus,
  type MiningTaskCreateRequest,
  type FactorLibraryStats,
} from '@/api/factors'

// ============== Mining Tasks Hook ==============

export function useMiningTasks(params?: { status?: string; limit?: number }) {
  const [tasks, setTasks] = useState<MiningTaskStatus[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)

  const fetchTasks = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await factorsApi.listMiningTasks(params)
      setTasks(response.tasks)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch mining tasks')
    } finally {
      setLoading(false)
    }
  }, [params?.status, params?.limit])

  const createTask = useCallback(async (request: MiningTaskCreateRequest) => {
    try {
      setCreating(true)
      setError(null)
      const response = await factorsApi.createMiningTask(request)
      if (response.success) {
        await fetchTasks()
      }
      return response
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to create mining task'
      setError(message)
      return { success: false, message, task_id: null }
    } finally {
      setCreating(false)
    }
  }, [fetchTasks])

  const cancelTask = useCallback(async (taskId: string) => {
    try {
      setError(null)
      const response = await factorsApi.cancelMiningTask(taskId)
      if (response.success) {
        await fetchTasks()
      }
      return response
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to cancel mining task'
      setError(message)
      return { success: false, message }
    }
  }, [fetchTasks])

  useEffect(() => {
    fetchTasks()
  }, [fetchTasks])

  // Auto-refresh for running tasks
  useEffect(() => {
    const hasRunningTasks = tasks.some(t => t.status === 'running' || t.status === 'pending')
    if (!hasRunningTasks) return

    const interval = setInterval(fetchTasks, 3000) // Refresh every 3 seconds
    return () => clearInterval(interval)
  }, [tasks, fetchTasks])

  return {
    tasks,
    loading,
    error,
    creating,
    createTask,
    cancelTask,
    refetch: fetchTasks,
  }
}

// ============== Single Mining Task Hook ==============

export function useMiningTask(taskId: string | null) {
  const [task, setTask] = useState<MiningTaskStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchTask = useCallback(async () => {
    if (!taskId) return

    try {
      setLoading(true)
      setError(null)
      const response = await factorsApi.getMiningTask(taskId)
      setTask(response)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch mining task')
    } finally {
      setLoading(false)
    }
  }, [taskId])

  useEffect(() => {
    fetchTask()
  }, [fetchTask])

  // Auto-refresh for running task
  useEffect(() => {
    if (!task || (task.status !== 'running' && task.status !== 'pending')) return

    const interval = setInterval(fetchTask, 2000) // Refresh every 2 seconds
    return () => clearInterval(interval)
  }, [task, fetchTask])

  return { task, loading, error, refetch: fetchTask }
}

// ============== Factor Library Stats Hook ==============

export function useFactorLibraryStats() {
  const [stats, setStats] = useState<FactorLibraryStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchStats = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await factorsApi.getLibraryStats()
      setStats(response)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch library stats')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchStats()
  }, [fetchStats])

  return { stats, loading, error, refetch: fetchStats }
}

// ============== Factor Comparison Hook ==============

export function useFactorComparison() {
  const [comparison, setComparison] = useState<{
    factors: any[]
    correlation_matrix: Record<string, Record<string, number>>
    ranking: string[]
  } | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const compare = useCallback(async (factorIds: string[]) => {
    if (factorIds.length < 2) {
      setError('At least 2 factors required for comparison')
      return null
    }

    try {
      setLoading(true)
      setError(null)
      const response = await factorsApi.compare(factorIds)
      setComparison(response)
      return response
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to compare factors')
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  const clear = useCallback(() => {
    setComparison(null)
    setError(null)
  }, [])

  return { comparison, loading, error, compare, clear }
}
