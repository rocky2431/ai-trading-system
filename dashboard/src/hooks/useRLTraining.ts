/**
 * useRLTraining - RL 训练 Hook
 *
 * 提供 RL 训练任务管理和状态查询
 */

import { useState, useEffect, useCallback } from 'react'
import {
  rlApi,
  RLTaskResponse,
  RLTaskStatus,
  RLModelInfo,
  RLStatsResponse,
  RLTrainingRequest,
  RLBacktestRequest,
} from '@/api/rl'

interface UseRLTrainingResult {
  // 数据
  tasks: RLTaskResponse[]
  models: RLModelInfo[]
  stats: RLStatsResponse | null

  // 分页
  page: number
  setPage: (page: number) => void
  total: number
  hasNext: boolean

  // 过滤
  taskType: 'training' | 'backtest' | undefined
  setTaskType: (type: 'training' | 'backtest' | undefined) => void
  statusFilter: RLTaskStatus | undefined
  setStatusFilter: (status: RLTaskStatus | undefined) => void

  // 状态
  loading: boolean
  error: Error | null

  // 操作
  refresh: () => Promise<void>
  submitTraining: (request: RLTrainingRequest) => Promise<RLTaskResponse>
  submitBacktest: (request: RLBacktestRequest) => Promise<RLTaskResponse>
  cancelTask: (taskId: string) => Promise<void>
}

export function useRLTraining(): UseRLTrainingResult {
  // 数据状态
  const [tasks, setTasks] = useState<RLTaskResponse[]>([])
  const [models, setModels] = useState<RLModelInfo[]>([])
  const [stats, setStats] = useState<RLStatsResponse | null>(null)

  // 分页状态
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const [hasNext, setHasNext] = useState(false)

  // 过滤状态
  const [taskType, setTaskType] = useState<'training' | 'backtest' | undefined>()
  const [statusFilter, setStatusFilter] = useState<RLTaskStatus | undefined>()

  // 加载状态
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  // 刷新数据
  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)

    try {
      const [tasksRes, modelsRes, statsRes] = await Promise.all([
        rlApi.getTasks({
          page,
          page_size: 10,
          task_type: taskType,
          status: statusFilter,
        }),
        rlApi.getModels(),
        rlApi.getStats(),
      ])

      setTasks(tasksRes.items)
      setTotal(tasksRes.total)
      setHasNext(tasksRes.has_next)
      setModels(modelsRes.models)
      setStats(statsRes)
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to load RL data'))
    } finally {
      setLoading(false)
    }
  }, [page, taskType, statusFilter])

  // 初始加载和条件变化时刷新
  useEffect(() => {
    refresh()
  }, [refresh])

  // 自动刷新运行中的任务
  useEffect(() => {
    const hasRunningTasks = tasks.some(
      (t) => t.status === 'started' || t.status === 'running'
    )

    if (hasRunningTasks) {
      const interval = setInterval(refresh, 5000) // 每5秒刷新
      return () => clearInterval(interval)
    }
  }, [tasks, refresh])

  // 提交训练
  const submitTraining = useCallback(
    async (request: RLTrainingRequest) => {
      const result = await rlApi.submitTraining(request)
      await refresh()
      return result
    },
    [refresh]
  )

  // 提交回测
  const submitBacktest = useCallback(
    async (request: RLBacktestRequest) => {
      const result = await rlApi.submitBacktest(request)
      await refresh()
      return result
    },
    [refresh]
  )

  // 取消任务
  const cancelTask = useCallback(
    async (taskId: string) => {
      await rlApi.cancelTask(taskId)
      await refresh()
    },
    [refresh]
  )

  return {
    tasks,
    models,
    stats,
    page,
    setPage,
    total,
    hasNext,
    taskType,
    setTaskType,
    statusFilter,
    setStatusFilter,
    loading,
    error,
    refresh,
    submitTraining,
    submitBacktest,
    cancelTask,
  }
}
