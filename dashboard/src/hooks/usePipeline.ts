/**
 * usePipeline - Pipeline 和 RD Loop 管理 Hook
 *
 * 提供 Pipeline/RD Loop 状态管理和 WebSocket 实时更新
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import {
  pipelineApi,
  PipelineStatusResponse,
  RDLoopRunInfo,
  RDLoopStateResponse,
  RDLoopStatisticsResponse,
  RDLoopCoreFactorResponse,
  RDLoopConfigRequest,
  RDLoopPhase,
} from '@/api/pipeline'

interface UsePipelineResult {
  // Pipeline 数据
  pipelineRuns: PipelineStatusResponse[]

  // RD Loop 数据
  rdLoopRuns: RDLoopRunInfo[]
  currentRDLoop: {
    runId: string | null
    isRunning: boolean
    phase: RDLoopPhase
    iteration: number
    totalHypothesesTested: number
    coreFactorsCount: number
  } | null
  selectedRDLoop: RDLoopStateResponse | null
  rdLoopStatistics: RDLoopStatisticsResponse | null
  rdLoopFactors: RDLoopCoreFactorResponse[]

  // 状态
  loading: boolean
  error: Error | null

  // 操作
  refresh: () => Promise<void>
  startRDLoop: (config?: RDLoopConfigRequest, dataSource?: string) => Promise<string>
  stopRDLoop: (runId: string) => Promise<void>
  selectRDLoop: (runId: string) => Promise<void>
  clearSelection: () => void
}

export function usePipeline(): UsePipelineResult {
  // Pipeline 状态
  const [pipelineRuns, setPipelineRuns] = useState<PipelineStatusResponse[]>([])

  // RD Loop 状态
  const [rdLoopRuns, setRDLoopRuns] = useState<RDLoopRunInfo[]>([])
  const [currentRDLoop, setCurrentRDLoop] = useState<UsePipelineResult['currentRDLoop']>(null)
  const [selectedRDLoop, setSelectedRDLoop] = useState<RDLoopStateResponse | null>(null)
  const [rdLoopStatistics, setRDLoopStatistics] = useState<RDLoopStatisticsResponse | null>(null)
  const [rdLoopFactors, setRDLoopFactors] = useState<RDLoopCoreFactorResponse[]>([])

  // 加载状态
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  // WebSocket 引用
  const wsRef = useRef<WebSocket | null>(null)

  // 刷新数据
  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)

    try {
      const [pipelineRes, rdLoopRunsRes, currentStateRes] = await Promise.all([
        pipelineApi.listRuns(),
        pipelineApi.listRDLoopRuns(),
        pipelineApi.getCurrentRDLoopState(),
      ])

      setPipelineRuns(pipelineRes.runs)
      setRDLoopRuns(rdLoopRunsRes.runs)
      setCurrentRDLoop({
        runId: currentStateRes.run_id,
        isRunning: currentStateRes.is_running,
        phase: currentStateRes.phase,
        iteration: currentStateRes.iteration,
        totalHypothesesTested: currentStateRes.total_hypotheses_tested,
        coreFactorsCount: currentStateRes.core_factors_count,
      })
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to load pipeline data'))
    } finally {
      setLoading(false)
    }
  }, [])

  // 初始加载
  useEffect(() => {
    refresh()
  }, [refresh])

  // 自动刷新运行中的任务
  useEffect(() => {
    if (currentRDLoop?.isRunning) {
      const interval = setInterval(refresh, 3000)
      return () => clearInterval(interval)
    }
  }, [currentRDLoop?.isRunning, refresh])

  // 启动 RD Loop
  const startRDLoop = useCallback(
    async (config?: RDLoopConfigRequest, dataSource?: string) => {
      const res = await pipelineApi.runRDLoop({ config, data_source: dataSource })
      await refresh()
      return res.run_id
    },
    [refresh]
  )

  // 停止 RD Loop
  const stopRDLoop = useCallback(
    async (runId: string) => {
      await pipelineApi.stopRDLoop(runId)
      await refresh()
    },
    [refresh]
  )

  // 选择 RD Loop 查看详情
  const selectRDLoop = useCallback(async (runId: string) => {
    try {
      const [stateRes, statsRes, factorsRes] = await Promise.all([
        pipelineApi.getRDLoopState(runId),
        pipelineApi.getRDLoopStatistics(runId).catch(() => null),
        pipelineApi.getRDLoopFactors(runId).catch(() => []),
      ])

      setSelectedRDLoop(stateRes)
      setRDLoopStatistics(statsRes)
      setRDLoopFactors(factorsRes)
    } catch (err) {
      console.error('Failed to load RD Loop details:', err)
    }
  }, [])

  // 清除选择
  const clearSelection = useCallback(() => {
    setSelectedRDLoop(null)
    setRDLoopStatistics(null)
    setRDLoopFactors([])
  }, [])

  // 清理 WebSocket
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  return {
    pipelineRuns,
    rdLoopRuns,
    currentRDLoop,
    selectedRDLoop,
    rdLoopStatistics,
    rdLoopFactors,
    loading,
    error,
    refresh,
    startRDLoop,
    stopRDLoop,
    selectRDLoop,
    clearSelection,
  }
}
