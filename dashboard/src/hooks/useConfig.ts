/**
 * Config API Hooks
 */

import { useState, useEffect, useCallback } from 'react'
import {
  configApi,
  type ConfigStatusResponse,
  type AvailableModelsResponse,
  type SavedAPIKeysResponse,
  type AgentConfigResponse,
  type DataConfigResponse,
  type FactorMiningConfigResponse,
  type RiskControlConfigResponse,
  type SetAPIKeysRequest,
  type SetAgentConfigRequest,
  type SetDataConfigRequest,
  type SetFactorMiningConfigRequest,
  type SetRiskControlConfigRequest,
  // P3 Types
  type SandboxConfigResponse,
  type SandboxConfigUpdate,
  type ExecutionLogResponse,
  type SecurityConfigResponse,
  type SecurityConfigUpdate,
  type LLMAdvancedConfigResponse,
  type LLMAdvancedConfigUpdate,
  type LLMTraceResponse,
  type LLMCostSummary,
  type DerivativeDataConfigResponse,
  type DerivativeDataConfigUpdate,
  type CheckpointThreadInfo,
  type CheckpointListResponse,
  type CheckpointStateResponse,
  type BenchmarkConfigResponse,
  type BenchmarkConfigUpdate,
  type BenchmarkResultsResponse,
} from '@/api/config'

// ============== Status Hook ==============

export function useConfigStatus() {
  const [status, setStatus] = useState<ConfigStatusResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchStatus = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await configApi.getStatus()
      setStatus(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch config status')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchStatus()
  }, [fetchStatus])

  return { status, loading, error, refetch: fetchStatus }
}

// ============== Models Hook ==============

export function useAvailableModels() {
  const [models, setModels] = useState<AvailableModelsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchModels = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await configApi.getModels()
      setModels(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch models')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchModels()
  }, [fetchModels])

  return { models, loading, error, refetch: fetchModels }
}

// ============== API Keys Hook ==============

export function useAPIKeys() {
  const [keys, setKeys] = useState<SavedAPIKeysResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [deleting, setDeleting] = useState(false)

  const fetchKeys = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await configApi.getAPIKeys()
      setKeys(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch API keys')
    } finally {
      setLoading(false)
    }
  }, [])

  const saveKeys = useCallback(async (data: SetAPIKeysRequest) => {
    try {
      setSaving(true)
      setError(null)
      const result = await configApi.setAPIKeys(data)
      if (result.success) {
        await fetchKeys()
      }
      return result
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to save API keys'
      setError(message)
      return { success: false, message }
    } finally {
      setSaving(false)
    }
  }, [fetchKeys])

  const deleteKeys = useCallback(async (keyType: 'llm' | 'exchange') => {
    try {
      setDeleting(true)
      setError(null)
      const result = await configApi.deleteAPIKeys(keyType)
      if (result.success) {
        await fetchKeys()
      }
      return result
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete API keys'
      setError(message)
      return { success: false, message }
    } finally {
      setDeleting(false)
    }
  }, [fetchKeys])

  useEffect(() => {
    fetchKeys()
  }, [fetchKeys])

  return { keys, loading, error, saving, deleting, saveKeys, deleteKeys, refetch: fetchKeys }
}

// ============== Test Connections ==============

export function useTestConnections() {
  const [testingLLM, setTestingLLM] = useState(false)
  const [testingExchange, setTestingExchange] = useState(false)
  const [llmResult, setLLMResult] = useState<{ success: boolean; message: string } | null>(null)
  const [exchangeResult, setExchangeResult] = useState<{ success: boolean; message: string } | null>(null)

  const testLLM = useCallback(async () => {
    try {
      setTestingLLM(true)
      setLLMResult(null)
      const result = await configApi.testLLM()
      setLLMResult({
        success: result.success,
        message: result.message
      })
      return result
    } catch (err) {
      const result = {
        success: false,
        message: err instanceof Error ? err.message : 'Failed to test LLM'
      }
      setLLMResult(result)
      return result
    } finally {
      setTestingLLM(false)
    }
  }, [])

  const testExchange = useCallback(async () => {
    try {
      setTestingExchange(true)
      setExchangeResult(null)
      const result = await configApi.testExchange()
      setExchangeResult({
        success: result.success,
        message: result.message
      })
      return result
    } catch (err) {
      const result = {
        success: false,
        message: err instanceof Error ? err.message : 'Failed to test exchange'
      }
      setExchangeResult(result)
      return result
    } finally {
      setTestingExchange(false)
    }
  }, [])

  return { testLLM, testExchange, testingLLM, testingExchange, llmResult, exchangeResult }
}

// ============== Agent Config Hook ==============

export function useAgentConfig() {
  const [config, setConfig] = useState<AgentConfigResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  const fetchConfig = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await configApi.getAgentConfig()
      setConfig(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch agent config')
    } finally {
      setLoading(false)
    }
  }, [])

  const saveConfig = useCallback(async (data: SetAgentConfigRequest) => {
    try {
      setSaving(true)
      setError(null)
      const result = await configApi.setAgentConfig(data)
      if (result.success) {
        await fetchConfig()
      }
      return result
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to save agent config'
      setError(message)
      return { success: false, message }
    } finally {
      setSaving(false)
    }
  }, [fetchConfig])

  useEffect(() => {
    fetchConfig()
  }, [fetchConfig])

  return { config, loading, error, saving, saveConfig, refetch: fetchConfig }
}

// ============== Data Config Hook ==============

export function useDataConfig() {
  const [config, setConfig] = useState<DataConfigResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  const fetchConfig = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await configApi.getDataConfig()
      setConfig(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data config')
    } finally {
      setLoading(false)
    }
  }, [])

  const saveConfig = useCallback(async (data: SetDataConfigRequest) => {
    try {
      setSaving(true)
      setError(null)
      const result = await configApi.setDataConfig(data)
      if (result.success) {
        await fetchConfig()
      }
      return result
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to save data config'
      setError(message)
      return { success: false, message }
    } finally {
      setSaving(false)
    }
  }, [fetchConfig])

  useEffect(() => {
    fetchConfig()
  }, [fetchConfig])

  return { config, loading, error, saving, saveConfig, refetch: fetchConfig }
}

// ============== Factor Mining Config Hook ==============

export function useFactorMiningConfig() {
  const [config, setConfig] = useState<FactorMiningConfigResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  const fetchConfig = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await configApi.getFactorMiningConfig()
      setConfig(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch factor mining config')
    } finally {
      setLoading(false)
    }
  }, [])

  const saveConfig = useCallback(async (data: SetFactorMiningConfigRequest) => {
    try {
      setSaving(true)
      setError(null)
      const result = await configApi.setFactorMiningConfig(data)
      if (result.success) {
        await fetchConfig()
      }
      return result
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to save factor mining config'
      setError(message)
      return { success: false, message }
    } finally {
      setSaving(false)
    }
  }, [fetchConfig])

  useEffect(() => {
    fetchConfig()
  }, [fetchConfig])

  return { config, loading, error, saving, saveConfig, refetch: fetchConfig }
}

// ============== Risk Control Config Hook ==============

export function useRiskControlConfig() {
  const [config, setConfig] = useState<RiskControlConfigResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  const fetchConfig = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await configApi.getRiskControlConfig()
      setConfig(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch risk control config')
    } finally {
      setLoading(false)
    }
  }, [])

  const saveConfig = useCallback(async (data: SetRiskControlConfigRequest) => {
    try {
      setSaving(true)
      setError(null)
      const result = await configApi.setRiskControlConfig(data)
      if (result.success) {
        await fetchConfig()
      }
      return result
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to save risk control config'
      setError(message)
      return { success: false, message }
    } finally {
      setSaving(false)
    }
  }, [fetchConfig])

  useEffect(() => {
    fetchConfig()
  }, [fetchConfig])

  return { config, loading, error, saving, saveConfig, refetch: fetchConfig }
}

// ============== P3: Sandbox Config Hook ==============

export function useSandboxConfig() {
  const [config, setConfig] = useState<SandboxConfigResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  const fetchConfig = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await configApi.getSandboxConfig()
      setConfig(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch sandbox config')
    } finally {
      setLoading(false)
    }
  }, [])

  const saveConfig = useCallback(async (data: SandboxConfigUpdate) => {
    try {
      setSaving(true)
      setError(null)
      const result = await configApi.updateSandboxConfig(data)
      if (result.success) {
        await fetchConfig()
      }
      return result
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to save sandbox config'
      setError(message)
      return { success: false, message }
    } finally {
      setSaving(false)
    }
  }, [fetchConfig])

  useEffect(() => {
    fetchConfig()
  }, [fetchConfig])

  return { config, loading, error, saving, saveConfig, refetch: fetchConfig }
}

// ============== P3: Execution Logs Hook ==============

export function useExecutionLogs(initialPage = 1, initialPageSize = 20) {
  const [logs, setLogs] = useState<ExecutionLogResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [page, setPage] = useState(initialPage)
  const [pageSize] = useState(initialPageSize)
  const [statusFilter, setStatusFilter] = useState<string | undefined>()

  const fetchLogs = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await configApi.getExecutionLogs(page, pageSize, statusFilter)
      setLogs(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch execution logs')
    } finally {
      setLoading(false)
    }
  }, [page, pageSize, statusFilter])

  useEffect(() => {
    fetchLogs()
  }, [fetchLogs])

  return { logs, loading, error, page, setPage, statusFilter, setStatusFilter, refetch: fetchLogs }
}

// ============== P3: Security Config Hook ==============

export function useSecurityConfig() {
  const [config, setConfig] = useState<SecurityConfigResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  const fetchConfig = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await configApi.getSecurityConfig()
      setConfig(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch security config')
    } finally {
      setLoading(false)
    }
  }, [])

  const saveConfig = useCallback(async (data: SecurityConfigUpdate) => {
    try {
      setSaving(true)
      setError(null)
      const result = await configApi.updateSecurityConfig(data)
      if (result.success) {
        await fetchConfig()
      }
      return result
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to save security config'
      setError(message)
      return { success: false, message }
    } finally {
      setSaving(false)
    }
  }, [fetchConfig])

  useEffect(() => {
    fetchConfig()
  }, [fetchConfig])

  return { config, loading, error, saving, saveConfig, refetch: fetchConfig }
}

// ============== P3: LLM Advanced Config Hook ==============

export function useLLMAdvancedConfig() {
  const [config, setConfig] = useState<LLMAdvancedConfigResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  const fetchConfig = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await configApi.getLLMAdvancedConfig()
      setConfig(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch LLM advanced config')
    } finally {
      setLoading(false)
    }
  }, [])

  const saveConfig = useCallback(async (data: LLMAdvancedConfigUpdate) => {
    try {
      setSaving(true)
      setError(null)
      const result = await configApi.updateLLMAdvancedConfig(data)
      if (result.success) {
        await fetchConfig()
      }
      return result
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to save LLM advanced config'
      setError(message)
      return { success: false, message }
    } finally {
      setSaving(false)
    }
  }, [fetchConfig])

  useEffect(() => {
    fetchConfig()
  }, [fetchConfig])

  return { config, loading, error, saving, saveConfig, refetch: fetchConfig }
}

// ============== P3: LLM Traces Hook ==============

export function useLLMTraces(initialPage = 1, initialPageSize = 20) {
  const [traces, setTraces] = useState<LLMTraceResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [page, setPage] = useState(initialPage)
  const [pageSize] = useState(initialPageSize)
  const [agentFilter, setAgentFilter] = useState<string | undefined>()
  const [modelFilter, setModelFilter] = useState<string | undefined>()

  const fetchTraces = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await configApi.getLLMTraces(page, pageSize, agentFilter, modelFilter)
      setTraces(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch LLM traces')
    } finally {
      setLoading(false)
    }
  }, [page, pageSize, agentFilter, modelFilter])

  useEffect(() => {
    fetchTraces()
  }, [fetchTraces])

  return { traces, loading, error, page, setPage, agentFilter, setAgentFilter, modelFilter, setModelFilter, refetch: fetchTraces }
}

// ============== P3: LLM Costs Hook ==============

export function useLLMCosts(hours = 24) {
  const [costs, setCosts] = useState<LLMCostSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchCosts = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await configApi.getLLMCosts(hours)
      setCosts(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch LLM costs')
    } finally {
      setLoading(false)
    }
  }, [hours])

  useEffect(() => {
    fetchCosts()
  }, [fetchCosts])

  return { costs, loading, error, refetch: fetchCosts }
}

// ============== P3: Derivative Data Config Hook ==============

export function useDerivativeDataConfig() {
  const [config, setConfig] = useState<DerivativeDataConfigResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  const fetchConfig = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await configApi.getDerivativeDataConfig()
      setConfig(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch derivative data config')
    } finally {
      setLoading(false)
    }
  }, [])

  const saveConfig = useCallback(async (data: DerivativeDataConfigUpdate) => {
    try {
      setSaving(true)
      setError(null)
      const result = await configApi.updateDerivativeDataConfig(data)
      if (result.success) {
        await fetchConfig()
      }
      return result
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to save derivative data config'
      setError(message)
      return { success: false, message }
    } finally {
      setSaving(false)
    }
  }, [fetchConfig])

  useEffect(() => {
    fetchConfig()
  }, [fetchConfig])

  return { config, loading, error, saving, saveConfig, refetch: fetchConfig }
}

// ============== P3: Checkpoints Hook ==============

export function useCheckpoints() {
  const [threads, setThreads] = useState<CheckpointThreadInfo[]>([])
  const [selectedThread, setSelectedThread] = useState<string | null>(null)
  const [checkpoints, setCheckpoints] = useState<CheckpointListResponse | null>(null)
  const [checkpointState, setCheckpointState] = useState<CheckpointStateResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [restoring, setRestoring] = useState(false)

  const fetchThreads = useCallback(async (limit = 20) => {
    try {
      setLoading(true)
      setError(null)
      const data = await configApi.listCheckpointThreads(limit)
      setThreads(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch checkpoint threads')
    } finally {
      setLoading(false)
    }
  }, [])

  const fetchCheckpoints = useCallback(async (threadId: string) => {
    try {
      setLoading(true)
      setError(null)
      const data = await configApi.listCheckpoints(threadId)
      setCheckpoints(data)
      setSelectedThread(threadId)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch checkpoints')
    } finally {
      setLoading(false)
    }
  }, [])

  const fetchCheckpointState = useCallback(async (threadId: string, checkpointId: string) => {
    try {
      setLoading(true)
      setError(null)
      const data = await configApi.getCheckpointState(threadId, checkpointId)
      setCheckpointState(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch checkpoint state')
    } finally {
      setLoading(false)
    }
  }, [])

  const restoreCheckpoint = useCallback(async (threadId: string, checkpointId: string) => {
    try {
      setRestoring(true)
      setError(null)
      const result = await configApi.restoreCheckpoint(threadId, checkpointId)
      return result
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to restore checkpoint'
      setError(message)
      return { success: false, message }
    } finally {
      setRestoring(false)
    }
  }, [])

  useEffect(() => {
    fetchThreads()
  }, [fetchThreads])

  return {
    threads,
    selectedThread,
    checkpoints,
    checkpointState,
    loading,
    error,
    restoring,
    fetchThreads,
    fetchCheckpoints,
    fetchCheckpointState,
    restoreCheckpoint,
  }
}

// ============== P3: Benchmark Config Hook ==============

export function useBenchmarkConfig() {
  const [config, setConfig] = useState<BenchmarkConfigResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  const fetchConfig = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await configApi.getBenchmarkConfig()
      setConfig(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch benchmark config')
    } finally {
      setLoading(false)
    }
  }, [])

  const saveConfig = useCallback(async (data: BenchmarkConfigUpdate) => {
    try {
      setSaving(true)
      setError(null)
      const result = await configApi.updateBenchmarkConfig(data)
      if (result.success) {
        await fetchConfig()
      }
      return result
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to save benchmark config'
      setError(message)
      return { success: false, message }
    } finally {
      setSaving(false)
    }
  }, [fetchConfig])

  useEffect(() => {
    fetchConfig()
  }, [fetchConfig])

  return { config, loading, error, saving, saveConfig, refetch: fetchConfig }
}

// ============== P3: Benchmark Results Hook ==============

export function useBenchmarkResults(initialPage = 1, initialPageSize = 20) {
  const [results, setResults] = useState<BenchmarkResultsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [page, setPage] = useState(initialPage)
  const [pageSize] = useState(initialPageSize)
  const [novelOnly, setNovelOnly] = useState(false)
  const [running, setRunning] = useState(false)

  const fetchResults = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await configApi.getBenchmarkResults(page, pageSize, novelOnly)
      setResults(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch benchmark results')
    } finally {
      setLoading(false)
    }
  }, [page, pageSize, novelOnly])

  const runBenchmark = useCallback(async (factorNames?: string[]) => {
    try {
      setRunning(true)
      setError(null)
      const result = await configApi.runBenchmark(factorNames)
      if (result.success) {
        // Refresh results after a delay
        setTimeout(fetchResults, 2000)
      }
      return result
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to run benchmark'
      setError(message)
      return { success: false, message }
    } finally {
      setRunning(false)
    }
  }, [fetchResults])

  useEffect(() => {
    fetchResults()
  }, [fetchResults])

  return { results, loading, error, page, setPage, novelOnly, setNovelOnly, running, runBenchmark, refetch: fetchResults }
}
