/**
 * Config API Hooks
 */

import { useState, useEffect, useCallback } from 'react'
import { configApi, type ConfigStatusResponse, type AvailableModelsResponse, type SavedAPIKeysResponse, type AgentConfigResponse, type DataConfigResponse, type FactorMiningConfigResponse, type RiskControlConfigResponse, type SetAPIKeysRequest, type SetAgentConfigRequest, type SetDataConfigRequest, type SetFactorMiningConfigRequest, type SetRiskControlConfigRequest } from '@/api/config'

// ============== Status Hook ==============

export function useConfigStatus() {
  const [status, setStatus] = useState<ConfigStatusResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchStatus = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await configApi.getStatus()
      setStatus(response.data)
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
      const response = await configApi.getModels()
      setModels(response.data)
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

  const fetchKeys = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await configApi.getAPIKeys()
      setKeys(response.data)
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
      const response = await configApi.setAPIKeys(data)
      if (response.data.success) {
        await fetchKeys()
      }
      return response.data
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to save API keys'
      setError(message)
      return { success: false, message }
    } finally {
      setSaving(false)
    }
  }, [fetchKeys])

  useEffect(() => {
    fetchKeys()
  }, [fetchKeys])

  return { keys, loading, error, saving, saveKeys, refetch: fetchKeys }
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
      const response = await configApi.testLLM()
      setLLMResult({
        success: response.data.success,
        message: response.data.message
      })
      return response.data
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
      const response = await configApi.testExchange()
      setExchangeResult({
        success: response.data.success,
        message: response.data.message
      })
      return response.data
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
      const response = await configApi.getAgentConfig()
      setConfig(response.data)
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
      const response = await configApi.setAgentConfig(data)
      if (response.data.success) {
        await fetchConfig()
      }
      return response.data
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
      const response = await configApi.getDataConfig()
      setConfig(response.data)
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
      const response = await configApi.setDataConfig(data)
      if (response.data.success) {
        await fetchConfig()
      }
      return response.data
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
      const response = await configApi.getFactorMiningConfig()
      setConfig(response.data)
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
      const response = await configApi.setFactorMiningConfig(data)
      if (response.data.success) {
        await fetchConfig()
      }
      return response.data
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
      const response = await configApi.getRiskControlConfig()
      setConfig(response.data)
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
      const response = await configApi.setRiskControlConfig(data)
      if (response.data.success) {
        await fetchConfig()
      }
      return response.data
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
