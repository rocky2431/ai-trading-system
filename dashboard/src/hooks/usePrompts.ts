/**
 * usePrompts - 提示词管理 Hook
 *
 * 提供提示词模板管理和系统配置功能
 */

import { useState, useEffect, useCallback } from 'react'
import {
  promptsApi,
  PromptTemplate,
  PromptHistoryEntry,
  SystemModeConfig,
} from '@/api/prompts'

interface UsePromptsResult {
  // 数据
  templates: PromptTemplate[]
  history: PromptHistoryEntry[]
  systemConfig: SystemModeConfig | null
  defaultConfig: SystemModeConfig | null

  // 状态
  loading: boolean
  error: Error | null

  // 操作
  refresh: () => Promise<void>
  updateTemplate: (agentId: string, systemPrompt: string | null) => Promise<void>
  resetTemplate: (agentId: string) => Promise<void>
  updateSystemConfig: (updates: Partial<SystemModeConfig>) => Promise<void>
  loadHistory: (agentId?: string) => Promise<void>
}

export function usePrompts(): UsePromptsResult {
  // 数据状态
  const [templates, setTemplates] = useState<PromptTemplate[]>([])
  const [history, setHistory] = useState<PromptHistoryEntry[]>([])
  const [systemConfig, setSystemConfig] = useState<SystemModeConfig | null>(null)
  const [defaultConfig, setDefaultConfig] = useState<SystemModeConfig | null>(null)

  // 加载状态
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  // 刷新所有数据
  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)

    try {
      const [templatesRes, systemRes] = await Promise.all([
        promptsApi.getTemplates(),
        promptsApi.getSystemMode(),
      ])

      setTemplates(templatesRes.templates)
      setSystemConfig(systemRes.config)
      setDefaultConfig(systemRes.defaults)
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to load prompts data'))
    } finally {
      setLoading(false)
    }
  }, [])

  // 加载历史
  const loadHistory = useCallback(async (agentId?: string) => {
    try {
      const res = await promptsApi.getHistory(agentId, 50)
      setHistory(res.entries)
    } catch (err) {
      console.error('Failed to load history:', err)
    }
  }, [])

  // 初始加载
  useEffect(() => {
    refresh()
    loadHistory()
  }, [refresh, loadHistory])

  // 更新模板
  const updateTemplate = useCallback(
    async (agentId: string, systemPrompt: string | null) => {
      await promptsApi.updateTemplate(agentId, systemPrompt)
      await refresh()
      await loadHistory(agentId)
    },
    [refresh, loadHistory]
  )

  // 重置模板
  const resetTemplate = useCallback(
    async (agentId: string) => {
      await promptsApi.resetTemplate(agentId)
      await refresh()
      await loadHistory(agentId)
    },
    [refresh, loadHistory]
  )

  // 更新系统配置
  const updateSystemConfig = useCallback(
    async (updates: Partial<SystemModeConfig>) => {
      const res = await promptsApi.updateSystemMode(updates)
      setSystemConfig(res.config)
    },
    []
  )

  return {
    templates,
    history,
    systemConfig,
    defaultConfig,
    loading,
    error,
    refresh,
    updateTemplate,
    resetTemplate,
    updateSystemConfig,
    loadHistory,
  }
}
