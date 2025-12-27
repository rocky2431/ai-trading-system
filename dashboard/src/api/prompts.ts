/**
 * Prompts API - 提示词模板管理 API 调用
 *
 * 对应后端: src/iqfmp/api/prompts/router.py
 */

import { api } from './client'

// ============== Types ==============

export interface PromptTemplate {
  agent_id: string
  agent_name: string
  prompt_id: string
  version: string
  system_prompt: string
  description: string | null
  is_custom: boolean
}

export interface PromptTemplateList {
  templates: PromptTemplate[]
  total: number
}

export interface PromptHistoryEntry {
  id: string
  agent_id: string
  old_prompt: string | null
  new_prompt: string | null
  changed_by: string
  changed_at: string
  change_type: 'created' | 'updated' | 'reset'
}

export interface PromptHistoryList {
  entries: PromptHistoryEntry[]
  total: number
}

export interface SystemModeConfig {
  // Strict mode
  strict_mode_enabled: boolean
  vector_strict_mode: boolean

  // Sandbox
  sandbox_enabled: boolean
  sandbox_timeout_seconds: number
  sandbox_memory_limit_mb: number
  sandbox_network_allowed: boolean

  // Human review
  human_review_enabled: boolean
  auto_reject_timeout_seconds: number

  // Feature flags
  ml_signal_enabled: boolean
  tool_context_enabled: boolean
  checkpoint_enabled: boolean
}

export interface SystemModeConfigResponse {
  config: SystemModeConfig
  defaults: SystemModeConfig
}

export interface UpdatePromptResponse {
  success: boolean
  message: string
  template: PromptTemplate | null
}

// ============== API ==============

export const promptsApi = {
  /**
   * 获取所有提示词模板
   */
  getTemplates: () =>
    api.get<PromptTemplateList>('/prompts/templates'),

  /**
   * 获取单个提示词模板
   */
  getTemplate: (agentId: string) =>
    api.get<PromptTemplate>(`/prompts/templates/${agentId}`),

  /**
   * 获取默认提示词模板
   */
  getDefaultTemplate: (agentId: string) =>
    api.get<PromptTemplate>(`/prompts/templates/${agentId}/default`),

  /**
   * 更新提示词模板
   */
  updateTemplate: (agentId: string, systemPrompt: string | null) =>
    api.put<UpdatePromptResponse>(`/prompts/templates/${agentId}`, {
      system_prompt: systemPrompt,
    }),

  /**
   * 重置提示词模板为默认
   */
  resetTemplate: (agentId: string) =>
    api.post<UpdatePromptResponse>(`/prompts/templates/${agentId}/reset`),

  /**
   * 获取提示词变更历史
   */
  getHistory: (agentId?: string, limit: number = 50) =>
    api.get<PromptHistoryList>('/prompts/history', {
      agent_id: agentId,
      limit,
    }),

  /**
   * 获取系统模式配置
   */
  getSystemMode: () =>
    api.get<SystemModeConfigResponse>('/prompts/system-mode'),

  /**
   * 更新系统模式配置
   */
  updateSystemMode: (updates: Partial<SystemModeConfig>) =>
    api.patch<SystemModeConfigResponse>('/prompts/system-mode', updates),
}
