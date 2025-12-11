/**
 * Config API - 系统配置 API
 */

import { api } from './client'

// ============== Types ==============

export interface FeaturesStatus {
  factor_generation: boolean
  factor_evaluation: boolean
  strategy_assembly: boolean
  backtest_optimization: boolean
  live_trading: boolean
}

export interface ConfigStatusResponse {
  llm_configured: boolean
  llm_provider: string | null
  llm_model: string | null
  exchange_configured: boolean
  exchange_id: string | null
  qlib_available: boolean
  timescaledb_connected: boolean
  redis_connected: boolean
  features: FeaturesStatus
}

export interface ModelInfo {
  id: string
  name: string
  provider: string
  context_length: number | null
  use_case: string | null
}

export interface EmbeddingModelInfo {
  id: string
  name: string
  dimensions: number
}

export interface AvailableModelsResponse {
  models: Record<string, ModelInfo[]>
  embedding_models: Record<string, EmbeddingModelInfo[]>
}

export interface SavedAPIKeysResponse {
  api_key: string | null
  model: string | null
  embedding_model: string | null
  exchange_id: string | null
  exchange_api_key: string | null
}

export interface SetAPIKeysRequest {
  provider?: string
  api_key?: string
  model?: string
  embedding_model?: string
  exchange_id?: string
  exchange_api_key?: string
  exchange_secret?: string
}

export interface SetAPIKeysResponse {
  success: boolean
  message: string
}

export interface TestLLMResponse {
  success: boolean
  model: string | null
  response_time_ms: number | null
  message: string
}

export interface TestExchangeResponse {
  success: boolean
  exchange_id: string | null
  latest_btc_price: number | null
  message: string
}

// Agent Config
export interface AgentModelConfig {
  agent_id: string
  agent_name: string
  description: string
  model_id: string
  enabled: boolean
}

export interface AgentConfigResponse {
  agents: AgentModelConfig[]
}

export interface SetAgentConfigRequest {
  agent_id: string
  model_id?: string
  enabled?: boolean
}

// Data Config
export interface FrequencyOption {
  id: string
  name: string
  description: string
}

export interface DataConfigResponse {
  data_frequency: string
  data_source: string
  symbols: string[]
  qlib_data_path: string | null
  frequency_options: FrequencyOption[]
}

export interface SetDataConfigRequest {
  data_frequency?: string
  data_source?: string
  symbols?: string[]
}

// Factor Mining Config
export interface FactorFamilyOption {
  id: string
  name: string
  description: string
  enabled: boolean
}

export interface EvaluationConfig {
  min_ic: number
  min_ir: number
  min_sharpe: number
  max_turnover: number
  cv_folds: number
  train_ratio: number
  valid_ratio: number
  test_ratio: number
  use_dynamic_threshold: boolean
  deflation_rate: number
}

export interface FactorMiningConfigResponse {
  factor_families: FactorFamilyOption[]
  evaluation: EvaluationConfig
  max_concurrent_generation: number
  code_execution_timeout: number
}

export interface SetFactorMiningConfigRequest {
  factor_families?: FactorFamilyOption[]
  evaluation?: EvaluationConfig
  max_concurrent_generation?: number
  code_execution_timeout?: number
}

// Risk Control Config
export interface RiskControlConfig {
  max_single_loss_pct: number
  max_daily_loss_pct: number
  max_position_pct: number
  max_total_position_pct: number
  emergency_close_threshold: number
}

export interface RiskControlConfigResponse {
  config: RiskControlConfig
  is_live_trading_enabled: boolean
}

export interface SetRiskControlConfigRequest {
  max_single_loss_pct?: number
  max_daily_loss_pct?: number
  max_position_pct?: number
  max_total_position_pct?: number
  emergency_close_threshold?: number
}

// ============== API ==============

export const configApi = {
  // Status
  getStatus: () => api.get<ConfigStatusResponse>('/config/status'),

  // Models
  getModels: () => api.get<AvailableModelsResponse>('/config/models'),

  // API Keys
  getAPIKeys: () => api.get<SavedAPIKeysResponse>('/config/api-keys'),
  setAPIKeys: (data: SetAPIKeysRequest) => api.post<SetAPIKeysResponse>('/config/api-keys', data),
  deleteAPIKeys: (keyType: 'llm' | 'exchange') => api.delete<SetAPIKeysResponse>(`/config/api-keys/${keyType}`),

  // Test Connections
  testLLM: () => api.post<TestLLMResponse>('/config/test-llm', {}),
  testExchange: () => api.post<TestExchangeResponse>('/config/test-exchange', {}),

  // Agent Config
  getAgentConfig: () => api.get<AgentConfigResponse>('/config/agents'),
  setAgentConfig: (data: SetAgentConfigRequest) => api.post<SetAPIKeysResponse>('/config/agents', data),

  // Data Config
  getDataConfig: () => api.get<DataConfigResponse>('/config/data'),
  setDataConfig: (data: SetDataConfigRequest) => api.post<SetAPIKeysResponse>('/config/data', data),

  // Factor Mining Config
  getFactorMiningConfig: () => api.get<FactorMiningConfigResponse>('/config/factor-mining'),
  setFactorMiningConfig: (data: SetFactorMiningConfigRequest) => api.post<SetAPIKeysResponse>('/config/factor-mining', data),

  // Risk Control Config
  getRiskControlConfig: () => api.get<RiskControlConfigResponse>('/config/risk-control'),
  setRiskControlConfig: (data: SetRiskControlConfigRequest) => api.post<SetAPIKeysResponse>('/config/risk-control', data),
}
