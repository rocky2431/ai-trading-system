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

// ============== P3: Sandbox/Security Config ==============

export interface SandboxConfigResponse {
  timeout_seconds: number
  max_memory_mb: number
  max_cpu_seconds: number
  use_subprocess: boolean
  allowed_modules: string[]
}

export interface SandboxConfigUpdate {
  timeout_seconds?: number
  max_memory_mb?: number
  max_cpu_seconds?: number
  use_subprocess?: boolean
  allowed_modules?: string[]
}

export interface ExecutionLogEntry {
  execution_id: string
  factor_name: string | null
  status: 'success' | 'error' | 'timeout' | 'security_violation' | 'resource_exceeded'
  execution_time: number
  memory_used_mb: number | null
  error_message: string | null
  created_at: string
}

export interface ExecutionLogResponse {
  items: ExecutionLogEntry[]
  total: number
  page: number
  page_size: number
  has_next: boolean
}

export interface SecurityConfigResponse {
  research_ledger_strict: boolean
  vector_strict_mode: boolean
  human_review_enabled: boolean
  ast_security_check: boolean
  sandbox_enabled: boolean
}

export interface SecurityConfigUpdate {
  research_ledger_strict?: boolean
  vector_strict_mode?: boolean
  human_review_enabled?: boolean
  ast_security_check?: boolean
  sandbox_enabled?: boolean
}

// ============== P3: LLM Advanced Config ==============

export interface RateLimitConfig {
  requests_per_minute: number
  tokens_per_minute: number
}

export interface FallbackChainConfig {
  models: string[]
  max_retries: number
}

export interface LLMAdvancedConfigResponse {
  default_model: string
  available_models: string[]
  rate_limit: RateLimitConfig
  fallback_chain: FallbackChainConfig
  cache_enabled: boolean
  cache_ttl: number
  total_requests: number
  total_tokens: number
  total_cost: number
  cache_hit_rate: number
}

export interface LLMAdvancedConfigUpdate {
  rate_limit?: RateLimitConfig
  fallback_chain?: FallbackChainConfig
  cache_enabled?: boolean
  cache_ttl?: number
}

export interface LLMTraceEntry {
  trace_id: string
  execution_id: string
  conversation_id: string | null
  agent: string | null
  model: string
  prompt_id: string | null
  prompt_version: string | null
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  cost_estimate: number
  latency_ms: number
  cached: boolean
  created_at: string
}

export interface LLMTraceResponse {
  items: LLMTraceEntry[]
  total: number
  page: number
  page_size: number
  has_next: boolean
}

export interface LLMCostSummary {
  total_cost: number
  total_tokens: number
  total_requests: number
  cache_hit_rate: number
  by_agent: Record<string, number>
  by_model: Record<string, number>
  hourly_costs: number[]
}

// ============== P3: Derivative Data Config ==============

export interface DerivativeDataConfigResponse {
  funding_rate_enabled: boolean
  open_interest_enabled: boolean
  liquidation_enabled: boolean
  long_short_ratio_enabled: boolean
  mark_price_enabled: boolean
  taker_buy_sell_enabled: boolean
  data_source: string
  exchanges: string[]
}

export interface DerivativeDataConfigUpdate {
  funding_rate_enabled?: boolean
  open_interest_enabled?: boolean
  liquidation_enabled?: boolean
  long_short_ratio_enabled?: boolean
  mark_price_enabled?: boolean
  taker_buy_sell_enabled?: boolean
  data_source?: string
  exchanges?: string[]
}

// ============== P3: Checkpoint Management ==============

export interface CheckpointThreadInfo {
  thread_id: string
  name: string | null
  created_at: string
  last_updated: string
  checkpoint_count: number
  current_phase: string | null
}

export interface CheckpointInfo {
  checkpoint_id: string
  thread_id: string
  phase: string
  created_at: string
  metadata: Record<string, unknown>
}

export interface CheckpointListResponse {
  thread_id: string
  checkpoints: CheckpointInfo[]
  total: number
}

export interface CheckpointStateResponse {
  checkpoint_id: string
  thread_id: string
  phase: string
  hypothesis: string | null
  factors: Record<string, unknown>[]
  evaluation_results: Record<string, unknown>
  strategy: Record<string, unknown> | null
  backtest_results: Record<string, unknown> | null
  messages: Record<string, unknown>[]
  created_at: string
}

// ============== P3: Alpha Benchmark Config ==============

export interface BenchmarkConfigResponse {
  benchmark_type: string
  enabled: boolean
  auto_run_on_evaluation: boolean
  novelty_threshold: number
  min_improvement_pct: number
}

export interface BenchmarkConfigUpdate {
  benchmark_type?: string
  enabled?: boolean
  auto_run_on_evaluation?: boolean
  novelty_threshold?: number
  min_improvement_pct?: number
}

export interface BenchmarkResultEntry {
  result_id: string
  factor_name: string
  factor_ic: number
  factor_ir: number
  factor_sharpe: number
  benchmark_avg_ic: number
  benchmark_avg_ir: number
  ic_improvement: number
  ir_improvement: number
  rank: number
  total_factors: number
  is_novel: boolean
  created_at: string
}

export interface BenchmarkResultsResponse {
  items: BenchmarkResultEntry[]
  total: number
  page: number
  page_size: number
  has_next: boolean
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

  // ============== P3: Sandbox/Security ==============
  getSandboxConfig: () => api.get<SandboxConfigResponse>('/config/sandbox'),
  updateSandboxConfig: (data: SandboxConfigUpdate) => api.put<SetAPIKeysResponse>('/config/sandbox', data),
  getExecutionLogs: (page = 1, pageSize = 20, status?: string) =>
    api.get<ExecutionLogResponse>(`/config/sandbox/executions?page=${page}&page_size=${pageSize}${status ? `&status=${status}` : ''}`),
  getSecurityConfig: () => api.get<SecurityConfigResponse>('/config/security'),
  updateSecurityConfig: (data: SecurityConfigUpdate) => api.put<SetAPIKeysResponse>('/config/security', data),

  // ============== P3: LLM Advanced ==============
  getLLMAdvancedConfig: () => api.get<LLMAdvancedConfigResponse>('/config/llm/advanced'),
  updateLLMAdvancedConfig: (data: LLMAdvancedConfigUpdate) => api.put<SetAPIKeysResponse>('/config/llm/advanced', data),
  getLLMTraces: (page = 1, pageSize = 20, agent?: string, model?: string) =>
    api.get<LLMTraceResponse>(`/config/llm/traces?page=${page}&page_size=${pageSize}${agent ? `&agent=${agent}` : ''}${model ? `&model=${model}` : ''}`),
  getLLMCosts: (hours = 24) => api.get<LLMCostSummary>(`/config/llm/costs?hours=${hours}`),

  // ============== P3: Derivative Data ==============
  getDerivativeDataConfig: () => api.get<DerivativeDataConfigResponse>('/config/derivative-data'),
  updateDerivativeDataConfig: (data: DerivativeDataConfigUpdate) => api.put<SetAPIKeysResponse>('/config/derivative-data', data),

  // ============== P3: Checkpoints ==============
  listCheckpointThreads: (limit = 20) => api.get<CheckpointThreadInfo[]>(`/config/checkpoints/threads?limit=${limit}`),
  listCheckpoints: (threadId: string) => api.get<CheckpointListResponse>(`/config/checkpoints/${threadId}`),
  getCheckpointState: (threadId: string, checkpointId: string) =>
    api.get<CheckpointStateResponse>(`/config/checkpoints/${threadId}/${checkpointId}`),
  restoreCheckpoint: (threadId: string, checkpointId: string) =>
    api.post<SetAPIKeysResponse>(`/config/checkpoints/${threadId}/restore/${checkpointId}`, {}),

  // ============== P3: Alpha Benchmark ==============
  getBenchmarkConfig: () => api.get<BenchmarkConfigResponse>('/config/benchmark'),
  updateBenchmarkConfig: (data: BenchmarkConfigUpdate) => api.put<SetAPIKeysResponse>('/config/benchmark', data),
  getBenchmarkResults: (page = 1, pageSize = 20, novelOnly = false) =>
    api.get<BenchmarkResultsResponse>(`/config/benchmark/results?page=${page}&page_size=${pageSize}${novelOnly ? '&novel_only=true' : ''}`),
  runBenchmark: (factorNames?: string[]) =>
    api.post<SetAPIKeysResponse>('/config/benchmark/run', factorNames ? { factor_names: factorNames } : {}),
}
