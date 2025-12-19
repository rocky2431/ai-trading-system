/**
 * API Module - 导出所有 API 函数
 *
 * 完整对齐后端 API (100% alignment)
 */

export { api, ApiError } from './client'

// ============== Auth API ==============
export { authApi, tokenStorage } from './auth'
export type {
  UserRole,
  UserResponse,
  TokenResponse,
  LoginRequest,
  RegisterRequest,
  RefreshTokenRequest,
} from './auth'

// ============== Factors API ==============
export { factorsApi } from './factors'
export type {
  FactorResponse,
  FactorListResponse,
  FactorMetricsResponse,
  StabilityResponse,
  FactorStatsResponse,
  FactorEvaluateResponse,
  MiningTaskStatus,
  MiningTaskListResponse,
  MiningTaskCreateRequest,
  MiningTaskCreateResponse,
  MiningTaskCancelResponse,
  FactorLibraryStats,
  FactorCompareResponse,
} from './factors'

// ============== Research API ==============
export { researchApi } from './research'
export type {
  TrialResponse,
  LedgerListResponse,
  StatisticsResponse,
  StatsResponse,
  ThresholdHistoryItem,
  ThresholdConfigResponse,
  ThresholdResponse,
  LedgerListParams,
} from './research'

// ============== System API ==============
export { systemApi } from './system'
export type {
  AgentResponse,
  AgentType,
  AgentConfigResponse,
  AgentConfigListResponse,
  AgentConfigUpdateRequest,
  AgentConfigOperationResponse,
  TaskQueueItemResponse,
  LLMMetricsResponse,
  CPUMetrics,
  MemoryMetrics,
  DiskMetrics,
  ResourceMetricsResponse,
  SystemStatusResponse,
  SystemWSMessageType,
  SystemWSMessage,
} from './system'

// ============== Pipeline API ==============
export { pipelineApi } from './pipeline'
export type {
  PipelineStatus,
  PipelineType,
  PipelineConfig,
  PipelineRunRequest,
  PipelineRunResponse,
  PipelineStatusResponse,
  PipelineListResponse,
  RDLoopPhase,
  HypothesisFamily,
  RDLoopConfigRequest,
  RDLoopRunRequest,
  RDLoopRunResponse,
  RDLoopStateResponse,
  RDLoopIterationResult,
  RDLoopStatisticsResponse,
  RDLoopCoreFactorResponse,
  RDLoopRunInfo,
  RDLoopRunsListResponse,
  PipelineWSMessageType,
  PipelineWSMessage,
} from './pipeline'

// ============== Strategies API (独立路由，推荐使用) ==============
export { strategiesApi } from './strategies'
export type {
  StrategyCreateRequest as StrategyCreate,
  StrategyUpdateRequest as StrategyUpdate,
  StrategyResponse as Strategy,
  StrategyListResponse as StrategyList,
  BacktestRequest as StrategyBacktestRequest,
  BacktestResultResponse as StrategyBacktestResult,
  BacktestListResponse as StrategyBacktestList,
} from './strategies'

// ============== Backtest API ==============
export { backtestApi } from './backtest'
export type {
  BacktestConfig,
  BacktestMetrics,
  BacktestResponse,
  BacktestDetailResponse,
  BacktestStatsResponse,
  BacktestEquityCurve,
  BacktestTrade,
  GenericResponse,
} from './backtest'
