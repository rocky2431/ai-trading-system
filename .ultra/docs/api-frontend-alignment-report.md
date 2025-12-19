  📊 API/前端对齐扫描报告

  一、总览

  | 模块              | 后端状态 | 前端状态  | 对齐程度    |
  |-----------------|------|-------|---------|
  | Factors         | ✅ 完整 | ✅ 完整  | ✅ 100%  |
  | Research        | ✅ 完整 | ✅ 完整  | ✅ 100%  |
  | Config          | ✅ 完整 | ✅ 完整  | ✅ 100%  |
  | Data            | ✅ 完整 | ✅ 完整  | ✅ 100%  |
  | Backtest        | ✅ 完整 | ✅ 完整  | ✅ 100%  |
  | System          | ✅ 完整 | ⚠️ 部分 | ⚠️ 75%  |
  | Auth            | ✅ 完整 | ❌ 缺失  | ❌ 0%    |
  | Pipeline        | ✅ 完整 | ❌ 缺失  | ❌ 0%    |
  | Strategies (独立) | ✅ 存在 | ⚠️ 冗余 | ⚠️ 架构重叠 |

  ---
  二、🔴 严重问题 (Critical)

  1. Auth API 前端完全缺失

  后端已实现 (src/iqfmp/api/auth/router.py):
  POST /api/v1/auth/register
  POST /api/v1/auth/login
  POST /api/v1/auth/refresh
  GET  /api/v1/auth/me

  前端状态: 无 dashboard/src/api/auth.ts 文件

  影响: 用户认证流程无法工作，无法登录/注册

  ---
  2. Pipeline/RD-Loop API 前端完全缺失

  后端已实现 (src/iqfmp/api/pipeline/router.py):
  POST /api/v1/pipeline/run
  GET  /api/v1/pipeline/{run_id}/status
  GET  /api/v1/pipeline/runs
  POST /api/v1/pipeline/{run_id}/cancel
  WS   /api/v1/pipeline/{run_id}/ws

  # RD Loop 核心功能
  POST /api/v1/pipeline/rd-loop/run
  GET  /api/v1/pipeline/rd-loop/{run_id}/state
  GET  /api/v1/pipeline/rd-loop/{run_id}/statistics
  GET  /api/v1/pipeline/rd-loop/{run_id}/factors
  POST /api/v1/pipeline/rd-loop/{run_id}/stop
  GET  /api/v1/pipeline/rdloop/state
  GET  /api/v1/pipeline/rd-loop/runs

  前端状态: 无 dashboard/src/api/pipeline.ts 文件

  影响: RD Loop 研发循环功能无法使用，因子挖掘自动化流程不可用

  ---
  三、🟡 中等问题 (Medium)

  1. System API 部分缺失

  后端已实现但前端未调用:
  GET  /api/v1/system/agents     ← 前端未调用
  GET  /api/v1/system/tasks      ← 前端未调用
  WS   /api/v1/system/ws         ← 前端未调用

  前端已调用 (system.ts):
  getStatus()       → GET /system/status     ✅
  getResources()    → GET /system/resources  ✅
  getLLMMetrics()   → GET /system/llm        ✅
  getAgentConfigs() → GET /system/agent-configs ✅
  // ... agent config CRUD ✅

  影响: 监控页面无法显示 Agent 列表和任务队列实时状态

  ---
  2. Strategies Router 架构冗余

  问题描述:
  - 后端同时存在 /strategies 和 /backtest/strategies 两个路由
  - src/iqfmp/api/strategies/router.py - 独立策略路由
  - src/iqfmp/api/backtest/router.py - 包含 /strategies 子路由

  前端调用: 仅使用 /backtest/strategies

  影响: 存在两套相似但不完全相同的策略 API，可能导致混淆

  ---
  四、🟢 对齐良好的模块

  Factors API (100% 对齐)

  | 前端调用               | 后端端点                        | 状态  |
  |--------------------|-----------------------------|-----|
  | generate()         | POST /factors/generate      | ✅   |
  | create()           | POST /factors               | ✅   |
  | list()             | GET /factors                | ✅   |
  | get()              | GET /factors/{id}           | ✅   |
  | stats()            | GET /factors/stats          | ✅   |
  | evaluate()         | POST /factors/{id}/evaluate | ✅   |
  | updateStatus()     | PUT /factors/{id}/status    | ✅   |
  | delete()           | DELETE /factors/{id}        | ✅   |
  | createMiningTask() | POST /factors/mining        | ✅   |
  | listMiningTasks()  | GET /factors/mining         | ✅   |
  | getMiningTask()    | GET /factors/mining/{id}    | ✅   |
  | cancelMiningTask() | DELETE /factors/mining/{id} | ✅   |
  | getLibraryStats()  | GET /factors/library/stats  | ✅   |
  | compare()          | POST /factors/compare       | ✅   |

  Research API (100% 对齐)

  | 前端调用            | 后端端点                     | 状态  |
  |-----------------|--------------------------|-----|
  | listLedger()    | GET /research/ledger     | ✅   |
  | getStats()      | GET /research/stats      | ✅   |
  | getThresholds() | GET /research/thresholds | ✅   |

  Config API (100% 对齐)

  所有 14 个端点完全对齐 ✅

  Data API (100% 对齐)

  所有 13 个端点完全对齐 ✅

  Backtest API (100% 对齐)

  所有 10 个端点完全对齐 ✅

  ---
  五、📋 Schema 不一致

  FactorStatsResponse 字段缺失

  后端返回 (factors/router.py:186-198):
  FactorStatsResponse(
      total_factors=...,
      by_status=...,
      total_trials=...,
      current_threshold=...,
      evaluated_count=...,     # ← 前端缺失
      pass_rate=...,           # ← 前端缺失
      avg_ic=...,              # ← 前端缺失
      avg_sharpe=...,          # ← 前端缺失
      pending_count=...,       # ← 前端缺失
  )

  前端类型 (factors.ts:47-53):
  export interface FactorStatsResponse {
    total_factors: number
    by_status: Record<string, number>
    total_trials: number
    current_threshold: number
    // 缺少 5 个字段!
  }

  ---
  六、📌 与本次修改的关系

  本次修改涉及的文件:

  1. orchestrator.py - PostgresCheckpointSaver (基础设施层)
  2. pipeline_builder.py - _default_checkpoint_saver (基础设施层)
  3. factor_engine.py - require_qlib 强制模式
  4. qlib_crypto.py - QlibExpressionEngine 强制模式

  对 API 的影响:

  - 无直接影响 - 修改均在核心计算层,未涉及 API 层
  - require_qlib=True 会在 Qlib 不可用时抛出 QlibUnavailableError
  - API 层错误处理仍然有效 (返回 422/500 错误)

  ---
  七、🎯 修复优先级建议

  | 优先级 | 问题                       | 工作量 | 建议               |
  |-----|--------------------------|-----|------------------|
  | P0  | Auth API 前端缺失            | 中   | 立即创建 auth.ts     |
  | P0  | Pipeline API 前端缺失        | 大   | 立即创建 pipeline.ts |
  | P1  | System agents/tasks 未调用  | 小   | 补充 3 个 API 调用    |
  | P1  | FactorStatsResponse 字段缺失 | 小   | 更新 TypeScript 类型 |
  | P2  | Strategies 架构冗余          | 中   | 考虑统一为一个路由        |

  ---
  八、结论

  诚实评估:
  - 核心业务功能 (Factors, Research, Backtest, Config, Data) API 对齐度 100%
  - 两个关键模块完全缺失前端实现: Auth 和 Pipeline
  - 本次 Qlib 强制模式修改不影响 API 层对齐
  - 系统整体可用,但认证流程和 RD Loop 功能暂时无法通过前端使用