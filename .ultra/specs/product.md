# Product Specification - IQFMP 智能量化因子挖掘平台

> **Source of Truth**: This document defines WHAT the system does and WHY. Technology choices belong in `architecture.md`.

## 1. Problem Statement

### 1.1 Core Problem

构建一个端到端的自动化量化研究平台，从因子生成到策略部署全流程自动化，超越 RD-Agent 的专业级加密货币量化研究系统。

**核心挑战**:
- 人工因子挖掘效率低、覆盖面窄
- 现有工具（如 RD-Agent）使用 Docker 隔离，反馈慢、调试难
- 加密货币市场特性（高波动、24/7、衍生品复杂）需要专门支持
- 过拟合是量化研究的头号敌人

### 1.2 Current Pain Points

1. **因子挖掘效率低**: 人工设计因子耗时长，探索空间有限
2. **过拟合严重**: 单市场、单时段验证导致策略上线后失效
3. **研究到实盘割裂**: 回测代码无法直接用于实盘
4. **缺乏系统性防护**: 没有机制约束"因子动物园"膨胀
5. **反馈循环慢**: Docker 隔离执行因子代码，调试困难

### 1.3 How Users Currently Solve This

- **手动因子设计**: 依赖经验和直觉，覆盖面有限
- **RD-Agent**: Docker 隔离，反馈慢，加密货币支持弱
- **Qlib 原生**: 功能强大但缺乏 Agent 驱动的自动化
- **多个独立系统**: 研究、回测、实盘分离，维护成本高

---

## 2. Users & Stakeholders

### 2.1 Primary User Segments

- **Segment 1**: 量化研究员
  - Size: 1-3 人（初期）
  - Priority: P0
  - 需求: 快速验证因子想法，系统性防过拟合

- **Segment 2**: 策略管理者
  - Size: 1 人
  - Priority: P0
  - 需求: 监控策略表现，管理因子库

### 2.2 User Characteristics

- **Technical Proficiency**: Advanced（熟悉 Python、量化研究）
- **Common Behaviors**: 日常研究、周度策略评审
- **Key Needs**: 效率、可靠性、防过拟合

---

## 3. User Stories

### 3.1 MVP Feature Scope

> **Research Decision**: 用户选择**端到端完整**模式，MVP 包含从因子生成到实盘执行的完整链路。

**MVP Features (12 weeks, ~53 工作日)**:

| Epic | 核心能力 | 优先级 | 预估 |
|------|----------|--------|------|
| **E1: 因子生成** | Agent 驱动的因子自动生成 | P0 | 10d |
| **E2: 防过拟合** | Research Ledger + 多切片验证 + 动态阈值 | P0 | 12d |
| **E3: 策略组装** | 多因子策略生成 + 回测优化 | P0 | 12d |
| **E4: 交易执行** | ccxt 实盘执行 + 风控模块 | P0 | 9d |
| **E5: Dashboard** | React 监控面板 (与后端并行开发) | P1 | 10d |

**Post-MVP Features (Phase 2+)**:
- 因子自动修复 (LLM 迭代优化)
- 因子聚类分析
- 贝叶斯参数优化
- 链上数据集成
- 社交情绪因子

### 3.2 Epic Breakdown

#### Epic 1: Agent 协作与因子生成

**User Story 1.1**
**As a** 量化研究员
**I want to** 通过自然语言描述因子思路，让系统自动生成因子代码
**So that** 我能快速探索更多因子可能性

**Acceptance Criteria**:
- [ ] 支持自然语言输入因子描述
- [ ] 自动生成符合 Qlib 规范的因子代码
- [ ] 代码通过安全检查（无危险操作）
- [ ] 执行成功率 ≥ 80%

**Priority**: P0
**Trace to**: Phase 1-2

---

**User Story 1.2**
**As a** 量化研究员
**I want to** 指定因子家族（趋势/波动/流动性等）进行定向挖掘
**So that** 我能在特定机制下系统性探索

**Acceptance Criteria**:
- [ ] 支持 6 大因子家族选择
- [ ] 生成的因子严格使用指定家族的数据字段
- [ ] 自动避免引用不存在的数据

**Priority**: P0
**Trace to**: advice-for-plan.md Section II

---

#### Epic 2: 防过拟合系统

**User Story 2.1**
**As a** 量化研究员
**I want to** 每个因子自动在多市场、多时段验证
**So that** 我能识别真正有效的因子

**Acceptance Criteria**:
- [ ] Train/Valid/Test 三段切分
- [ ] 多市场验证（BTC/ETH + Altcoins + 小市值）
- [ ] 多频率验证（1h/4h/1d）
- [ ] 多环境验证（牛/熊 × 高/低波动）

**Priority**: P0
**Trace to**: advice-for-plan.md Section I.1

---

**User Story 2.2**
**As a** 系统
**I want to** 维护研究账本并动态调整阈值
**So that** 防止 p-hacking

**Acceptance Criteria**:
- [ ] 每次因子评估记录到研究账本
- [ ] 全局试验次数 N 实时追踪
- [ ] 评价阈值随 N 增加而变严
- [ ] 支持 Deflated Sharpe Ratio 计算

**Priority**: P0
**Trace to**: advice-for-plan.md Section I.2

---

**User Story 2.3**
**As a** 量化研究员
**I want to** 查看因子的多维度稳定性报告
**So that** 我能了解因子在不同条件下的表现

**Acceptance Criteria**:
- [ ] 时间稳定性（按月/季度 IC）
- [ ] 市场稳定性（大/中/小市值分组）
- [ ] 环境稳定性（regime 划分）
- [ ] 可视化报告

**Priority**: P0
**Trace to**: advice-for-plan.md Section I.3

---

#### Epic 3: 策略组装引擎

**User Story 3.1** (S-001)
**As a** 量化研究员
**I want to** 从验证通过的因子中选择组合
**So that** 我能构建多因子策略

**Acceptance Criteria**:
- [ ] 支持因子库筛选（按家族/稳定性/IC）
- [ ] 支持手动选择因子组合
- [ ] 显示因子间相关性矩阵
- [ ] 警告高度相关的因子组合

**Priority**: P0
**Estimated**: 2d

---

**User Story 3.2** (S-002)
**As a** 系统
**I want to** 自动生成多因子策略代码
**So that** 减少手动编码工作

**Acceptance Criteria**:
- [ ] LLM 生成策略框架代码
- [ ] 支持多因子加权组合
- [ ] 代码符合 Qlib Strategy 规范
- [ ] 生成代码可直接执行

**Priority**: P0
**Estimated**: 3d

---

**User Story 3.3** (S-003)
**As a** 量化研究员
**I want to** 自定义开仓/平仓/止盈止损逻辑
**So that** 我能实现复杂的多空策略

**Acceptance Criteria**:
- [ ] 支持多空双向信号
- [ ] 自定义止盈止损条件
- [ ] 仓位管理（凯利公式/固定比例）
- [ ] 时间止损支持

**Priority**: P0
**Estimated**: 2d

---

**User Story 3.4** (S-004)
**As a** 系统
**I want to** 执行策略回测并生成报告
**So that** 验证策略有效性

**Acceptance Criteria**:
- [ ] 多空双向回测
- [ ] 完整绩效指标（Sharpe/MaxDD/Win Rate）
- [ ] 分时段/分市场绩效分解
- [ ] 可视化回测报告

**Priority**: P0
**Estimated**: 3d

---

#### Epic 4: 交易执行层

**User Story 4.1** (T-001)
**As a** 系统
**I want to** 通过 ccxt 连接 Binance/OKX
**So that** 获取实时数据和执行订单

**Acceptance Criteria**:
- [ ] 支持 Binance Futures API
- [ ] 支持 OKX Swap API
- [ ] 统一的交易所抽象接口
- [ ] 自动重连机制

**Priority**: P0
**Estimated**: 2d

---

**User Story 4.2** (T-002)
**As a** 系统
**I want to** 执行多空双向订单
**So that** 实现策略信号落地

**Acceptance Criteria**:
- [ ] 支持限价/市价订单
- [ ] 支持开多/开空/平多/平空
- [ ] 订单状态追踪
- [ ] 部分成交处理

**Priority**: P0
**Estimated**: 2d

---

**User Story 4.3** (T-003)
**As a** 系统
**I want to** 实时监控持仓和 PnL
**So that** 掌握账户状态

**Acceptance Criteria**:
- [ ] 实时持仓同步
- [ ] 未实现/已实现 PnL 计算
- [ ] 保证金使用率监控
- [ ] WebSocket 实时推送

**Priority**: P0
**Estimated**: 2d

---

**User Story 4.4** (T-004)
**As a** 系统
**I want to** 执行风控检查
**So that** 防止重大亏损

**Acceptance Criteria**:
- [ ] 最大回撤限制（可配置，默认 20%）
- [ ] 单笔亏损上限（可配置）
- [ ] 持仓集中度检查
- [ ] 触发风控时自动减仓/平仓

**Priority**: P0
**Estimated**: 2d

---

**User Story 4.5** (T-005)
**As a** 用户
**I want to** 紧急情况一键平仓所有头寸
**So that** 快速止损

**Acceptance Criteria**:
- [ ] Dashboard 一键平仓按钮
- [ ] API 紧急平仓接口
- [ ] 平仓前确认提示
- [ ] 平仓结果通知

**Priority**: P0
**Estimated**: 1d

---

#### Epic 5: 因子库管理

**User Story 4.1**
**As a** 量化研究员
**I want to** 因子自动去重和聚类
**So that** 避免因子库膨胀和冗余

**Acceptance Criteria**:
- [ ] 向量化因子表示
- [ ] 相似度检索（阈值 0.85）
- [ ] 自动聚类（层次聚类）
- [ ] 标记代表因子和冗余因子

**Priority**: P1
**Trace to**: advice-for-plan.md Section I.5

---

#### Epic 5: 监控与可视化

**User Story 5.1**
**As a** 策略管理者
**I want to** 实时查看 Agent 运行状态和任务队列
**So that** 我能监控系统健康度

**Acceptance Criteria**:
- [ ] Agent 状态展示
- [ ] 任务队列可视化
- [ ] 性能指标（LLM 延迟、成功率）
- [ ] 资源使用监控

**Priority**: P1
**Trace to**: Phase 5

---

## 4. Functional Requirements

### 4.1 Core Capabilities

1. **因子生成引擎**
   - Input: 自然语言描述 + 因子家族选择
   - Output: Qlib 兼容的因子代码
   - Business Rules: 必须使用已定义的数据字段，禁止引用不存在数据
   - **Trace to**: User Story 1.1, 1.2

2. **因子评估引擎**
   - Input: 因子代码 + 评估配置
   - Output: 多维度评估报告（IC/IR/Sharpe/稳定性）
   - Business Rules: 必须通过所有数据切片验证
   - **Trace to**: User Story 2.1, 2.3

3. **研究账本**
   - Input: 因子评估结果
   - Output: 全局试验统计 + 动态阈值
   - Business Rules: 阈值随试验次数变严
   - **Trace to**: User Story 2.2

4. **因子知识库**
   - Input: 验证通过的因子
   - Output: 向量化存储 + 相似度检索
   - Business Rules: 相似度 > 0.85 标记为重复
   - **Trace to**: User Story 4.1

### 4.2 Data Operations

- **Create**: 因子、策略、回测任务
- **Read**: 因子库查询、历史评估结果、市场数据
- **Update**: 因子状态（candidate/rejected/core）
- **Delete**: 软删除冗余因子

### 4.3 Integration Requirements

- **Binance API**: 市场数据获取 (ccxt)
- **OKX API**: 备选数据源 (ccxt)
- **OpenRouter**: LLM 调用
- **Qlib**: 研究核心框架

---

## 5. Non-Functional Requirements

### 5.1 Performance Requirements

- **因子生成**: < 30s/个
- **因子评估**: < 5min/因子（含多维度验证）
- **端到端Pipeline**: < 4h（100因子→10策略→最优策略）
- **LLM 响应**: < 10s/请求
- **Dashboard 加载**: < 2s

### 5.2 Security Requirements

- **代码执行**: AST 安全检查，禁止危险操作
- **API Keys**: 加密存储，环境变量
- **数据加密**: TLS 1.3 传输
- **访问控制**: API 认证（JWT）

### 5.3 Scalability Requirements

- **因子生成并发**: 10 并发
- **回测分布式**: Ray/Celery 支持
- **数据存储**: TimescaleDB 自动分区

### 5.4 Reliability Requirements

- **Uptime Target**: 99.5%
- **LLM 备份**: 多供应商（OpenRouter 支持切换）
- **数据备份**: 每日增量备份

---

## 6. Constraints

### 6.1 Technical Constraints

- **Must use**: Qlib 作为研究核心（非 Docker）
- **Must use**: ccxt 作为交易所连接层
- **Cannot use**: 未经验证的第三方量化库

### 6.2 Business Constraints

- **Team Size**: 1 人开发（初期）
- **Timeline**: 12 周完成端到端 MVP
- **Development Strategy**: 分 3 阶段递进
  - Phase 1 (Week 1-4): 核心研究引擎
  - Phase 2 (Week 5-8): 策略与执行
  - Phase 3 (Week 9-12): Dashboard 与优化

---

## 7. Risks & Mitigation

### 7.1 Critical Risks

| Risk | Probability | Impact | Category |
|------|------------|--------|----------|
| LLM 生成质量不稳定 | 高 | 高 | Technical |
| Qlib 加密货币兼容性 | 中 | 高 | Technical |
| 过拟合控制不充分 | 中 | 高 | Business |
| 性能不达标 | 中 | 中 | Technical |

### 7.2 Mitigation Strategies

1. **LLM 质量问题**
   - Mitigation: 多模型 ensemble、Prompt 版本控制、人工审核
   - Contingency: 降级到模板库

2. **Qlib 兼容性**
   - Mitigation: 深度测试、扩展加密货币支持
   - Contingency: Fork 维护

3. **过拟合**
   - Mitigation: 多维度验证、研究账本、动态阈值
   - Contingency: 人工审核强制介入

---

## 8. Success Metrics

### 8.1 Key Performance Indicators (KPIs)

**研究效率**:
- 因子生成速度: > 100 因子/天
- 有效因子率: > 5%（通过所有验证）

**因子质量**:
- 平均 IC: > 0.03
- 平均 IR: > 1.0
- 稳定性得分: > 0.6

**系统性能**:
- 端到端延迟: < 4h/pipeline
- 系统可用性: > 99.5%

---

## 9. Out of Scope

- **链上数据**: Phase 2+ 考虑
- **社交情绪**: Phase 2+ 考虑
- **多交易所套利**: Phase 3+ 考虑
- **高频策略 (< 1min)**: 当前不支持

---

## 10. Dependencies

### 10.1 External Dependencies

- **OpenRouter API**: LLM 调用
- **Binance/OKX API**: 市场数据
- **Qlib**: 量化研究框架

### 10.2 Internal Dependencies

- 已有 `plan.md` 和 `advice-for-plan.md` 作为设计参考

---

**Document Status**: Draft
**Last Updated**: 2025-12-09
