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

**MVP Features (Phase 1-2)**:
1. Agent 驱动的因子自动生成
2. 多维度因子评估与防过拟合
3. Qlib 深度集成（非 Docker）
4. 因子知识库与去重
5. 基础监控面板

**Post-MVP Features**:
- 策略自动组装
- 实盘交易执行
- 高级风控模块
- 链上数据集成

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

#### Epic 3: Qlib 深度集成

**User Story 3.1**
**As a** 系统
**I want to** 直接执行因子代码（非 Docker 隔离）
**So that** 反馈速度更快、调试更方便

**Acceptance Criteria**:
- [ ] Qlib 直接初始化
- [ ] 因子代码实时执行
- [ ] 支持加密货币数据源
- [ ] 执行时间 < 30s/因子

**Priority**: P0
**Trace to**: plan.md Module 3

---

**User Story 3.2**
**As a** 量化研究员
**I want to** 自定义开仓平仓逻辑
**So that** 我能实现复杂的多空策略

**Acceptance Criteria**:
- [ ] 支持多空双向
- [ ] 自定义止盈止损
- [ ] 仓位管理（凯利公式/固定比例）
- [ ] 风控检查（最大回撤/单笔亏损上限）

**Priority**: P1
**Trace to**: Phase 3

---

#### Epic 4: 因子库管理

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
- **Timeline**: Phase 1-2（8周）为 MVP

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
