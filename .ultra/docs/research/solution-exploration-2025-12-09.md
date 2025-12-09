# Round 2: Solution Exploration Report

**Date**: 2025-12-09
**Project**: IQFMP 智能量化因子挖掘平台

---

## 1. User Requirements Summary

| 决策点 | 用户选择 | 影响 |
|--------|----------|------|
| MVP 范围 | 端到端完整 | 包含因子→策略→实盘全链路 |
| 防过拟合 | 核心优先 | Research Ledger + 多切片必须完整实现 |
| Dashboard | 同步开发 | 与后端并行，方便实时监控 |
| 交易范围 | 全功能实盘 | 由风控模块约束 |
| 频率范围 | 自适应 | 根据因子特性自动选择 |

---

## 2. 6D Analysis Summary

### 2.1 Technical Dimension
- 5 层架构：UI → API → Agent → Engine → Data
- 关键路径：LLM Engine → Qlib Core → Anti-Overfitting
- 技术风险：LLM 生成质量、Qlib 加密货币兼容

### 2.2 Business Dimension
- 5 个 Epic，25 个 User Stories
- MVP 预估：53 工作日 (约 12 周)
- 信心度：81%

### 2.3 Team Dimension
- 单人开发可行性：✅ 可行（需严格优先级）
- 分 3 阶段递进开发
- 每周明确交付物

### 2.4 Ecosystem Dimension
- Qlib 需扩展加密货币支持
- LangGraph 状态管理需 Checkpointer
- ccxt 多交易所接口稳定

### 2.5 Strategic Dimension
- 选择端到端：量化系统需完整链路验证
- 竞争优势：非 Docker 执行 + 系统化防过拟合

### 2.6 Meta Dimension
- 关键假设：OpenRouter 稳定、Qlib 可扩展
- 决策可逆性：ccxt 高、Qlib 低

---

## 3. Epic Structure

### Epic 1: 因子生成 (10d)
- F-001: 自然语言因子描述
- F-002: 因子家族约束生成
- F-003: AST 安全检查
- F-004: 非 Docker 执行

### Epic 2: 防过拟合 (12d)
- A-001: 多切片验证
- A-002: 研究账本
- A-003: 动态阈值
- A-004: 稳定性报告

### Epic 3: 策略组装 (12d)
- S-001: 因子选择组合
- S-002: 策略代码生成
- S-003: 开平仓逻辑
- S-004: 回测报告

### Epic 4: 交易执行 (9d)
- T-001: ccxt 连接
- T-002: 多空订单
- T-003: 持仓监控
- T-004: 风控检查
- T-005: 紧急平仓

### Epic 5: Dashboard (10d)
- D-001: Agent 状态
- D-002: 因子浏览
- D-003: 研究账本视图
- D-004: 实盘监控

---

## 4. Implementation Timeline

```
Week 1-2:  因子生成核心
Week 3-4:  防过拟合核心
Week 5-6:  策略组装
Week 7-8:  交易执行
Week 9-10: Dashboard
Week 11-12: 集成测试与优化
```

---

## 5. Outputs

- ✅ Updated: `.ultra/specs/product.md` (Epic 3, 4 User Stories)
- ✅ Created: This research report

---

**Next Round**: Technology Selection (Round 3)
