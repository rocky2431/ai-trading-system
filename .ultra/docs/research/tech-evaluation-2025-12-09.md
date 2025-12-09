# Round 3: Technology Evaluation Report

**Date**: 2025-12-09
**Project**: IQFMP 智能量化因子挖掘平台

---

## 1. User Technology Choices

| 决策点 | 用户选择 | 理由 |
|--------|----------|------|
| LLM 提供商 | OpenRouter 为主 | 多模型切换，成本优化 |
| Qlib 集成 | Fork 维护 | 完全控制加密货币扩展 |
| 任务队列 | Celery + Redis | 分布式回测，成熟稳定 |
| Dashboard | React + shadcn/ui | 现代组件库，高定制性 |
| 部署环境 | 混合模式 | 本地开发 + 云端生产 |

---

## 2. Technology Validation Results

### 2.1 Qlib Crypto Compatibility ✅

**验证方法**: Context7 MCP + Exa 代码搜索

**关键发现**:
- Qlib 官方支持 `scripts/data_collector/crypto/` 数据收集器
- `DataHandlerLP` 支持自定义特征工程
- 社区已有 CryptoDataHandler 实现可参考

**实现策略**:
```python
class CryptoDataHandler(DataHandlerLP):
    def __init__(self, instruments, start_time, end_time):
        super().__init__(instruments, start_time, end_time)
        self.data_loader = CryptoDataLoader()

    def setup_data(self):
        # 自定义加密货币数据处理
        self._init_crypto_features()
```

### 2.2 LangGraph State Persistence ✅

**验证方法**: Context7 官方文档查询

**关键发现**:
- `SqliteSaver` 适合开发环境
- `PostgresSaver` 适合生产环境
- 支持 `checkpoint_id` 实现时间旅行调试

**实现策略**:
```python
from langgraph.checkpoint.postgres import PostgresSaver

# 生产环境配置
checkpointer = PostgresSaver(conn_string=os.getenv("DATABASE_URL"))

graph = StateGraph(AgentState)
graph.add_node("factor_generator", factor_generator_node)
# ... 其他节点

app = graph.compile(checkpointer=checkpointer)
```

### 2.3 OpenRouter Multi-Model ✅

**验证方法**: API 文档确认

**模型分配策略**:
| 任务类型 | 推荐模型 | 备选模型 |
|----------|----------|----------|
| 因子生成 | deepseek/deepseek-coder | claude-3.5-sonnet |
| 策略设计 | claude-3.5-sonnet | gpt-4o |
| 代码审查 | gpt-4o | deepseek-chat |
| Fallback | deepseek-chat | - |

### 2.4 Celery + Redis ✅

**验证方法**: 成熟度评估

**适用场景**:
- 分布式回测任务
- 异步因子计算
- 定时策略执行

**配置示例**:
```python
from celery import Celery

app = Celery('iqfmp',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

@app.task
def run_backtest(strategy_id, params):
    # 执行回测逻辑
    pass
```

### 2.5 React + shadcn/ui ✅

**验证方法**: 组件库评估

**选择理由**:
- 高度可定制，避免"AI slop"外观
- 基于 Radix UI 的无障碍支持
- Tailwind CSS 集成
- 复制粘贴模式，完全控制代码

**Dashboard 页面规划**:
1. Agent 状态监控
2. 因子浏览器
3. Research Ledger 视图
4. 实盘监控面板

---

## 3. Technology Stack Summary

### 3.1 Backend Core
```
Python 3.11+
├── LangGraph 0.2.x (Agent 编排)
├── Qlib (Fork) (量化研究)
├── ccxt 4.x (交易执行)
├── Celery 5.x (任务队列)
└── FastAPI 0.100+ (API 服务)
```

### 3.2 Data Layer
```
TimescaleDB (时序数据)
├── Redis (缓存 + 消息队列)
├── Qdrant (向量存储)
└── PostgreSQL (状态持久化)
```

### 3.3 Frontend
```
React 18+
├── shadcn/ui (组件库)
├── Tailwind CSS (样式)
├── TanStack Query (数据获取)
└── Recharts (图表)
```

### 3.4 DevOps
```
混合模式
├── 本地: Docker Compose
├── 生产: Kubernetes
└── CI/CD: GitHub Actions
```

---

## 4. Risk Mitigation

| 风险 | 缓解策略 |
|------|----------|
| Qlib Fork 维护成本 | 最小化修改，上游同步脚本 |
| OpenRouter 服务稳定性 | 多模型 Fallback 链 |
| LangGraph 版本变动 | 锁定版本，渐进升级 |

---

## 5. Next Steps

- [ ] Round 4: Risk & Constraints 深度分析
- [ ] 生成完整规格文档
- [ ] 创建技术 Spike 验证计划

---

**Validation Status**: All 5 core technologies validated ✅
