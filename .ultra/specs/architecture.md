# Architecture Design - IQFMP 智能量化因子挖掘平台

> **Purpose**: This document defines HOW the system is built, based on requirements in `product.md`.

## 1. System Overview

### 1.1 Architecture Vision

**架构风格**: 分层架构 + 多 Agent 协作

```
┌─────────────────────────────────────────────────────────┐
│                    IQFMP Platform                        │
├─────────────────────────────────────────────────────────┤
│  Layer 1: Agent协作层                                    │
│  ┌──────────┬──────────┬──────────┬──────────┐          │
│  │FactorGen │FactorEval│ StrategyA│ BacktestO│          │
│  │  Agent   │  Agent   │  Agent   │  Agent   │          │
│  └──────────┴──────────┴──────────┴──────────┘          │
├─────────────────────────────────────────────────────────┤
│  Layer 2: 引擎层                                          │
│  ┌──────────┬──────────┬──────────┬──────────┐          │
│  │LLM Engine│Code Exec │Qlib Core │Risk Mgmt │          │
│  └──────────┴──────────┴──────────┴──────────┘          │
├─────────────────────────────────────────────────────────┤
│  Layer 3: 数据层                                          │
│  ┌──────────┬──────────┬──────────┬──────────┐          │
│  │TimescaleDB│Redis Cache│Vector DB│File Store│          │
│  └──────────┴──────────┴──────────┴──────────┘          │
└─────────────────────────────────────────────────────────┘
```

**关键质量属性**:
- **可扩展性**: 因子家族、数据源、交易所可插拔
- **可靠性**: 防过拟合机制系统化
- **性能**: 因子执行 < 30s，非 Docker 隔离

### 1.2 Key Components

1. **Multi-Agent Orchestrator**: 状态机驱动的 Agent 协作
2. **Factor Generation Agent**: LLM 驱动的因子代码生成
3. **Factor Evaluation Engine**: 多维度验证 + 防过拟合
4. **Research Ledger**: 试验记录 + 动态阈值
5. **Factor Knowledge Base**: 向量存储 + 去重
6. **Qlib Integration**: 深度集成（非 Docker）
7. **Execution Engine**: ccxt 驱动的交易执行（Phase 3+）

### 1.3 Data Flow Overview

```
用户输入 (因子描述 + 家族选择)
        ↓
┌───────────────────────────────┐
│   FactorGenerationAgent       │
│   - Prompt 渲染               │
│   - LLM 调用                  │
│   - 代码生成                  │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│   Code Executor (非 Docker)   │
│   - AST 安全检查              │
│   - Qlib 直接执行             │
│   - 因子值计算                │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│   FactorEvaluationAgent       │
│   - 多切片验证                │
│   - 稳定性分析                │
│   - 研究账本更新              │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│   Factor Knowledge Base       │
│   - 向量化存储                │
│   - 去重检查                  │
│   - 聚类标记                  │
└───────────────────────────────┘
```

---

## 2. Architecture Principles

**Inherited from `.ultra/constitution.md`**:
- Specification-Driven
- Test-First Development
- Anti-Overfitting First
- Minimal Abstraction

**Project-Specific Principles**:

1. **非 Docker 执行**: Qlib 直接调用，减少反馈延迟
2. **家族约束生成**: 因子必须在指定家族内生成
3. **全程可追溯**: 每个因子从生成到评估全程记录
4. **插件化数据源**: 交易所、数据类型可扩展

---

## 3. Technology Stack

> **Research Validation**: 技术选型已通过 Round 3 验证，关键组件文档确认完成。

### 3.1 Backend Stack

#### 3.1.1 Runtime & Framework Selection

**Decision**: Python 3.12 + FastAPI

**Rationale**:
- **Traces to**: product.md - Qlib 兼容性要求
- **Workload Type**: I/O-bound (LLM 调用) + CPU-bound (因子计算)
- **Team Expertise**: Python 量化生态成熟
- **Ecosystem**: Qlib、ccxt、LangGraph 均为 Python

#### 3.1.2 Technical Details

| 组件 | 技术选型 | 版本 | 理由 | 验证状态 |
|------|---------|------|------|----------|
| Runtime | Python | 3.12+ | Qlib 兼容、性能优化 | ✅ |
| API Framework | FastAPI | 0.110+ | 异步支持、类型提示 | ✅ |
| Agent Framework | **LangGraph** | 0.2+ | 状态机编排、Checkpoint 支持 | ✅ 文档验证 |
| LLM Provider | **OpenRouter** | v1 | 多模型切换 (DeepSeek/Claude/GPT) | ✅ |
| Quant Kernel | **Qlib (Fork)** | latest | 深度定制加密货币支持 | ✅ 文档验证 |
| Trading | ccxt | 4.x | 统一交易所接口 | ✅ |
| Task Queue | **Celery + Redis** | 5.3+ | 异步任务、分布式回测 | ✅ |

#### 3.1.3 LLM Provider Strategy

```python
# OpenRouter 多模型配置
LLM_MODELS = {
    "factor_generation": "deepseek/deepseek-coder",      # 代码生成
    "strategy_design": "anthropic/claude-3.5-sonnet",    # 策略设计
    "code_review": "openai/gpt-4o",                      # 代码审查
    "fallback": "deepseek/deepseek-chat"                 # 备用
}
```

#### 3.1.4 Qlib Deep Fork 策略

> **用户决策 (Round 4)**: 深度 Fork - 完全定制加密货币支持

**文档验证结果**:
- ✅ 官方有 `scripts/data_collector/crypto/` 数据收集器
- ✅ DataHandlerLP 支持自定义特征和数据加载
- ✅ Provider.register 可注册自定义数据提供者

**深度 Fork 范围**:

```
qlib-crypto/
├── qlib/                        # 核心 Fork
│   ├── data/
│   │   ├── crypto_handler.py    # CryptoDataHandler (新增)
│   │   ├── funding_rate.py      # 资金费率处理 (新增)
│   │   └── orderbook.py         # 订单簿数据 (新增)
│   ├── contrib/
│   │   └── crypto_factors/      # 加密货币因子库 (新增)
│   └── backtest/
│       └── crypto_executor.py   # 多空执行器 (新增)
├── scripts/
│   └── sync_upstream.sh         # 上游同步脚本
└── tests/
    └── crypto/                  # 加密货币专属测试
```

**CryptoDataHandler 完整实现**:

```python
class CryptoDataHandler(DataHandlerLP):
    """深度定制加密货币数据处理器"""

    CRYPTO_FIELDS = [
        # 基础 OHLCV
        'open', 'high', 'low', 'close', 'volume',
        # 衍生品特有
        'funding_rate', 'open_interest', 'basis', 'premium',
        # 订单簿
        'bid_volume', 'ask_volume', 'spread', 'depth_imbalance',
        # 链上数据 (可选)
        'whale_flow', 'exchange_reserve'
    ]

    def __init__(self, instruments, start_time, end_time, **kwargs):
        super().__init__(instruments, start_time, end_time, **kwargs)
        self._init_crypto_features()
        self._setup_funding_rate_handler()

    def _setup_funding_rate_handler(self):
        """资金费率时间对齐处理"""
        self.funding_schedule = {
            'binance': ['00:00', '08:00', '16:00'],
            'okx': ['00:00', '08:00', '16:00'],
        }
```

**上游同步策略**:
- 每周检查 qlib/qlib 上游更新
- 使用 git rebase 合并非冲突更新
- 关键变更手动评估

#### 3.1.5 LangGraph 状态持久化

**文档验证结果**:
- ✅ 支持 SqliteSaver (开发) / PostgresSaver (生产)
- ✅ thread_id 隔离不同会话
- ✅ checkpoint_id 支持 time-travel 调试

```python
from langgraph.checkpoint.postgres import PostgresSaver

# 生产环境使用 PostgreSQL
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost/iqfmp"
)

# 编译 Agent 图
graph = workflow.compile(checkpointer=checkpointer)

# 执行（支持断点续传）
config = {"configurable": {"thread_id": "factor-mining-session-1"}}
result = graph.invoke(state, config)
```

---

### 3.2 Database Stack

#### 3.2.1 Database Selection

**Primary**: TimescaleDB (时序数据)
**Cache**: Redis (结果缓存、实时数据)
**Vector**: Qdrant (因子向量检索)

**Rationale**:
- **TimescaleDB**: 时序数据优化、SQL 兼容、自动分区
- **Redis**: 高速缓存、Pub/Sub 支持
- **Qdrant**: 专业向量数据库、相似度检索高效

#### 3.2.2 Technical Details

| 数据库 | 用途 | 版本 |
|--------|------|------|
| TimescaleDB | OHLCV、因子值、回测结果 | 2.14+ |
| Redis | LLM 响应缓存、因子值缓存、实时数据 | 7.0+ |
| Qdrant | 因子向量存储、相似度检索 | 1.8+ |

---

### 3.3 Message Queue Stack

**Decision**: Celery + RabbitMQ

**Rationale**:
- **Traces to**: product.md - 因子生成并发 10
- **Use Cases**: 异步因子评估、分布式回测
- **Ecosystem**: Python 生态成熟方案

---

### 3.4 Frontend Stack (Dashboard)

#### 3.4.1 Framework Selection

**Decision**: React + TypeScript + shadcn/ui

**Rationale**:
- **Traces to**: plan.md - 已有 React 基础
- **User Choice**: 现代设计、高度可定制
- **Ecosystem**: shadcn/ui 基于 Radix，可访问性好

#### 3.4.2 Technical Details

| 组件 | 技术 | 版本 | 说明 |
|------|------|------|------|
| Framework | React | 18.3+ | 稳定版本 |
| Language | TypeScript | 5.x | 类型安全 |
| UI Library | **shadcn/ui** | latest | 可定制组件 |
| Styling | Tailwind CSS | 3.x | 原子化 CSS |
| State | Zustand | 4.x | 轻量状态管理 |
| Charts | Recharts | latest | React 原生图表 |
| Data Fetching | TanStack Query | 5.x | 缓存 + 自动重试 |
| Build | Vite | 5.x | 快速构建 |
| WebSocket | socket.io-client | 4.x | 实时推送 |

#### 3.4.3 Dashboard 核心页面

```
dashboard/
├── pages/
│   ├── AgentMonitor.tsx      # Agent 运行状态
│   ├── FactorExplorer.tsx    # 因子库浏览
│   ├── StrategyBuilder.tsx   # 策略构建器
│   ├── Backtest.tsx          # 回测结果
│   ├── LiveTrading.tsx       # 实盘监控
│   ├── ResearchLedger.tsx    # 研究账本
│   └── Settings.tsx          # 系统配置
├── components/
│   ├── charts/               # 图表组件
│   ├── tables/               # 数据表格
│   └── forms/                # 表单组件
└── hooks/
    ├── useWebSocket.ts       # WebSocket 连接
    └── useFactorData.ts      # 因子数据查询
```

---

### 3.5 Infrastructure Stack

| 组件 | 技术 | 用途 |
|------|------|------|
| Containerization | Docker | 开发环境一致性 |
| Orchestration | Docker Compose (dev) | 本地开发 |
| Monitoring | Prometheus + Grafana | 指标监控 |
| Logging | ELK / Loki | 日志聚合 |
| CI/CD | GitHub Actions | 持续集成 |

---

## 4. Component Architecture

### 4.1 Multi-Agent Orchestrator

```python
class MultiAgentOrchestrator:
    """多智能体编排引擎"""

    def __init__(self):
        self.state_graph = StateGraph()
        self.agents = {
            'factor_gen': FactorGenerationAgent(),
            'factor_eval': FactorEvaluationAgent(),
            'strategy_assembly': StrategyAssemblyAgent(),
            'backtest_opt': BacktestOptimizationAgent(),
            'risk_check': RiskCheckAgent(),
        }

    def run_pipeline(self, objective: str) -> Pipeline:
        """运行完整流程"""
        # 状态机编排
        # Agent 协作
        # 结果聚合
        pass
```

**Trace to**: product.md#Epic1

---

### 4.2 Factor Generation Agent

```python
class FactorGenerationAgent(BaseAgent):
    """因子生成 Agent"""

    def run(self, state: State) -> State:
        # 1. 解析用户输入
        # 2. 检索已有因子（避免重复）
        # 3. Prompt 渲染（家族约束）
        # 4. LLM 调用
        # 5. 代码解析与验证
        pass
```

**关键约束**:
- 必须在指定因子家族内生成
- 只能使用已定义的数据字段
- 代码必须通过 AST 安全检查

**Trace to**: product.md#UserStory1.1, 1.2

---

### 4.3 Factor Evaluation Engine

```python
class FactorEvaluator:
    """多维度因子评估"""

    def evaluate(self, factor_code: str, config: EvalConfig) -> FactorReport:
        # 1. 多切片数据准备
        splits = self.cv_splitter.get_splits(config)

        # 2. 因子计算
        factor_values = self.compute_factor(factor_code, splits)

        # 3. 指标计算
        metrics = {
            'IC': self.calc_ic(factor_values),
            'IR': self.calc_ir(factor_values),
            'sharpe': self.calc_sharpe(factor_values),
            'stability': self.stability_analyzer.analyze(factor_values),
        }

        # 4. 研究账本更新
        self.research_ledger.log(factor_code, metrics)

        # 5. 动态阈值检查
        passed = self.check_thresholds(metrics)

        return FactorReport(metrics=metrics, passed=passed)
```

**Trace to**: product.md#Epic2

---

### 4.4 Research Ledger (防过拟合核心)

```python
class ResearchLedger:
    """研究账本 - 防止 p-hacking"""

    def __init__(self, db: TimescaleDB):
        self.db = db
        self.experiment_count = 0

    def log(self, factor_id: str, metrics: dict):
        """记录每次试验"""
        record = {
            'timestamp': datetime.now(),
            'factor_id': factor_id,
            'code_hash': hash(factor_code),
            'metrics': metrics,
            'experiment_number': self.experiment_count,
        }
        self.db.insert(record)
        self.experiment_count += 1

    def get_dynamic_threshold(self, metric: str) -> float:
        """动态阈值 - 试验越多，要求越严"""
        N = self.experiment_count
        base_threshold = self.base_thresholds[metric]
        # Deflated Sharpe Ratio 简化版
        adjustment = 1 + 0.1 * log(N + 1)
        return base_threshold * adjustment
```

**Trace to**: product.md#UserStory2.2, advice-for-plan.md Section I.2

---

### 4.5 Stability Analyzer

```python
class StabilityAnalyzer:
    """稳定性分析器"""

    def analyze(self, factor_values: pd.DataFrame) -> StabilityReport:
        return StabilityReport(
            time_stability=self._calc_time_stability(factor_values),
            market_stability=self._calc_market_stability(factor_values),
            regime_stability=self._calc_regime_stability(factor_values),
        )

    def _calc_time_stability(self, values):
        """按月/季度 IC 稳定性"""
        monthly_ic = values.groupby(pd.Grouper(freq='M')).apply(calc_ic)
        return {
            'positive_ic_ratio': (monthly_ic > 0).mean(),
            'ic_std': monthly_ic.std(),
            'max_drawdown': calc_max_drawdown(monthly_ic),
        }

    def _calc_market_stability(self, values):
        """跨市场稳定性"""
        # 大/中/小市值分组
        pass

    def _calc_regime_stability(self, values):
        """跨环境稳定性"""
        # 牛/熊 × 高/低波动
        pass
```

**Trace to**: product.md#UserStory2.3, advice-for-plan.md Section I.3

---

### 4.6 Qlib Integration

```python
class QlibIntegration:
    """Qlib 深度集成（非 Docker）"""

    def __init__(self):
        self.qlib_initialized = False

    def init_qlib(self, provider_uri: str):
        """初始化 Qlib"""
        import qlib
        qlib.init(provider_uri=provider_uri)
        self.qlib_initialized = True

    def execute_factor(self, factor_code: str) -> pd.DataFrame:
        """直接执行因子代码"""
        # AST 安全检查
        self._validate_code(factor_code)
        # 执行
        exec(factor_code, self._get_safe_globals())
        return result

    def _validate_code(self, code: str):
        """AST 安全检查"""
        import ast
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['system', 'exec', 'eval']:
                        raise SecurityError("Dangerous operation detected")
```

**Trace to**: product.md#UserStory3.1

---

### 4.7 CryptoCVSplitter (防过拟合)

```python
class CryptoCVSplitter:
    """加密货币专用交叉验证切分器"""

    def get_splits(
        self,
        symbols: List[str],
        date_range: Tuple[datetime, datetime],
        config: SplitConfig
    ) -> List[Split]:
        """生成多维度切分"""
        splits = []

        # 时间切分: Train → Valid → Test
        time_splits = self._time_split(date_range)

        # 市场切分: BTC/ETH + Altcoins + 小市值
        market_groups = self._market_split(symbols)

        # 频率切分: 1h / 4h / 1d
        frequencies = ['1h', '4h', '1d']

        for time_split in time_splits:
            for market_group in market_groups:
                for freq in frequencies:
                    splits.append(Split(
                        time=time_split,
                        market=market_group,
                        frequency=freq,
                    ))

        return splits
```

**Trace to**: advice-for-plan.md Section I.1

---

## 5. Data Architecture

### 5.1 Core Data Models

#### Factor

```python
@dataclass
class Factor:
    id: str                    # UUID
    name: str                  # 因子名称
    family: List[str]          # 因子家族 (trend, vol, liquidity, ...)
    code: str                  # 因子代码
    code_hash: str             # 代码哈希（去重用）
    target_task: str           # 目标任务 (1h_trend, 4h_mean_reversion, ...)

    # 评估指标
    metrics: FactorMetrics
    stability: StabilityReport

    # 状态
    status: FactorStatus       # candidate / rejected / core / redundant
    cluster_id: Optional[str]  # 聚类 ID

    # 元数据
    created_at: datetime
    experiment_number: int     # 研究账本序号
```

#### FactorMetrics

```python
@dataclass
class FactorMetrics:
    ic_mean: float
    ic_std: float
    ir: float                  # IC / IC_std
    sharpe: float
    max_drawdown: float
    turnover: float

    # 分切片指标
    ic_by_split: Dict[str, float]
    sharpe_by_split: Dict[str, float]
```

#### ResearchExperiment

```python
@dataclass
class ResearchExperiment:
    id: str
    timestamp: datetime
    factor_id: str
    code_hash: str
    prompt: str                # 生成 Prompt
    config: EvalConfig         # 评估配置
    metrics: FactorMetrics
    passed: bool
    rejection_reason: Optional[str]
```

### 5.2 Database Schema

#### TimescaleDB Tables

```sql
-- 市场数据（时序表）
CREATE TABLE market_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open DECIMAL,
    high DECIMAL,
    low DECIMAL,
    close DECIMAL,
    volume DECIMAL,
    open_interest DECIMAL,
    funding_rate DECIMAL
);
SELECT create_hypertable('market_data', 'time');

-- 因子值（时序表）
CREATE TABLE factor_values (
    time TIMESTAMPTZ NOT NULL,
    factor_id UUID NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    value DECIMAL
);
SELECT create_hypertable('factor_values', 'time');

-- 研究账本
CREATE TABLE research_ledger (
    id UUID PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    factor_id UUID,
    code_hash VARCHAR(64),
    experiment_number INTEGER,
    metrics JSONB,
    passed BOOLEAN
);
```

---

## 6. API Design

### 6.1 REST Endpoints

```
POST   /api/v1/factors/generate       - 生成因子
GET    /api/v1/factors                - 查询因子库
GET    /api/v1/factors/:id            - 获取因子详情
POST   /api/v1/factors/:id/evaluate   - 评估因子
PUT    /api/v1/factors/:id/status     - 更新因子状态

GET    /api/v1/research/ledger        - 查询研究账本
GET    /api/v1/research/stats         - 研究统计

POST   /api/v1/pipeline/run           - 运行完整 Pipeline
GET    /api/v1/pipeline/:id/status    - Pipeline 状态

GET    /api/v1/metrics/thresholds     - 获取当前动态阈值
```

---

## 7. Project Structure

```
trading-system-v3/
├── .ultra/                     # Ultra Builder Pro 配置
├── src/
│   ├── agents/                 # Agent 实现
│   │   ├── base.py
│   │   ├── factor_gen.py
│   │   ├── factor_eval.py
│   │   └── orchestrator.py
│   ├── core/                   # 核心引擎
│   │   ├── qlib_integration.py
│   │   ├── code_executor.py
│   │   ├── research_ledger.py
│   │   └── stability_analyzer.py
│   ├── data/                   # 数据层
│   │   ├── cv_splitter.py
│   │   ├── market_data.py
│   │   └── factor_store.py
│   ├── llm/                    # LLM 集成
│   │   ├── provider.py
│   │   └── prompts/
│   ├── api/                    # FastAPI
│   │   ├── main.py
│   │   └── routers/
│   ├── models/                 # 数据模型
│   └── utils/
├── dashboard/                  # React Dashboard
│   ├── src/
│   └── package.json
├── tests/
├── docker-compose.yml
└── pyproject.toml
```

---

## 8. Security Architecture

> **用户决策 (Round 4)**: 严格模式 - 生产级安全要求

### 8.1 Code Execution Safety (严格模式)

**三重防护机制**:

```
用户输入 → [1. AST 静态分析] → [2. 沙箱执行] → [3. 人工审核] → 执行
                 ↓                  ↓                ↓
            禁止危险函数        RestrictedPython   所有代码需确认
```

**8.1.1 AST 静态分析 (Layer 1)**

```python
class ASTSecurityChecker:
    """严格模式 AST 安全检查"""

    FORBIDDEN_FUNCTIONS = {
        'eval', 'exec', 'compile', 'open', 'input',
        '__import__', 'getattr', 'setattr', 'delattr',
    }

    FORBIDDEN_MODULES = {
        'os', 'sys', 'subprocess', 'shutil', 'socket',
        'requests', 'urllib', 'ftplib', 'smtplib',
    }

    def check(self, code: str) -> SecurityReport:
        tree = ast.parse(code)
        violations = []
        for node in ast.walk(tree):
            # 检查危险函数调用
            # 检查危险模块导入
            # 检查属性访问模式
        return SecurityReport(safe=len(violations) == 0, violations=violations)
```

**8.1.2 沙箱执行 (Layer 2)**

```python
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins

class SandboxExecutor:
    """隔离执行环境"""

    ALLOWED_GLOBALS = {
        '__builtins__': safe_builtins,
        'pd': pd,
        'np': np,
        'qlib': qlib_safe_subset,
    }

    def execute(self, code: str, timeout: int = 60) -> Result:
        byte_code = compile_restricted(code, '<factor>', 'exec')
        # 资源限制: CPU/Memory
        with resource_limits(timeout=timeout, memory_mb=512):
            exec(byte_code, self.ALLOWED_GLOBALS)
```

**8.1.3 人工审核流程 (Layer 3)**

```python
class HumanReviewGate:
    """所有生成代码必须人工确认"""

    async def submit_for_review(self, factor: Factor) -> ReviewStatus:
        # 1. 生成代码摘要
        summary = self.generate_summary(factor.code)

        # 2. 发送通知 (Telegram/Slack)
        await self.notify_reviewer(summary)

        # 3. 等待确认 (默认阻塞)
        return await self.wait_for_approval(factor.id, timeout=3600)
```

- **AST 解析**: 禁止 os.system, exec, eval, 动态 import
- **沙箱环境**: RestrictedPython + 资源限制
- **超时控制**: 单因子执行 < 60s
- **人工审核**: 所有代码必须经过人工确认后才能执行

### 8.2 API Security

- **JWT 认证**: 所有 API 需要认证
- **Rate Limiting**: 防止滥用
- **Input Validation**: Pydantic 严格验证

### 8.3 Data Protection

- **API Keys 加密**: 使用环境变量 + Secrets Manager
- **传输加密**: TLS 1.3
- **日志脱敏**: 不记录敏感数据

---

## 9. Testing Strategy

### 9.1 Test Pyramid

```
       /\
      /E2E\        - 10%  (Pipeline 端到端)
     /------\
    /Integra\      - 30%  (Agent + Qlib 集成)
   /----------\
  /Unit Tests \    - 60%  (因子计算、稳定性分析)
 /--------------\
```

### 9.2 Test Coverage Targets

See `.ultra/config.json`:
- Overall: ≥ 85%
- Critical paths: 100%
- Branch: ≥ 75%
- Function: ≥ 85%

### 9.3 Quant-Specific Testing

- **因子计算正确性**: 对比已知因子值
- **防过拟合验证**: 确保阈值动态调整
- **稳定性计算**: 验证多维度分析正确

---

## 10. Deployment Architecture

### 10.1 Environments

- **Development**: Docker Compose 本地
- **Staging**: 单机部署 + 模拟数据
- **Production**: 云服务器 + 实盘数据

### 10.2 CI/CD Pipeline

```
1. Code push → GitHub
2. Run tests (pytest --cov)
3. Build Docker images
4. Deploy to staging (auto)
5. Manual approval for production
6. Deploy to production
7. Health check
```

---

## 11. Risk Mitigation Architecture

> **用户决策 (Round 4)**: 生产系统定位 - 完整风控 + 高可用

### 11.1 Production System Requirements

**系统可用性目标**:
- SLA: 99.5% (允许每月约 3.5 小时停机)
- RTO: < 30 分钟 (恢复时间目标)
- RPO: < 5 分钟 (数据丢失容忍)

**关键组件冗余**:

```
┌─────────────────────────────────────────────────────────────┐
│                   Production Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐              │
│   │ Primary │────▶│  Redis  │◀────│ Backup  │              │
│   │  Node   │     │ Sentinel│     │  Node   │              │
│   └─────────┘     └─────────┘     └─────────┘              │
│        │                                │                    │
│        ▼                                ▼                    │
│   ┌─────────────────────────────────────────┐              │
│   │         TimescaleDB (Primary + Replica)  │              │
│   └─────────────────────────────────────────┘              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 11.2 Trading Risk Control

**硬性风控规则 (不可覆盖)**:

```python
class RiskController:
    """生产级风控引擎"""

    # 硬性限制 (不可通过配置修改)
    MAX_SINGLE_LOSS_PCT = 0.02        # 单笔最大亏损 2%
    MAX_DAILY_LOSS_PCT = 0.05         # 日最大亏损 5%
    MAX_POSITION_PCT = 0.10           # 单策略最大仓位 10%
    MAX_TOTAL_POSITION_PCT = 0.50     # 总仓位上限 50%
    EMERGENCY_CLOSE_THRESHOLD = 0.08  # 紧急平仓阈值 8%

    def check_order(self, order: Order) -> RiskDecision:
        checks = [
            self._check_single_loss(order),
            self._check_daily_loss(order),
            self._check_position_limit(order),
            self._check_market_hours(order),
            self._check_liquidity(order),
        ]
        return RiskDecision.combine(checks)

    async def emergency_close_all(self, reason: str):
        """紧急全平仓 - 需要人工确认"""
        await self.notify_admin(f"Emergency close triggered: {reason}")
        if await self.wait_admin_confirm(timeout=300):
            await self.close_all_positions()
```

### 11.3 Monitoring & Alerting

**监控维度**:

| 维度 | 指标 | 告警阈值 |
|------|------|----------|
| 系统健康 | CPU/Memory/Disk | > 80% |
| 交易延迟 | Order latency | > 500ms |
| 数据延迟 | Market data lag | > 10s |
| PnL | Daily drawdown | > 3% |
| 异常 | Error rate | > 1% |

**告警通道**:
- Telegram Bot (实时)
- Email (非紧急)
- PagerDuty (7x24 紧急)

### 11.4 Disaster Recovery

**备份策略**:
- TimescaleDB: 每小时增量备份，每日全量备份
- Redis: RDB + AOF 持久化
- 因子代码: Git 版本控制

**恢复流程**:
1. 自动切换到备用节点 (< 30s)
2. 暂停所有活跃策略
3. 验证数据一致性
4. 人工确认后恢复交易

---

## 12. Cost Analysis

### 12.1 Monthly Cost Estimate

| 项目 | 成本 | 说明 |
|------|------|------|
| 云服务器 (主节点) | $200 | 4 vCPU, 16GB RAM |
| 云服务器 (备份) | $100 | 2 vCPU, 8GB RAM |
| TimescaleDB 托管 | $50 | 基础版 |
| Redis 托管 | $30 | 基础版 |
| LLM API (OpenRouter) | $210 | ~1M tokens/day |
| Monitoring (Grafana Cloud) | $20 | 基础版 |
| **Total** | **~$610/月** | |

### 12.2 Scaling Considerations

- 初期: 单主节点 + 备份
- 中期: 分离回测与交易节点
- 长期: Kubernetes 集群

---

## 13. Open Questions

### 13.1 Technical Uncertainties

1. ~~**Qlib 加密货币数据格式**~~: ✅ 已通过 Round 3 验证
2. ~~**LangGraph 状态管理**~~: ✅ 已确认 PostgresSaver 方案
3. **向量化因子表示**: 最佳 embedding 方法待调研

### 13.2 Performance Targets Validation

- **因子执行 < 30s**: 需要实际测试验证
- **10 并发生成**: LLM API 限流处理

### 13.3 Round 4 Decision Impact

| 决策 | 影响 | 应对 |
|------|------|------|
| 完整版 MVP | 时间压力 ↑ | 严格优先级管理 |
| 严格安全模式 | 开发速度 ↓ | 提前构建安全框架 |
| 深度 Fork | 维护成本 ↑ | 自动化同步脚本 |
| 生产系统 | 质量要求 ↑ | 增加测试投入 |

---

**Document Status**: Draft → In Review
**Last Updated**: 2025-12-09
**Round 4 Decisions Integrated**: ✅
