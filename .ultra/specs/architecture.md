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

### 3.1 Backend Stack

#### 3.1.1 Runtime & Framework Selection

**Decision**: Python 3.12 + FastAPI

**Rationale**:
- **Traces to**: product.md - Qlib 兼容性要求
- **Workload Type**: I/O-bound (LLM 调用) + CPU-bound (因子计算)
- **Team Expertise**: Python 量化生态成熟
- **Ecosystem**: Qlib、ccxt、LangGraph 均为 Python

#### 3.1.2 Technical Details

| 组件 | 技术选型 | 版本 | 理由 |
|------|---------|------|------|
| Runtime | Python | 3.12+ | Qlib 兼容、性能优化 |
| API Framework | FastAPI | 0.110+ | 异步支持、类型提示 |
| Agent Framework | LangGraph + LangChain | 0.2+ | 状态机编排、官方维护 |
| LLM Provider | OpenRouter | v1 | 多模型支持、无 vendor lock-in |
| Quant Kernel | Qlib | latest | 微软维护、社区丰富 |
| Trading | ccxt | 4.x | 统一交易所接口 |

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

**Decision**: React + TypeScript + Ant Design

**Rationale**:
- **Traces to**: plan.md - 已有 React 基础
- **Team Expertise**: 熟悉 React 生态
- **Ecosystem**: Ant Design 适合数据密集型 Dashboard

#### 3.4.2 Technical Details

| 组件 | 技术 | 版本 |
|------|------|------|
| Framework | React | 18.3+ |
| Language | TypeScript | 5.x |
| UI Library | Ant Design | 5.x |
| State | Zustand | 4.x |
| Charts | ECharts / Recharts | latest |
| Build | Vite | 5.x |

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

### 8.1 Code Execution Safety

- **AST 解析**: 禁止 os.system, exec, eval
- **沙箱环境**: 限制可访问的模块
- **超时控制**: 单因子执行 < 60s

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

## 11. Open Questions

### 11.1 Technical Uncertainties

1. **Qlib 加密货币数据格式**: 需要验证现有数据结构兼容性
2. **LangGraph 状态管理**: 复杂 Pipeline 的状态持久化方案
3. **向量化因子表示**: 最佳 embedding 方法待调研

### 11.2 Performance Targets Validation

- **因子执行 < 30s**: 需要实际测试验证
- **10 并发生成**: LLM API 限流处理

---

**Document Status**: Draft
**Last Updated**: 2025-12-09
