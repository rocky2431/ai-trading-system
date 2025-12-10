# IQFMP 系统优化方案 v1.0

> 基于外部测试反馈的深度扫描分析报告

---

## 一、现状诊断总结

### 1.1 核心问题矩阵

| 问题域 | 严重程度 | 现状 | 影响 |
|--------|---------|------|------|
| **评估链路断裂** | 🔴 Critical | evaluation 模块未接入 API | 因子评估返回0指标 |
| **数据持久化不完整** | 🔴 Critical | Redis stub + 内存存储 | 数据丢失风险 |
| **Qlib 集成表面化** | 🟡 High | 手写解析器，未用 D.features | 表达式支持有限 |
| **Agent 编排孤立** | 🟡 High | 未与 API/Celery 绑定 | 无法分布式执行 |
| **向量库未启用** | 🟡 High | store.py 完整但未调用 | 无因子去重 |
| **System 静态假数据** | 🟠 Medium | 硬编码 Agent 状态 | 监控不真实 |
| **前端部分占位** | 🟠 Medium | Pipeline/LiveTrading 模拟 | 交互不完整 |
| **WebSocket 缺失** | 🟠 Medium | 仅轮询模式 | 实时性差 |

### 1.2 模块完成度评估

```
┌─────────────────────────────────────────────────────────────┐
│ 模块                    │ 代码完成 │ 集成完成 │ 可用性  │
├─────────────────────────┼──────────┼──────────┼─────────┤
│ API Routes              │   95%    │   70%    │   70%   │
│ Database Layer          │   90%    │   60%    │   55%   │
│ Factor Evaluation       │   95%    │   30%    │   25%   │
│ Qlib Integration        │   80%    │   50%    │   45%   │
│ Agent Orchestration     │   90%    │   40%    │   35%   │
│ Vector Store            │   85%    │   10%    │   10%   │
│ Celery Tasks            │   60%    │   40%    │   35%   │
│ Frontend Pages          │   80%    │   70%    │   65%   │
│ WebSocket               │    5%    │    0%    │    0%   │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、阶段1：打通可运行闭环（2-3周）

### 2.1 数据与持久层改造

#### 任务 1.1: 因子/策略从 Redis 切换到 TimescaleDB

**问题**: `api/factors/service.py` 第 120-150 行使用 Redis 存储因子，未持久化到 DB

**修改文件**:
- `src/iqfmp/api/factors/service.py`
- `src/iqfmp/db/repositories.py`

**具体改动**:
```python
# factors/service.py 修改
# 原代码 (L120-125):
async def create_factor(self, request: FactorCreateRequest) -> Factor:
    factor = Factor(id=str(uuid4()), ...)
    await self.redis_client.hset("factors", factor.id, factor.model_dump_json())
    return factor

# 改为:
async def create_factor(self, request: FactorCreateRequest) -> Factor:
    factor = Factor(id=str(uuid4()), ...)
    # 1. 持久化到 TimescaleDB
    db_factor = await self.factor_repo.create(factor)
    # 2. Redis 仅做缓存
    await self.redis_client.setex(f"factor:{factor.id}", 3600, factor.model_dump_json())
    return db_factor
```

#### 任务 1.2: Celery 任务结果写回数据库

**问题**: `celery_app/tasks.py` 第 200-250 行回测任务仅模拟执行

**修改文件**:
- `src/iqfmp/celery_app/tasks.py`
- `src/iqfmp/db/repositories.py` (添加 BacktestResultRepository)

**具体改动**:
```python
# tasks.py 添加数据库写入
@celery_app.task(bind=True)
def backtest_task(self, backtest_id: str, strategy_id: str, config: dict):
    try:
        # 真实执行回测
        from iqfmp.core.backtest_engine import BacktestEngine
        engine = BacktestEngine()
        result = engine.run_factor_backtest(...)

        # 写入数据库 (新增)
        with get_db_session() as session:
            repo = BacktestResultRepository(session)
            repo.save_result(backtest_id, result.metrics, result.equity_curve)

        return {"status": "completed", "metrics": result.metrics.dict()}
    except Exception as e:
        # 错误也要记录
        with get_db_session() as session:
            repo.mark_failed(backtest_id, str(e))
        raise
```

#### 任务 1.3: docker-compose 补充 RabbitMQ

**问题**: Celery broker 使用 Redis，生产建议用 RabbitMQ

**修改文件**: `docker-compose.yml`

```yaml
# 添加 RabbitMQ 服务
rabbitmq:
  image: rabbitmq:3.12-management-alpine
  ports:
    - "5672:5672"
    - "15672:15672"
  environment:
    RABBITMQ_DEFAULT_USER: iqfmp
    RABBITMQ_DEFAULT_PASS: ${RABBITMQ_PASSWORD:-iqfmp_secret}
  volumes:
    - rabbitmq_data:/var/lib/rabbitmq
  healthcheck:
    test: rabbitmq-diagnostics -q ping
    interval: 30s
    timeout: 10s
    retries: 5

# 修改 celery-worker 环境变量
celery-worker:
  environment:
    - CELERY_BROKER_URL=amqp://iqfmp:${RABBITMQ_PASSWORD}@rabbitmq:5672//
```

---

### 2.2 因子评估链路打通

#### 任务 2.1: 接入完整的 FactorEvaluator

**核心问题**: 存在两个同名 `FactorEvaluator`:
- `core/factor_engine.py` (简化版，当前使用)
- `evaluation/factor_evaluator.py` (完整版，未使用)

**修改文件**: `src/iqfmp/api/factors/service.py`

**具体改动**:
```python
# 第 256-382 行 evaluate_factor 方法重构

# 原代码:
from iqfmp.core.factor_engine import FactorEngine, FactorEvaluator

# 改为:
from iqfmp.core.factor_engine import FactorEngine
from iqfmp.evaluation.factor_evaluator import FactorEvaluator, EvaluationConfig
from iqfmp.evaluation.cv_splitter import CryptoCVSplitter, CVSplitConfig
from iqfmp.evaluation.stability_analyzer import StabilityAnalyzer

async def evaluate_factor(
    self,
    factor_id: str,
    splits: list[str],
    market_splits: list[str] = None,
) -> tuple[FactorMetrics, StabilityReport, bool, int]:

    factor = await self.get_factor(factor_id)

    # 1. 使用 FactorEngine 计算因子值
    engine = FactorEngine(data_path=get_default_data_path())
    factor_values = engine.compute_factor(factor.code, factor.name)

    # 2. 配置完整评估 (NEW)
    eval_config = EvaluationConfig(
        use_cv_splits=True,
        run_stability_analysis=True,
        include_transaction_costs=True,  # 新增
    )

    # 3. 使用 CryptoCVSplitter 进行多维切分 (NEW)
    cv_config = CVSplitConfig(
        time_split=True,
        market_split=market_splits is not None,
        regime_split=True,  # 波动率制度
    )
    cv_splitter = CryptoCVSplitter(cv_config)

    # 4. 使用完整版 FactorEvaluator (NEW)
    evaluator = FactorEvaluator(config=eval_config)
    metrics = evaluator.evaluate(
        factor_values=factor_values,
        forward_returns=engine.get_forward_returns(),
        cv_splitter=cv_splitter,
    )

    # 5. 稳定性分析 (NEW)
    stability_analyzer = StabilityAnalyzer()
    stability_report = stability_analyzer.analyze(
        factor_values=factor_values,
        returns=engine.get_forward_returns(),
        market_data=engine.data,
    )

    # 6. 动态阈值检查
    threshold = await self._get_dynamic_threshold(factor.family[0])
    passed = metrics.sharpe > threshold

    # 7. 记录研究试验
    trial_number = await self.trial_repo.create(...)

    return metrics, stability_report, passed, trial_number
```

#### 任务 2.2: 添加交易成本模型

**问题**: `FactorEvaluator` 缺少交易成本/容量估算

**修改文件**: `src/iqfmp/evaluation/factor_evaluator.py`

**新增代码**:
```python
# 在 FactorEvaluator 类中添加

def _estimate_transaction_costs(
    self,
    factor_values: pd.Series,
    volume: pd.Series,
    config: TransactionCostConfig = None,
) -> TransactionCostMetrics:
    """
    估算交易成本和容量约束

    Args:
        factor_values: 因子值序列
        volume: 成交量序列
        config: 成本配置 (默认: taker_fee=0.0004, slippage_bps=2)

    Returns:
        TransactionCostMetrics:
            - turnover: 换手率
            - estimated_cost_bps: 预估成本(基点)
            - capacity_usd: 容量估算(美元)
            - implementability: 可实施性评分 (0-1)
    """
    config = config or TransactionCostConfig()

    # 1. 计算换手率
    position_changes = factor_values.diff().abs()
    turnover = position_changes.mean()

    # 2. 估算交易成本
    taker_fee = config.taker_fee  # 0.04%
    slippage = config.slippage_bps / 10000  # 2 bps
    estimated_cost = turnover * (taker_fee + slippage) * 252  # 年化

    # 3. 容量估算 (基于成交量的 1%)
    avg_volume_usd = volume.mean() * config.price_assumption
    capacity_usd = avg_volume_usd * 0.01 * 252  # 年化

    # 4. 可实施性评分
    if estimated_cost < 0.005:  # < 0.5% 年化成本
        implementability = 1.0
    elif estimated_cost < 0.02:  # < 2%
        implementability = 0.7
    else:
        implementability = 0.3

    return TransactionCostMetrics(
        turnover=turnover,
        estimated_cost_bps=estimated_cost * 10000,
        capacity_usd=capacity_usd,
        implementability=implementability,
    )
```

---

### 2.3 Qlib 集成完善

#### 任务 3.1: 修复表达式解析器

**问题**: `core/factor_engine.py` 手写正则解析，嵌套表达式支持有限

**修改文件**: `src/iqfmp/core/factor_engine.py`

**改进策略**: 优先使用 Qlib 官方解析，手写作为降级

```python
# 第 181-262 行 _evaluate_expression 方法重构

def _evaluate_expression(self, expr: str, df: pd.DataFrame) -> pd.Series:
    """
    评估 Qlib 表达式

    优先级:
    1. 尝试 Qlib D.features (如果已初始化)
    2. 降级到手写解析器
    """
    # 1. 优先使用 Qlib 官方解析
    if self._qlib_initialized and self._can_use_d_features(expr):
        try:
            return self._evaluate_with_qlib(expr, df)
        except Exception as e:
            logger.warning(f"Qlib evaluation failed, falling back: {e}")

    # 2. 降级到增强版手写解析器
    return self._evaluate_with_custom_parser(expr, df)

def _evaluate_with_qlib(self, expr: str, df: pd.DataFrame) -> pd.Series:
    """使用 Qlib D.features 计算"""
    from qlib.data import D

    # 转换 DataFrame 索引为 Qlib 格式
    instruments = df.index.get_level_values('symbol').unique().tolist()
    result = D.features(
        instruments=instruments,
        fields=[expr],
        start_time=df.index.get_level_values('timestamp').min(),
        end_time=df.index.get_level_values('timestamp').max(),
    )
    return result[expr]

def _evaluate_with_custom_parser(self, expr: str, df: pd.DataFrame) -> pd.Series:
    """增强版手写解析器 (支持嵌套)"""
    # 使用 tokenizer 而非正则
    tokens = self._tokenize(expr)
    ast = self._parse_tokens(tokens)
    return self._evaluate_ast(ast, df)
```

#### 任务 3.2: 完善 provider_uri 配置

**修改文件**: `.ultra/config.json` + `src/iqfmp/core/factor_engine.py`

```json
// .ultra/config.json 添加
{
  "qlib": {
    "provider_uri": "~/.qlib/qlib_data/crypto",
    "region": "crypto",
    "default_exchange": "binance",
    "supported_timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"]
  }
}
```

```python
# factor_engine.py 修改 __init__
def __init__(self, config_path: Optional[Path] = None, ...):
    # 加载项目配置
    self._config = self._load_config(config_path)
    qlib_config = self._config.get("qlib", {})

    # 自动初始化 Qlib
    if qlib_config.get("provider_uri"):
        self.init_qlib(
            provider_uri=qlib_config["provider_uri"],
            region=qlib_config.get("region", "crypto"),
        )
```

---

### 2.4 前后端打通

#### 任务 4.1: System API 返回真实数据

**问题**: `api/system/service.py` 第 39-84 行返回硬编码 Agent 状态

**修改文件**: `src/iqfmp/api/system/service.py`

```python
# 改造 get_agents() 方法

async def get_agents(self) -> list[AgentResponse]:
    """返回真实 Agent 状态"""
    agents = []

    # 1. 查询 Celery 活跃任务
    from iqfmp.celery_app.app import celery_app
    active_tasks = celery_app.control.inspect().active() or {}

    # 2. 聚合 Agent 状态
    agent_definitions = [
        ("agent-factor-gen", "Factor Generator", "factors"),
        ("agent-evaluator", "Factor Evaluator", "evaluation"),
        ("agent-backtest", "Backtest Engine", "backtest"),
        ("agent-orchestrator", "Pipeline Orchestrator", "pipeline"),
    ]

    for agent_id, name, task_prefix in agent_definitions:
        # 检查是否有活跃任务
        active_count = sum(
            1 for worker_tasks in active_tasks.values()
            for task in worker_tasks
            if task.get("name", "").startswith(f"iqfmp.celery_app.tasks.{task_prefix}")
        )

        status = "busy" if active_count > 0 else "idle"
        current_task = None

        if active_count > 0:
            # 获取当前任务信息
            for worker_tasks in active_tasks.values():
                for task in worker_tasks:
                    if task.get("name", "").startswith(f"iqfmp.celery_app.tasks.{task_prefix}"):
                        current_task = task.get("id")
                        break

        agents.append(AgentResponse(
            id=agent_id,
            name=name,
            status=status,
            current_task=current_task,
            last_activity=datetime.now(),
            tasks_completed=await self._get_completed_count(task_prefix),
        ))

    return agents
```

#### 任务 4.2: WebSocket 推送实现

**新增文件**: `src/iqfmp/api/websocket.py`

```python
"""WebSocket 推送服务"""
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import asyncio
import json

class ConnectionManager:
    """WebSocket 连接管理器"""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {
            "pipeline": set(),
            "mining": set(),
            "trading": set(),
            "system": set(),
        }

    async def connect(self, websocket: WebSocket, channel: str):
        await websocket.accept()
        self.active_connections[channel].add(websocket)

    def disconnect(self, websocket: WebSocket, channel: str):
        self.active_connections[channel].discard(websocket)

    async def broadcast(self, channel: str, message: dict):
        """广播消息到指定频道"""
        for connection in self.active_connections[channel]:
            try:
                await connection.send_json(message)
            except:
                self.disconnect(connection, channel)

manager = ConnectionManager()

# 在 main.py 中添加路由
@app.websocket("/ws/{channel}")
async def websocket_endpoint(websocket: WebSocket, channel: str):
    if channel not in manager.active_connections:
        await websocket.close(code=4000)
        return

    await manager.connect(websocket, channel)
    try:
        while True:
            # 保持连接，等待客户端消息
            data = await websocket.receive_text()
            # 处理订阅请求等
    except WebSocketDisconnect:
        manager.disconnect(websocket, channel)
```

#### 任务 4.3: 前端 LiveTrading 页面接入真实 API

**问题**: `dashboard/src/hooks/useLiveTrading.ts` 完全是客户端模拟

**修改文件**: `dashboard/src/hooks/useLiveTrading.ts`

```typescript
// 从模拟改为真实 API + WebSocket

export function useLiveTrading() {
  const [positions, setPositions] = useState<Position[]>([]);
  const [orders, setOrders] = useState<Order[]>([]);
  const [account, setAccount] = useState<AccountInfo | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // 1. 初始加载
    const loadInitialData = async () => {
      const [positionsRes, ordersRes, accountRes] = await Promise.all([
        tradingApi.getPositions(),
        tradingApi.getOrders(),
        tradingApi.getAccount(),
      ]);
      setPositions(positionsRes.data);
      setOrders(ordersRes.data);
      setAccount(accountRes.data);
    };
    loadInitialData();

    // 2. WebSocket 实时更新
    const ws = new WebSocket(`${WS_BASE_URL}/ws/trading`);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      switch (data.type) {
        case 'position_update':
          setPositions(prev => updatePosition(prev, data.payload));
          break;
        case 'order_update':
          setOrders(prev => updateOrder(prev, data.payload));
          break;
        case 'account_update':
          setAccount(data.payload);
          break;
      }
    };
    wsRef.current = ws;

    return () => ws.close();
  }, []);

  // ... 其他方法
}
```

---

### 2.5 运行与校验

#### 启动顺序

```bash
# 1. 启动基础设施
docker compose up -d timescaledb redis qdrant rabbitmq

# 2. 等待数据库就绪
sleep 10

# 3. 初始化数据库
python scripts/init_db.py

# 4. 启动 Celery Worker
celery -A iqfmp.celery_app.app worker -l info -Q high,default,low &

# 5. 启动后端
uvicorn iqfmp.api.main:app --reload --host 0.0.0.0 --port 8000 &

# 6. 启动前端
cd dashboard && npm run dev &
```

#### 验证清单

```bash
# 健康检查
curl http://localhost:8000/health

# 创建因子
curl -X POST http://localhost:8000/api/v1/factors/ \
  -H "Content-Type: application/json" \
  -d '{"name":"test_momentum","family":["momentum"],"code":"..."}'

# 评估因子 (应返回真实指标)
curl -X POST http://localhost:8000/api/v1/factors/{factor_id}/evaluate \
  -H "Content-Type: application/json" \
  -d '{"splits":["train","valid","test"]}'

# 检查研究账本
curl http://localhost:8000/api/v1/research/ledger

# 验证 Celery 任务
curl http://localhost:8000/api/v1/backtest/create \
  -H "Content-Type: application/json" \
  -d '{"strategy_id":"...","config":{...}}'

# 检查任务状态 (应显示真实进度)
curl http://localhost:8000/api/v1/backtest/{backtest_id}
```

---

## 三、阶段2：超越 RD-Agent（4-6周）

### 3.1 LangGraph 编排与 RD-Loop 集成

#### 任务 5.1: 将 RDLoop 暴露为 API

**新增文件**: `src/iqfmp/api/pipeline/rd_loop_router.py`

```python
from fastapi import APIRouter, BackgroundTasks
from iqfmp.core.rd_loop import RDLoop, LoopConfig

router = APIRouter(prefix="/pipeline/rd-loop", tags=["RD Loop"])

@router.post("/run")
async def run_rd_loop(
    config: LoopConfig,
    background_tasks: BackgroundTasks,
):
    """启动 RD 循环"""
    run_id = str(uuid4())

    # 后台执行
    background_tasks.add_task(
        _execute_rd_loop,
        run_id=run_id,
        config=config,
    )

    return {"run_id": run_id, "status": "started"}

async def _execute_rd_loop(run_id: str, config: LoopConfig):
    """执行 RD 循环并广播进度"""
    loop = RDLoop(config=config)

    # 注册阶段回调
    async def on_phase_change(phase, progress):
        await manager.broadcast("pipeline", {
            "type": "rd_loop_progress",
            "run_id": run_id,
            "phase": phase.value,
            "progress": progress,
        })

    loop.on_phase_change = on_phase_change
    results = loop.run()

    # 保存结果到数据库
    await save_rd_loop_results(run_id, results)
```

#### 任务 5.2: 使用 LangGraph 检查点持久化

**修改文件**: `src/iqfmp/agents/orchestrator.py`

```python
# 添加 PostgreSQL 检查点保存器

from langgraph.checkpoint.postgres import PostgresSaver

class DatabaseCheckpointSaver(PostgresSaver):
    """PostgreSQL 检查点保存器"""

    def __init__(self, connection_string: str):
        super().__init__(connection_string)

    async def save(self, checkpoint: Checkpoint) -> str:
        """保存检查点到 PostgreSQL"""
        # 使用 LangGraph 官方实现
        return await super().save(checkpoint)

    async def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """加载检查点"""
        return await super().load(checkpoint_id)

# 修改 AgentOrchestrator 初始化
class AgentOrchestrator:
    def __init__(self, config: OrchestratorConfig):
        # 使用数据库检查点
        self._checkpoint_saver = DatabaseCheckpointSaver(
            connection_string=config.database_url
        )
```

### 3.2 防过拟合体系升级

#### 任务 6.1: 完善 CryptoCVSplitter 多维切分

**修改文件**: `src/iqfmp/evaluation/cv_splitter.py`

```python
# 添加 Regime 切分支持

class CryptoCVSplitter:
    def split(
        self,
        data: pd.DataFrame,
        regime_column: Optional[str] = None,
    ) -> Iterator[CVSplit]:
        """
        多维交叉验证切分

        支持:
        - 时间切分 (60/20/20)
        - 市场切分 (大/中/小盘)
        - 频率切分 (1h/4h/1d)
        - Regime 切分 (高波/低波, 趋势/震荡)
        """
        if self.config.time_split:
            yield from self._time_split(data)

        if self.config.market_split:
            yield from self._market_cap_split(data)

        if self.config.frequency_split:
            yield from self._frequency_split(data)

        if self.config.regime_split:
            yield from self._regime_split(data, regime_column)

    def _regime_split(
        self,
        data: pd.DataFrame,
        regime_column: Optional[str] = None,
    ) -> Iterator[CVSplit]:
        """按市场制度切分"""
        if regime_column is None:
            # 自动检测制度
            volatility = data['close'].pct_change().rolling(20).std()
            regime = pd.cut(
                volatility,
                bins=[0, 0.02, 0.05, float('inf')],
                labels=['low_vol', 'medium_vol', 'high_vol']
            )
        else:
            regime = data[regime_column]

        for regime_name in regime.unique():
            mask = regime == regime_name
            yield CVSplit(
                name=f"regime_{regime_name}",
                train_mask=mask & self._get_train_mask(data),
                test_mask=mask & self._get_test_mask(data),
            )
```

#### 任务 6.2: Deflated Sharpe Ratio 动态阈值

**修改文件**: `src/iqfmp/evaluation/research_ledger.py`

```python
class DynamicThreshold:
    """
    基于 Deflated Sharpe Ratio 的动态阈值

    参考: Bailey & López de Prado (2014)
    "The Deflated Sharpe Ratio: Correcting for Selection Bias,
    Backtest Overfitting and Non-Normality"
    """

    def calculate(
        self,
        n_trials: int,
        expected_sharpe: float = 0.0,
        variance_of_sharpe: float = 1.0,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """
        计算 Deflated Sharpe Ratio 阈值

        Args:
            n_trials: 已进行的试验次数
            expected_sharpe: 预期 Sharpe (通常为 0)
            variance_of_sharpe: Sharpe 方差
            skewness: 收益偏度
            kurtosis: 收益峰度

        Returns:
            调整后的 Sharpe 阈值
        """
        # 1. 计算 Expected Maximum Sharpe (基于 n_trials)
        e_max_sharpe = self._expected_max_sharpe(n_trials, variance_of_sharpe)

        # 2. 计算 Sharpe 的标准误差 (考虑非正态性)
        se_sharpe = self._sharpe_standard_error(
            n_observations=252,  # 假设一年
            skewness=skewness,
            kurtosis=kurtosis,
        )

        # 3. Deflated Sharpe Ratio 阈值
        # 要求: SR_observed > E[max(SR)] + z_alpha * SE(SR)
        z_alpha = 1.96  # 95% 置信度
        threshold = e_max_sharpe + z_alpha * se_sharpe

        return max(threshold, self.min_threshold)

    def _expected_max_sharpe(self, n: int, variance: float) -> float:
        """期望最大 Sharpe (基于正态分布的 Order Statistics)"""
        from scipy.stats import norm

        # E[max] ≈ Φ^(-1)(1 - 1/n) * sqrt(variance)
        if n <= 1:
            return 0.0

        quantile = norm.ppf(1 - 1 / n)
        return quantile * np.sqrt(variance)

    def _sharpe_standard_error(
        self,
        n_observations: int,
        skewness: float,
        kurtosis: float,
    ) -> float:
        """
        Sharpe Ratio 标准误差 (Lo, 2002)

        SE(SR) = sqrt((1 + 0.5*SR^2 - γ3*SR + (γ4-3)/4*SR^2) / n)
        """
        sr = 1.0  # 假设 SR=1 进行估算
        se_squared = (
            1 + 0.5 * sr**2
            - skewness * sr
            + (kurtosis - 3) / 4 * sr**2
        ) / n_observations

        return np.sqrt(se_squared)
```

### 3.3 因子库去重与向量检索

#### 任务 7.1: 因子入库 Qdrant

**修改文件**: `src/iqfmp/api/factors/service.py`

```python
# 在因子创建/评估通过后入库向量

async def _index_factor_to_vector_store(self, factor: Factor, metrics: FactorMetrics):
    """将因子索引到 Qdrant 向量库"""
    from iqfmp.vector.store import VectorStore
    from iqfmp.vector.embedding import get_factor_embedding

    # 1. 生成因子嵌入
    embedding = await get_factor_embedding(
        code=factor.code,
        description=factor.description,
        family=factor.family,
    )

    # 2. 准备元数据
    metadata = {
        "factor_id": factor.id,
        "name": factor.name,
        "family": factor.family,
        "sharpe": metrics.sharpe,
        "ic_mean": metrics.ic_mean,
        "created_at": factor.created_at.isoformat(),
    }

    # 3. 存入 Qdrant
    store = VectorStore()
    await store.upsert(
        collection="factors",
        id=factor.id,
        vector=embedding,
        metadata=metadata,
    )
```

#### 任务 7.2: 因子生成前查重

**修改文件**: `src/iqfmp/api/factors/service.py`

```python
async def generate_factor(self, request: FactorGenerateRequest) -> Factor:
    """生成因子 (带去重检查)"""

    # 1. 调用 LLM 生成因子代码
    result = await self._generate_factor_code(request)

    # 2. 相似因子检查 (NEW)
    similar_factors = await self._check_similarity(result.code)

    if similar_factors:
        # 返回最相似的因子信息，让用户决定
        top_similar = similar_factors[0]
        if top_similar.similarity > 0.95:
            raise FactorDuplicateError(
                f"因子与 '{top_similar.name}' 高度相似 (相似度: {top_similar.similarity:.2%})"
            )

        # 相似度较高但不完全重复，添加警告
        result.warnings.append(
            f"发现相似因子: {top_similar.name} (相似度: {top_similar.similarity:.2%})"
        )

    # 3. 创建因子
    return await self.create_factor(result)

async def _check_similarity(self, code: str) -> list[SimilarFactor]:
    """检查因子代码相似度"""
    from iqfmp.vector.store import VectorStore
    from iqfmp.vector.embedding import get_factor_embedding

    embedding = await get_factor_embedding(code=code)

    store = VectorStore()
    results = await store.search(
        collection="factors",
        vector=embedding,
        top_k=5,
        threshold=0.8,  # 相似度阈值
    )

    return [
        SimilarFactor(
            id=r.metadata["factor_id"],
            name=r.metadata["name"],
            similarity=r.score,
        )
        for r in results
    ]
```

### 3.4 策略与执行

#### 任务 8.1: Qlib 回测驱动策略生成

**修改文件**: `src/iqfmp/strategy/generator.py`

```python
class StrategyGenerator:
    """基于因子评估结果生成策略"""

    async def generate_from_factors(
        self,
        factor_ids: list[str],
        combination_method: str = "equal_weight",
    ) -> Strategy:
        """
        从验证通过的因子生成策略

        Args:
            factor_ids: 因子 ID 列表 (应来自不同 cluster)
            combination_method: 组合方法 (equal_weight, ic_weight, optimization)
        """
        # 1. 加载因子
        factors = await self._load_factors(factor_ids)

        # 2. 检查因子多样性 (不同 cluster)
        clusters = set(f.cluster_id for f in factors if f.cluster_id)
        if len(clusters) < len(factors) * 0.5:
            warnings.warn("因子多样性不足，建议选择不同聚类的因子")

        # 3. 生成组合权重
        if combination_method == "equal_weight":
            weights = {f.id: 1.0 / len(factors) for f in factors}
        elif combination_method == "ic_weight":
            weights = self._ic_weighted(factors)
        else:
            weights = await self._optimize_weights(factors)

        # 4. 生成 Qlib 策略配置
        strategy_config = self._generate_qlib_strategy(factors, weights)

        # 5. 创建策略记录
        strategy = Strategy(
            id=str(uuid4()),
            name=f"combined_{len(factors)}factors",
            factor_weights=weights,
            qlib_config=strategy_config,
            created_at=datetime.now(),
        )

        return strategy
```

#### 任务 8.2: 风险控制硬性阈值

**修改文件**: `src/iqfmp/exchange/risk.py`

```python
class RiskController:
    """风险控制器 (带硬性阈值)"""

    # 硬性阈值 (不可调整)
    MAX_DRAWDOWN_THRESHOLD = 0.15  # 15% 最大回撤触发平仓
    MAX_POSITION_RATIO = 0.3      # 单一持仓不超过 30%
    MAX_LEVERAGE = 3.0            # 最大杠杆 3x
    EMERGENCY_LOSS_THRESHOLD = 0.05  # 5% 单日亏损触发紧急平仓

    async def check_risk(self, position: Position, account: Account) -> RiskCheckResult:
        """检查风险并返回建议动作"""
        violations = []

        # 1. 回撤检查
        drawdown = self._calculate_drawdown(account)
        if drawdown > self.MAX_DRAWDOWN_THRESHOLD:
            violations.append(RiskViolation(
                type="max_drawdown",
                severity="critical",
                action="emergency_close_all",
                message=f"最大回撤 {drawdown:.2%} 超过阈值 {self.MAX_DRAWDOWN_THRESHOLD:.2%}",
            ))

        # 2. 持仓集中度检查
        position_ratio = position.value / account.equity
        if position_ratio > self.MAX_POSITION_RATIO:
            violations.append(RiskViolation(
                type="position_concentration",
                severity="high",
                action="reduce_position",
                message=f"持仓比例 {position_ratio:.2%} 超过阈值 {self.MAX_POSITION_RATIO:.2%}",
            ))

        # 3. 杠杆检查
        leverage = account.total_position_value / account.equity
        if leverage > self.MAX_LEVERAGE:
            violations.append(RiskViolation(
                type="leverage",
                severity="high",
                action="reduce_leverage",
                message=f"杠杆 {leverage:.2f}x 超过阈值 {self.MAX_LEVERAGE}x",
            ))

        # 4. 单日亏损检查
        daily_pnl = self._get_daily_pnl(account)
        daily_loss_ratio = -daily_pnl / account.equity if daily_pnl < 0 else 0
        if daily_loss_ratio > self.EMERGENCY_LOSS_THRESHOLD:
            violations.append(RiskViolation(
                type="daily_loss",
                severity="critical",
                action="emergency_close_all",
                message=f"单日亏损 {daily_loss_ratio:.2%} 超过阈值 {self.EMERGENCY_LOSS_THRESHOLD:.2%}",
            ))

        return RiskCheckResult(
            is_safe=len(violations) == 0,
            violations=violations,
            recommended_action=self._get_recommended_action(violations),
        )
```

### 3.5 监控与可视化

#### 任务 9.1: Prometheus 指标暴露

**新增文件**: `src/iqfmp/monitoring/metrics.py`

```python
from prometheus_client import Counter, Histogram, Gauge

# LLM 指标
LLM_REQUEST_TOTAL = Counter(
    'iqfmp_llm_requests_total',
    'Total LLM API requests',
    ['model', 'status']
)
LLM_REQUEST_LATENCY = Histogram(
    'iqfmp_llm_request_latency_seconds',
    'LLM request latency',
    ['model'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60]
)
LLM_TOKEN_USAGE = Counter(
    'iqfmp_llm_tokens_total',
    'Total tokens used',
    ['model', 'type']  # type: prompt, completion
)

# 因子指标
FACTOR_GENERATION_TOTAL = Counter(
    'iqfmp_factors_generated_total',
    'Total factors generated',
    ['family', 'status']
)
FACTOR_EVALUATION_LATENCY = Histogram(
    'iqfmp_factor_evaluation_latency_seconds',
    'Factor evaluation latency',
    buckets=[1, 5, 10, 30, 60, 120]
)
FACTOR_PASS_RATE = Gauge(
    'iqfmp_factor_pass_rate',
    'Factor pass rate',
    ['family']
)

# 回测指标
BACKTEST_DURATION = Histogram(
    'iqfmp_backtest_duration_seconds',
    'Backtest execution duration',
    buckets=[10, 30, 60, 120, 300, 600]
)

# 任务队列指标
TASK_QUEUE_LENGTH = Gauge(
    'iqfmp_task_queue_length',
    'Number of tasks in queue',
    ['queue']
)
```

#### 任务 9.2: 前端监控大屏

**新增文件**: `dashboard/src/pages/MonitoringDashboardPage.tsx`

```typescript
export function MonitoringDashboardPage() {
  const { data: metrics, isLoading } = useSystemMetrics();

  return (
    <div className="grid grid-cols-3 gap-4 p-4">
      {/* LLM 性能 */}
      <Card>
        <CardHeader>LLM 性能</CardHeader>
        <CardContent>
          <div className="space-y-2">
            <MetricRow label="平均延迟" value={`${metrics?.llm.avgLatency}ms`} />
            <MetricRow label="成功率" value={`${metrics?.llm.successRate}%`} />
            <MetricRow label="今日 Token" value={metrics?.llm.tokensToday} />
          </div>
        </CardContent>
      </Card>

      {/* 因子统计 */}
      <Card>
        <CardHeader>因子生成</CardHeader>
        <CardContent>
          <div className="space-y-2">
            <MetricRow label="今日生成" value={metrics?.factors.generatedToday} />
            <MetricRow label="通过率" value={`${metrics?.factors.passRate}%`} />
            <MetricRow label="平均 Sharpe" value={metrics?.factors.avgSharpe?.toFixed(2)} />
          </div>
        </CardContent>
      </Card>

      {/* 任务队列 */}
      <Card>
        <CardHeader>任务队列</CardHeader>
        <CardContent>
          <div className="space-y-2">
            <MetricRow label="待处理" value={metrics?.queue.pending} />
            <MetricRow label="处理中" value={metrics?.queue.active} />
            <MetricRow label="已完成" value={metrics?.queue.completed} />
          </div>
        </CardContent>
      </Card>

      {/* 因子稳定性图表 */}
      <Card className="col-span-2">
        <CardHeader>因子稳定性趋势</CardHeader>
        <CardContent>
          <StabilityChart data={metrics?.stabilityTrend} />
        </CardContent>
      </Card>

      {/* 实时日志 */}
      <Card>
        <CardHeader>实时日志</CardHeader>
        <CardContent>
          <LogStream channel="system" maxLines={20} />
        </CardContent>
      </Card>
    </div>
  );
}
```

---

## 四、任务优先级矩阵

| 优先级 | 任务 | 影响 | 工作量 | 依赖 |
|--------|------|------|--------|------|
| **P0** | 1.1 因子持久化切换 DB | 🔴 Critical | 2d | 无 |
| **P0** | 2.1 接入完整 FactorEvaluator | 🔴 Critical | 3d | 1.1 |
| **P0** | 1.2 Celery 任务写回 DB | 🔴 Critical | 2d | 1.1 |
| **P1** | 2.2 交易成本模型 | 🟡 High | 2d | 2.1 |
| **P1** | 3.1 Qlib 表达式解析修复 | 🟡 High | 3d | 无 |
| **P1** | 4.1 System API 真实数据 | 🟡 High | 1d | 1.2 |
| **P1** | 4.2 WebSocket 推送 | 🟡 High | 2d | 无 |
| **P2** | 5.1 RDLoop API 暴露 | 🟠 Medium | 2d | 2.1 |
| **P2** | 6.1 CryptoCVSplitter 完善 | 🟠 Medium | 2d | 2.1 |
| **P2** | 6.2 Deflated Sharpe 阈值 | 🟠 Medium | 1d | 无 |
| **P2** | 7.1 因子入库 Qdrant | 🟠 Medium | 2d | 2.1 |
| **P2** | 7.2 因子生成前查重 | 🟠 Medium | 1d | 7.1 |
| **P3** | 8.1 策略生成器 | 🟢 Low | 3d | 7.1 |
| **P3** | 8.2 风险控制硬阈值 | 🟢 Low | 2d | 无 |
| **P3** | 9.1 Prometheus 指标 | 🟢 Low | 1d | 无 |
| **P3** | 9.2 监控大屏 | 🟢 Low | 2d | 9.1 |

---

## 五、预期成果

### 阶段1完成后 (2-3周)

- ✅ 因子创建→评估→查询完整闭环
- ✅ 研究账本持久化并可查询
- ✅ Celery 任务真实执行并写回 DB
- ✅ 前端可展示真实数据
- ✅ WebSocket 实时推送

### 阶段2完成后 (4-6周)

- ✅ RD-Loop 可通过 API 启动
- ✅ 因子去重和向量检索
- ✅ Deflated Sharpe 动态阈值
- ✅ 策略自动生成
- ✅ 完整监控体系

---

## 六、风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| Qlib 表达式兼容性问题 | 高 | 中 | 保留手写解析器作为降级 |
| 向量库嵌入模型依赖 | 中 | 中 | 使用本地 sentence-transformers |
| Celery 任务卡死 | 中 | 高 | 设置 soft/hard timeout |
| WebSocket 连接数过多 | 低 | 中 | 使用连接池和限流 |

---

*文档生成时间: 2025-12-10*
*版本: 1.0*
