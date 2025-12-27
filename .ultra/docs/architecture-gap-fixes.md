# 架构缺口修复跟踪

> 基于 2025-12-26 团队架构审查反馈

## 问题总览

| 优先级 | 问题 | 状态 | 负责 |
|--------|------|------|------|
| P0 | 安全三层未落地 | 🔴 待修 | - |
| P1 | 研究账本无 TimescaleDB | 🔴 待修 | - |
| P1 | Pipeline 未与 API/Celery 接线 | 🔴 待修 | - |
| P1 | 防过拟合默认关闭 | 🔴 待修 | - |
| P2 | Qlib RL/ML 能力未利用 | 🟡 低优 | - |
| P2 | Alpha 数据集未对齐基准 | 🟡 低优 | - |

---

## P0: 安全三层落地

### 问题描述
1. `sandbox.py` 使用原生 `exec()` + 白名单，缺少 RestrictedPython
2. `review.py` 的 `HumanReviewGate` 已实现但未在任何节点调用
3. 无 CPU/内存资源限制

### 修复任务

#### 任务 1.1: RestrictedPython 集成
```bash
pip install RestrictedPython
```

修改 `src/iqfmp/core/sandbox.py`:
```python
from RestrictedPython import compile_restricted, safe_builtins

# 替换原有 compile()
compiled = compile_restricted(code, "<sandbox>", "exec")
```

#### 任务 1.2: 资源限制
```python
import resource

# 添加到 SandboxConfig
max_cpu_seconds: int = 30
max_memory_bytes: int = 512 * 1024 * 1024  # 512MB

# 在执行前设置
resource.setrlimit(resource.RLIMIT_CPU, (config.max_cpu_seconds, config.max_cpu_seconds))
resource.setrlimit(resource.RLIMIT_AS, (config.max_memory_bytes, config.max_memory_bytes))
```

#### 任务 1.3: HumanReviewGate 接入
在以下节点前调用：
- `FactorGenerationAgent.generate()` 返回后
- `EvaluationAgent.evaluate()` 执行前
- `BacktestAgent.run_backtest()` 执行前

---

## P1: 研究账本 TimescaleDB 持久化

### 问题描述
- `research_ledger.py` 仅 MemoryStorage/FileStorage
- `factor_evaluator.py` 默认 MemoryStorage
- 动态阈值未入库

### 修复任务

#### 任务 2.1: TimescaleDB 后端

创建 `src/iqfmp/evaluation/timescale_storage.py`:
```python
class TimescaleDBStorage(StorageBackend):
    """TimescaleDB 持久化后端"""

    async def save_trial(self, trial: TrialRecord) -> str:
        # INSERT INTO research_ledger
        pass

    async def get_trial(self, trial_id: str) -> Optional[TrialRecord]:
        # SELECT FROM research_ledger
        pass
```

#### 任务 2.2: 数据库表
```sql
CREATE TABLE research_ledger (
    trial_id TEXT PRIMARY KEY,
    factor_name TEXT NOT NULL,
    factor_family TEXT NOT NULL,
    sharpe_ratio DOUBLE PRECISION,
    ic_mean DOUBLE PRECISION,
    ir DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    win_rate DOUBLE PRECISION,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('research_ledger', 'created_at');
```

---

## P1: Pipeline 编排接线

### 问题描述
- `langgraph_orchestrator.py` 定义了 StateGraph
- `api/pipeline/service.py` 仅管理状态，不触发 LangGraph

### 修复任务

#### 任务 3.1: API 调用 LangGraph

修改 `src/iqfmp/api/pipeline/service.py`:
```python
from iqfmp.agents.langgraph_orchestrator import create_factor_pipeline

async def start_pipeline(self, run_id: str, config: dict):
    graph = create_factor_pipeline()
    result = await graph.ainvoke({"thread_id": run_id, **config})
    return result
```

#### 任务 3.2: Celery 集成

修改 `src/iqfmp/celery_app/tasks.py`:
```python
@celery_app.task
def run_factor_pipeline(run_id: str, config: dict):
    graph = create_factor_pipeline()
    asyncio.run(graph.ainvoke({"thread_id": run_id, **config}))
```

---

## P1: 防过拟合流程

### 问题描述
- `EvaluationConfig.use_cv_splits = False` 默认关闭
- CryptoCVSplitter 已实现但未被接入

### 修复任务

#### 任务 4.1: 默认启用 CV

修改 `src/iqfmp/evaluation/factor_evaluator.py`:
```python
@dataclass
class EvaluationConfig:
    use_cv_splits: bool = True  # 改为 True
    run_stability_analysis: bool = True  # 改为 True
```

#### 任务 4.2: Purged CV 接入

确保评估路径调用：
```python
from iqfmp.evaluation.cv_splitter import CryptoCVSplitter

splitter = CryptoCVSplitter(n_splits=5, purge_gap=10)
for train_idx, test_idx in splitter.split(data):
    # 评估每个 fold
```

---

## P2: Qlib RL/ML 集成

### 问题描述
- `qlib_rl_adapter.py` 只包装环境
- 未调用 `qlib.rl.contrib.train_onpolicy`

### 修复任务

#### 任务 5.1: 验证 Qlib RL 可用
```bash
cd vendor/qlib/examples/rl_order_execution
python -m qlib.rl.contrib.train_onpolicy --config_path exp_configs/train_ppo.yml
```

#### 任务 5.2: 集成到 RL Adapter
```python
from qlib.rl.contrib.train_onpolicy import train
from qlib.rl.contrib.backtest import backtest

# 在 qlib_rl_adapter.py 中调用
```

---

## P2: Alpha 数据集基准对齐

### 问题描述
- 未使用 Qlib workflow 的 Alpha158/360 dataset+model 配置
- 缺乏与基准的对比

### 修复任务

#### 任务 6.1: 验证 Qlib Model Zoo
```bash
python vendor/qlib/examples/run_all_model.py \
  --dataset Alpha360 \
  --qlib_data_path ~/.qlib/qlib_data/cn_data \
  --models lightgbm xgboost
```

#### 任务 6.2: 基准配置集成
使用 Qlib 官方 benchmarks 配置作为评估基准

---

## 实施顺序建议

```
Phase 1 (本周): P0 安全三层
  └─ RestrictedPython + HumanReviewGate 接入

Phase 2 (下周): P1 核心功能
  ├─ TimescaleDB 持久化
  ├─ Pipeline 接线
  └─ CV 验证启用

Phase 3 (后续): P2 能力增强
  ├─ Qlib RL 集成
  └─ Alpha 基准对齐
```

---

*Last updated: 2025-12-26*
