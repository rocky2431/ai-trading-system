# IQFMP 架构重新设计方案 v3

**核心问题**: 当前IQFMP对Qlib的使用和LLM的使用完全不对
**目标**: 设计一个能够工作的多Agent架构，结合RD-Agent与Qlib集成能力 + 成熟多Agent框架

---

## 📍 问题1: 知识库应该存在哪里？

### 选项分析

| 选项 | 位置 | 优点 | 缺点 |
|------|------|------|------|
| **A. qtrade PSQL** | 现有qtrade数据库 | 统一管理、已有连接 | 可能与交易数据混合 |
| **B. 独立PSQL实例** | 新建dedicated DB | 隔离清晰、独立扩展 | 需要额外维护 |
| **C. SQLite本地** | 项目目录内 | 简单、无依赖 | 不适合分布式 |
| **D. Redis + PSQL混合** | 热数据Redis + 冷数据PSQL | 高性能 | 架构复杂 |

### 推荐方案: **B. 独立PSQL实例 (专用schema)**

```
# 在现有PSQL服务器上创建独立schema
qtrade_db
├── public          # 现有交易数据
└── knowledge       # 新建: 知识库专用schema
    ├── factor_traces          # 因子尝试记录
    ├── factor_successes       # 成功因子
    ├── error_patterns         # 错误模式
    ├── component_mappings     # 组件映射
    └── embeddings_cache       # 向量缓存
```

**理由**:
1. 复用现有PSQL连接，无需新建实例
2. Schema隔离确保交易数据不受影响
3. 支持关系型查询 + pgvector扩展做向量搜索
4. 可独立备份/迁移知识库

**配置示例**:
```python
# iqfmp/config.py
KNOWLEDGE_DB_SCHEMA = "knowledge"
KNOWLEDGE_DB_URL = os.getenv("DATABASE_URL")  # 复用现有连接
```

---

## 📍 问题2: RD-Agent如何用Python函数连接Qlib？

### RD-Agent的完整机制

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RD-Agent Qlib集成架构                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. LLM生成Python代码 (factor.py)                                            │
│     ┌────────────────────────────────────────────────┐                       │
│     │ import pandas as pd                            │                       │
│     │                                                │                       │
│     │ def factor(df: pd.DataFrame) -> pd.Series:    │                       │
│     │     close = df['$close']                       │                       │
│     │     result = close.rolling(20).mean() / close │                       │
│     │     return result                              │                       │
│     └────────────────────────────────────────────────┘                       │
│                           │                                                  │
│                           ▼                                                  │
│  2. FactorFBWorkspace 执行Python代码                                         │
│     ┌────────────────────────────────────────────────┐                       │
│     │ subprocess.check_output(                       │                       │
│     │     f"python factor.py",                       │                       │
│     │     cwd=workspace_path  # 包含源数据            │                       │
│     │ )                                              │                       │
│     │ → 输出: result.h5 (HDF5格式的factor值)         │                       │
│     └────────────────────────────────────────────────┘                       │
│                           │                                                  │
│                           ▼                                                  │
│  3. process_factor_data 收集所有factor                                       │
│     ┌────────────────────────────────────────────────┐                       │
│     │ factor_dfs = []                                │                       │
│     │ for impl in exp.sub_workspace_list:            │                       │
│     │     msg, df = impl.execute("All")              │                       │
│     │     if df is not None:                         │                       │
│     │         factor_dfs.append(df)                  │                       │
│     │ combined = pd.concat(factor_dfs, axis=1)       │                       │
│     │ combined.to_parquet("combined_factors.parquet")│                       │
│     └────────────────────────────────────────────────┘                       │
│                           │                                                  │
│                           ▼                                                  │
│  4. Qlib配置文件 (conf_combined_factors.yaml)                                 │
│     ┌────────────────────────────────────────────────┐                       │
│     │ data_handler_config:                           │                       │
│     │   data_loader:                                 │                       │
│     │     class: NestedDataLoader                    │                       │
│     │     kwargs:                                    │                       │
│     │       dataloader_l:                            │                       │
│     │         - class: Alpha158DL  # 内置因子        │                       │
│     │         - class: StaticDataLoader              │                       │
│     │           kwargs:                              │                       │
│     │             config: "combined_factors.parquet" │ ◄── Python因子输出     │
│     └────────────────────────────────────────────────┘                       │
│                           │                                                  │
│                           ▼                                                  │
│  5. Docker/Conda环境执行Qlib回测                                              │
│     ┌────────────────────────────────────────────────┐                       │
│     │ qtde.check_output(                             │                       │
│     │     local_path=workspace_path,                 │                       │
│     │     entry="qrun conf_combined_factors.yaml"    │                       │
│     │ )                                              │                       │
│     │ → 输出: qlib_res.csv (IC, Sharpe, IR等指标)    │                       │
│     └────────────────────────────────────────────────┘                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 关键发现

**RD-Agent不是用Qlib表达式，而是**:
1. LLM生成**完整Python函数**
2. Python函数读取源数据，计算factor值，输出到**HDF5/Parquet文件**
3. Qlib通过**StaticDataLoader**加载这个文件作为feature
4. Qlib配置文件**动态组合**内置因子 + 自定义因子
5. 整个回测在**隔离环境(Docker/Conda)**中执行

### 为什么这样设计？

```python
# RD-Agent的评估需要:
# 1. 代码执行反馈 - Python代码能运行吗？有语法错误吗？
# 2. 数值输出反馈 - 输出格式对吗？数值合理吗？
# 3. 与Ground Truth对比 - 精确匹配验证

# Qlib表达式做不到这些，因为:
# - 无法获取中间执行状态
# - 无法进行精确数值对比
# - 无法自定义复杂逻辑
```

---

## 📍 问题3: IQFMP当前架构的核心问题

### 当前架构 vs 正确架构

```
当前IQFMP架构 (问题重重):
┌────────────────────────────────────────────────────┐
│  用户假设                                           │
│      │                                              │
│      ▼                                              │
│  LLM生成Qlib表达式 ────────────────┐               │
│  "RSI($close, 14)"                 │               │
│      │                              │               │
│      ▼                              ▼               │
│  FactorEngine.compute_factor()   FactorEvaluator   │
│  (内部eval执行)                  (IC/IR计算)       │
│      │                              │               │
│      └──────────────┬───────────────┘               │
│                     ▼                               │
│               阈值判断 → 存储/丢弃                  │
│                                                     │
│  ❌ 问题:                                           │
│  - 没有代码执行反馈循环                             │
│  - 没有知识积累和复用                               │
│  - 没有与Qlib回测系统的真正集成                     │
│  - Qlib表达式能力有限，无法实现复杂因子              │
└────────────────────────────────────────────────────┘

正确的架构 (参考RD-Agent + 成熟多Agent框架):
┌────────────────────────────────────────────────────────────────────────────┐
│                         Multi-Agent Factor Mining System                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ Hypothesis  │    │   Coder     │    │  Executor   │    │  Evaluator  │  │
│  │   Agent     │───▶│   Agent     │───▶│   Agent     │───▶│   Agent     │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                  │                  │                  │           │
│        └──────────────────┴──────────────────┴──────────────────┘           │
│                                      │                                      │
│                                      ▼                                      │
│                          ┌─────────────────────┐                            │
│                          │   Knowledge Base    │                            │
│                          │   (PSQL + Vector)   │                            │
│                          └─────────────────────┘                            │
│                                      │                                      │
│                                      ▼                                      │
│                          ┌─────────────────────┐                            │
│                          │  Qlib Integration   │                            │
│                          │  (Docker/Subprocess)│                            │
│                          └─────────────────────┘                            │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 📍 问题4: 成熟多Agent架构对比

### 主流框架对比

| 特性 | LangGraph | CrewAI | AutoGen |
|------|-----------|--------|---------|
| **设计哲学** | 图状态机 | 角色扮演团队 | 对话式协作 |
| **流程控制** | 显式图定义 | 隐式任务链 | 消息传递 |
| **状态管理** | 中央State | 分布式Memory | SharedContext |
| **适合场景** | 复杂工作流 | 明确分工任务 | 开放对话 |
| **学习曲线** | 中等 | 低 | 中等 |

### 推荐: LangGraph风格

**理由**:
1. **显式状态图** - Factor Mining需要明确的阶段(假设→编码→执行→评估→反馈)
2. **循环支持** - 自然支持迭代优化循环
3. **状态持久化** - 可以保存/恢复实验状态
4. **条件分支** - 根据评估结果决定下一步动作

---

## 🎯 新架构设计: IQFMP Multi-Agent System

### 架构总览

```python
"""
IQFMP Multi-Agent Factor Mining System
基于LangGraph风格的状态图架构
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
import pandas as pd

# ============================================================================
# 状态定义
# ============================================================================

class FactorMiningState(Enum):
    """状态机状态"""
    IDLE = "idle"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    FACTOR_CODING = "factor_coding"
    CODE_EXECUTION = "code_execution"
    FACTOR_EVALUATION = "factor_evaluation"
    QLIB_BACKTEST = "qlib_backtest"
    FEEDBACK_ANALYSIS = "feedback_analysis"
    KNOWLEDGE_UPDATE = "knowledge_update"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SystemState:
    """中央状态对象 - 在所有Agent间共享"""

    # 当前状态
    current_state: FactorMiningState = FactorMiningState.IDLE
    iteration: int = 0
    max_iterations: int = 5

    # 假设相关
    hypothesis: Optional[str] = None
    hypothesis_family: Optional[str] = None

    # 代码相关
    factor_code: Optional[str] = None  # Python函数代码
    code_is_valid: bool = False

    # 执行相关
    execution_output: Optional[pd.DataFrame] = None
    execution_error: Optional[str] = None

    # 评估相关
    factor_metrics: dict = field(default_factory=dict)  # IC, IR, Sharpe
    qlib_backtest_result: Optional[pd.Series] = None

    # 反馈相关
    feedback: Optional[str] = None
    should_retry: bool = False

    # 知识库查询结果
    similar_successes: list = field(default_factory=list)
    similar_errors: list = field(default_factory=list)


# ============================================================================
# Agent定义
# ============================================================================

class HypothesisAgent:
    """假设生成Agent - 基于知识库生成因子假设"""

    def __init__(self, llm_provider, knowledge_base):
        self.llm = llm_provider
        self.kb = knowledge_base

    async def generate(self, state: SystemState) -> SystemState:
        """生成因子假设"""

        # 1. 从知识库查询历史信息
        history = self.kb.get_recent_hypotheses(limit=10)
        successful_patterns = self.kb.get_successful_patterns()

        # 2. 构建动态prompt
        prompt = self._build_prompt(history, successful_patterns, state.feedback)

        # 3. 调用LLM生成假设
        response = await self.llm.complete(prompt)

        # 4. 更新状态
        state.hypothesis = response.hypothesis
        state.hypothesis_family = response.family
        state.current_state = FactorMiningState.FACTOR_CODING

        return state


class CoderAgent:
    """代码生成Agent - 生成Python因子函数"""

    def __init__(self, llm_provider, knowledge_base):
        self.llm = llm_provider
        self.kb = knowledge_base

    async def generate(self, state: SystemState) -> SystemState:
        """生成因子Python代码"""

        # 1. 查询相似成功案例
        state.similar_successes = self.kb.query_similar_tasks(state.hypothesis)

        # 2. 如果是重试，查询相似错误解决方案
        if state.execution_error:
            state.similar_errors = self.kb.query_similar_errors(state.execution_error)

        # 3. 构建动态prompt (RD-Agent风格)
        prompt = self._build_prompt(
            hypothesis=state.hypothesis,
            similar_successes=state.similar_successes,
            similar_errors=state.similar_errors,
            previous_code=state.factor_code if state.should_retry else None,
            previous_error=state.execution_error if state.should_retry else None,
        )

        # 4. 生成Python函数代码
        response = await self.llm.complete(prompt)

        # 5. 更新状态
        state.factor_code = response.code
        state.current_state = FactorMiningState.CODE_EXECUTION

        return state

    def _build_prompt(self, **kwargs) -> str:
        """构建动态prompt - 注入知识库内容"""
        parts = [FACTOR_CODING_SYSTEM_PROMPT]

        # 注入相似成功案例
        if kwargs.get("similar_successes"):
            parts.append("## Similar Successful Implementations:")
            for success in kwargs["similar_successes"][:3]:
                parts.append(f"### {success.task_info}")
                parts.append(f"```python\n{success.code}\n```")

        # 注入错误解决方案
        if kwargs.get("similar_errors"):
            parts.append("## Similar Errors and Solutions:")
            for err_solution in kwargs["similar_errors"][:2]:
                parts.append(f"Error: {err_solution.error}")
                parts.append(f"Solution: ```python\n{err_solution.fixed_code}\n```")

        # 如果是重试，包含之前的错误
        if kwargs.get("previous_error"):
            parts.append(f"## Previous Attempt Failed:")
            parts.append(f"Code: ```python\n{kwargs['previous_code']}\n```")
            parts.append(f"Error: {kwargs['previous_error']}")

        parts.append(f"## Current Task: {kwargs['hypothesis']}")

        return "\n\n".join(parts)


class ExecutorAgent:
    """代码执行Agent - 在隔离环境中执行Python代码"""

    def __init__(self, workspace_manager):
        self.workspace = workspace_manager

    async def execute(self, state: SystemState) -> SystemState:
        """执行因子代码"""

        # 1. 创建工作空间
        ws_path = self.workspace.create(state.hypothesis)

        # 2. 写入代码文件
        self.workspace.write_file(ws_path / "factor.py", state.factor_code)

        # 3. 链接源数据
        self.workspace.link_data(ws_path)

        # 4. 执行代码 (subprocess)
        try:
            result = await self.workspace.execute(
                ws_path,
                command="python factor.py",
                timeout=60,
            )

            # 5. 读取输出
            output_path = ws_path / "result.h5"
            if output_path.exists():
                state.execution_output = pd.read_hdf(output_path)
                state.code_is_valid = True
                state.execution_error = None
                state.current_state = FactorMiningState.FACTOR_EVALUATION
            else:
                state.execution_error = "No output file generated"
                state.code_is_valid = False
                state.current_state = FactorMiningState.FEEDBACK_ANALYSIS

        except Exception as e:
            state.execution_error = str(e)
            state.code_is_valid = False
            state.current_state = FactorMiningState.FEEDBACK_ANALYSIS

        return state


class EvaluatorAgent:
    """评估Agent - 多维度评估因子质量"""

    def __init__(self, llm_provider):
        self.llm = llm_provider

    async def evaluate(self, state: SystemState) -> SystemState:
        """多维度评估因子"""

        if state.execution_output is None:
            state.current_state = FactorMiningState.FEEDBACK_ANALYSIS
            return state

        # 1. 数值评估
        metrics = self._compute_metrics(state.execution_output)
        state.factor_metrics = metrics

        # 2. 代码质量评估 (LLM)
        code_feedback = await self._evaluate_code_quality(
            state.factor_code,
            state.hypothesis,
        )

        # 3. 综合判断
        is_success = (
            metrics.get("ic_mean", 0) >= 0.03 and
            metrics.get("ir", 0) >= 0.5 and
            code_feedback.is_valid
        )

        if is_success:
            state.current_state = FactorMiningState.QLIB_BACKTEST
        else:
            state.feedback = self._generate_feedback(metrics, code_feedback)
            state.current_state = FactorMiningState.FEEDBACK_ANALYSIS

        return state


class QlibBacktestAgent:
    """Qlib回测Agent - 集成Qlib进行完整回测"""

    def __init__(self, qlib_config):
        self.config = qlib_config

    async def backtest(self, state: SystemState) -> SystemState:
        """执行Qlib回测"""

        # 1. 准备因子数据
        factor_path = self._save_factor_data(state.execution_output)

        # 2. 生成Qlib配置 (类似RD-Agent)
        config_path = self._generate_qlib_config(factor_path)

        # 3. 执行Qlib回测 (Docker/Subprocess)
        result, stdout = await self._run_qlib(config_path)

        if result is not None:
            state.qlib_backtest_result = result
            state.current_state = FactorMiningState.KNOWLEDGE_UPDATE
        else:
            state.execution_error = stdout
            state.current_state = FactorMiningState.FEEDBACK_ANALYSIS

        return state


class FeedbackAgent:
    """反馈分析Agent - 决定是否重试"""

    async def analyze(self, state: SystemState) -> SystemState:
        """分析反馈，决定下一步"""

        if state.iteration >= state.max_iterations:
            state.current_state = FactorMiningState.FAILED
            return state

        # 决定是否重试
        if state.execution_error or not state.code_is_valid:
            state.should_retry = True
            state.iteration += 1
            state.current_state = FactorMiningState.FACTOR_CODING
        else:
            state.should_retry = False
            state.current_state = FactorMiningState.COMPLETED

        return state


class KnowledgeAgent:
    """知识更新Agent - 更新知识库"""

    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    async def update(self, state: SystemState) -> SystemState:
        """更新知识库"""

        # 1. 记录成功案例
        self.kb.add_success(
            task_info=state.hypothesis,
            code=state.factor_code,
            metrics=state.factor_metrics,
        )

        # 2. 清理工作轨迹
        self.kb.clear_working_trace(state.hypothesis)

        state.current_state = FactorMiningState.COMPLETED
        return state


# ============================================================================
# 状态图定义 (LangGraph风格)
# ============================================================================

class FactorMiningGraph:
    """因子挖掘状态图"""

    def __init__(self, agents: dict):
        self.agents = agents
        self.transitions = {
            FactorMiningState.IDLE: self._hypothesis,
            FactorMiningState.HYPOTHESIS_GENERATION: self._coding,
            FactorMiningState.FACTOR_CODING: self._execute,
            FactorMiningState.CODE_EXECUTION: self._evaluate_or_feedback,
            FactorMiningState.FACTOR_EVALUATION: self._backtest_or_feedback,
            FactorMiningState.QLIB_BACKTEST: self._knowledge_or_feedback,
            FactorMiningState.FEEDBACK_ANALYSIS: self._retry_or_fail,
            FactorMiningState.KNOWLEDGE_UPDATE: self._complete,
        }

    async def run(self, initial_state: SystemState) -> SystemState:
        """执行状态图"""
        state = initial_state

        while state.current_state not in [
            FactorMiningState.COMPLETED,
            FactorMiningState.FAILED,
        ]:
            transition = self.transitions.get(state.current_state)
            if transition:
                state = await transition(state)
            else:
                break

        return state

    async def _hypothesis(self, state):
        return await self.agents["hypothesis"].generate(state)

    async def _coding(self, state):
        return await self.agents["coder"].generate(state)

    async def _execute(self, state):
        return await self.agents["executor"].execute(state)

    async def _evaluate_or_feedback(self, state):
        if state.code_is_valid:
            return await self.agents["evaluator"].evaluate(state)
        else:
            return await self.agents["feedback"].analyze(state)

    # ... 其他转换方法
```

---

## 📍 安全重构策略 (不破坏现有功能)

### 阶段式迁移

```
Phase 1: 知识库基础设施 (1-2天)
├── 创建knowledge schema
├── 实现基础表结构
├── 实现KnowledgeBase类
└── ✅ 不影响现有功能

Phase 2: Agent接口定义 (2-3天)
├── 定义Agent协议/接口
├── 实现状态机框架
├── 保持旧接口兼容
└── ✅ 新旧系统并行

Phase 3: 逐个Agent迁移 (3-5天)
├── HypothesisAgent (用新知识库)
├── CoderAgent (支持Python函数)
├── ExecutorAgent (隔离执行)
├── EvaluatorAgent (多维评估)
└── ✅ 每个Agent独立测试

Phase 4: Qlib集成 (2-3天)
├── 实现StaticDataLoader集成
├── 配置文件生成
├── Docker/Subprocess执行
└── ✅ 完整回测流程

Phase 5: 切换和验证 (1-2天)
├── 特性开关控制新旧系统
├── A/B测试验证
├── 逐步切换流量
└── ✅ 平滑过渡
```

### 代码结构

```
src/iqfmp/
├── agents/                    # 现有 - 保持不变
│   ├── factor_generation.py   # 现有 - 保持兼容
│   └── hypothesis_agent.py    # 现有 - 保持兼容
│
├── multi_agent/               # 新增 - 新架构
│   ├── __init__.py
│   ├── state.py              # 状态定义
│   ├── graph.py              # 状态图
│   ├── agents/
│   │   ├── hypothesis.py     # 假设Agent
│   │   ├── coder.py          # 编码Agent
│   │   ├── executor.py       # 执行Agent
│   │   ├── evaluator.py      # 评估Agent
│   │   ├── backtest.py       # 回测Agent
│   │   └── knowledge.py      # 知识Agent
│   └── knowledge/
│       ├── base.py           # 知识库基类
│       ├── postgres.py       # PSQL实现
│       └── queries.py        # 查询方法
│
├── qlib_integration/          # 新增 - Qlib集成
│   ├── __init__.py
│   ├── workspace.py          # 工作空间管理
│   ├── config_generator.py   # 配置生成
│   ├── executor.py           # 隔离执行
│   └── templates/
│       ├── conf_baseline.yaml
│       └── conf_combined.yaml
│
└── core/
    ├── rd_loop.py            # 现有 - 添加特性开关
    └── rd_loop_v2.py         # 新增 - 新架构入口
```

---

## 📊 总结

| 问题 | 解答 |
|------|------|
| **知识库存在哪里？** | 现有PSQL的独立schema `knowledge`，复用连接，隔离数据 |
| **RD-Agent如何用Python？** | LLM生成Python函数 → subprocess执行 → 输出HDF5 → Qlib StaticDataLoader加载 |
| **如何安全重构？** | 阶段式迁移，特性开关控制，新旧系统并行 |
| **多Agent架构？** | LangGraph风格状态图，6个专职Agent，中央状态共享 |

**下一步**: 需要我开始实现知识库基础设施吗？
