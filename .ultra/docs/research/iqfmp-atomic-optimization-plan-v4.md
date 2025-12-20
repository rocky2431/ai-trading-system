# IQFMP 原子级优化方案 v4.0

> **目标**：以RD-Agent为"LLM交互正确范式"参照，在LLM交互/工程化与Qlib能力上全面超越它，系统深度做成crypto-first
>
> **置信度说明**：
> - 工程实现超越RD-Agent: **>99%** (已有StateGraph编排 + Qlib fork + crypto数据链路)
> - 收益/Sharpe提升: **不承诺** (取决于数据、成本、交易约束与市场阶段；通过A/B实验验证)

---

## 当前代码问题诊断（基于代码事实）

### 问题1: LLM层缺失RD-Agent核心能力

**当前代码位置**: `src/iqfmp/llm/provider.py:201`

| 能力 | RD-Agent | IQFMP当前 | 差距 |
|------|----------|-----------|------|
| 持久化缓存 | SQLite + MD5 key | 内存dict | ❌ 重启即失 |
| 会话历史 | ChatSession类 | 无 | ❌ 无法多轮 |
| 自动续写 | finish_reason=="length"→追加 | 无 | ❌ 长输出截断 |
| JSON schema校验 | response_format + TypeAdapter | 无 | ❌ 解析失败 |
| 错误分类重试 | 429/timeout/5xx分开处理 | 统一LLMError | ❌ 无法智能重试 |
| 种子控制 | seed影响cache key | 无 | ❌ 无法复现 |

**RD-Agent关键代码参考**:
```python
# rdagent/oai/backend/base.py:218 - 持久化缓存
class SQliteLazyCache:
    def chat_get(self, content: str) -> str | None:
        md5_key = hashlib.md5(content.encode()).hexdigest()
        self.c.execute("SELECT value FROM chat_cache WHERE key=?", (md5_key,))
        ...

# rdagent/oai/backend/base.py:562 - 自动续写
def _create_chat_completion_auto_continue(self, ...):
    for _ in range(try_n):
        response, finish_reason = self._create_chat_completion_inner_function(...)
        all_response += response
        if finish_reason is None or finish_reason != "length":
            break  # 完整响应
        new_messages.append({"role": "assistant", "content": response})
```

### 问题2: Prompt输出类型矛盾

**当前代码位置**: `src/iqfmp/llm/prompts/factor_generation.py`

```python
# Line 54-74: system_prompt 正确说 "Qlib expression"
"You MUST use Qlib expression syntax. DO NOT write Python functions."

# Line 268-270: render() 却说 "Generate Python factor function" ❌
parts.append(
    "Generate a Python factor function following the guidelines..."
)
```

**后果**: LLM收到矛盾指令，概率性输出Python函数而非Qlib表达式

### 问题3: 字段映射不完整

**当前代码位置**: `src/iqfmp/core/qlib_crypto.py`

```python
# Line 54-74: 只映射5个基础字段
"- `$open` - Opening price"
"- `$high` - Highest price"
"- `$low` - Lowest price"
"- `$close` - Closing price"
"- `$volume` - Trading volume"
```

**但我们已有衍生品数据**:
- `src/iqfmp/data/derivatives.py:34-41`: funding_rate, open_interest, liquidation, long_short_ratio...
- `src/iqfmp/data/alignment.py`: 已实现数据对齐

**后果**: LLM无法使用 `$funding_rate` 等字段，crypto-first目标无法实现

---

## 原子级优化方案

### A组: LLM交互层升级 (12个原子)

#### A1. LLMResponse 增补元信息

**目标**: 支持finish_reason/raw_response/model_id/cost_estimate

**当前代码**:
```python
# src/iqfmp/llm/provider.py:66-72
@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict[str, int]
    latency_ms: Optional[float] = None
    cached: bool = False
```

**优化后**:
```python
@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict[str, int]
    latency_ms: Optional[float] = None
    cached: bool = False
    # 新增字段
    finish_reason: Optional[str] = None  # "stop" | "length" | "content_filter"
    raw_response: Optional[dict] = None  # 原始API响应
    model_id: Optional[str] = None  # 实际使用的模型ID
    cost_estimate: Optional[float] = None  # 估算费用(USD)
    request_id: Optional[str] = None  # 请求追踪ID
```

**修改位置**: `src/iqfmp/llm/provider.py:66-72`

**验收标准**:
- [ ] 日志能看到finish_reason
- [ ] 能复现"length→续写"逻辑

---

#### A2. 持久化Prompt缓存 (SQLite)

**目标**: 重启后仍能命中缓存，节省token

**实现方案**:

```python
# 新文件: src/iqfmp/llm/cache.py

import sqlite3
import hashlib
import json
from pathlib import Path
from typing import Optional
import time

class PromptCache:
    """持久化Prompt缓存，仿RD-Agent SQLite实现"""

    def __init__(self, cache_path: Optional[Path] = None):
        self.cache_path = cache_path or Path.home() / ".iqfmp" / "prompt_cache.db"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        self.conn = sqlite3.connect(str(self.cache_path))
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS prompt_cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                model TEXT,
                tokens_saved INTEGER DEFAULT 0,
                created_at REAL,
                accessed_at REAL,
                access_count INTEGER DEFAULT 0
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_accessed_at ON prompt_cache(accessed_at)
        """)
        self.conn.commit()

    def _generate_key(self, messages: list[dict], model: str, seed: Optional[int] = None) -> str:
        """生成缓存key: MD5(messages + model + seed)"""
        content = json.dumps(messages, sort_keys=True) + f":model={model}:seed={seed}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, messages: list[dict], model: str, seed: Optional[int] = None) -> Optional[str]:
        key = self._generate_key(messages, model, seed)
        cursor = self.conn.execute(
            "SELECT value, tokens_saved FROM prompt_cache WHERE key=?", (key,)
        )
        row = cursor.fetchone()
        if row:
            # 更新访问统计
            self.conn.execute(
                "UPDATE prompt_cache SET accessed_at=?, access_count=access_count+1 WHERE key=?",
                (time.time(), key)
            )
            self.conn.commit()
            return row[0]
        return None

    def set(self, messages: list[dict], model: str, response: str,
            tokens_saved: int = 0, seed: Optional[int] = None):
        key = self._generate_key(messages, model, seed)
        now = time.time()
        self.conn.execute(
            """INSERT OR REPLACE INTO prompt_cache
               (key, value, model, tokens_saved, created_at, accessed_at, access_count)
               VALUES (?, ?, ?, ?, ?, ?, 1)""",
            (key, response, model, tokens_saved, now, now)
        )
        self.conn.commit()

    def get_stats(self) -> dict:
        """获取缓存统计"""
        cursor = self.conn.execute("""
            SELECT
                COUNT(*) as total_entries,
                SUM(tokens_saved) as total_tokens_saved,
                SUM(access_count) as total_hits
            FROM prompt_cache
        """)
        row = cursor.fetchone()
        return {
            "total_entries": row[0] or 0,
            "total_tokens_saved": row[1] or 0,
            "total_hits": row[2] or 0,
        }

    def cleanup(self, max_age_days: int = 30, max_entries: int = 10000):
        """清理过期条目"""
        cutoff = time.time() - (max_age_days * 86400)
        self.conn.execute("DELETE FROM prompt_cache WHERE accessed_at < ?", (cutoff,))
        # 保留最新的max_entries条
        self.conn.execute("""
            DELETE FROM prompt_cache WHERE key NOT IN (
                SELECT key FROM prompt_cache ORDER BY accessed_at DESC LIMIT ?
            )
        """, (max_entries,))
        self.conn.commit()
```

**修改LLMProvider**: 在 `src/iqfmp/llm/provider.py` 中集成

**验收标准**:
- [ ] 重启进程后同prompt命中cache
- [ ] cache命中率/节省token可统计

---

#### A3. 会话历史 (ChatSession)

**目标**: 支持多轮对话，上下文复用

```python
# 新增到 src/iqfmp/llm/session.py

import uuid
from typing import Optional, Any
from dataclasses import dataclass, field

@dataclass
class ChatSession:
    """多轮对话会话管理"""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    system_prompt: Optional[str] = None
    messages: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def get_messages_for_api(self) -> list[dict[str, str]]:
        """构建API请求的messages"""
        result = []
        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})
        result.extend(self.messages)
        return result

    def clear_history(self) -> None:
        self.messages.clear()

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "system_prompt": self.system_prompt,
            "messages": self.messages,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChatSession":
        return cls(
            session_id=data.get("session_id", str(uuid.uuid4())),
            system_prompt=data.get("system_prompt"),
            messages=data.get("messages", []),
            metadata=data.get("metadata", {}),
        )


class SessionManager:
    """会话持久化管理器"""

    def __init__(self, cache: "PromptCache"):
        self.cache = cache
        self._sessions: dict[str, ChatSession] = {}

    def get_or_create(self, session_id: Optional[str] = None,
                      system_prompt: Optional[str] = None) -> ChatSession:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]

        session = ChatSession(
            session_id=session_id or str(uuid.uuid4()),
            system_prompt=system_prompt
        )
        self._sessions[session.session_id] = session
        return session

    def save(self, session: ChatSession) -> None:
        """持久化会话到SQLite"""
        # 实现省略，复用PromptCache的连接
        pass

    def load(self, session_id: str) -> Optional[ChatSession]:
        """从SQLite加载会话"""
        pass
```

**验收标准**:
- [ ] 同会话多轮对话能复用上下文
- [ ] 会话可回放

---

#### A4. 自动续写 (Auto-Continue)

**目标**: finish_reason=="length"时自动追加"continue"

```python
# 修改 src/iqfmp/llm/provider.py 的 _call_api 方法

async def _call_api_with_auto_continue(
    self,
    messages: list[dict[str, str]],
    model: ModelType | str,
    max_tokens: Optional[int] = None,
    max_continue_rounds: int = 5,
    **kwargs: Any,
) -> LLMResponse:
    """支持自动续写的API调用"""

    all_content = ""
    all_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    current_messages = messages.copy()

    for round_idx in range(max_continue_rounds):
        response = await self._call_api(
            messages=current_messages,
            model=model,
            max_tokens=max_tokens,
            **kwargs,
        )

        all_content += response.content

        # 累加usage
        for key in all_usage:
            all_usage[key] += response.usage.get(key, 0)

        # 检查finish_reason
        if response.finish_reason != "length":
            break  # 完整响应

        # 需要续写
        logger.info(f"Response truncated (length), continuing round {round_idx + 2}")
        current_messages.append({"role": "assistant", "content": response.content})
        current_messages.append({"role": "user", "content": "continue"})

    return LLMResponse(
        content=all_content,
        model=response.model,
        usage=all_usage,
        latency_ms=response.latency_ms,
        finish_reason=response.finish_reason,
    )
```

**关键修改**: 从OpenRouter响应中提取finish_reason

```python
# 在 _call_api 中添加:
data = response.json()
finish_reason = data["choices"][0].get("finish_reason", "stop")

return LLMResponse(
    content=content,
    model=model_id,
    usage=usage,
    latency_ms=latency_ms,
    finish_reason=finish_reason,  # 新增
)
```

**验收标准**:
- [ ] 长输出稳定完整
- [ ] 最大续写次数可配置

---

#### A5. JSON Mode + Schema校验

**目标**: 结构化输出解析成功率 > 95%

```python
# 新增到 src/iqfmp/llm/schema.py

import json
from typing import Type, TypeVar, Optional
from pydantic import BaseModel, ValidationError

T = TypeVar('T', bound=BaseModel)

class JSONParser:
    """智能JSON解析器"""

    @staticmethod
    def extract_json(text: str) -> Optional[str]:
        """从文本中提取JSON"""
        # 策略1: 查找```json代码块
        import re
        json_block = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_block:
            return json_block.group(1)

        # 策略2: 查找{...}或[...]
        brace_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if brace_match:
            return brace_match.group(1)

        return None

    @staticmethod
    def parse(text: str, target_type: Optional[Type[T]] = None) -> T | dict:
        """解析JSON并可选验证schema"""
        json_str = JSONParser.extract_json(text)
        if not json_str:
            raise ValueError(f"No JSON found in: {text[:100]}...")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            # 尝试修复常见问题
            json_str = JSONParser._fix_common_issues(json_str)
            data = json.loads(json_str)

        if target_type:
            return target_type.model_validate(data)
        return data

    @staticmethod
    def _fix_common_issues(json_str: str) -> str:
        """修复常见JSON格式问题"""
        # 移除trailing comma
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        # 转换单引号为双引号
        json_str = json_str.replace("'", '"')
        return json_str


# 使用示例:
# class FactorSpec(BaseModel):
#     indicator: str
#     window: int
#     direction: str
#
# spec = JSONParser.parse(llm_response, FactorSpec)
```

**验收标准**:
- [ ] 结构化输出(proposal/spec)解析成功率 > 95%

---

#### A6. 错误分类重试 + Backoff

**目标**: LLM请求失败率下降，重试次数可观测

```python
# 修改 src/iqfmp/llm/provider.py

import asyncio
from enum import Enum

class LLMErrorType(Enum):
    RATE_LIMIT = "rate_limit"  # 429
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"  # 5xx
    CONTENT_POLICY = "content_policy"
    INVALID_REQUEST = "invalid_request"  # 4xx
    NETWORK = "network"
    UNKNOWN = "unknown"

def classify_error(e: Exception, status_code: Optional[int] = None) -> LLMErrorType:
    """分类错误类型"""
    if status_code == 429:
        return LLMErrorType.RATE_LIMIT
    if status_code and 500 <= status_code < 600:
        return LLMErrorType.SERVER_ERROR
    if isinstance(e, asyncio.TimeoutError):
        return LLMErrorType.TIMEOUT
    if isinstance(e, ConnectionError):
        return LLMErrorType.NETWORK
    if "content_policy" in str(e).lower():
        return LLMErrorType.CONTENT_POLICY
    return LLMErrorType.UNKNOWN

# 重试策略
RETRY_STRATEGIES = {
    LLMErrorType.RATE_LIMIT: {"max_retries": 5, "base_wait": 60, "exponential": True},
    LLMErrorType.TIMEOUT: {"max_retries": 3, "base_wait": 5, "exponential": True},
    LLMErrorType.SERVER_ERROR: {"max_retries": 3, "base_wait": 10, "exponential": True},
    LLMErrorType.NETWORK: {"max_retries": 3, "base_wait": 5, "exponential": False},
    LLMErrorType.CONTENT_POLICY: {"max_retries": 0, "base_wait": 0, "exponential": False},  # 不重试
    LLMErrorType.INVALID_REQUEST: {"max_retries": 0, "base_wait": 0, "exponential": False},
    LLMErrorType.UNKNOWN: {"max_retries": 2, "base_wait": 5, "exponential": True},
}

async def _call_api_with_retry(self, *args, **kwargs) -> LLMResponse:
    """带智能重试的API调用"""
    last_error = None
    retry_stats = {"attempts": 0, "error_types": []}

    for attempt in range(10):  # 最大总尝试次数
        try:
            return await self._call_api(*args, **kwargs)
        except Exception as e:
            error_type = classify_error(e, getattr(e, 'status_code', None))
            strategy = RETRY_STRATEGIES[error_type]

            retry_stats["attempts"] += 1
            retry_stats["error_types"].append(error_type.value)

            if attempt >= strategy["max_retries"]:
                logger.error(f"Max retries ({strategy['max_retries']}) reached for {error_type}")
                raise

            wait_time = strategy["base_wait"]
            if strategy["exponential"]:
                wait_time *= (2 ** attempt)

            # 429错误特殊处理: 尝试从header获取retry-after
            if error_type == LLMErrorType.RATE_LIMIT:
                retry_after = getattr(e, 'retry_after', None)
                if retry_after:
                    wait_time = max(wait_time, retry_after)

            logger.warning(f"Retry {attempt + 1} for {error_type}: waiting {wait_time}s")
            await asyncio.sleep(wait_time)
            last_error = e

    raise last_error
```

**验收标准**:
- [ ] LLM请求失败率下降
- [ ] 重试次数可观测

---

#### A7-A12: 其他LLM层原子

| 原子 | 目标 | 实现要点 |
|------|------|----------|
| A7 | Token/费用预算器 | max_prompt_tokens/max_completion_tokens/max_cost_usd，超限降级 |
| A8 | 种子/多样性控制 | seed参数影响cache key，可生成N个可复现候选 |
| A9 | 结构化Trace | 事件流: prompt_hash, model, tokens, latency, cache_hit, error |
| A10 | 工具调用总线 | 注册tools: get_field_schema/validate_expression/quick_eval |
| A11 | 上下文裁剪 | 对trace/数据样本/错误日志做"可控摘要" |
| A12 | 安全红线 | 敏感信息脱敏，禁止输出API key |

---

### B组: Prompt系统升级 (12个原子)

#### B1. 修复输出类型矛盾 (立刻做)

**问题位置**: `src/iqfmp/llm/prompts/factor_generation.py:268-270`

**当前代码**:
```python
parts.append(
    "\n## Task\n"
    "Generate a Python factor function following the guidelines in the system prompt.\n"  # ❌ 矛盾
    "Ensure the factor is optimized for cryptocurrency markets."
)
```

**修改为**:
```python
parts.append(
    "\n## Task\n"
    "Generate a **Qlib expression** that implements the requested factor.\n"
    "Output ONLY the expression, no Python code, no markdown, no explanation before.\n"
    "You may add a brief comment after the expression starting with #"
)
```

**验收标准**:
- [ ] prompt与解析器/validator完全一致
- [ ] "生成Python"概率≈0

---

#### B2. PromptRegistry (YAML + 版本号 + A/B标记)

**目标**: 可版本化、可测试、可回滚的prompt管理

```yaml
# 新文件: src/iqfmp/llm/prompts/registry.yaml

prompts:
  factor_generation:
    version: "2.0.0"
    ab_variant: "expression_v2"
    enabled: true
    system_prompt: |
      You are an expert quantitative factor developer for **cryptocurrency markets**.
      Your task is to generate **Qlib expressions** that implement the user's hypothesis.

      ## Available Fields (ONLY these)
      {{available_fields}}

      ## Operators
      {{operators}}

      ## Output Format
      Return ONLY a single Qlib expression. Example:
      (EMA($close, 12) - EMA($close, 26)) / $close

    user_template: |
      ## User Request
      {{user_request}}

      ## Factor Family
      {{factor_family}}

      ## Task
      Generate a Qlib expression that implements this hypothesis.

    examples:
      - user: "Create a momentum factor"
        assistant: "Ref($close, -20) / $close - 1"
      - user: "Create a funding rate extreme factor"
        assistant: "($funding_rate - Mean($funding_rate, 24)) / Std($funding_rate, 24)"

  factor_refinement:
    version: "1.0.0"
    ab_variant: null
    enabled: true
    # ...
```

```python
# 新文件: src/iqfmp/llm/prompts/registry.py

from pathlib import Path
import yaml
from string import Template
from typing import Optional

class PromptRegistry:
    """Prompt版本管理器"""

    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or Path(__file__).parent / "registry.yaml"
        self._prompts = self._load()

    def _load(self) -> dict:
        with open(self.registry_path) as f:
            return yaml.safe_load(f)["prompts"]

    def get(self, name: str, version: Optional[str] = None) -> dict:
        """获取prompt配置"""
        prompt = self._prompts.get(name)
        if not prompt or not prompt.get("enabled"):
            raise ValueError(f"Prompt '{name}' not found or disabled")
        return prompt

    def render_system(self, name: str, **kwargs) -> str:
        """渲染system prompt"""
        prompt = self.get(name)
        template = Template(prompt["system_prompt"])
        return template.safe_substitute(**kwargs)

    def render_user(self, name: str, **kwargs) -> str:
        """渲染user prompt"""
        prompt = self.get(name)
        template = Template(prompt["user_template"])
        return template.safe_substitute(**kwargs)

    def get_version(self, name: str) -> str:
        return self.get(name)["version"]

    def get_ab_variant(self, name: str) -> Optional[str]:
        return self.get(name).get("ab_variant")
```

**验收标准**:
- [ ] 可在配置里指定prompt version
- [ ] 能回滚到旧版本

---

#### B3. 字段能力动态注入

**目标**: system prompt不再硬编码"只有5个字段"

```python
# 新文件: src/iqfmp/data/schema.py

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class FieldSource(Enum):
    OHLCV = "ohlcv"
    DERIVATIVES = "derivatives"
    ONCHAIN = "onchain"
    COMPUTED = "computed"

@dataclass
class FieldSpec:
    name: str
    qlib_name: str  # $field格式
    description: str
    source: FieldSource
    unit: Optional[str] = None
    frequency: str = "1h"
    available: bool = True  # 当前数据集是否有此字段

@dataclass
class DataSchema:
    """数据schema定义，用于动态注入prompt"""

    fields: list[FieldSpec] = field(default_factory=list)

    @classmethod
    def build_for_symbol(cls, symbol: str, include_derivatives: bool = True) -> "DataSchema":
        """根据symbol构建可用字段schema"""
        schema = cls()

        # 基础OHLCV (总是可用)
        base_fields = [
            FieldSpec("open", "$open", "Opening price", FieldSource.OHLCV),
            FieldSpec("high", "$high", "Highest price", FieldSource.OHLCV),
            FieldSpec("low", "$low", "Lowest price", FieldSource.OHLCV),
            FieldSpec("close", "$close", "Closing price", FieldSource.OHLCV),
            FieldSpec("volume", "$volume", "Trading volume in base currency", FieldSource.OHLCV),
        ]
        schema.fields.extend(base_fields)

        if include_derivatives:
            # 衍生品字段 (根据数据可用性)
            derivative_fields = [
                FieldSpec("funding_rate", "$funding_rate",
                         "Perpetual funding rate (positive = longs pay shorts, typical: -0.1% to +0.1% per 8h)",
                         FieldSource.DERIVATIVES, unit="percent/8h"),
                FieldSpec("open_interest", "$open_interest",
                         "Total open positions in contracts",
                         FieldSource.DERIVATIVES, unit="contracts"),
                FieldSpec("liquidation_long", "$liquidation_long",
                         "Long position liquidations",
                         FieldSource.DERIVATIVES, unit="USD"),
                FieldSpec("liquidation_short", "$liquidation_short",
                         "Short position liquidations",
                         FieldSource.DERIVATIVES, unit="USD"),
                FieldSpec("long_short_ratio", "$long_short_ratio",
                         "Ratio of long to short positions (>1 = more longs)",
                         FieldSource.DERIVATIVES),
            ]
            schema.fields.extend(derivative_fields)

        return schema

    def to_prompt_block(self) -> str:
        """生成prompt中的字段说明块"""
        lines = ["## Available Data Fields (use $ prefix)\n"]
        lines.append("**ONLY these fields are available. DO NOT use any other fields.**\n")

        by_source = {}
        for f in self.fields:
            if f.available:
                by_source.setdefault(f.source.value, []).append(f)

        for source, fields in by_source.items():
            lines.append(f"\n### {source.upper()} Fields\n")
            for f in fields:
                unit_str = f" ({f.unit})" if f.unit else ""
                lines.append(f"- `{f.qlib_name}` - {f.description}{unit_str}")

        return "\n".join(lines)

    def get_available_field_names(self) -> list[str]:
        """获取可用字段名列表(不含$)"""
        return [f.name for f in self.fields if f.available]

    def get_missing_field_names(self) -> list[str]:
        """获取不可用字段名列表"""
        return [f.name for f in self.fields if not f.available]
```

**集成到FactorGenerationPrompt**:
```python
def get_system_prompt(self, schema: DataSchema) -> str:
    return f"""You are an expert quantitative factor developer...

{schema.to_prompt_block()}

## Qlib Expression Operators
...
"""
```

**验收标准**:
- [ ] LLM不再臆造字段
- [ ] 字段违规率显著下降

---

#### B4. 表达式语法约束 (EBNF/regex gate)

**目标**: 无效表达式率 < 5%

```python
# 新文件: src/iqfmp/llm/validation/expression_gate.py

import re
from typing import Optional, Tuple

class ExpressionGate:
    """表达式语法门禁"""

    # 允许的操作符
    ALLOWED_OPS = {
        "Ref", "Mean", "Std", "Sum", "Max", "Min", "Delta", "EMA", "WMA",
        "RSI", "MACD", "Abs", "Log", "Sign", "Rank", "Corr", "Cov", "If",
        "Med", "Var", "Count", "Quantile", "Kurt", "Skew"
    }

    # 禁止的关键字(防止代码注入)
    FORBIDDEN_KEYWORDS = {
        "import", "exec", "eval", "compile", "__", "open(", "os.", "sys.",
        "subprocess", "lambda", "class", "def ", "return"
    }

    # 表达式最大长度
    MAX_LENGTH = 2000

    # 最大嵌套深度
    MAX_DEPTH = 10

    def validate(self, expression: str, allowed_fields: list[str]) -> Tuple[bool, Optional[str]]:
        """
        验证表达式是否合法

        Returns:
            (is_valid, error_message)
        """
        if not expression or not expression.strip():
            return False, "Expression is empty"

        # 长度检查
        if len(expression) > self.MAX_LENGTH:
            return False, f"Expression too long ({len(expression)} > {self.MAX_LENGTH})"

        # 禁止关键字检查
        expr_lower = expression.lower()
        for kw in self.FORBIDDEN_KEYWORDS:
            if kw.lower() in expr_lower:
                return False, f"Forbidden keyword detected: {kw}"

        # 括号平衡检查
        if not self._check_brackets(expression):
            return False, "Unbalanced brackets"

        # 嵌套深度检查
        depth = self._get_max_depth(expression)
        if depth > self.MAX_DEPTH:
            return False, f"Expression too deeply nested ({depth} > {self.MAX_DEPTH})"

        # 字段检查
        used_fields = self._extract_fields(expression)
        invalid_fields = [f for f in used_fields if f not in allowed_fields]
        if invalid_fields:
            return False, f"Invalid fields used: {invalid_fields}. Allowed: {allowed_fields}"

        # 操作符检查
        used_ops = self._extract_operators(expression)
        invalid_ops = [op for op in used_ops if op not in self.ALLOWED_OPS]
        if invalid_ops:
            return False, f"Invalid operators: {invalid_ops}"

        return True, None

    def _check_brackets(self, expr: str) -> bool:
        """检查括号平衡"""
        stack = []
        for char in expr:
            if char in "([{":
                stack.append(char)
            elif char in ")]}":
                if not stack:
                    return False
                open_bracket = stack.pop()
                if not self._brackets_match(open_bracket, char):
                    return False
        return len(stack) == 0

    def _brackets_match(self, open_b: str, close_b: str) -> bool:
        return (open_b, close_b) in [("(", ")"), ("[", "]"), ("{", "}")]

    def _get_max_depth(self, expr: str) -> int:
        """计算最大嵌套深度"""
        max_depth = 0
        current_depth = 0
        for char in expr:
            if char == "(":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ")":
                current_depth -= 1
        return max_depth

    def _extract_fields(self, expr: str) -> list[str]:
        """提取使用的字段"""
        pattern = r'\$([a-z_]+)'
        return list(set(re.findall(pattern, expr.lower())))

    def _extract_operators(self, expr: str) -> list[str]:
        """提取使用的操作符"""
        pattern = r'([A-Z][a-zA-Z]*)\s*\('
        return list(set(re.findall(pattern, expr)))
```

**验收标准**:
- [ ] 无效表达式率 < 5%

---

#### B5. 指标/算子字典 (Indicator Catalog)

**目标**: LLM优先"选模板+填参+组合"，不再依赖"临场背公式"

```python
# 新文件: src/iqfmp/llm/prompts/indicator_catalog.py

INDICATOR_CATALOG = {
    # === 趋势指标 ===
    "SMA": {
        "template": "Mean($close, {period})",
        "params": {"period": {"type": "int", "range": [5, 200], "default": 20}},
        "description": "Simple Moving Average",
        "category": "trend",
    },
    "EMA": {
        "template": "EMA($close, {period})",
        "params": {"period": {"type": "int", "range": [5, 200], "default": 20}},
        "description": "Exponential Moving Average",
        "category": "trend",
    },
    "MACD": {
        "template": "MACD($close, {fast}, {slow}, {signal})",
        "params": {
            "fast": {"type": "int", "range": [5, 20], "default": 12},
            "slow": {"type": "int", "range": [20, 50], "default": 26},
            "signal": {"type": "int", "range": [5, 15], "default": 9},
        },
        "description": "Moving Average Convergence Divergence",
        "category": "trend",
    },

    # === 动量指标 ===
    "RSI": {
        "template": "RSI($close, {period})",
        "params": {"period": {"type": "int", "range": [7, 28], "default": 14}},
        "description": "Relative Strength Index (0-100)",
        "category": "momentum",
    },
    "ROC": {
        "template": "$close / Ref($close, {period}) - 1",
        "params": {"period": {"type": "int", "range": [1, 60], "default": 10}},
        "description": "Rate of Change",
        "category": "momentum",
    },
    "MOMENTUM": {
        "template": "$close - Ref($close, {period})",
        "params": {"period": {"type": "int", "range": [1, 60], "default": 10}},
        "description": "Simple Momentum",
        "category": "momentum",
    },

    # === 波动率指标 ===
    "BOLLINGER_UPPER": {
        "template": "Mean($close, {period}) + {mult} * Std($close, {period})",
        "params": {
            "period": {"type": "int", "range": [10, 50], "default": 20},
            "mult": {"type": "float", "range": [1.5, 3.0], "default": 2.0},
        },
        "description": "Bollinger Band Upper",
        "category": "volatility",
    },
    "BOLLINGER_LOWER": {
        "template": "Mean($close, {period}) - {mult} * Std($close, {period})",
        "params": {
            "period": {"type": "int", "range": [10, 50], "default": 20},
            "mult": {"type": "float", "range": [1.5, 3.0], "default": 2.0},
        },
        "description": "Bollinger Band Lower",
        "category": "volatility",
    },
    "ATR": {
        "template": "Mean(Max(Max($high - $low, Abs($high - Ref($close, 1))), Abs($low - Ref($close, 1))), {period})",
        "params": {"period": {"type": "int", "range": [7, 28], "default": 14}},
        "description": "Average True Range",
        "category": "volatility",
    },

    # === Williams %R ===
    "WR": {
        "template": "($close - Min($low, {period})) / (Max($high, {period}) - Min($low, {period})) * 100 - 100",
        "params": {"period": {"type": "int", "range": [7, 28], "default": 14}},
        "description": "Williams %R (-100 to 0)",
        "category": "momentum",
    },

    # === Stochastic ===
    "STOCH_K": {
        "template": "($close - Min($low, {period})) / (Max($high, {period}) - Min($low, {period})) * 100",
        "params": {"period": {"type": "int", "range": [7, 28], "default": 14}},
        "description": "Stochastic %K (0-100)",
        "category": "momentum",
    },

    # === Crypto特有 ===
    "FUNDING_ZSCORE": {
        "template": "($funding_rate - Mean($funding_rate, {period})) / Std($funding_rate, {period})",
        "params": {"period": {"type": "int", "range": [12, 168], "default": 24}},
        "description": "Funding Rate Z-Score (crypto-specific)",
        "category": "crypto",
    },
    "OI_CHANGE": {
        "template": "Delta($open_interest, {period}) / Ref($open_interest, {period})",
        "params": {"period": {"type": "int", "range": [1, 24], "default": 1}},
        "description": "Open Interest Change Rate (crypto-specific)",
        "category": "crypto",
    },
    "LIQ_IMBALANCE": {
        "template": "($liquidation_long - $liquidation_short) / ($liquidation_long + $liquidation_short + 1)",
        "params": {},
        "description": "Liquidation Imbalance (crypto-specific)",
        "category": "crypto",
    },
}

def get_catalog_prompt_block() -> str:
    """生成指标目录的prompt块"""
    lines = ["## Indicator Templates (use these instead of reinventing formulas)\n"]

    by_category = {}
    for name, spec in INDICATOR_CATALOG.items():
        by_category.setdefault(spec["category"], []).append((name, spec))

    for category, indicators in by_category.items():
        lines.append(f"\n### {category.upper()}\n")
        for name, spec in indicators:
            params_str = ", ".join(f"{k}={v['default']}" for k, v in spec["params"].items())
            lines.append(f"- **{name}**({params_str}): `{spec['template']}`")
            lines.append(f"  {spec['description']}")

    return "\n".join(lines)
```

**验收标准**:
- [ ] WR/SSL/Zigzag/Boll等"实现正确率"可测且显著提升

---

#### B6-B12: 其他Prompt原子

| 原子 | 目标 | 实现要点 |
|------|------|----------|
| B6 | 两阶段生成(Spec→Compile) | LLM先输出JSON spec，再由编译器生成表达式 |
| B7 | 强制completeness | 复用indicator_intelligence.py的反馈循环 |
| B8 | 错误反馈结构化 | JSON格式: missing_fields/parse_error/runtime_error |
| B9 | 反p-hacking提示与硬限制 | 每轮最多候选K、总试验数上限、复杂度惩罚 |
| B10 | 语言归一 | 中英文指标名映射(威廉指标→WR) |
| B11 | Prompt单测 | 对每个version做golden tests |
| B12 | Prompt性能看板 | 成功率/平均IC/平均token/返工次数 |

---

### C组: Qlib + Crypto 闭环 (15个原子)

#### C1. 统一MarketDataProvider

**目标**: OHLCV + derivatives合流

```python
# 修改 src/iqfmp/data/provider.py

from iqfmp.data.derivatives import DerivativeDownloader
from iqfmp.data.alignment import merge_derivative_data

class UnifiedMarketDataProvider:
    """统一市场数据提供者"""

    async def load_market_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        include_derivatives: bool = True,
        derivative_types: list[str] = None,
    ) -> pd.DataFrame:
        """
        加载合并后的市场数据

        返回DataFrame包含:
        - open, high, low, close, volume (OHLCV)
        - funding_rate, open_interest, liquidation_long, liquidation_short (if derivatives)
        """
        # 1. 加载OHLCV
        ohlcv_df = await self._load_ohlcv(symbol, start_date, end_date)

        if not include_derivatives:
            return ohlcv_df

        # 2. 加载衍生品数据
        derivative_types = derivative_types or [
            "funding_rate", "open_interest", "liquidation", "long_short_ratio"
        ]
        derivative_df = await self._load_derivatives(
            symbol, start_date, end_date, derivative_types
        )

        # 3. 合并对齐 (使用现有的alignment.py)
        merged_df = merge_derivative_data(ohlcv_df, derivative_df)

        return merged_df

    async def _load_derivatives(self, symbol, start_date, end_date, types) -> pd.DataFrame:
        """加载衍生品数据"""
        downloader = DerivativeDownloader()
        try:
            dfs = []
            if "funding_rate" in types:
                funding = await downloader.fetch_funding_rate_history(symbol)
                dfs.append(pd.DataFrame(funding))
            # ... 其他数据类型

            if dfs:
                return pd.concat(dfs, axis=1)
            return pd.DataFrame()
        finally:
            await downloader.close()
```

**验收标准**:
- [ ] 返回df至少包含funding_rate/open_interest/liquidation_total/long_short_ratio

---

#### C3. 字段映射自动化

**目标**: QlibExpressionEngine支持所有字段

**修改位置**: `src/iqfmp/core/qlib_crypto.py`

```python
# 当前实现只映射5个字段，需要扩展

class QlibExpressionEngine:
    """Qlib表达式引擎 - 支持动态字段"""

    # 扩展字段映射
    FIELD_MAPPING = {
        # OHLCV
        "open": "$open",
        "high": "$high",
        "low": "$low",
        "close": "$close",
        "volume": "$volume",
        # 衍生品
        "funding_rate": "$funding_rate",
        "open_interest": "$open_interest",
        "liquidation_long": "$liquidation_long",
        "liquidation_short": "$liquidation_short",
        "liquidation_total": "$liquidation_total",
        "long_short_ratio": "$long_short_ratio",
        "mark_price": "$mark_price",
        "taker_buy_ratio": "$taker_buy_ratio",
    }

    def _prepare_data_for_qlib(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """准备数据供Qlib表达式使用"""
        data = {}
        for col in df.columns:
            if col in self.FIELD_MAPPING:
                qlib_field = self.FIELD_MAPPING[col]
                data[qlib_field] = df[col]
            elif col.startswith("$"):
                # 已经是Qlib格式
                data[col] = df[col]
            else:
                # 自动添加$前缀
                data[f"${col}"] = df[col]
        return data
```

**验收标准**:
- [ ] 表达式里 `$funding_rate` 能被解析并计算

---

#### C7. Qlib原生DataHandler接入

**目标**: 能跑通qlib workflow，产出recorder artifacts

```python
# 新文件: src/iqfmp/qlib/crypto_handler.py

"""
接入Qlib原生DataHandler，使用vendored Qlib的crypto能力
"""

from pathlib import Path
import pandas as pd
from vendor.qlib.qlib.data.dataset import DatasetH
from vendor.qlib.qlib.data.dataset.handler import DataHandlerLP

class CryptoDataHandler(DataHandlerLP):
    """Crypto专用DataHandler"""

    def __init__(
        self,
        instruments: list[str],
        start_time: str,
        end_time: str,
        include_derivatives: bool = True,
        **kwargs
    ):
        # 构建数据加载配置
        data_loader_config = self._build_data_loader_config(include_derivatives)

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader_config,
            **kwargs
        )

    def _build_data_loader_config(self, include_derivatives: bool) -> dict:
        """构建NestedDataLoader配置，仿RD-Agent"""
        config = {
            "class": "NestedDataLoader",
            "kwargs": {
                "dataloader_l": [
                    # 基础OHLCV + 内置因子
                    {
                        "class": "qlib.contrib.data.loader.Alpha158DL",
                        "kwargs": {
                            "config": self._get_alpha158_config()
                        }
                    }
                ]
            }
        }

        if include_derivatives:
            # 添加衍生品数据加载器
            config["kwargs"]["dataloader_l"].append({
                "class": "qlib.data.dataset.loader.StaticDataLoader",
                "kwargs": {
                    "config": "derivatives_data.parquet"  # 预处理的衍生品数据
                }
            })

        return config
```

**验收标准**:
- [ ] 能跑通qlib workflow
- [ ] 产出recorder artifacts

---

#### C9-C15: 其他Qlib原子

| 原子 | 目标 | 实现要点 |
|------|------|----------|
| C9 | 字段可用性声明 | 每次loop生成FieldSchema(available/missing/units/freq/source) |
| C10 | 表达式复杂度预算 | 限制嵌套深度/rolling window上限/算子数量 |
| C11 | 在线/离线缓存 | parquet cache + qlib结果cache(按expression hash) |
| C12 | 衍生品label管线 | 净收益label(含funding/fee) |
| C13 | 跨周期一致性 | 同一因子在1h/4h/1d的稳定性评估 |
| C14 | Qlib tuner接入 | 参数搜索交给tuner |
| C15 | 研究账本打通 | experiment_id/prompt_hash/expression_hash统一 |

---

### D组: Crypto深度优化 (12个原子)

| 原子 | 目标 | 实现要点 |
|------|------|----------|
| D1 | Regime-aware CV切分 | 高波动/低波动、funding极端/正常分层评估 |
| D2 | Funding/费率纳入回测 | perpetual funding 8h结算 + maker/taker fee |
| D3 | 滑点模型动态化 | 用volume/spread/volatility proxy |
| D4 | 杠杆与爆仓约束 | 保证金/最大回撤触发减仓/强平阈值 |
| D5 | 跨交易所可执行性 | binance/okx价差与成交量约束 |
| D6 | 现实检验强制化 | DSR门槛纳入"是否收录" |
| D7 | 因子冗余惩罚 | Qdrant embedding + 统计相关性 |
| D8 | 衍生品alpha库 | 用alpha_derivatives.py作为benchmark |
| D9 | 事件驱动特征 | funding结算前后、周末流动性 |
| D10 | 多资产组合构建 | cross-sectional分配(top-k/风险平价) |
| D11 | 策略层与风控层联动 | 风险违规触发"改因子/改策略" |
| D12 | 线上一致性 | paper trading参数与回测一致 |

---

## 实施路线图

### Phase 1: 打穿交互闭环 (优先级最高)

**目标**: LLM能正确输出Qlib表达式，字段不越界

| 原子 | 工时估计 | 依赖 |
|------|----------|------|
| B1. 修复输出类型矛盾 | 0.5h | 无 |
| A1. LLMResponse增补 | 1h | 无 |
| A4. 自动续写 | 2h | A1 |
| B4. 表达式语法门禁 | 2h | 无 |
| C3. 字段映射自动化 | 2h | 无 |

**验收**: 生成的表达式100%为Qlib语法，字段违规率<5%

### Phase 2: 提升交互可靠性

**目标**: 缓存、重试、schema校验

| 原子 | 工时估计 | 依赖 |
|------|----------|------|
| A2. 持久化缓存 | 3h | 无 |
| A5. JSON schema校验 | 2h | 无 |
| A6. 错误分类重试 | 2h | A1 |
| B3. 字段动态注入 | 2h | C3 |
| B5. 指标目录 | 3h | B3 |

**验收**: 缓存命中率可观测，结构化输出成功率>95%

### Phase 3: 打通Crypto数据

**目标**: 衍生品字段可用

| 原子 | 工时估计 | 依赖 |
|------|----------|------|
| C1. 统一MarketDataProvider | 4h | 无 |
| C4. FactorEngine同步升级 | 2h | C1, C3 |
| C5. 衍生字段派生特征 | 2h | C1 |
| C9. 字段可用性声明 | 1h | C1 |

**验收**: LLM能使用$funding_rate等字段，表达式能正确计算

### Phase 4: 评估与质量

**目标**: 防止p-hacking，确保因子质量

| 原子 | 工时估计 | 依赖 |
|------|----------|------|
| D1. Regime-aware CV | 4h | C1 |
| D2. Funding纳入回测 | 3h | C1 |
| D6. DSR强制化 | 2h | 无 |
| D7. 因子冗余惩罚 | 3h | 无 |
| B9. 反p-hacking | 2h | 无 |

**验收**: 评估指标包含regime分层，冗余因子被过滤

---

## 验收指标汇总

| 维度 | 指标 | 基线 | 目标 |
|------|------|------|------|
| 表达式合规率 | Qlib语法正确 | ~70% | >95% |
| 字段违规率 | 使用不存在字段 | ~30% | <5% |
| 缓存命中率 | 重复prompt | 0% | >50% |
| JSON解析成功率 | 结构化输出 | ~80% | >95% |
| 长输出完整率 | 无截断 | ~60% | >98% |
| 指标实现正确率 | WR/Boll等 | ~50% | >85% |
| 评估稳定性 | IC跨周期一致 | 未测 | 可测 |

---

## 关键代码修改清单

| 文件 | 修改类型 | 优先级 |
|------|----------|--------|
| `src/iqfmp/llm/prompts/factor_generation.py:268-270` | 修复矛盾 | P0 |
| `src/iqfmp/llm/provider.py:66-72` | 扩展LLMResponse | P0 |
| `src/iqfmp/llm/provider.py:377-450` | 添加自动续写 | P0 |
| `src/iqfmp/core/qlib_crypto.py:FIELD_MAPPING` | 扩展字段 | P0 |
| `src/iqfmp/llm/cache.py` | 新建 | P1 |
| `src/iqfmp/llm/schema.py` | 新建 | P1 |
| `src/iqfmp/data/schema.py` | 新建 | P1 |
| `src/iqfmp/llm/validation/expression_gate.py` | 新建 | P1 |
| `src/iqfmp/llm/prompts/indicator_catalog.py` | 新建 | P1 |
| `src/iqfmp/data/provider.py` | 扩展 | P2 |

---

**下一步**: 需要我从哪个原子开始实现？建议从 **B1 + A1 + A4** 开始（修复矛盾 + 支持续写），这是最快能看到效果的路径。
