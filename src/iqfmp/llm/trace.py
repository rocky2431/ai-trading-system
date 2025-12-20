"""LLM call tracing (RD-Agent parity).

Stores per-call records for replay/debugging with a Redis (L1) + Postgres (L2) backend.
All writes are best-effort: tracing must never break core execution.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select

from iqfmp.db.database import get_async_session, get_redis_client
from iqfmp.db.models import LLMTraceORM


@dataclass(frozen=True)
class LLMTraceRecord:
    """Single LLM call trace record."""

    execution_id: str
    conversation_id: str
    created_at: str

    agent: Optional[str]
    model: str

    prompt_id: Optional[str]
    prompt_version: Optional[str]

    messages: list[dict[str, str]]
    response: str
    usage: dict[str, int]
    cached: bool
    cost_estimate: Optional[float]
    request_id: Optional[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str) -> "LLMTraceRecord":
        data = json.loads(payload)
        return cls(**data)

    @classmethod
    def now(
        cls,
        *,
        execution_id: str,
        conversation_id: str,
        agent: Optional[str],
        model: str,
        prompt_id: Optional[str],
        prompt_version: Optional[str],
        messages: list[dict[str, str]],
        response: str,
        usage: dict[str, int],
        cached: bool,
        cost_estimate: Optional[float],
        request_id: Optional[str],
    ) -> "LLMTraceRecord":
        return cls(
            execution_id=execution_id,
            conversation_id=conversation_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            agent=agent,
            model=model,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            messages=messages,
            response=response,
            usage=usage,
            cached=cached,
            cost_estimate=cost_estimate,
            request_id=request_id,
        )


class LLMTraceStore:
    """Two-tier LLM trace store (Redis list + Postgres table)."""

    REDIS_PREFIX = "llm:trace:"
    DEFAULT_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days

    async def record(self, record: LLMTraceRecord) -> None:
        """Persist a trace record (best-effort)."""
        await self._record_redis(record)
        await self._record_postgres(record)

    async def list_records(self, execution_id: str) -> list[LLMTraceRecord]:
        """Fetch trace records by execution_id (Redis first, then Postgres)."""
        records = await self._list_redis(execution_id)
        if records:
            return records
        return await self._list_postgres(execution_id)

    async def _record_redis(self, record: LLMTraceRecord) -> None:
        try:
            redis = get_redis_client()
            key = f"{self.REDIS_PREFIX}{record.execution_id}"
            await redis.rpush(key, record.to_json())
            await redis.expire(key, self.DEFAULT_TTL_SECONDS)
        except Exception:
            return

    async def _list_redis(self, execution_id: str) -> list[LLMTraceRecord]:
        try:
            redis = get_redis_client()
            key = f"{self.REDIS_PREFIX}{execution_id}"
            raw_items = await redis.lrange(key, 0, -1)
            if not raw_items:
                return []
            records: list[LLMTraceRecord] = []
            for item in raw_items:
                if isinstance(item, bytes):
                    item = item.decode("utf-8")
                if isinstance(item, str):
                    records.append(LLMTraceRecord.from_json(item))
            return records
        except Exception:
            return []

    async def _record_postgres(self, record: LLMTraceRecord) -> None:
        try:
            async with get_async_session() as session:
                session.add(
                    LLMTraceORM(
                        execution_id=record.execution_id,
                        conversation_id=record.conversation_id,
                        agent=record.agent,
                        model=record.model,
                        prompt_id=record.prompt_id,
                        prompt_version=record.prompt_version,
                        messages={"messages": record.messages},
                        response=record.response,
                        usage=record.usage,
                        cost_estimate=record.cost_estimate,
                        cached=record.cached,
                        request_id=record.request_id,
                    )
                )
                await session.commit()
        except Exception:
            return

    async def _list_postgres(self, execution_id: str) -> list[LLMTraceRecord]:
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    select(LLMTraceORM)
                    .where(LLMTraceORM.execution_id == execution_id)
                    .order_by(LLMTraceORM.id)
                )
                rows = result.scalars().all()
        except Exception:
            return []

        records: list[LLMTraceRecord] = []
        for row in rows:
            # messages stored as {"messages": [...]}
            messages_obj = row.messages.get("messages", []) if isinstance(row.messages, dict) else []
            messages: list[dict[str, str]] = []
            if isinstance(messages_obj, list):
                for m in messages_obj:
                    if isinstance(m, dict):
                        role = str(m.get("role", "user"))
                        content = str(m.get("content", ""))
                        messages.append({"role": role, "content": content})

            records.append(
                LLMTraceRecord(
                    execution_id=row.execution_id,
                    conversation_id=row.conversation_id or row.execution_id,
                    created_at=row.created_at.isoformat() if row.created_at else "",
                    agent=row.agent,
                    model=row.model,
                    prompt_id=row.prompt_id,
                    prompt_version=row.prompt_version,
                    messages=messages,
                    response=row.response,
                    usage=row.usage or {},
                    cached=bool(row.cached),
                    cost_estimate=row.cost_estimate,
                    request_id=row.request_id,
                )
            )

        return records


_TRACE_STORE: Optional[LLMTraceStore] = None


def get_llm_trace_store() -> LLMTraceStore:
    global _TRACE_STORE
    if _TRACE_STORE is None:
        _TRACE_STORE = LLMTraceStore()
    return _TRACE_STORE


class TracingLLMProvider:
    """Wrapper that injects execution/conversation metadata into LLM calls."""

    def __init__(
        self,
        inner: Any,
        *,
        execution_id: str,
        conversation_id: Optional[str] = None,
        agent: Optional[str] = None,
        prompt_id: Optional[str] = None,
        prompt_version: Optional[str] = None,
    ) -> None:
        self._inner = inner
        self._execution_id = execution_id
        self._conversation_id = conversation_id or execution_id
        self._agent = agent
        self._prompt_id = prompt_id
        self._prompt_version = prompt_version

    async def complete(self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any) -> Any:
        return await self._inner.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            execution_id=self._execution_id,
            conversation_id=self._conversation_id,
            agent=self._agent,
            prompt_id=self._prompt_id,
            prompt_version=self._prompt_version,
            **kwargs,
        )

    async def complete_structured(self, prompt: str, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(self._inner, "complete_structured"):
            raise AttributeError("Inner provider does not support complete_structured")
        return await self._inner.complete_structured(
            prompt=prompt,
            execution_id=self._execution_id,
            conversation_id=self._conversation_id,
            agent=self._agent,
            prompt_id=self._prompt_id,
            prompt_version=self._prompt_version,
            *args,
            **kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

