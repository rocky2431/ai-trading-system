# Task 26: Pipeline 运行 API

## Overview

实现 Pipeline 运行 API，支持异步任务执行、状态查询和 WebSocket 进度推送。

## API Endpoints

### 1. POST /api/v1/pipeline/run

启动 Pipeline 运行。

**Request Body:**
```json
{
  "pipeline_type": "factor_evaluation|strategy_backtest|full_pipeline",
  "config": {
    "factor_id": "...",
    "strategy_id": "...",
    "date_range": ["2024-01-01", "2024-12-31"],
    "symbols": ["BTC/USDT", "ETH/USDT"]
  }
}
```

**Response:**
```json
{
  "run_id": "run-uuid",
  "status": "pending",
  "created_at": "2025-01-01T00:00:00Z"
}
```

### 2. GET /api/v1/pipeline/{run_id}/status

获取 Pipeline 运行状态。

**Response:**
```json
{
  "run_id": "run-uuid",
  "status": "pending|running|completed|failed",
  "progress": 0.75,
  "current_step": "evaluating_factors",
  "started_at": "...",
  "completed_at": "...",
  "result": {...},
  "error": null
}
```

### 3. WebSocket /api/v1/pipeline/{run_id}/ws

实时进度推送。

**Message Format:**
```json
{
  "type": "progress|status|error|result",
  "data": {...}
}
```

## Pipeline Types

1. **factor_evaluation**: 因子评估流程
2. **strategy_backtest**: 策略回测流程
3. **full_pipeline**: 完整流程 (生成 → 评估 → 回测)

## File Structure

```
src/iqfmp/api/pipeline/
├── __init__.py
├── schemas.py      # Pydantic schemas
├── service.py      # Pipeline service
├── runner.py       # Pipeline runner (async execution)
└── router.py       # FastAPI router + WebSocket
```

## Dependencies

- Task 17: BacktestEngine (策略回测)
- Task 23: FastAPI framework

## Test Coverage

- Schema validation tests
- Service unit tests
- Router endpoint tests
- WebSocket tests
- Integration tests
- Status transition tests
