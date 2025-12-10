# Task 25: Research Ledger API

## Overview

实现研究账本 API，提供试验记录查询、统计分析和动态阈值接口。

## API Endpoints

### 1. GET /api/v1/research/ledger

获取试验记录列表，支持分页和筛选。

**Query Parameters:**
- `page` (int, default=1): 页码
- `page_size` (int, default=10, max=100): 每页记录数
- `family` (str, optional): 按因子家族筛选
- `min_sharpe` (float, optional): 最小 Sharpe 比率筛选
- `start_date` (datetime, optional): 开始日期筛选
- `end_date` (datetime, optional): 结束日期筛选

**Response:**
```json
{
  "trials": [...],
  "total": 100,
  "page": 1,
  "page_size": 10
}
```

### 2. GET /api/v1/research/stats

获取研究账本统计信息。

**Query Parameters:**
- `group_by_family` (bool, default=false): 是否按家族分组

**Response:**
```json
{
  "overall": {
    "total_trials": 100,
    "mean_sharpe": 1.5,
    "std_sharpe": 0.5,
    "max_sharpe": 3.2,
    "min_sharpe": -0.5,
    "median_sharpe": 1.4
  },
  "by_family": {
    "momentum": {...},
    "volatility": {...}
  }
}
```

### 3. GET /api/v1/metrics/thresholds

获取当前动态阈值信息。

**Response:**
```json
{
  "current_threshold": 2.15,
  "n_trials": 50,
  "config": {
    "base_sharpe_threshold": 2.0,
    "confidence_level": 0.95,
    "min_trials_for_adjustment": 1
  },
  "threshold_history": [
    {"n_trials": 10, "threshold": 2.05},
    {"n_trials": 25, "threshold": 2.10},
    {"n_trials": 50, "threshold": 2.15}
  ]
}
```

## File Structure

```
src/iqfmp/api/research/
├── __init__.py
├── schemas.py      # Pydantic schemas
├── service.py      # Research service
└── router.py       # FastAPI router
```

## Dependencies

- Task 11: ResearchLedger, TrialRecord, DynamicThreshold
- Task 23: FastAPI framework, JWT auth

## Test Coverage

- Schema validation tests
- Service unit tests
- Router endpoint tests
- Integration tests
- Boundary tests (pagination, filtering)
- Exception tests (invalid params)
