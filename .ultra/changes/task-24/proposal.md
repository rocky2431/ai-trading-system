# Feature: 因子管理 API

**Task ID**: 24
**Status**: In Progress
**Branch**: feat/task-24-factor-api

## Overview

实现因子管理 REST API：
- POST /factors/generate - 生成因子
- GET /factors - 因子列表
- GET /factors/:id - 因子详情
- POST /factors/:id/evaluate - 评估因子
- PUT /factors/:id/status - 更新状态

## Implementation Plan

1. Factor Schemas - API 请求/响应模型
   - FactorGenerateRequest
   - FactorResponse
   - FactorListResponse
   - FactorEvaluateRequest/Response

2. Factor Service - 业务逻辑
   - FactorService: CRUD + 生成/评估

3. Factor Router - API 路由
   - 5 个端点实现
   - 认证保护
   - 分页筛选

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /factors/generate | 根据描述生成因子 |
| GET | /factors | 获取因子列表 |
| GET | /factors/{id} | 获取因子详情 |
| POST | /factors/{id}/evaluate | 评估因子 |
| PUT | /factors/{id}/status | 更新因子状态 |

## Requirements Trace
- Traces to: specs/architecture.md#api-design
