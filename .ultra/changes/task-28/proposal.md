# Feature: Agent 状态监控页面 (D-001)

**Task ID**: 28
**Status**: In Progress
**Branch**: feat/task-28-agent-monitor

## Overview

实现 AgentMonitor.tsx 组件，提供 LangGraph Agent 运行状态的实时监控界面。包括：
- Agent 运行状态展示
- 任务队列可视化
- LLM 延迟和成功率指标
- 资源使用监控（CPU/Memory）

## Rationale

用户需要实时了解 Agent 系统的运行状态，以便：
1. 监控因子生成任务的进度
2. 识别系统瓶颈和性能问题
3. 快速发现和响应异常情况

## Impact Assessment

- **User Stories Affected**: specs/product.md#user-story-51
- **Architecture Changes**: No - 使用现有 Dashboard 架构
- **Breaking Changes**: No

## Requirements Trace

- Traces to: specs/product.md#user-story-51

## Implementation Plan

### Components

1. **AgentMonitor.tsx** - 主页面组件
2. **AgentStatusCard.tsx** - Agent 状态卡片
3. **TaskQueueChart.tsx** - 任务队列图表
4. **LLMMetricsCard.tsx** - LLM 指标卡片
5. **ResourceUsageChart.tsx** - 资源使用图表

### Data Flow

```
Backend API (GET /agents/status) → React Query → Components → UI
WebSocket (ws://localhost:8000/ws/agents) → Real-time Updates
```

### Mock Data Strategy

由于后端 Agent API 已在 Task 24 实现，本任务使用：
1. 开发阶段：Mock 数据模拟
2. 集成阶段：连接真实 API
