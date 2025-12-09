# Feature: 实现 LangGraph Agent 编排框架

**Task ID**: 7
**Status**: In Progress
**Branch**: feat/task-7-langgraph-orchestrator

## Overview

搭建基于 LangGraph 的 Agent 编排框架，实现 StateGraph 状态机和 AgentOrchestrator 编排器。
支持状态持久化（PostgresSaver）和 checkpoint/time-travel 调试能力。

## Rationale

IQFMP 的因子生成和策略构建需要多个 Agent 协作完成复杂任务：
- 因子生成 Agent
- 代码安全检查 Agent
- 回测评估 Agent
- 策略优化 Agent

LangGraph 提供了成熟的 Agent 编排能力：
- StateGraph 状态机模式
- 内置 checkpoint 支持
- PostgreSQL 持久化
- Time-travel 调试

## Implementation Plan

### 1. AgentState 状态定义
- 定义统一的 Agent 状态结构
- 支持消息历史、工具调用、中间结果
- 类型安全的状态访问

### 2. StateGraph 状态机
- 实现节点 (Node) 和边 (Edge) 抽象
- 支持条件路由
- 支持循环和分支

### 3. AgentOrchestrator 编排器
- 管理多个 Agent 的生命周期
- 协调 Agent 之间的数据流
- 提供统一的执行接口

### 4. PostgresSaver 持久化
- 基于 PostgreSQL 的状态存储
- Checkpoint 自动保存
- Time-travel 状态回溯

### 5. 调试支持
- Checkpoint 查询和恢复
- 执行历史可视化
- 状态快照导出

## Impact Assessment

- **User Stories Affected**: US 1.1 (因子生成 Agent)
- **Architecture Changes**: Yes - 新增 agents 模块核心框架
- **Breaking Changes**: No

## Requirements Trace

- Traces to: specs/architecture.md#langgraph-state-persistence
