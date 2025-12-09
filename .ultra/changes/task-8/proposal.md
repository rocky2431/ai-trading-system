# Feature: 实现自然语言因子生成 (User Story 1.1)

**Task ID**: 8
**Status**: In Progress
**Branch**: feat/task-8-factor-generation-agent

## Overview

实现 FactorGenerationAgent，用户可以通过自然语言描述因子逻辑，
Agent 自动生成 Qlib 兼容的 Python 因子代码，并通过 AST 安全检查。

## Rationale

这是 IQFMP 的核心功能之一，让用户无需编写代码即可生成量化因子：
- 降低量化投资门槛
- 加速因子研究迭代
- 确保生成代码的安全性

## Implementation Plan

### 1. Prompt Templates
- 因子生成系统提示词
- 因子家族约束模板
- 代码格式要求模板
- Few-shot 示例

### 2. FactorGenerationAgent
- 基于 AgentOrchestrator 的 Agent 实现
- 集成 LLMProvider 调用
- 支持因子家族约束
- 代码提取和后处理

### 3. AST Security Integration
- 调用 ASTSecurityChecker 验证生成代码
- 禁止危险函数和模块
- 提供详细的安全报告

### 4. Factor Output
- Qlib 兼容的因子代码格式
- 因子元数据 (名称、描述、家族)
- 生成日志和追踪

## Impact Assessment

- **User Stories Affected**: US 1.1 (自然语言因子生成)
- **Architecture Changes**: No
- **Breaking Changes**: No

## Requirements Trace

- Traces to: specs/product.md#user-story-11
