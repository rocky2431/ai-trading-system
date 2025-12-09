# Feature: 配置 OpenRouter LLM 集成

**Task ID**: 6
**Status**: In Progress
**Branch**: feat/task-6-openrouter-llm

## Overview

实现 LLMProvider 类，提供统一的 LLM 调用接口，支持多模型切换和 Fallback 机制。
通过 OpenRouter API 统一访问 DeepSeek、Claude、GPT 等多种模型。

## Rationale

IQFMP 需要 LLM 能力来生成因子代码和策略代码。通过 OpenRouter：
- 统一 API 接口，无需为每个模型维护不同的集成
- 支持模型切换和故障转移
- 提供成本控制和速率限制

## Implementation Plan

### 1. LLMProvider Core
- 基于 OpenRouter API 的统一接口
- 支持 DeepSeek-V3, Claude-3.5-Sonnet, GPT-4o 等模型
- 配置通过环境变量和 YAML 文件

### 2. Model Selection Strategy
- 默认模型配置
- 基于任务类型的模型选择
- 手动模型覆盖

### 3. Fallback Chain
- 主模型失败时自动切换备用模型
- 可配置的重试策略
- 错误分类和处理

### 4. Caching & Rate Limiting
- 请求缓存 (相同输入返回缓存结果)
- 速率限制 (防止超过 API 限制)
- 成本追踪

## Impact Assessment

- **User Stories Affected**: US 1.1 (因子生成依赖 LLM)
- **Architecture Changes**: No - 新增 LLM 模块
- **Breaking Changes**: No

## Requirements Trace

- Traces to: specs/architecture.md#llm-provider-strategy
