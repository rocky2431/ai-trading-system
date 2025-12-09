# Feature: 实现因子家族约束生成 (User Story 1.2)

**Task ID**: 9
**Status**: In Progress
**Branch**: feat/task-9-factor-family-constraints

## Overview

扩展 FactorGenerationAgent，实现因子家族约束验证功能。
用户选择因子家族后，系统自动验证生成的代码只使用该家族允许的数据字段。

## Rationale

确保生成的因子代码符合家族定义的数据字段约束：
- 防止 Momentum 因子使用 Sentiment 数据
- 确保因子逻辑与家族定义一致
- 提供详细的违规报告

## Implementation Plan

### 1. FactorFieldValidator
- 扫描生成代码中使用的数据字段 (df['xxx'])
- 对比因子家族允许的字段列表
- 返回验证结果和违规详情

### 2. Enhanced FactorFamily
- 扩展每个家族的字段定义
- 添加字段描述和用途说明
- 支持字段别名映射

### 3. Agent Integration
- 在 FactorGenerationAgent.generate() 中集成验证
- 验证失败时抛出 FieldConstraintViolationError
- 可配置是否启用字段约束

### 4. Prompt Enhancement
- 在 Prompt 中明确列出允许的字段
- 添加字段使用示例
- 强调字段约束要求

## Impact Assessment

- **User Stories Affected**: US 1.2 (因子家族约束生成)
- **Architecture Changes**: No
- **Breaking Changes**: No (新增功能，向后兼容)

## Requirements Trace

- Traces to: specs/product.md#user-story-12
