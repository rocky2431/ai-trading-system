# Feature: 人工审核网关 (严格模式 Layer 3)

**Task ID**: 3
**Status**: In Progress
**Branch**: feat/task-3-human-review-gate

## Overview

实现 HumanReviewGate 类，作为 LLM 生成代码的第三道安全防线。在代码通过 AST 检查和沙箱验证后，需要人工确认才能最终执行。

## Rationale

即使代码通过了静态检查和沙箱测试，仍可能存在逻辑错误或业务风险。人工审核提供最终确认，确保：
1. 代码逻辑符合业务预期
2. 生成的因子策略合理
3. 提供审计追踪记录

## Security Features

### 审核队列
- 待审核代码入队
- 支持优先级排序
- 超时自动拒绝

### 通知渠道
- Telegram Bot 通知
- Slack Webhook 通知
- 内置回调接口

### 审核决策
- APPROVE: 批准执行
- REJECT: 拒绝执行
- TIMEOUT: 超时自动拒绝

## Impact Assessment

- **User Stories Affected**: 无直接用户故事，安全基础设施
- **Architecture Changes**: 添加 core/review.py 模块
- **Breaking Changes**: 无

## Requirements Trace

- Traces to: specs/architecture.md#code-execution-safety
- Depends on: Task 2 (Sandbox Executor)

## Implementation Checklist

- [ ] 定义 ReviewRequest/ReviewDecision 数据模型
- [ ] 实现 HumanReviewGate 类
- [ ] 实现审核队列管理
- [ ] 实现通知发送接口 (抽象)
- [ ] 实现超时处理
- [ ] 编写 6 维度测试用例
