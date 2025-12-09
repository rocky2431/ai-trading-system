# Feature: 沙箱执行器 (严格模式 Layer 2)

**Task ID**: 2
**Status**: In Progress
**Branch**: feat/task-2-sandbox-executor

## Overview

基于 RestrictedPython 实现 SandboxExecutor 类，作为 LLM 生成代码的第二道安全防线。在运行时隔离环境中执行因子代码，限制可访问的模块和函数，配置资源限制。

## Rationale

即使代码通过了 AST 静态检查，在执行时仍可能存在风险。沙箱执行器提供运行时隔离，确保：
1. 代码只能访问白名单内的模块和函数
2. 资源使用受限（CPU、内存、执行时间）
3. 无法进行网络、文件系统等危险操作

## Security Features

### 模块白名单
- `pandas`, `numpy`, `math`, `statistics`
- `datetime`, `collections`, `itertools`
- `functools`, `operator`, `typing`

### 资源限制
- **CPU**: 最大执行时间 60 秒
- **Memory**: 最大内存 512MB
- **Timeout**: 可配置超时时间

### 受限环境
- 自定义 `__builtins__` (移除危险函数)
- 禁用 `import` 语句 (使用预注入模块)
- 使用 RestrictedPython 编译

## Impact Assessment

- **User Stories Affected**: 无直接用户故事，安全基础设施
- **Architecture Changes**: 添加 core/sandbox.py 模块
- **Breaking Changes**: 无

## Requirements Trace

- Traces to: specs/architecture.md#code-execution-safety
- Depends on: Task 1 (AST Security Checker)

## Implementation Checklist

- [ ] 定义 ExecutionResult 数据模型
- [ ] 实现 SandboxExecutor 类
- [ ] 实现模块白名单机制
- [ ] 实现资源限制 (timeout)
- [ ] 集成 AST 安全检查器
- [ ] 编写 6 维度测试用例
