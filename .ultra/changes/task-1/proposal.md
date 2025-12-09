# Feature: AST 安全检查器 (严格模式 Layer 1)

**Task ID**: 1
**Status**: In Progress
**Branch**: feat/task-1-ast-security-checker

## Overview

实现 ASTSecurityChecker 类，作为 LLM 生成代码的第一道安全防线。使用 Python AST 模块静态分析代码，检测并阻止危险操作。

## Rationale

LLM 生成的因子代码可能包含恶意或危险操作。AST 安全检查器通过静态分析在代码执行前拦截威胁，是三层安全架构的第一层。

## Security Checks

### 禁止的危险函数
- `eval()`, `exec()`, `compile()`
- `os.system()`, `os.popen()`, `subprocess.*`
- `open()` (文件操作)
- `__import__()`, `importlib.*`

### 禁止的危险模块
- `os`, `sys`, `subprocess`
- `socket`, `requests`, `urllib`
- `pickle`, `marshal`
- `ctypes`, `cffi`

### 禁止的危险属性访问
- `__builtins__`, `__globals__`
- `__code__`, `__class__`
- `__subclasses__`, `__mro__`

## Impact Assessment

- **User Stories Affected**: 无直接用户故事，安全基础设施
- **Architecture Changes**: 添加 core/security.py 模块
- **Breaking Changes**: 无

## Requirements Trace

- Traces to: specs/architecture.md#code-execution-safety
- Traces to: specs/architecture.md#security-architecture

## Implementation Checklist

- [ ] 定义 SecurityViolation 数据模型
- [ ] 实现 ASTSecurityChecker 类
- [ ] 实现危险函数检测
- [ ] 实现危险模块检测
- [ ] 实现危险属性访问检测
- [ ] 生成详细违规报告
- [ ] 编写 6 维度测试用例
