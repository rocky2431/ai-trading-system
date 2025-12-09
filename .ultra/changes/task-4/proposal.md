# Feature: Qlib Deep Fork 基础结构

**Task ID**: 4
**Status**: In Progress
**Branch**: feat/task-4-qlib-deep-fork

## Overview

搭建 Qlib Deep Fork 基础结构，创建 qlib-crypto 目录，实现 CryptoDataHandler 骨架类，为加密货币量化分析提供 Qlib 兼容的数据处理框架。

## Rationale

Qlib 是微软开源的量化投资平台，但原生不支持加密货币特有的数据字段（资金费率、持仓量、基差等）。通过 Deep Fork 方式扩展 Qlib，可以：
1. 保持与 Qlib 生态兼容
2. 支持加密货币特有数据
3. 复用 Qlib 的回测和模型训练框架

## Architecture

```
src/iqfmp/
├── qlib_crypto/              # Qlib Deep Fork
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── handler.py        # CryptoDataHandler
│   ├── contrib/
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
```

## Key Components

### CryptoDataHandler
- 继承 Qlib DataHandlerLP
- 支持加密货币字段: funding_rate, open_interest, basis
- 多交易所数据格式适配

## Impact Assessment

- **User Stories Affected**: 无直接用户故事，基础设施
- **Architecture Changes**: 添加 qlib_crypto 模块
- **Breaking Changes**: 无

## Requirements Trace

- Traces to: specs/architecture.md#qlib-deep-fork

## Implementation Checklist

- [ ] 创建 qlib_crypto 目录结构
- [ ] 实现 CryptoDataHandler 骨架类
- [ ] 定义加密货币数据字段
- [ ] 实现数据验证逻辑
- [ ] 编写 6 维度测试用例
