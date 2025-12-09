# Feature: 实现 Research Ledger 研究账本 (User Story 2.2)

**Task ID**: 11
**Status**: In Progress
**Branch**: feat/task-11-research-ledger

## Overview

实现研究账本，记录所有因子评估历史，追踪试验次数，
实现动态阈值计算以防止过拟合。

## Rationale

多重检验问题是量化研究中的核心挑战：
- 试验次数越多，偶然发现显著因子的概率越高
- 需要动态调整显著性阈值
- 提供透明的研究记录供审计

## Implementation Plan

### 1. TrialRecord
- 记录单次因子评估结果
- 包含因子信息、评估指标、时间戳
- 支持序列化存储

### 2. ResearchLedger
- 追踪全局试验次数 N
- 记录所有试验历史
- 提供查询和统计功能

### 3. DynamicThreshold
- 实现 Deflated Sharpe Ratio 简化版
- 基于试验次数 N 调整阈值
- 支持不同置信水平

### 4. Storage
- 支持内存存储 (开发)
- 支持文件存储 (生产)
- 可扩展到数据库

## Impact Assessment

- **User Stories Affected**: US 2.2 (研究账本)
- **Architecture Changes**: No
- **Breaking Changes**: No

## Requirements Trace

- Traces to: specs/product.md#user-story-22
