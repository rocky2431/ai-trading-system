# Feature: 实现 CryptoCVSplitter 多切片验证 (User Story 2.1)

**Task ID**: 10
**Status**: In Progress
**Branch**: feat/task-10-crypto-cv-splitter

## Overview

实现加密货币专用的交叉验证切分器，支持多维度数据切分：
- 时间切分：Train/Valid/Test
- 市场切分：BTC/ETH 大市值 vs Altcoins 小市值
- 频率切分：1h/4h/1d 多时间框架

## Rationale

加密货币市场具有独特的特性，需要专门的验证策略：
- 高波动性要求更稳健的时间切分
- 市场分层测试因子在不同市值股票上的表现
- 多频率验证确保因子跨时间框架的稳定性

## Implementation Plan

### 1. TimeSplitter
- 按时间比例切分 (默认 60/20/20)
- 滚动窗口切分
- 避免数据泄露的严格时间边界

### 2. MarketSplitter
- 大市值组：BTC, ETH
- 中市值组：Top 10-50
- 小市值组：Altcoins
- 支持自定义分组规则

### 3. FrequencySplitter
- 支持 1h, 4h, 1d 等多种频率
- 数据重采样
- 频率间对齐

### 4. CryptoCVSplitter
- 组合多个 Splitter
- 生成切片组合
- 支持迭代器模式

## Impact Assessment

- **User Stories Affected**: US 2.1 (多切片验证)
- **Architecture Changes**: No
- **Breaking Changes**: No

## Requirements Trace

- Traces to: specs/product.md#user-story-21
