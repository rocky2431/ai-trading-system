# Feature: 实现 StabilityAnalyzer 稳定性分析 (User Story 2.3)

**Task ID**: 12
**Status**: In Progress
**Branch**: feat/task-12-stability-analyzer

## Overview

实现因子稳定性分析器，从三个维度评估因子的稳健性：
- 时间稳定性：按月/季度分析 IC 变化
- 市场稳定性：大/中/小市值分组分析
- 环境稳定性：不同市场环境 (regime) 下的表现

## Rationale

因子挖掘的核心挑战是避免过拟合：
- 时间稳定性确保因子不是特定时期的偶然发现
- 市场稳定性确保因子跨不同市值股票有效
- 环境稳定性确保因子在牛熊市都能工作

## Implementation Plan

### 1. TimeStabilityAnalyzer
- 按月/季度计算 IC 序列
- 计算 IC 均值、标准差、IR
- 检测 IC 衰减趋势
- 计算滚动 IC 稳定性

### 2. MarketStabilityAnalyzer
- 按市值分组计算 IC
- 比较大/中/小市值表现差异
- 计算市场维度一致性得分

### 3. RegimeStabilityAnalyzer
- 基于波动率/趋势划分市场 regime
- 分 regime 计算因子表现
- 评估 regime 敏感度

### 4. StabilityAnalyzer (组合)
- 整合三个维度分析
- 生成综合稳定性得分
- 输出 StabilityReport

## Impact Assessment

- **User Stories Affected**: US 2.3 (稳定性分析)
- **Architecture Changes**: No
- **Breaking Changes**: No

## Requirements Trace

- Traces to: specs/product.md#user-story-23
