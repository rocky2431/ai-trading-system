# Feature: 策略回测引擎 (S-004)

**Task ID**: 17
**Status**: In Progress
**Branch**: feat/task-17-backtest-engine

## Overview

实现完整的策略回测系统：
- 多空双向回测支持
- 完整绩效指标计算 (Sharpe/MaxDD/Win Rate)
- 分时段/分市场绩效分解
- 可视化回测报告生成

## Implementation Plan

1. BacktestEngine - 回测核心引擎
   - 事件驱动回测
   - 订单执行模拟
   - 滑点/手续费模拟

2. PerformanceMetrics - 绩效指标
   - Sharpe Ratio
   - Maximum Drawdown
   - Win Rate / Profit Factor
   - Sortino Ratio
   - Calmar Ratio

3. BacktestResult - 回测结果
   - 交易记录
   - 净值曲线
   - 分时段分析
   - 分市场分析

4. BacktestReport - 回测报告
   - 统计摘要
   - 图表生成
   - 导出功能

## Requirements Trace
- Traces to: specs/product.md#user-story-34
