# Feature: 开平仓止盈止损逻辑 (S-003)

**Task ID**: 16
**Status**: In Progress
**Branch**: feat/task-16-position-management

## Overview

实现完整的仓位管理系统：
- 多空双向信号处理
- 止盈止损条件配置
- 仓位规模计算（凯利公式/固定比例）
- 时间止损支持

## Implementation Plan

1. PositionSizing - 仓位规模计算
   - KellySizer: 凯利公式
   - FixedSizer: 固定比例
   - RiskParitySizer: 风险平价

2. StopLossManager - 止损管理
   - PriceStopLoss: 价格止损
   - PercentStopLoss: 百分比止损
   - TrailingStop: 移动止损
   - TimeStop: 时间止损

3. TakeProfitManager - 止盈管理
   - FixedTakeProfit: 固定止盈
   - TrailingTakeProfit: 移动止盈

4. PositionManager - 仓位管理器
   - 开仓/平仓逻辑
   - 信号处理
   - 仓位跟踪

## Requirements Trace
- Traces to: specs/product.md#user-story-33
