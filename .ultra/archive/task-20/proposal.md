# Feature: 持仓与 PnL 监控 (T-003)

**Task ID**: 20
**Status**: In Progress
**Branch**: feat/task-20-position-monitoring

## Overview

实现实时持仓与 PnL 监控系统：
- 实时持仓同步
- 未实现/已实现 PnL 计算
- 保证金使用率监控
- WebSocket 实时推送

## Implementation Plan

1. PositionTracker - 持仓追踪器
   - 实时持仓同步
   - 持仓状态更新
   - 多交易所持仓聚合

2. PnLCalculator - PnL 计算器
   - 未实现 PnL 计算
   - 已实现 PnL 计算
   - 总 PnL 汇总

3. MarginMonitor - 保证金监控
   - 保证金使用率计算
   - 维持保证金检查
   - 保证金告警

4. RealtimeUpdater - 实时推送
   - WebSocket 连接管理
   - 数据推送订阅
   - 断线重连

## Requirements Trace
- Traces to: specs/product.md#user-story-43-t-003
