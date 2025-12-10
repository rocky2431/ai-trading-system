# Feature: 风控检查模块 (T-004)

**Task ID**: 21
**Status**: In Progress
**Branch**: feat/task-21-risk-controller

## Overview

实现完整的风控检查系统：
- 最大回撤限制 (20%)
- 单笔亏损上限 (2%)
- 持仓集中度检查
- 触发风控时自动减仓/平仓

## Implementation Plan

1. RiskController - 风控控制器
   - 风控规则管理
   - 风控检查执行
   - 风控事件触发

2. DrawdownMonitor - 回撤监控
   - 实时回撤计算
   - 最大回撤追踪
   - 回撤阈值告警

3. LossLimiter - 单笔亏损限制
   - 单笔亏损检查
   - 累计亏损监控
   - 亏损限制触发

4. ConcentrationChecker - 持仓集中度检查
   - 单一资产集中度
   - 行业/板块集中度
   - 集中度告警

5. RiskAction - 风控动作
   - 自动减仓
   - 自动平仓
   - 交易暂停

## Risk Parameters (Default)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| max_drawdown | 20% | 最大回撤限制 |
| max_single_loss | 2% | 单笔亏损上限 |
| max_position_concentration | 30% | 单一持仓集中度上限 |
| daily_loss_limit | 5% | 日内亏损限制 |

## Requirements Trace
- Traces to: specs/product.md#user-story-44-t-004
