# Feature: 紧急平仓功能 (T-005)

**Task ID**: 22
**Status**: In Progress
**Branch**: feat/task-22-emergency-close

## Overview

实现紧急平仓系统，支持一键平仓所有持仓：
- EmergencyCloseManager: 紧急平仓管理器
- ConfirmationGate: 人工确认网关
- TelegramNotifier: Telegram 通知服务
- CloseResult: 平仓结果记录

## Implementation Plan

1. EmergencyCloseManager - 紧急平仓管理器
   - 获取所有持仓
   - 批量平仓执行
   - 平仓结果汇总
   - 失败重试机制

2. ConfirmationGate - 人工确认网关
   - 生成确认令牌
   - 等待确认 (超时设置)
   - 确认状态追踪
   - 取消平仓支持

3. TelegramNotifier - Telegram 通知
   - 发送平仓请求通知
   - 发送确认链接
   - 发送平仓结果
   - 错误告警通知

4. CloseResult - 平仓结果
   - 成功/失败记录
   - 平仓价格
   - 滑点统计
   - 时间戳

## Requirements Trace
- Traces to: specs/product.md#user-story-45-t-005
