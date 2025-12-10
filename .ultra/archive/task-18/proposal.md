# Feature: ccxt 交易所连接 (T-001)

**Task ID**: 18
**Status**: In Progress
**Branch**: feat/task-18-exchange-connection

## Overview

实现统一的加密货币交易所连接层：
- 基于 ccxt 库的交易所抽象
- 支持 Binance Futures 和 OKX Swap
- 自动重连和错误处理机制

## Implementation Plan

1. ExchangeAdapter - 交易所抽象接口
   - 统一的市场数据接口
   - 统一的订单接口
   - 统一的账户接口

2. BinanceAdapter - Binance Futures 实现
   - USDT-M 永续合约
   - WebSocket 实时数据

3. OKXAdapter - OKX Swap 实现
   - USDT 永续合约
   - WebSocket 实时数据

4. ConnectionManager - 连接管理
   - 自动重连机制
   - 心跳检测
   - 错误处理

## Requirements Trace
- Traces to: specs/product.md#user-story-41
