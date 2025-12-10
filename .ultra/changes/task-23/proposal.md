# Feature: FastAPI 后端框架

**Task ID**: 23
**Status**: In Progress
**Branch**: feat/task-23-fastapi-backend

## Overview

搭建完整的 FastAPI 后端框架：
- 路由结构设计
- JWT 认证系统
- CORS 配置
- OpenAPI 文档
- 依赖注入

## Implementation Plan

1. Auth Module - JWT 认证
   - JWTConfig: JWT 配置
   - TokenService: Token 生成/验证
   - PasswordService: 密码加密/验证
   - AuthDependencies: 认证依赖注入

2. User Module - 用户管理
   - User model: 用户数据模型
   - UserService: 用户业务逻辑
   - UserRouter: 用户 API 路由

3. Router Structure - 路由结构
   - /api/v1/auth - 认证路由
   - /api/v1/users - 用户路由
   - /api/v1/factors - 因子路由 (后续任务)
   - /api/v1/research - 研究路由 (后续任务)

4. Middleware & Config
   - CORS 中间件
   - 请求日志中间件
   - 异常处理器
   - 配置管理

## Requirements Trace
- Traces to: specs/architecture.md#api-design
