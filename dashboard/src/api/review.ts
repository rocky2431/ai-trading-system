/**
 * Review API - 人工审核 API 调用
 *
 * 对应后端: src/iqfmp/api/review/router.py
 */

import { api } from './client'

// ============== Types ==============

export type ReviewStatus = 'pending' | 'approved' | 'rejected' | 'timeout'

export interface ReviewRequest {
  request_id: string
  code: string
  code_summary: string
  factor_name: string
  metadata: Record<string, unknown>
  priority: number
  status: ReviewStatus
  created_at: string
}

export interface ReviewDecision {
  request_id: string
  status: ReviewStatus
  reviewer: string | null
  reason: string | null
  decided_at: string
}

export interface ReviewQueueStats {
  pending_count: number
  approved_count: number
  rejected_count: number
  timeout_count: number
  average_review_time_seconds: number | null
  oldest_pending_age_seconds: number | null
}

export interface ReviewConfig {
  timeout_seconds: number
  max_queue_size: number
  auto_reject_on_timeout: boolean
}

export interface PaginatedReviewResponse {
  items: ReviewRequest[]
  total: number
  page: number
  page_size: number
  has_next: boolean
}

export interface PaginatedDecisionResponse {
  items: ReviewDecision[]
  total: number
  page: number
  page_size: number
  has_next: boolean
}

export interface ReviewRequestCreate {
  code: string
  code_summary: string
  factor_name: string
  metadata?: Record<string, unknown>
  priority?: number
}

export interface ReviewDecisionRequest {
  approved: boolean
  reason?: string
}

export interface ReviewConfigUpdate {
  timeout_seconds?: number
  max_queue_size?: number
  auto_reject_on_timeout?: boolean
}

// ============== API ==============

export const reviewApi = {
  /**
   * 提交审核请求
   */
  submitRequest: (data: ReviewRequestCreate) =>
    api.post<ReviewRequest>('/review/requests', data),

  /**
   * 获取待审核请求列表
   */
  getPendingRequests: (page = 1, pageSize = 20) =>
    api.get<PaginatedReviewResponse>('/review/requests/pending', {
      page,
      page_size: pageSize,
    }),

  /**
   * 获取指定审核请求详情
   */
  getRequest: (requestId: string) =>
    api.get<ReviewRequest>(`/review/requests/${requestId}`),

  /**
   * 做出审核决策
   */
  decide: (requestId: string, data: ReviewDecisionRequest) =>
    api.post<ReviewDecision>(`/review/requests/${requestId}/decide`, data),

  /**
   * 获取审核历史
   */
  getDecisionHistory: (page = 1, pageSize = 20) =>
    api.get<PaginatedDecisionResponse>('/review/decisions', {
      page,
      page_size: pageSize,
    }),

  /**
   * 获取审核队列统计
   */
  getStats: () =>
    api.get<ReviewQueueStats>('/review/stats'),

  /**
   * 获取审核配置
   */
  getConfig: () =>
    api.get<ReviewConfig>('/review/config'),

  /**
   * 更新审核配置
   */
  updateConfig: (data: ReviewConfigUpdate) =>
    api.post<ReviewConfig>('/review/config', data),

  /**
   * 处理超时请求
   */
  processTimeouts: () =>
    api.post<{ message: string; timed_out_count: number }>('/review/process-timeouts'),
}
