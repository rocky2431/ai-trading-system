/**
 * Auth API - 用户认证 API
 *
 * 对应后端: src/iqfmp/api/auth/router.py
 */

import { api } from './client'

// ============== Types ==============

export type UserRole = 'admin' | 'user' | 'readonly'

export interface UserResponse {
  id: string
  username: string
  email: string
  role: UserRole
  is_active: boolean
}

export interface TokenResponse {
  access_token: string
  refresh_token?: string
  token_type: string
}

export interface LoginRequest {
  username: string
  password: string
}

export interface RegisterRequest {
  username: string
  email: string
  password: string
}

export interface RefreshTokenRequest {
  refresh_token: string
}

// ============== Token Storage ==============

const TOKEN_KEY = 'iqfmp_access_token'
const REFRESH_TOKEN_KEY = 'iqfmp_refresh_token'

export const tokenStorage = {
  getAccessToken: (): string | null => {
    return localStorage.getItem(TOKEN_KEY)
  },

  getRefreshToken: (): string | null => {
    return localStorage.getItem(REFRESH_TOKEN_KEY)
  },

  setTokens: (accessToken: string, refreshToken?: string): void => {
    localStorage.setItem(TOKEN_KEY, accessToken)
    if (refreshToken) {
      localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken)
    }
  },

  clearTokens: (): void => {
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(REFRESH_TOKEN_KEY)
  },

  isAuthenticated: (): boolean => {
    return !!localStorage.getItem(TOKEN_KEY)
  },
}

// ============== API ==============

export const authApi = {
  /**
   * 用户注册
   */
  register: (data: RegisterRequest) =>
    api.post<UserResponse>('/auth/register', data),

  /**
   * 用户登录
   */
  login: async (data: LoginRequest): Promise<TokenResponse> => {
    const response = await api.post<TokenResponse>('/auth/login', data)
    // 自动保存 token
    tokenStorage.setTokens(response.access_token, response.refresh_token)
    return response
  },

  /**
   * 刷新 access token
   */
  refresh: async (refreshToken?: string): Promise<TokenResponse> => {
    const token = refreshToken || tokenStorage.getRefreshToken()
    if (!token) {
      throw new Error('No refresh token available')
    }
    const response = await api.post<TokenResponse>('/auth/refresh', {
      refresh_token: token,
    })
    // 更新 access token
    tokenStorage.setTokens(response.access_token)
    return response
  },

  /**
   * 获取当前用户信息
   */
  me: () => api.get<UserResponse>('/auth/me'),

  /**
   * 登出 (清除本地 token)
   */
  logout: (): void => {
    tokenStorage.clearTokens()
  },

  /**
   * 检查是否已登录
   */
  isAuthenticated: (): boolean => {
    return tokenStorage.isAuthenticated()
  },
}
