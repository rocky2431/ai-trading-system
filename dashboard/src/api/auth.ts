/**
 * Auth API - 用户认证 API
 *
 * 对应后端: src/iqfmp/api/auth/router.py
 */

import { api } from './client'

// ============== Types ==============

export type UserRole = 'admin' | 'user' | 'viewer'

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

// In-memory fallback for when localStorage is unavailable (private browsing, etc.)
let memoryTokens: { access?: string; refresh?: string } = {}

function safeGetItem(key: string): string | null {
  try {
    return localStorage.getItem(key)
  } catch {
    // localStorage unavailable (private browsing, security restrictions)
    return key === TOKEN_KEY ? memoryTokens.access ?? null : memoryTokens.refresh ?? null
  }
}

function safeSetItem(key: string, value: string): void {
  try {
    localStorage.setItem(key, value)
  } catch {
    // Fallback to memory storage
    if (key === TOKEN_KEY) {
      memoryTokens.access = value
    } else if (key === REFRESH_TOKEN_KEY) {
      memoryTokens.refresh = value
    }
  }
}

function safeRemoveItem(key: string): void {
  try {
    localStorage.removeItem(key)
  } catch {
    // Clear from memory fallback
  }
  if (key === TOKEN_KEY) {
    delete memoryTokens.access
  } else if (key === REFRESH_TOKEN_KEY) {
    delete memoryTokens.refresh
  }
}

export const tokenStorage = {
  getAccessToken: (): string | null => {
    return safeGetItem(TOKEN_KEY)
  },

  getRefreshToken: (): string | null => {
    return safeGetItem(REFRESH_TOKEN_KEY)
  },

  setTokens: (accessToken: string, refreshToken?: string): void => {
    safeSetItem(TOKEN_KEY, accessToken)
    if (refreshToken) {
      safeSetItem(REFRESH_TOKEN_KEY, refreshToken)
    }
  },

  clearTokens: (): void => {
    safeRemoveItem(TOKEN_KEY)
    safeRemoveItem(REFRESH_TOKEN_KEY)
  },

  isAuthenticated: (): boolean => {
    return !!safeGetItem(TOKEN_KEY)
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
