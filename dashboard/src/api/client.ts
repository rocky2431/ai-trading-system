/**
 * API Client - HTTP 客户端配置
 *
 * Features:
 * - Automatic token refresh on 401 responses
 * - Bearer token authentication
 * - Query parameter support
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'
const TOKEN_KEY = 'iqfmp_access_token'
const REFRESH_TOKEN_KEY = 'iqfmp_refresh_token'

// Safe localStorage helpers - handle private browsing and security restrictions
const safeStorage = {
  get(key: string): string | null {
    try {
      return localStorage.getItem(key)
    } catch (err) {
      console.warn('[Storage] Failed to read key:', key, err)
      return null
    }
  },
  set(key: string, value: string): void {
    try {
      localStorage.setItem(key, value)
    } catch (err) {
      console.error('[Storage] Failed to write key:', key, err)
      // Storage unavailable - session will not persist
    }
  },
  remove(key: string): void {
    try {
      localStorage.removeItem(key)
    } catch (err) {
      console.warn('[Storage] Failed to remove key:', key, err)
    }
  },
}

interface RequestOptions extends RequestInit {
  params?: Record<string, string | number | undefined>
  _isRetry?: boolean // Internal flag to prevent infinite retry loops
}

class ApiError extends Error {
  constructor(
    public status: number,
    public statusText: string,
    public data?: unknown
  ) {
    super(`API Error: ${status} ${statusText}`)
    this.name = 'ApiError'
  }
}

// Track ongoing refresh to prevent multiple simultaneous refresh requests
let refreshPromise: Promise<boolean> | null = null

async function tryRefreshToken(): Promise<boolean> {
  // If already refreshing, wait for that to complete
  if (refreshPromise) {
    return refreshPromise
  }

  refreshPromise = (async () => {
    try {
      const refreshToken = safeStorage.get(REFRESH_TOKEN_KEY)
      if (!refreshToken) {
        return false
      }

      const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refreshToken }),
      })

      if (!response.ok) {
        return false
      }

      const data = await response.json()
      if (data.access_token) {
        safeStorage.set(TOKEN_KEY, data.access_token)
        return true
      }
      return false
    } catch (err) {
      console.error('[Auth] Token refresh failed:', err)
      return false
    } finally {
      refreshPromise = null
    }
  })()

  return refreshPromise
}

async function request<T>(endpoint: string, options: RequestOptions = {}): Promise<T> {
  const { params, _isRetry, ...fetchOptions } = options

  let url = `${API_BASE_URL}${endpoint}`

  // Add query params
  if (params) {
    const searchParams = new URLSearchParams()
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        searchParams.append(key, String(value))
      }
    })
    const queryString = searchParams.toString()
    if (queryString) {
      url += `?${queryString}`
    }
  }

  // Get auth token if available (direct localStorage access to avoid circular import)
  const token = safeStorage.get(TOKEN_KEY)

  const response = await fetch(url, {
    ...fetchOptions,
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` }),
      ...fetchOptions.headers,
    },
  })

  // Handle 401 Unauthorized - try to refresh token
  if (response.status === 401 && !_isRetry && !endpoint.includes('/auth/')) {
    const refreshed = await tryRefreshToken()
    if (refreshed) {
      // Retry the original request with new token
      return request<T>(endpoint, { ...options, _isRetry: true })
    }
    // Refresh failed - clear tokens
    safeStorage.remove(TOKEN_KEY)
    safeStorage.remove(REFRESH_TOKEN_KEY)
  }

  if (!response.ok) {
    let errorData
    try {
      errorData = await response.json()
    } catch (parseErr) {
      console.warn('[API] Failed to parse error response:', response.status, parseErr)
      errorData = null
    }
    throw new ApiError(response.status, response.statusText, errorData)
  }

  // Handle 204 No Content
  if (response.status === 204) {
    return undefined as T
  }

  return response.json()
}

export const api = {
  get: <T>(endpoint: string, params?: Record<string, string | number | undefined>) =>
    request<T>(endpoint, { method: 'GET', params }),

  post: <T>(endpoint: string, data?: unknown) =>
    request<T>(endpoint, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    }),

  put: <T>(endpoint: string, data?: unknown) =>
    request<T>(endpoint, {
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
    }),

  patch: <T>(endpoint: string, data?: unknown) =>
    request<T>(endpoint, {
      method: 'PATCH',
      body: data ? JSON.stringify(data) : undefined,
    }),

  delete: <T>(endpoint: string) =>
    request<T>(endpoint, { method: 'DELETE' }),
}

export { ApiError }
