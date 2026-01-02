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
      let refreshToken: string | null = null
      try {
        refreshToken = localStorage.getItem(REFRESH_TOKEN_KEY)
      } catch {
        // localStorage unavailable
      }

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
        try {
          localStorage.setItem(TOKEN_KEY, data.access_token)
        } catch {
          // localStorage unavailable
        }
        return true
      }
      return false
    } catch {
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
  let token: string | null = null
  try {
    token = localStorage.getItem(TOKEN_KEY)
  } catch {
    // localStorage unavailable (private browsing, security restrictions)
  }

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
    // Refresh failed - clear tokens and redirect to login
    try {
      localStorage.removeItem(TOKEN_KEY)
      localStorage.removeItem(REFRESH_TOKEN_KEY)
    } catch {
      // localStorage unavailable
    }
  }

  if (!response.ok) {
    let errorData
    try {
      errorData = await response.json()
    } catch {
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

  delete: <T>(endpoint: string) =>
    request<T>(endpoint, { method: 'DELETE' }),
}

export { ApiError }
