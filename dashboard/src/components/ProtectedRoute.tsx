/**
 * ProtectedRoute - 路由认证守卫
 *
 * 检查用户是否已登录且 token 有效，未登录或 token 无效则重定向到登录页面
 */

import { useEffect, useState } from 'react'
import { Navigate, useLocation } from 'react-router-dom'
import { authApi, tokenStorage } from '@/api/auth'
import { useAppStore } from '@/store/useAppStore'

interface ProtectedRouteProps {
  children: React.ReactNode
}

export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const location = useLocation()
  const setUser = useAppStore((s) => s.setUser)
  const [isValidating, setIsValidating] = useState(true)
  const [isValid, setIsValid] = useState(false)

  useEffect(() => {
    let isMounted = true

    async function validateToken() {
      // First check if token exists
      if (!authApi.isAuthenticated()) {
        if (isMounted) {
          setIsValidating(false)
          setIsValid(false)
        }
        return
      }

      // Validate token by calling /auth/me
      try {
        const user = await authApi.me()
        if (isMounted) {
          setUser({
            id: user.id,
            email: user.email,
            name: user.username,
            role: user.role === 'admin' ? 'admin' : user.role === 'viewer' ? 'viewer' : 'user',
          })
          setIsValid(true)
        }
      } catch {
        // Token invalid or expired - clear it
        tokenStorage.clearTokens()
        if (isMounted) {
          setIsValid(false)
        }
      } finally {
        if (isMounted) {
          setIsValidating(false)
        }
      }
    }

    validateToken()

    return () => {
      isMounted = false
    }
  }, [setUser])

  // Show loading while validating
  if (isValidating) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
        <div className="text-gray-500 dark:text-gray-400">Loading...</div>
      </div>
    )
  }

  // Redirect to login if not valid
  if (!isValid) {
    return <Navigate to="/login" state={{ from: location }} replace />
  }

  return <>{children}</>
}
