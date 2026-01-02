/**
 * ProtectedRoute - 路由认证守卫
 *
 * 检查用户是否已登录，未登录则重定向到登录页面
 */

import { Navigate, useLocation } from 'react-router-dom'
import { authApi } from '@/api/auth'

interface ProtectedRouteProps {
  children: React.ReactNode
}

export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const location = useLocation()

  // Check if user has valid token
  if (!authApi.isAuthenticated()) {
    // Redirect to login, preserving the intended destination
    return <Navigate to="/login" state={{ from: location }} replace />
  }

  return <>{children}</>
}
