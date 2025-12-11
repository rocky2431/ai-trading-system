import { create } from 'zustand'
import { devtools, persist } from 'zustand/middleware'

interface User {
  id: string
  email: string
  name: string
  role: 'admin' | 'user' | 'viewer'
}

// Helper function to apply theme to DOM
const applyThemeToDOM = (theme: 'light' | 'dark') => {
  if (typeof document !== 'undefined') {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }
}

interface AppState {
  // Theme
  theme: 'light' | 'dark'
  setTheme: (theme: 'light' | 'dark') => void
  toggleTheme: () => void

  // User
  user: User | null
  isAuthenticated: boolean
  setUser: (user: User | null) => void
  logout: () => void

  // Sidebar
  sidebarCollapsed: boolean
  toggleSidebar: () => void
}

export const useAppStore = create<AppState>()(
  devtools(
    persist(
      (set) => ({
        // Theme
        theme: 'light',
        setTheme: (theme) => {
          applyThemeToDOM(theme)
          set({ theme })
        },
        toggleTheme: () =>
          set((state) => {
            const newTheme = state.theme === 'light' ? 'dark' : 'light'
            applyThemeToDOM(newTheme)
            return { theme: newTheme }
          }),

        // User
        user: null,
        isAuthenticated: false,
        setUser: (user) =>
          set({
            user,
            isAuthenticated: !!user,
          }),
        logout: () =>
          set({
            user: null,
            isAuthenticated: false,
          }),

        // Sidebar
        sidebarCollapsed: false,
        toggleSidebar: () =>
          set((state) => ({
            sidebarCollapsed: !state.sidebarCollapsed,
          })),
      }),
      {
        name: 'iqfmp-storage',
        partialize: (state) => ({
          theme: state.theme,
          sidebarCollapsed: state.sidebarCollapsed,
        }),
        onRehydrateStorage: () => (state) => {
          // Apply theme to DOM when state is rehydrated from localStorage
          if (state?.theme) {
            applyThemeToDOM(state.theme)
          }
        },
      }
    )
  )
)
