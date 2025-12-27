import { Outlet, Link, useLocation } from 'react-router-dom'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { useAppStore } from '@/store/useAppStore'
import {
  LayoutDashboard,
  LineChart,
  Activity,
  Settings,
  Menu,
  Moon,
  Sun,
  Bot,
  Beaker,
  Zap,
  Database,
  Pickaxe,
  Wrench,
  FlaskConical,
  Gauge,
  ClipboardCheck,
  Brain,
  FileCode,
  GitBranch,
} from 'lucide-react'

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Monitoring', href: '/monitoring', icon: Gauge },
  { name: 'Data Center', href: '/data', icon: Database },
  { name: 'Agents', href: '/agents', icon: Bot },
  { name: 'Factor Mining', href: '/mining', icon: Pickaxe },
  { name: 'Factors', href: '/factors', icon: LineChart },
  { name: 'Strategies', href: '/strategies', icon: Wrench },
  { name: 'Backtest', href: '/backtest', icon: FlaskConical },
  { name: 'Research', href: '/research', icon: Beaker },
  { name: 'Review', href: '/review', icon: ClipboardCheck },
  { name: 'RL Training', href: '/rl-training', icon: Brain },
  { name: 'Prompts', href: '/prompts', icon: FileCode },
  { name: 'Checkpoints', href: '/checkpoints', icon: GitBranch },
  // DISABLED: Live trading uses mock data, backend API not implemented
  // { name: 'Trading', href: '/trading', icon: Zap },
  { name: 'Pipeline', href: '/pipeline', icon: Activity },
  { name: 'Settings', href: '/settings', icon: Settings },
]

export function DashboardLayout() {
  const location = useLocation()
  const { theme, toggleTheme, sidebarCollapsed, toggleSidebar } = useAppStore()

  return (
    <div className={cn('min-h-screen', theme === 'dark' && 'dark')}>
      <div className="flex h-screen bg-background">
        {/* Sidebar */}
        <aside
          className={cn(
            'flex flex-col border-r bg-card transition-all duration-300',
            sidebarCollapsed ? 'w-16' : 'w-64'
          )}
        >
          {/* Logo */}
          <div className="flex h-14 items-center border-b px-4">
            {!sidebarCollapsed && (
              <span className="font-semibold text-lg">IQFMP</span>
            )}
          </div>

          {/* Navigation */}
          <nav className="flex-1 space-y-1 p-2">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={cn(
                    'flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                    isActive
                      ? 'bg-primary text-primary-foreground'
                      : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                  )}
                >
                  <item.icon className="h-5 w-5" />
                  {!sidebarCollapsed && <span>{item.name}</span>}
                </Link>
              )
            })}
          </nav>
        </aside>

        {/* Main content */}
        <div className="flex flex-1 flex-col overflow-hidden">
          {/* Header */}
          <header className="flex h-14 items-center justify-between border-b bg-card px-4">
            <Button variant="ghost" size="icon" onClick={toggleSidebar}>
              <Menu className="h-5 w-5" />
            </Button>

            <div className="flex items-center gap-2">
              <Button variant="ghost" size="icon" onClick={toggleTheme}>
                {theme === 'light' ? (
                  <Moon className="h-5 w-5" />
                ) : (
                  <Sun className="h-5 w-5" />
                )}
              </Button>
            </div>
          </header>

          {/* Page content */}
          <main className="flex-1 overflow-auto p-6">
            <Outlet />
          </main>
        </div>
      </div>
    </div>
  )
}
