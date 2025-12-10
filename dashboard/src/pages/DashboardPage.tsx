/**
 * Dashboard Page - 系统概览和关键指标
 * 集成 Factor、Mining、Backtest、Agent 状态数据
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { useFactors } from '@/hooks/useFactors'
import { useMiningTasks, useFactorLibraryStats } from '@/hooks/useMining'
import { useBacktestStats, useBacktests } from '@/hooks/useBacktest'
import { useAgentStatus } from '@/hooks/useAgentStatus'
import {
  LineChart,
  Pickaxe,
  FlaskConical,
  TrendingUp,
  TrendingDown,
  Activity,
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  RefreshCw,
  ArrowRight,
} from 'lucide-react'
import { Link } from 'react-router-dom'

function StatCard({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  loading,
}: {
  title: string
  value: string | number
  subtitle?: string
  icon: React.ElementType
  trend?: 'up' | 'down' | 'neutral'
  loading?: boolean
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        {loading ? (
          <Loader2 className="h-6 w-6 animate-spin" />
        ) : (
          <>
            <div className="flex items-center gap-2">
              <span className="text-2xl font-bold">{value}</span>
              {trend === 'up' && <TrendingUp className="h-4 w-4 text-emerald-500" />}
              {trend === 'down' && <TrendingDown className="h-4 w-4 text-rose-500" />}
            </div>
            {subtitle && (
              <p className="text-xs text-muted-foreground">{subtitle}</p>
            )}
          </>
        )}
      </CardContent>
    </Card>
  )
}

function RecentFactorsCard() {
  const { factors, loading } = useFactors()

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle>Recent Factors</CardTitle>
          <CardDescription>Latest generated factors</CardDescription>
        </div>
        <Link to="/factors">
          <Button variant="ghost" size="sm">
            View All <ArrowRight className="h-4 w-4 ml-1" />
          </Button>
        </Link>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="flex justify-center py-4">
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
        ) : factors.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-4">
            No factors yet. Start mining!
          </p>
        ) : (
          <div className="space-y-3">
            {factors.slice(0, 5).map((factor) => (
              <div
                key={factor.id}
                className="flex items-center justify-between border-b pb-2 last:border-0"
              >
                <div>
                  <p className="font-medium text-sm">{factor.name}</p>
                  <p className="text-xs text-muted-foreground capitalize">
                    {factor.family}
                  </p>
                </div>
                <div className="text-right">
                  {factor.latestMetrics?.sharpe ? (
                    <Badge
                      variant={factor.latestMetrics.sharpe > 1.5 ? 'default' : 'secondary'}
                    >
                      SR: {factor.latestMetrics.sharpe.toFixed(2)}
                    </Badge>
                  ) : (
                    <Badge variant="outline">New</Badge>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function ActiveMiningCard() {
  const { tasks, loading } = useMiningTasks({ limit: 10 })
  const activeTasks = tasks.filter((t) => t.status === 'running' || t.status === 'pending')

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle>Mining Tasks</CardTitle>
          <CardDescription>Factor mining progress</CardDescription>
        </div>
        <Link to="/mining">
          <Button variant="ghost" size="sm">
            View All <ArrowRight className="h-4 w-4 ml-1" />
          </Button>
        </Link>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="flex justify-center py-4">
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
        ) : activeTasks.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-4">
            No active mining tasks
          </p>
        ) : (
          <div className="space-y-4">
            {activeTasks.slice(0, 3).map((task) => (
              <div key={task.id} className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">{task.name}</span>
                  <Badge
                    variant={task.status === 'running' ? 'default' : 'secondary'}
                  >
                    {task.status}
                  </Badge>
                </div>
                <Progress value={task.progress} className="h-2" />
                <p className="text-xs text-muted-foreground">
                  {task.generated_count || 0} generated • {task.progress}% complete
                </p>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function RecentBacktestsCard() {
  const { backtests, loading } = useBacktests({ page_size: 5 })

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle>Recent Backtests</CardTitle>
          <CardDescription>Latest backtest results</CardDescription>
        </div>
        <Link to="/backtest">
          <Button variant="ghost" size="sm">
            View All <ArrowRight className="h-4 w-4 ml-1" />
          </Button>
        </Link>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="flex justify-center py-4">
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
        ) : backtests.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-4">
            No backtests yet. Create a strategy first!
          </p>
        ) : (
          <div className="space-y-3">
            {backtests.slice(0, 5).map((backtest) => (
              <div
                key={backtest.id}
                className="flex items-center justify-between border-b pb-2 last:border-0"
              >
                <div>
                  <p className="font-medium text-sm">{backtest.name || backtest.strategy_name}</p>
                  <p className="text-xs text-muted-foreground">
                    {backtest.config.start_date} → {backtest.config.end_date}
                  </p>
                </div>
                <div className="text-right">
                  {backtest.status === 'completed' && backtest.metrics ? (
                    <div className="flex flex-col items-end gap-1">
                      <Badge
                        variant={backtest.metrics.sharpe_ratio > 1.5 ? 'default' : 'secondary'}
                      >
                        SR: {backtest.metrics.sharpe_ratio.toFixed(2)}
                      </Badge>
                      <span className="text-xs text-muted-foreground">
                        {(backtest.metrics.total_return * 100).toFixed(1)}% return
                      </span>
                    </div>
                  ) : backtest.status === 'running' ? (
                    <div className="flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span className="text-xs">{backtest.progress}%</span>
                    </div>
                  ) : backtest.status === 'failed' ? (
                    <Badge variant="destructive">Failed</Badge>
                  ) : (
                    <Badge variant="outline">{backtest.status}</Badge>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function AgentStatusCard() {
  const { status, loading } = useAgentStatus()

  if (loading || !status) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Agent Status</CardTitle>
          <CardDescription>AI agent system health</CardDescription>
        </CardHeader>
        <CardContent className="flex justify-center py-4">
          <Loader2 className="h-6 w-6 animate-spin" />
        </CardContent>
      </Card>
    )
  }

  const runningAgents = status.agents.filter((a) => a.status === 'running').length
  const healthColor = {
    healthy: 'text-emerald-500',
    degraded: 'text-amber-500',
    unhealthy: 'text-rose-500',
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle>Agent Status</CardTitle>
          <CardDescription>AI agent system health</CardDescription>
        </div>
        <Link to="/agents">
          <Button variant="ghost" size="sm">
            Monitor <ArrowRight className="h-4 w-4 ml-1" />
          </Button>
        </Link>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* System Health */}
          <div className="flex items-center justify-between">
            <span className="text-sm">System Health</span>
            <Badge
              variant={status.systemHealth === 'healthy' ? 'default' : 'destructive'}
              className={healthColor[status.systemHealth]}
            >
              {status.systemHealth.toUpperCase()}
            </Badge>
          </div>

          {/* Agents Summary */}
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center p-2 bg-muted rounded-md">
              <div className="text-xl font-bold text-emerald-500">{runningAgents}</div>
              <div className="text-xs text-muted-foreground">Running</div>
            </div>
            <div className="text-center p-2 bg-muted rounded-md">
              <div className="text-xl font-bold">{status.agents.length}</div>
              <div className="text-xs text-muted-foreground">Total</div>
            </div>
          </div>

          {/* Agent List */}
          <div className="space-y-2">
            {status.agents.slice(0, 5).map((agent) => (
              <div key={agent.id} className="flex items-center justify-between text-sm">
                <span className="truncate max-w-[150px]">{agent.name}</span>
                <div className="flex items-center gap-1">
                  {agent.status === 'running' ? (
                    <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                  ) : agent.status === 'error' ? (
                    <XCircle className="h-4 w-4 text-rose-500" />
                  ) : (
                    <Clock className="h-4 w-4 text-amber-500" />
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* LLM Stats */}
          <div className="border-t pt-3">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">LLM Requests</span>
              <span>{status.llmMetrics.totalRequests}</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Success Rate</span>
              <span>{status.llmMetrics.successRate.toFixed(1)}%</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export function DashboardPage() {
  const { factors, stats: factorStats, loading: factorsLoading } = useFactors()
  const { stats: libraryStats, loading: libraryLoading } = useFactorLibraryStats()
  const { tasks: miningTasks, loading: miningLoading } = useMiningTasks({ limit: 100 })
  const { stats: backtestStats, loading: backtestLoading } = useBacktestStats()

  const runningMiningTasks = miningTasks.filter((t) => t.status === 'running').length
  const completedMiningTasks = miningTasks.filter((t) => t.status === 'completed').length

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-muted-foreground">
            Welcome to IQFMP - Intelligent Quantitative Factor Mining Platform
          </p>
        </div>
        <Button variant="outline" onClick={() => window.location.reload()}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Top Stats */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Total Factors"
          value={libraryStats?.total_factors || factorStats?.total || 0}
          subtitle={libraryStats?.by_family ? `${libraryStats.by_family.momentum || 0} momentum` : undefined}
          icon={LineChart}
          loading={factorsLoading && libraryLoading}
        />

        <StatCard
          title="Mining Tasks"
          value={runningMiningTasks}
          subtitle={`${completedMiningTasks} completed`}
          icon={Pickaxe}
          loading={miningLoading}
        />

        <StatCard
          title="Best Sharpe"
          value={backtestStats?.best_sharpe?.toFixed(2) || '-'}
          subtitle={backtestStats?.avg_sharpe ? `Avg: ${backtestStats.avg_sharpe.toFixed(2)}` : undefined}
          icon={TrendingUp}
          trend={backtestStats?.best_sharpe && backtestStats.best_sharpe > 2 ? 'up' : 'neutral'}
          loading={backtestLoading}
        />

        <StatCard
          title="Total Backtests"
          value={backtestStats?.total_backtests || 0}
          subtitle={`${backtestStats?.running_backtests || 0} running`}
          icon={FlaskConical}
          loading={backtestLoading}
        />
      </div>

      {/* Factor Library Stats */}
      {libraryStats && libraryStats.by_family && Object.keys(libraryStats.by_family).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Factor Library Overview
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
              {Object.entries(libraryStats.by_family).map(([family, count]) => (
                <div
                  key={family}
                  className="text-center p-3 bg-muted rounded-lg"
                >
                  <div className="text-xl font-bold">{count}</div>
                  <div className="text-xs text-muted-foreground capitalize">{family}</div>
                </div>
              ))}
            </div>
            <div className="mt-4 flex items-center justify-between text-sm">
              <span className="text-muted-foreground">
                Average IC: {libraryStats.avg_ic?.toFixed(4) || 'N/A'}
              </span>
              <span className="text-muted-foreground">
                Average Sharpe: {libraryStats.avg_sharpe?.toFixed(2) || 'N/A'}
              </span>
              <span className="text-muted-foreground">
                Best Sharpe: {libraryStats.best_sharpe?.toFixed(2) || 'N/A'}
              </span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Main Content Grid */}
      <div className="grid gap-6 md:grid-cols-2">
        <RecentFactorsCard />
        <ActiveMiningCard />
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <RecentBacktestsCard />
        <AgentStatusCard />
      </div>
    </div>
  )
}
