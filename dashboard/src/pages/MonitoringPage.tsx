/**
 * Monitoring Page - IQFMP 监控大屏
 * 展示 LLM、因子、RD Loop、风险等关键指标
 */

import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Button } from '@/components/ui/button'
import { useMonitoring } from '@/hooks/useMonitoring'
import {
  Activity,
  Brain,
  LineChart,
  RefreshCw,
  Loader2,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Cpu,
  HardDrive,
  Database,
  Zap,
  Target,
  TrendingUp,
  Shield,
  Clock,
  BarChart3,
} from 'lucide-react'

function formatUptime(seconds: number): string {
  const days = Math.floor(seconds / 86400)
  const hours = Math.floor((seconds % 86400) / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  if (days > 0) return `${days}d ${hours}h ${minutes}m`
  if (hours > 0) return `${hours}h ${minutes}m`
  return `${minutes}m`
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`
}

function MetricCard({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  status,
}: {
  title: string
  value: string | number
  subtitle?: string
  icon: React.ElementType
  trend?: 'up' | 'down' | 'neutral'
  status?: 'success' | 'warning' | 'danger'
}) {
  const statusColors = {
    success: 'text-emerald-500',
    warning: 'text-amber-500',
    danger: 'text-rose-500',
  }

  return (
    <Card className="relative overflow-hidden">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <Icon className={`h-4 w-4 ${status ? statusColors[status] : 'text-muted-foreground'}`} />
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-2">
          <span className={`text-2xl font-bold ${status ? statusColors[status] : ''}`}>
            {value}
          </span>
          {trend === 'up' && <TrendingUp className="h-4 w-4 text-emerald-500" />}
          {trend === 'down' && <TrendingUp className="h-4 w-4 text-rose-500 rotate-180" />}
        </div>
        {subtitle && <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>}
      </CardContent>
    </Card>
  )
}

function GaugeCard({
  title,
  value,
  max,
  unit,
  thresholds,
  icon: Icon,
}: {
  title: string
  value: number
  max: number
  unit?: string
  thresholds: { warning: number; danger: number }
  icon: React.ElementType
}) {
  const percentage = (value / max) * 100
  const status =
    percentage >= thresholds.danger
      ? 'danger'
      : percentage >= thresholds.warning
        ? 'warning'
        : 'success'

  const statusColors = {
    success: 'bg-emerald-500',
    warning: 'bg-amber-500',
    danger: 'bg-rose-500',
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-2xl font-bold">
            {value.toFixed(1)}
            {unit}
          </span>
          <span className="text-sm text-muted-foreground">/ {max}{unit}</span>
        </div>
        <div className="h-2 bg-muted rounded-full overflow-hidden">
          <div
            className={`h-full transition-all ${statusColors[status]}`}
            style={{ width: `${Math.min(percentage, 100)}%` }}
          />
        </div>
      </CardContent>
    </Card>
  )
}

function LLMMetricsSection({
  metrics,
}: {
  metrics: NonNullable<ReturnType<typeof useMonitoring>['metrics']>['llm']
}) {
  return (
    <Card className="col-span-full lg:col-span-2">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5" />
          LLM Metrics
        </CardTitle>
        <CardDescription>Large Language Model usage statistics</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div className="text-center p-3 bg-muted rounded-lg">
            <div className="text-2xl font-bold">{metrics.totalRequests.toLocaleString()}</div>
            <div className="text-xs text-muted-foreground">Total Requests</div>
          </div>
          <div className="text-center p-3 bg-muted rounded-lg">
            <div className={`text-2xl font-bold ${metrics.successRate >= 95 ? 'text-emerald-500' : metrics.successRate >= 80 ? 'text-amber-500' : 'text-rose-500'}`}>
              {metrics.successRate.toFixed(1)}%
            </div>
            <div className="text-xs text-muted-foreground">Success Rate</div>
          </div>
          <div className="text-center p-3 bg-muted rounded-lg">
            <div className="text-2xl font-bold">{metrics.avgLatency.toFixed(0)}ms</div>
            <div className="text-xs text-muted-foreground">Avg Latency</div>
          </div>
          <div className="text-center p-3 bg-muted rounded-lg">
            <div className="text-2xl font-bold">{(metrics.tokensUsed / 1000).toFixed(1)}K</div>
            <div className="text-xs text-muted-foreground">Tokens Used</div>
          </div>
          <div className="text-center p-3 bg-muted rounded-lg">
            <div className="text-2xl font-bold">${metrics.costEstimate.toFixed(2)}</div>
            <div className="text-xs text-muted-foreground">Cost Estimate</div>
          </div>
          <div className="text-center p-3 bg-muted rounded-lg">
            <div className="text-2xl font-bold">{metrics.requestsPerMinute}</div>
            <div className="text-xs text-muted-foreground">Req/min</div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function FactorMetricsSection({
  metrics,
}: {
  metrics: NonNullable<ReturnType<typeof useMonitoring>['metrics']>['factors']
}) {
  return (
    <Card className="col-span-full lg:col-span-2">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <LineChart className="h-5 w-5" />
          Factor Metrics
        </CardTitle>
        <CardDescription>Factor generation and evaluation statistics</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div className="text-center p-3 bg-muted rounded-lg">
            <div className="text-2xl font-bold">{metrics.totalGenerated}</div>
            <div className="text-xs text-muted-foreground">Generated</div>
          </div>
          <div className="text-center p-3 bg-muted rounded-lg">
            <div className="text-2xl font-bold">{metrics.totalEvaluated}</div>
            <div className="text-xs text-muted-foreground">Evaluated</div>
          </div>
          <div className="text-center p-3 bg-muted rounded-lg">
            <div className={`text-2xl font-bold ${metrics.passRate >= 50 ? 'text-emerald-500' : 'text-amber-500'}`}>
              {(metrics.passRate * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-muted-foreground">Pass Rate</div>
          </div>
          <div className="text-center p-3 bg-muted rounded-lg">
            <div className="text-2xl font-bold">{metrics.avgIC.toFixed(4)}</div>
            <div className="text-xs text-muted-foreground">Avg IC</div>
          </div>
          <div className="text-center p-3 bg-muted rounded-lg">
            <div className="text-2xl font-bold">{metrics.avgSharpe.toFixed(2)}</div>
            <div className="text-xs text-muted-foreground">Avg Sharpe</div>
          </div>
          <div className="text-center p-3 bg-muted rounded-lg">
            <div className="text-2xl font-bold">{metrics.pendingEvaluation}</div>
            <div className="text-xs text-muted-foreground">Pending</div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function RDLoopSection({
  metrics,
}: {
  metrics: NonNullable<ReturnType<typeof useMonitoring>['metrics']>['rdLoop']
}) {
  const progress = (metrics.currentIteration / metrics.totalIterations) * 100

  return (
    <Card className="col-span-full lg:col-span-2">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Target className="h-5 w-5" />
          RD Loop Status
          {metrics.isRunning ? (
            <Badge variant="default" className="ml-2">
              Running
            </Badge>
          ) : (
            <Badge variant="secondary" className="ml-2">
              Idle
            </Badge>
          )}
        </CardTitle>
        <CardDescription>Research & Development Loop progress</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span>Progress</span>
            <span className="font-medium">
              {metrics.currentIteration} / {metrics.totalIterations} iterations
            </span>
          </div>
          <Progress value={progress} className="h-2" />
        </div>
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center p-3 bg-muted rounded-lg">
            <div className="text-xl font-bold">{metrics.coreFactorsCount}</div>
            <div className="text-xs text-muted-foreground">Core Factors</div>
          </div>
          <div className="text-center p-3 bg-muted rounded-lg">
            <div className="text-xl font-bold">{metrics.hypothesesTested}</div>
            <div className="text-xs text-muted-foreground">Hypotheses Tested</div>
          </div>
          <div className="text-center p-3 bg-muted rounded-lg">
            <div className="text-sm font-medium capitalize">{metrics.currentPhase}</div>
            <div className="text-xs text-muted-foreground">Current Phase</div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function RiskSection({
  metrics,
}: {
  metrics: NonNullable<ReturnType<typeof useMonitoring>['metrics']>['risk']
}) {
  const riskIcon =
    metrics.riskLevel === 'safe'
      ? CheckCircle2
      : metrics.riskLevel === 'warning'
        ? AlertTriangle
        : XCircle

  const riskColor =
    metrics.riskLevel === 'safe'
      ? 'text-emerald-500'
      : metrics.riskLevel === 'warning'
        ? 'text-amber-500'
        : 'text-rose-500'

  return (
    <Card className="col-span-full lg:col-span-2">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Shield className="h-5 w-5" />
          Risk Status
          <Badge
            variant={
              metrics.riskLevel === 'safe'
                ? 'default'
                : metrics.riskLevel === 'warning'
                  ? 'secondary'
                  : 'destructive'
            }
            className="ml-2"
          >
            {metrics.riskLevel.toUpperCase()}
          </Badge>
        </CardTitle>
        <CardDescription>Real-time risk monitoring with hard thresholds</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          {/* Drawdown */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span>Drawdown</span>
              <span className="font-medium">
                {(metrics.currentDrawdown * 100).toFixed(2)}% / {(metrics.maxDrawdownThreshold * 100).toFixed(0)}%
              </span>
            </div>
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <div
                className={`h-full transition-all ${
                  metrics.currentDrawdown >= metrics.maxDrawdownThreshold
                    ? 'bg-rose-500'
                    : metrics.currentDrawdown >= metrics.maxDrawdownThreshold * 0.7
                      ? 'bg-amber-500'
                      : 'bg-emerald-500'
                }`}
                style={{
                  width: `${Math.min((metrics.currentDrawdown / metrics.maxDrawdownThreshold) * 100, 100)}%`,
                }}
              />
            </div>
          </div>
          {/* Leverage */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span>Leverage</span>
              <span className="font-medium">
                {metrics.currentLeverage.toFixed(2)}x / {metrics.maxLeverage}x
              </span>
            </div>
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <div
                className={`h-full transition-all ${
                  metrics.currentLeverage >= metrics.maxLeverage
                    ? 'bg-rose-500'
                    : metrics.currentLeverage >= metrics.maxLeverage * 0.7
                      ? 'bg-amber-500'
                      : 'bg-emerald-500'
                }`}
                style={{
                  width: `${Math.min((metrics.currentLeverage / metrics.maxLeverage) * 100, 100)}%`,
                }}
              />
            </div>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center p-3 bg-muted rounded-lg">
            <div className="text-xl font-bold">
              {(metrics.positionConcentration * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-muted-foreground">Position Concentration</div>
          </div>
          <div className="text-center p-3 bg-muted rounded-lg">
            <div className={`text-xl font-bold ${metrics.violationsCount > 0 ? 'text-rose-500' : 'text-emerald-500'}`}>
              {metrics.violationsCount}
            </div>
            <div className="text-xs text-muted-foreground">Violations</div>
          </div>
          <div className="text-center p-3 bg-muted rounded-lg flex items-center justify-center">
            {React.createElement(riskIcon, { className: `h-8 w-8 ${riskColor}` })}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function VectorStoreSection({
  metrics,
}: {
  metrics: NonNullable<ReturnType<typeof useMonitoring>['metrics']>['vectorStore']
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Database className="h-5 w-5" />
          Vector Store
        </CardTitle>
        <CardDescription>Factor similarity and deduplication</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center p-3 bg-muted rounded-lg">
            <div className="text-xl font-bold">{metrics.totalVectors}</div>
            <div className="text-xs text-muted-foreground">Total Vectors</div>
          </div>
          <div className="text-center p-3 bg-muted rounded-lg">
            <div className="text-xl font-bold">{metrics.searchLatency.toFixed(0)}ms</div>
            <div className="text-xs text-muted-foreground">Search Latency</div>
          </div>
          <div className="text-center p-3 bg-muted rounded-lg">
            <div className="text-xl font-bold">{metrics.similarityChecks}</div>
            <div className="text-xs text-muted-foreground">Checks</div>
          </div>
          <div className="text-center p-3 bg-muted rounded-lg">
            <div className="text-xl font-bold">{metrics.duplicatesFound}</div>
            <div className="text-xs text-muted-foreground">Duplicates</div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function SystemResourcesSection({
  metrics,
}: {
  metrics: NonNullable<ReturnType<typeof useMonitoring>['metrics']>['resources']
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-5 w-5" />
          System Resources
        </CardTitle>
        <CardDescription>CPU, Memory, and Disk usage</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* CPU */}
        <div className="space-y-1">
          <div className="flex items-center justify-between text-sm">
            <span className="flex items-center gap-2">
              <Cpu className="h-4 w-4" /> CPU
            </span>
            <span className="font-medium">{metrics.cpuUsage.toFixed(1)}%</span>
          </div>
          <div className="h-2 bg-muted rounded-full overflow-hidden">
            <div
              className={`h-full transition-all ${
                metrics.cpuUsage >= 90
                  ? 'bg-rose-500'
                  : metrics.cpuUsage >= 70
                    ? 'bg-amber-500'
                    : 'bg-emerald-500'
              }`}
              style={{ width: `${metrics.cpuUsage}%` }}
            />
          </div>
        </div>
        {/* Memory */}
        <div className="space-y-1">
          <div className="flex items-center justify-between text-sm">
            <span className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" /> Memory
            </span>
            <span className="font-medium">
              {formatBytes(metrics.memoryUsed)} / {formatBytes(metrics.memoryTotal)}
            </span>
          </div>
          <div className="h-2 bg-muted rounded-full overflow-hidden">
            <div
              className={`h-full transition-all ${
                metrics.memoryUsage >= 90
                  ? 'bg-rose-500'
                  : metrics.memoryUsage >= 70
                    ? 'bg-amber-500'
                    : 'bg-emerald-500'
              }`}
              style={{ width: `${metrics.memoryUsage}%` }}
            />
          </div>
        </div>
        {/* Disk */}
        <div className="space-y-1">
          <div className="flex items-center justify-between text-sm">
            <span className="flex items-center gap-2">
              <HardDrive className="h-4 w-4" /> Disk
            </span>
            <span className="font-medium">{metrics.diskUsage.toFixed(1)}%</span>
          </div>
          <div className="h-2 bg-muted rounded-full overflow-hidden">
            <div
              className={`h-full transition-all ${
                metrics.diskUsage >= 90
                  ? 'bg-rose-500'
                  : metrics.diskUsage >= 70
                    ? 'bg-amber-500'
                    : 'bg-emerald-500'
              }`}
              style={{ width: `${metrics.diskUsage}%` }}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export function MonitoringPage() {
  const { metrics, loading, error, refresh } = useMonitoring(3000) // 3s refresh

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-muted-foreground">Loading monitoring data...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <Card className="max-w-md">
          <CardHeader>
            <CardTitle className="text-destructive">Error</CardTitle>
            <CardDescription>{error.message}</CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={refresh}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (!metrics) return null

  const healthConfig = {
    healthy: { label: 'Healthy', color: 'text-emerald-500', bg: 'bg-emerald-500' },
    degraded: { label: 'Degraded', color: 'text-amber-500', bg: 'bg-amber-500' },
    unhealthy: { label: 'Unhealthy', color: 'text-rose-500', bg: 'bg-rose-500' },
  }
  const health = healthConfig[metrics.systemHealth]

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Monitoring Dashboard</h1>
          <p className="text-muted-foreground">
            Real-time IQFMP system metrics and health status
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className={`h-3 w-3 rounded-full ${health.bg} animate-pulse`} />
            <span className={`font-medium ${health.color}`}>{health.label}</span>
          </div>
          <div className="text-sm text-muted-foreground flex items-center gap-1">
            <Clock className="h-4 w-4" />
            Uptime: {formatUptime(metrics.uptime)}
          </div>
          <Button variant="outline" size="sm" onClick={refresh}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Top Stats */}
      <div className="grid gap-4 grid-cols-2 md:grid-cols-4">
        <MetricCard
          title="LLM Requests"
          value={metrics.llm.totalRequests.toLocaleString()}
          subtitle={`${metrics.llm.successRate.toFixed(1)}% success`}
          icon={Brain}
          status={metrics.llm.successRate >= 95 ? 'success' : metrics.llm.successRate >= 80 ? 'warning' : 'danger'}
        />
        <MetricCard
          title="Factors Generated"
          value={metrics.factors.totalGenerated}
          subtitle={`${metrics.factors.pendingEvaluation} pending`}
          icon={LineChart}
        />
        <MetricCard
          title="RD Loop Progress"
          value={`${metrics.rdLoop.currentIteration}/${metrics.rdLoop.totalIterations}`}
          subtitle={metrics.rdLoop.isRunning ? 'Running' : 'Idle'}
          icon={Target}
          status={metrics.rdLoop.isRunning ? 'success' : undefined}
        />
        <MetricCard
          title="Risk Status"
          value={metrics.risk.riskLevel.toUpperCase()}
          subtitle={`${metrics.risk.violationsCount} violations`}
          icon={Shield}
          status={metrics.risk.riskLevel === 'safe' ? 'success' : metrics.risk.riskLevel === 'warning' ? 'warning' : 'danger'}
        />
      </div>

      {/* Main Sections */}
      <div className="grid gap-6 grid-cols-1 lg:grid-cols-4">
        <LLMMetricsSection metrics={metrics.llm} />
        <FactorMetricsSection metrics={metrics.factors} />
      </div>

      <div className="grid gap-6 grid-cols-1 lg:grid-cols-4">
        <RDLoopSection metrics={metrics.rdLoop} />
        <RiskSection metrics={metrics.risk} />
      </div>

      <div className="grid gap-6 grid-cols-1 md:grid-cols-2">
        <VectorStoreSection metrics={metrics.vectorStore} />
        <SystemResourcesSection metrics={metrics.resources} />
      </div>
    </div>
  )
}
