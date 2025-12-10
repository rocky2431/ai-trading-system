/**
 * Agent 状态监控页面
 * 展示 Agent 运行状态、任务队列、LLM 指标和资源使用
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { AgentStatusCard } from '@/components/agents/AgentStatusCard'
import { TaskQueueCard } from '@/components/agents/TaskQueueCard'
import { LLMMetricsCard } from '@/components/agents/LLMMetricsCard'
import { ResourceUsageCard } from '@/components/agents/ResourceUsageCard'
import { useAgentStatus } from '@/hooks/useAgentStatus'
import { Bot, RefreshCw, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'

function formatUptime(seconds: number): string {
  const days = Math.floor(seconds / 86400)
  const hours = Math.floor((seconds % 86400) / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)

  if (days > 0) return `${days}d ${hours}h ${minutes}m`
  if (hours > 0) return `${hours}h ${minutes}m`
  return `${minutes}m`
}

const healthConfig = {
  healthy: { label: 'Healthy', variant: 'success' as const },
  degraded: { label: 'Degraded', variant: 'warning' as const },
  unhealthy: { label: 'Unhealthy', variant: 'destructive' as const },
}

export function AgentMonitorPage() {
  const { status, loading, error } = useAgentStatus()

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-muted-foreground">Loading agent status...</p>
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
            <Button onClick={() => window.location.reload()}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (!status) return null

  const health = healthConfig[status.systemHealth]
  const runningAgents = status.agents.filter(a => a.status === 'running').length

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Agent Monitor</h1>
          <p className="text-muted-foreground">
            Real-time monitoring of AI agents and system resources
          </p>
        </div>
        <div className="flex items-center gap-4">
          <Badge variant={health.variant} className="text-sm px-3 py-1">
            {health.label}
          </Badge>
          <div className="text-sm text-muted-foreground">
            Uptime: {formatUptime(status.uptime)}
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Agents</CardTitle>
            <Bot className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{runningAgents}</div>
            <p className="text-xs text-muted-foreground">
              of {status.agents.length} total agents
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Queue Size</CardTitle>
            <RefreshCw className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {status.taskQueue.filter(t => t.status === 'pending').length}
            </div>
            <p className="text-xs text-muted-foreground">
              {status.taskQueue.filter(t => t.status === 'running').length} running
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">LLM Requests</CardTitle>
            <RefreshCw className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{status.llmMetrics.totalRequests}</div>
            <p className="text-xs text-muted-foreground">
              {status.llmMetrics.successRate.toFixed(1)}% success rate
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">CPU Usage</CardTitle>
            <RefreshCw className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {status.resources.cpu.usage.toFixed(1)}%
            </div>
            <p className="text-xs text-muted-foreground">
              Memory: {status.resources.memory.percentage.toFixed(1)}%
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Agent Status Grid */}
      <div>
        <h2 className="text-lg font-semibold mb-4">Agent Status</h2>
        <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
          {status.agents.map((agent) => (
            <AgentStatusCard key={agent.id} agent={agent} />
          ))}
        </div>
      </div>

      {/* Bottom Section: Task Queue, LLM Metrics, Resources */}
      <div className="grid gap-6 grid-cols-1 lg:grid-cols-3">
        <TaskQueueCard tasks={status.taskQueue} />
        <LLMMetricsCard metrics={status.llmMetrics} />
        <ResourceUsageCard resources={status.resources} />
      </div>
    </div>
  )
}
