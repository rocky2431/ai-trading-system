/**
 * Agent 状态卡片组件
 * 展示单个 Agent 的运行状态、当前任务和进度
 * 点击可打开配置对话框
 */

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import type { Agent, AgentStatus } from '@/types/agent'
import { Bot, Pause, Play, AlertCircle, Clock, Settings } from 'lucide-react'

interface AgentStatusCardProps {
  agent: Agent
  onClick?: () => void
}

const statusConfig: Record<AgentStatus, { label: string; variant: 'default' | 'secondary' | 'destructive' | 'success' | 'warning' }> = {
  idle: { label: 'Idle', variant: 'secondary' },
  running: { label: 'Running', variant: 'success' },
  paused: { label: 'Paused', variant: 'warning' },
  error: { label: 'Error', variant: 'destructive' },
}

const typeLabels: Record<Agent['type'], string> = {
  factor_generation: 'Factor Generation',
  evaluation: 'Evaluation',
  strategy: 'Strategy',
  backtest: 'Backtest',
}

function formatRelativeTime(dateString: string | null): string {
  if (!dateString) return 'N/A'
  const date = new Date(dateString)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffSec = Math.floor(diffMs / 1000)
  const diffMin = Math.floor(diffSec / 60)
  const diffHour = Math.floor(diffMin / 60)

  if (diffSec < 60) return 'just now'
  if (diffMin < 60) return `${diffMin}m ago`
  if (diffHour < 24) return `${diffHour}h ago`
  return `${Math.floor(diffHour / 24)}d ago`
}

export function AgentStatusCard({ agent, onClick }: AgentStatusCardProps) {
  const config = statusConfig[agent.status]

  return (
    <Card
      className={onClick ? 'cursor-pointer hover:shadow-md hover:border-blue-300 dark:hover:border-blue-600 transition-all' : ''}
      onClick={onClick}
    >
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Bot className="h-4 w-4" />
          {agent.name}
        </CardTitle>
        <div className="flex items-center gap-2">
          {onClick && (
            <Settings className="h-4 w-4 text-gray-400 hover:text-blue-500 transition-colors" />
          )}
          <Badge variant={config.variant}>{config.label}</Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {/* Agent Type */}
          <div className="text-xs text-muted-foreground">
            {typeLabels[agent.type]}
          </div>

          {/* Current Task */}
          {agent.currentTask ? (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm">
                {agent.status === 'running' && <Play className="h-3 w-3 text-emerald-500" />}
                {agent.status === 'paused' && <Pause className="h-3 w-3 text-amber-500" />}
                {agent.status === 'error' && <AlertCircle className="h-3 w-3 text-red-500" />}
                <span className="truncate">{agent.currentTask}</span>
              </div>
              <div className="flex items-center gap-2">
                <Progress value={agent.progress} className="flex-1" />
                <span className="text-xs text-muted-foreground w-10 text-right">
                  {Math.round(agent.progress)}%
                </span>
              </div>
            </div>
          ) : (
            <div className="text-sm text-muted-foreground">No active task</div>
          )}

          {/* Last Activity */}
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <div className="flex items-center gap-1">
              <Clock className="h-3 w-3" />
              Last activity: {formatRelativeTime(agent.lastActivity)}
            </div>
            {onClick && (
              <span className="text-blue-500 hover:underline">Configure</span>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
