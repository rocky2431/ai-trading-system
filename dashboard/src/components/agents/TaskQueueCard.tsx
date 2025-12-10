/**
 * 任务队列可视化组件
 * 展示待执行、执行中、已完成的任务列表
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { TaskQueueItem } from '@/types/agent'
import { ListTodo, Clock, CheckCircle2, XCircle, Loader2 } from 'lucide-react'

interface TaskQueueCardProps {
  tasks: TaskQueueItem[]
}

const statusConfig: Record<TaskQueueItem['status'], {
  label: string
  variant: 'default' | 'secondary' | 'destructive' | 'success' | 'warning'
  icon: typeof Clock
}> = {
  pending: { label: 'Pending', variant: 'secondary', icon: Clock },
  running: { label: 'Running', variant: 'default', icon: Loader2 },
  completed: { label: 'Completed', variant: 'success', icon: CheckCircle2 },
  failed: { label: 'Failed', variant: 'destructive', icon: XCircle },
}

const priorityConfig: Record<TaskQueueItem['priority'], { label: string; className: string }> = {
  low: { label: 'Low', className: 'text-slate-500' },
  normal: { label: 'Normal', className: 'text-blue-500' },
  high: { label: 'High', className: 'text-amber-500' },
}

const typeLabels: Record<string, string> = {
  factor_generation: 'Factor Gen',
  evaluation: 'Evaluation',
  strategy: 'Strategy',
  backtest: 'Backtest',
}

function formatTime(dateString: string): string {
  const date = new Date(dateString)
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false
  })
}

export function TaskQueueCard({ tasks }: TaskQueueCardProps) {
  const pendingCount = tasks.filter(t => t.status === 'pending').length
  const runningCount = tasks.filter(t => t.status === 'running').length

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ListTodo className="h-5 w-5" />
          Task Queue
        </CardTitle>
        <CardDescription>
          {runningCount} running, {pendingCount} pending
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {tasks.length === 0 ? (
            <div className="text-sm text-muted-foreground text-center py-4">
              No tasks in queue
            </div>
          ) : (
            tasks.slice(0, 5).map((task) => {
              const status = statusConfig[task.status]
              const priority = priorityConfig[task.priority]
              const StatusIcon = status.icon

              return (
                <div
                  key={task.id}
                  className="flex items-center justify-between p-2 rounded-md bg-muted/50"
                >
                  <div className="flex items-center gap-3">
                    <StatusIcon
                      className={`h-4 w-4 ${
                        task.status === 'running' ? 'animate-spin text-primary' :
                        task.status === 'completed' ? 'text-emerald-500' :
                        task.status === 'failed' ? 'text-red-500' :
                        'text-muted-foreground'
                      }`}
                    />
                    <div>
                      <div className="text-sm font-medium">
                        {typeLabels[task.type] || task.type}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Created: {formatTime(task.createdAt)}
                        {task.startedAt && ` | Started: ${formatTime(task.startedAt)}`}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`text-xs font-medium ${priority.className}`}>
                      {priority.label}
                    </span>
                    <Badge variant={status.variant} className="text-xs">
                      {status.label}
                    </Badge>
                  </div>
                </div>
              )
            })
          )}
          {tasks.length > 5 && (
            <div className="text-xs text-muted-foreground text-center">
              +{tasks.length - 5} more tasks
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
