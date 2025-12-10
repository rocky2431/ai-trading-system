/**
 * Factor Mining Page - 因子挖掘任务管理
 */

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { useMiningTasks, useFactorLibraryStats } from '@/hooks/useMining'
import type { MiningTaskStatus } from '@/api/factors'
import {
  Pickaxe,
  Play,
  Pause,
  Trash2,
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  TrendingUp,
  Target,
  Layers,
  BarChart3,
  RefreshCw,
} from 'lucide-react'

const FACTOR_FAMILIES = [
  { value: 'momentum', label: 'Momentum', color: 'bg-blue-500' },
  { value: 'volatility', label: 'Volatility', color: 'bg-purple-500' },
  { value: 'value', label: 'Value', color: 'bg-green-500' },
  { value: 'liquidity', label: 'Liquidity', color: 'bg-amber-500' },
  { value: 'sentiment', label: 'Sentiment', color: 'bg-rose-500' },
  { value: 'fundamental', label: 'Fundamental', color: 'bg-cyan-500' },
]

function StatusBadge({ status }: { status: MiningTaskStatus['status'] }) {
  const config = {
    pending: { label: 'Pending', variant: 'secondary' as const, icon: Clock },
    running: { label: 'Running', variant: 'default' as const, icon: Loader2 },
    completed: { label: 'Completed', variant: 'outline' as const, icon: CheckCircle2 },
    failed: { label: 'Failed', variant: 'destructive' as const, icon: XCircle },
    cancelled: { label: 'Cancelled', variant: 'secondary' as const, icon: Pause },
  }[status]

  const Icon = config.icon
  return (
    <Badge variant={config.variant} className="gap-1">
      <Icon className={`h-3 w-3 ${status === 'running' ? 'animate-spin' : ''}`} />
      {config.label}
    </Badge>
  )
}

function ProgressBar({ value, className = '' }: { value: number; className?: string }) {
  return (
    <div className={`h-2 bg-muted rounded-full overflow-hidden ${className}`}>
      <div
        className="h-full bg-primary transition-all duration-300"
        style={{ width: `${Math.min(value, 100)}%` }}
      />
    </div>
  )
}

function MiningTaskCard({
  task,
  onCancel,
}: {
  task: MiningTaskStatus
  onCancel: (id: string) => void
}) {
  const isActive = task.status === 'running' || task.status === 'pending'

  return (
    <Card className={isActive ? 'border-primary/50' : ''}>
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between">
          <div>
            <CardTitle className="text-base">{task.name}</CardTitle>
            <CardDescription className="text-xs mt-1">
              {task.description || 'No description'}
            </CardDescription>
          </div>
          <StatusBadge status={task.status} />
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Progress */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Progress</span>
            <span className="font-medium">{task.progress.toFixed(1)}%</span>
          </div>
          <ProgressBar value={task.progress} />
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-xl font-bold">{task.generated_count}</div>
            <div className="text-xs text-muted-foreground">Generated</div>
          </div>
          <div>
            <div className="text-xl font-bold text-emerald-500">{task.passed_count}</div>
            <div className="text-xs text-muted-foreground">Passed</div>
          </div>
          <div>
            <div className="text-xl font-bold text-rose-500">{task.failed_count}</div>
            <div className="text-xs text-muted-foreground">Failed</div>
          </div>
        </div>

        {/* Families */}
        <div className="flex flex-wrap gap-1">
          {task.factor_families.map((family) => (
            <Badge key={family} variant="outline" className="text-xs">
              {family}
            </Badge>
          ))}
        </div>

        {/* Error */}
        {task.error_message && (
          <div className="text-sm text-destructive bg-destructive/10 rounded p-2">
            {task.error_message}
          </div>
        )}

        {/* Actions */}
        {isActive && (
          <Button
            variant="destructive"
            size="sm"
            className="w-full"
            onClick={() => onCancel(task.id)}
          >
            <Pause className="h-4 w-4 mr-2" />
            Cancel Task
          </Button>
        )}

        {/* Timestamps */}
        <div className="text-xs text-muted-foreground space-y-1">
          <div>Created: {new Date(task.created_at).toLocaleString()}</div>
          {task.started_at && <div>Started: {new Date(task.started_at).toLocaleString()}</div>}
          {task.completed_at && <div>Completed: {new Date(task.completed_at).toLocaleString()}</div>}
        </div>
      </CardContent>
    </Card>
  )
}

export function FactorMiningPage() {
  const { tasks, loading, creating, createTask, cancelTask, refetch } = useMiningTasks()
  const { stats, loading: statsLoading } = useFactorLibraryStats()

  // Form state
  const [taskName, setTaskName] = useState('')
  const [taskDescription, setTaskDescription] = useState('')
  const [targetCount, setTargetCount] = useState(10)
  const [autoEvaluate, setAutoEvaluate] = useState(true)
  const [selectedFamilies, setSelectedFamilies] = useState<string[]>(['momentum', 'volatility'])

  const handleCreateTask = async () => {
    if (!taskName.trim()) return

    await createTask({
      name: taskName,
      description: taskDescription,
      factor_families: selectedFamilies,
      target_count: targetCount,
      auto_evaluate: autoEvaluate,
    })

    // Reset form
    setTaskName('')
    setTaskDescription('')
    setTargetCount(10)
    setSelectedFamilies(['momentum', 'volatility'])
  }

  const toggleFamily = (family: string) => {
    setSelectedFamilies((prev) =>
      prev.includes(family) ? prev.filter((f) => f !== family) : [...prev, family]
    )
  }

  const runningTasks = tasks.filter((t) => t.status === 'running' || t.status === 'pending')
  const completedTasks = tasks.filter(
    (t) => t.status === 'completed' || t.status === 'failed' || t.status === 'cancelled'
  )

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Factor Mining</h1>
          <p className="text-muted-foreground">
            Automated factor discovery and evaluation using AI agents
          </p>
        </div>
        <Button variant="outline" onClick={refetch}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Library Stats */}
      {stats && (
        <div className="grid gap-4 grid-cols-2 sm:grid-cols-4 lg:grid-cols-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Factors</CardTitle>
              <Layers className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.total_factors}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Core</CardTitle>
              <CheckCircle2 className="h-4 w-4 text-emerald-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-emerald-500">{stats.core_factors}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Candidate</CardTitle>
              <Clock className="h-4 w-4 text-amber-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-amber-500">{stats.candidate_factors}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Avg Sharpe</CardTitle>
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.avg_sharpe.toFixed(2)}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Avg IC</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{(stats.avg_ic * 100).toFixed(1)}%</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Best Sharpe</CardTitle>
              <Target className="h-4 w-4 text-primary" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-primary">{stats.best_sharpe.toFixed(2)}</div>
            </CardContent>
          </Card>
        </div>
      )}

      <Tabs defaultValue="create">
        <TabsList>
          <TabsTrigger value="create">Create Task</TabsTrigger>
          <TabsTrigger value="active">
            Active Tasks
            {runningTasks.length > 0 && (
              <Badge variant="secondary" className="ml-2">
                {runningTasks.length}
              </Badge>
            )}
          </TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
        </TabsList>

        {/* Create Task Tab */}
        <TabsContent value="create" className="space-y-6 mt-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Pickaxe className="h-5 w-5" />
                New Mining Task
              </CardTitle>
              <CardDescription>
                Configure and start a new factor mining session
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Task Name */}
              <div className="space-y-2">
                <Label htmlFor="task-name">Task Name *</Label>
                <Input
                  id="task-name"
                  placeholder="e.g., Momentum Factor Exploration v1"
                  value={taskName}
                  onChange={(e) => setTaskName(e.target.value)}
                />
              </div>

              {/* Description */}
              <div className="space-y-2">
                <Label htmlFor="task-desc">Description</Label>
                <Input
                  id="task-desc"
                  placeholder="Optional description for this mining task"
                  value={taskDescription}
                  onChange={(e) => setTaskDescription(e.target.value)}
                />
              </div>

              {/* Factor Families */}
              <div className="space-y-2">
                <Label>Factor Families</Label>
                <div className="flex flex-wrap gap-2">
                  {FACTOR_FAMILIES.map((family) => (
                    <Button
                      key={family.value}
                      type="button"
                      variant={selectedFamilies.includes(family.value) ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => toggleFamily(family.value)}
                    >
                      <span
                        className={`w-2 h-2 rounded-full mr-2 ${
                          selectedFamilies.includes(family.value) ? 'bg-white' : family.color
                        }`}
                      />
                      {family.label}
                    </Button>
                  ))}
                </div>
              </div>

              {/* Target Count */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label>Target Factor Count</Label>
                  <span className="text-2xl font-bold">{targetCount}</span>
                </div>
                <Slider
                  value={[targetCount]}
                  onValueChange={(v) => setTargetCount(v[0])}
                  min={1}
                  max={50}
                  step={1}
                />
                <p className="text-sm text-muted-foreground">
                  The mining task will generate up to {targetCount} factors
                </p>
              </div>

              {/* Auto Evaluate */}
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Auto Evaluate</Label>
                  <p className="text-sm text-muted-foreground">
                    Automatically evaluate generated factors against thresholds
                  </p>
                </div>
                <Switch checked={autoEvaluate} onCheckedChange={setAutoEvaluate} />
              </div>

              {/* Submit */}
              <Button
                className="w-full"
                size="lg"
                onClick={handleCreateTask}
                disabled={creating || !taskName.trim() || selectedFamilies.length === 0}
              >
                {creating ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Creating...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Start Mining Task
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Active Tasks Tab */}
        <TabsContent value="active" className="mt-6">
          {loading ? (
            <div className="flex items-center justify-center h-48">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
          ) : runningTasks.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Pickaxe className="h-12 w-12 text-muted-foreground mb-4" />
                <p className="text-lg font-medium">No active mining tasks</p>
                <p className="text-sm text-muted-foreground">
                  Create a new task to start mining factors
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {runningTasks.map((task) => (
                <MiningTaskCard key={task.id} task={task} onCancel={cancelTask} />
              ))}
            </div>
          )}
        </TabsContent>

        {/* History Tab */}
        <TabsContent value="history" className="mt-6">
          {loading ? (
            <div className="flex items-center justify-center h-48">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
          ) : completedTasks.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Clock className="h-12 w-12 text-muted-foreground mb-4" />
                <p className="text-lg font-medium">No completed tasks</p>
                <p className="text-sm text-muted-foreground">
                  Completed tasks will appear here
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {completedTasks.map((task) => (
                <MiningTaskCard key={task.id} task={task} onCancel={cancelTask} />
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}
