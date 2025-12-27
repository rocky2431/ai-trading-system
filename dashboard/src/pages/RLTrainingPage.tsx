/**
 * RL Training Page - RL 强化学习训练页面
 *
 * 功能:
 * - 提交 RL 训练任务
 * - 查看训练任务状态
 * - 管理训练模型
 * - 运行模型回测
 */

import { useState } from 'react'
import {
  Brain,
  Play,
  Square,
  RefreshCw,
  CheckCircle,
  XCircle,
  Clock,
  Loader2,
  FileText,
  TrendingUp,
  Settings,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { useRLTraining } from '@/hooks/useRLTraining'
import { RLTaskResponse, RLTaskStatus, RLModelInfo } from '@/api/rl'

const statusConfig: Record<
  RLTaskStatus,
  { label: string; variant: 'default' | 'secondary' | 'destructive' | 'outline'; icon: typeof Clock }
> = {
  pending: { label: '等待中', variant: 'secondary', icon: Clock },
  started: { label: '已启动', variant: 'default', icon: Play },
  running: { label: '运行中', variant: 'default', icon: Loader2 },
  success: { label: '成功', variant: 'outline', icon: CheckCircle },
  failed: { label: '失败', variant: 'destructive', icon: XCircle },
  revoked: { label: '已取消', variant: 'secondary', icon: Square },
}

export function RLTrainingPage() {
  const {
    tasks,
    models,
    stats,
    page,
    setPage,
    total,
    hasNext,
    taskType,
    setTaskType,
    statusFilter,
    setStatusFilter,
    loading,
    error,
    refresh,
    submitTraining,
    submitBacktest,
    cancelTask,
  } = useRLTraining()

  const [trainingDialogOpen, setTrainingDialogOpen] = useState(false)
  const [backtestDialogOpen, setBacktestDialogOpen] = useState(false)
  const [detailsDialogOpen, setDetailsDialogOpen] = useState(false)
  const [selectedTask, setSelectedTask] = useState<RLTaskResponse | null>(null)
  const [selectedModel, setSelectedModel] = useState<RLModelInfo | null>(null)

  // 训练表单状态
  const [trainDataPath, setTrainDataPath] = useState('')
  const [testDataPath, setTestDataPath] = useState('')
  const [trainingName, setTrainingName] = useState('')
  const [timesteps, setTimesteps] = useState('10000')
  const [learningRate, setLearningRate] = useState('0.0003')

  // 回测表单状态
  const [backtestDataPath, setBacktestDataPath] = useState('')
  const [backtestName, setBacktestName] = useState('')

  const handleSubmitTraining = async () => {
    try {
      await submitTraining({
        train_data_path: trainDataPath,
        test_data_path: testDataPath,
        name: trainingName || undefined,
        config: {
          total_timesteps: parseInt(timesteps),
          learning_rate: parseFloat(learningRate),
          save_model: true,
        },
      })
      setTrainingDialogOpen(false)
      // 重置表单
      setTrainDataPath('')
      setTestDataPath('')
      setTrainingName('')
    } catch (err) {
      console.error('Failed to submit training:', err)
    }
  }

  const handleSubmitBacktest = async () => {
    if (!selectedModel) return

    try {
      await submitBacktest({
        model_path: selectedModel.path,
        data_path: backtestDataPath,
        name: backtestName || undefined,
      })
      setBacktestDialogOpen(false)
      setBacktestDataPath('')
      setBacktestName('')
      setSelectedModel(null)
    } catch (err) {
      console.error('Failed to submit backtest:', err)
    }
  }

  const formatDuration = (seconds: number | null) => {
    if (!seconds) return '-'
    if (seconds < 60) return `${Math.round(seconds)}秒`
    if (seconds < 3600) return `${Math.round(seconds / 60)}分钟`
    return `${(seconds / 3600).toFixed(1)}小时`
  }

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return '-'
    return new Date(dateStr).toLocaleString('zh-CN')
  }

  if (loading && tasks.length === 0) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4">
        <XCircle className="h-12 w-12 text-destructive" />
        <p className="text-lg text-muted-foreground">{error.message}</p>
        <Button onClick={refresh}>重试</Button>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* 页面标题 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">RL 强化学习训练</h1>
          <p className="text-muted-foreground">
            使用 Qlib 进行订单执行优化的 PPO 强化学习训练
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={refresh}>
            <RefreshCw className={cn('mr-2 h-4 w-4', loading && 'animate-spin')} />
            刷新
          </Button>
          <Button onClick={() => setTrainingDialogOpen(true)}>
            <Brain className="mr-2 h-4 w-4" />
            新建训练
          </Button>
        </div>
      </div>

      {/* 统计卡片 */}
      {stats && (
        <div className="grid gap-4 md:grid-cols-5">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                总训练任务
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.total_training_jobs}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                成功
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">{stats.successful_jobs}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                失败
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-600">{stats.failed_jobs}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                运行中
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-600">{stats.running_jobs}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                平均训练时间
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {formatDuration(stats.average_training_time_seconds)}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* 标签页 */}
      <Tabs defaultValue="tasks">
        <TabsList>
          <TabsTrigger value="tasks">任务列表</TabsTrigger>
          <TabsTrigger value="models">模型管理</TabsTrigger>
        </TabsList>

        {/* 任务列表 */}
        <TabsContent value="tasks" className="space-y-4">
          {/* 过滤器 */}
          <div className="flex gap-4">
            <Select
              value={taskType || 'all'}
              onValueChange={(v) => setTaskType(v === 'all' ? undefined : (v as 'training' | 'backtest'))}
            >
              <SelectTrigger className="w-[150px]">
                <SelectValue placeholder="任务类型" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">全部类型</SelectItem>
                <SelectItem value="training">训练</SelectItem>
                <SelectItem value="backtest">回测</SelectItem>
              </SelectContent>
            </Select>

            <Select
              value={statusFilter || 'all'}
              onValueChange={(v) => setStatusFilter(v === 'all' ? undefined : (v as RLTaskStatus))}
            >
              <SelectTrigger className="w-[150px]">
                <SelectValue placeholder="状态" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">全部状态</SelectItem>
                <SelectItem value="pending">等待中</SelectItem>
                <SelectItem value="running">运行中</SelectItem>
                <SelectItem value="success">成功</SelectItem>
                <SelectItem value="failed">失败</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* 任务表格 */}
          <Card>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>任务 ID</TableHead>
                  <TableHead>名称</TableHead>
                  <TableHead>类型</TableHead>
                  <TableHead>状态</TableHead>
                  <TableHead>创建时间</TableHead>
                  <TableHead>完成时间</TableHead>
                  <TableHead className="text-right">操作</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {tasks.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={7} className="text-center text-muted-foreground">
                      暂无任务
                    </TableCell>
                  </TableRow>
                ) : (
                  tasks.map((task) => {
                    const config = statusConfig[task.status]
                    const StatusIcon = config.icon
                    return (
                      <TableRow key={task.task_id}>
                        <TableCell className="font-mono text-sm">
                          {task.task_id.slice(0, 16)}...
                        </TableCell>
                        <TableCell>{task.name || '-'}</TableCell>
                        <TableCell>
                          <Badge variant="outline">
                            {task.task_type === 'training' ? '训练' : '回测'}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <Badge variant={config.variant} className="gap-1">
                            <StatusIcon
                              className={cn(
                                'h-3 w-3',
                                task.status === 'running' && 'animate-spin'
                              )}
                            />
                            {config.label}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-sm text-muted-foreground">
                          {formatDate(task.created_at)}
                        </TableCell>
                        <TableCell className="text-sm text-muted-foreground">
                          {formatDate(task.completed_at)}
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex justify-end gap-2">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => {
                                setSelectedTask(task)
                                setDetailsDialogOpen(true)
                              }}
                            >
                              <FileText className="h-4 w-4" />
                            </Button>
                            {(task.status === 'started' || task.status === 'running') && (
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => cancelTask(task.task_id)}
                              >
                                <Square className="h-4 w-4" />
                              </Button>
                            )}
                          </div>
                        </TableCell>
                      </TableRow>
                    )
                  })
                )}
              </TableBody>
            </Table>
          </Card>

          {/* 分页 */}
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">
              共 {total} 条记录
            </p>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={page === 1}
                onClick={() => setPage(page - 1)}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <span className="flex items-center px-2 text-sm">
                第 {page} 页
              </span>
              <Button
                variant="outline"
                size="sm"
                disabled={!hasNext}
                onClick={() => setPage(page + 1)}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </TabsContent>

        {/* 模型管理 */}
        <TabsContent value="models" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>已训练模型</CardTitle>
              <CardDescription>
                成功训练完成的 RL 策略模型，可用于回测评估
              </CardDescription>
            </CardHeader>
            <CardContent>
              {models.length === 0 ? (
                <p className="text-center text-muted-foreground py-8">
                  暂无已训练模型
                </p>
              ) : (
                <div className="space-y-4">
                  {models.map((model) => (
                    <div
                      key={model.model_id}
                      className="flex items-center justify-between rounded-lg border p-4"
                    >
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <Brain className="h-4 w-4 text-primary" />
                          <span className="font-medium">{model.model_id}</span>
                        </div>
                        <p className="text-sm text-muted-foreground font-mono">
                          {model.path}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          训练于 {formatDate(model.created_at)}
                        </p>
                      </div>
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => {
                            setSelectedModel(model)
                            setBacktestDialogOpen(true)
                          }}
                        >
                          <TrendingUp className="mr-2 h-4 w-4" />
                          回测
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* 新建训练对话框 */}
      <Dialog open={trainingDialogOpen} onOpenChange={setTrainingDialogOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>新建 RL 训练任务</DialogTitle>
            <DialogDescription>
              配置 PPO 强化学习训练参数
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="train-name">任务名称（可选）</Label>
              <Input
                id="train-name"
                placeholder="例如: BTC订单优化训练"
                value={trainingName}
                onChange={(e) => setTrainingName(e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="train-data">训练数据路径</Label>
              <Input
                id="train-data"
                placeholder="data/train.parquet"
                value={trainDataPath}
                onChange={(e) => setTrainDataPath(e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="test-data">测试数据路径</Label>
              <Input
                id="test-data"
                placeholder="data/test.parquet"
                value={testDataPath}
                onChange={(e) => setTestDataPath(e.target.value)}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="timesteps">训练步数</Label>
                <Input
                  id="timesteps"
                  type="number"
                  value={timesteps}
                  onChange={(e) => setTimesteps(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="lr">学习率</Label>
                <Input
                  id="lr"
                  type="number"
                  step="0.0001"
                  value={learningRate}
                  onChange={(e) => setLearningRate(e.target.value)}
                />
              </div>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setTrainingDialogOpen(false)}>
              取消
            </Button>
            <Button
              onClick={handleSubmitTraining}
              disabled={!trainDataPath || !testDataPath}
            >
              <Brain className="mr-2 h-4 w-4" />
              开始训练
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* 回测对话框 */}
      <Dialog open={backtestDialogOpen} onOpenChange={setBacktestDialogOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>运行模型回测</DialogTitle>
            <DialogDescription>
              使用已训练模型在历史数据上进行回测
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            {selectedModel && (
              <div className="rounded-lg bg-muted p-3">
                <p className="text-sm font-medium">选中模型</p>
                <p className="text-sm text-muted-foreground font-mono">
                  {selectedModel.path}
                </p>
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="backtest-name">任务名称（可选）</Label>
              <Input
                id="backtest-name"
                placeholder="例如: BTC回测"
                value={backtestName}
                onChange={(e) => setBacktestName(e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="backtest-data">回测数据路径</Label>
              <Input
                id="backtest-data"
                placeholder="data/backtest.parquet"
                value={backtestDataPath}
                onChange={(e) => setBacktestDataPath(e.target.value)}
              />
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setBacktestDialogOpen(false)}>
              取消
            </Button>
            <Button
              onClick={handleSubmitBacktest}
              disabled={!backtestDataPath}
            >
              <TrendingUp className="mr-2 h-4 w-4" />
              开始回测
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* 任务详情对话框 */}
      <Dialog open={detailsDialogOpen} onOpenChange={setDetailsDialogOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>任务详情</DialogTitle>
          </DialogHeader>

          {selectedTask && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">任务 ID</p>
                  <p className="font-mono">{selectedTask.task_id}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Celery 任务 ID</p>
                  <p className="font-mono text-sm">{selectedTask.celery_task_id}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground">类型</p>
                  <p>{selectedTask.task_type === 'training' ? '训练' : '回测'}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground">状态</p>
                  <Badge variant={statusConfig[selectedTask.status].variant}>
                    {statusConfig[selectedTask.status].label}
                  </Badge>
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground">创建时间</p>
                  <p>{formatDate(selectedTask.created_at)}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground">完成时间</p>
                  <p>{formatDate(selectedTask.completed_at)}</p>
                </div>
              </div>

              {selectedTask.error && (
                <div>
                  <p className="text-sm font-medium text-muted-foreground">错误信息</p>
                  <pre className="mt-1 rounded bg-destructive/10 p-3 text-sm text-destructive overflow-auto">
                    {selectedTask.error}
                  </pre>
                </div>
              )}

              {selectedTask.result && (
                <div>
                  <p className="text-sm font-medium text-muted-foreground">结果</p>
                  <pre className="mt-1 rounded bg-muted p-3 text-sm overflow-auto max-h-64">
                    {JSON.stringify(selectedTask.result, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}
