/**
 * Pipeline Page - 完整的 Pipeline 和 RD Loop 管理页面
 *
 * 功能:
 * - RD Loop 运行管理
 * - 实时状态监控
 * - 核心因子查看
 * - 迭代统计
 * - 检查点恢复
 */

import { useState } from 'react'
import {
  Play,
  Square,
  RefreshCw,
  Loader2,
  XCircle,
  CheckCircle,
  Clock,
  Beaker,
  TrendingUp,
  Code,
  ChevronRight,
  Zap,
  AlertTriangle,
  BarChart3,
  Settings2,
  Eye,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
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
import { usePipeline } from '@/hooks/usePipeline'
import { RDLoopPhase, RDLoopConfigRequest, HypothesisFamily } from '@/api/pipeline'

const phaseConfig: Record<
  RDLoopPhase,
  { label: string; color: string; icon: typeof Beaker; progress: number }
> = {
  idle: { label: '空闲', color: 'bg-muted', icon: Clock, progress: 0 },
  initialization: { label: '初始化', color: 'bg-blue-500', icon: Settings2, progress: 10 },
  hypothesis_generation: { label: '假设生成', color: 'bg-purple-500', icon: Beaker, progress: 25 },
  factor_coding: { label: '因子编码', color: 'bg-amber-500', icon: Code, progress: 45 },
  evaluation: { label: '因子评估', color: 'bg-cyan-500', icon: BarChart3, progress: 65 },
  benchmark: { label: '基准对比', color: 'bg-green-500', icon: TrendingUp, progress: 80 },
  selection: { label: '因子筛选', color: 'bg-emerald-500', icon: CheckCircle, progress: 90 },
  completed: { label: '已完成', color: 'bg-green-600', icon: CheckCircle, progress: 100 },
  failed: { label: '失败', color: 'bg-red-500', icon: XCircle, progress: 0 },
}

const familyOptions: { value: HypothesisFamily; label: string }[] = [
  { value: 'momentum', label: '动量' },
  { value: 'mean_reversion', label: '均值回归' },
  { value: 'volatility', label: '波动率' },
  { value: 'volume', label: '成交量' },
  { value: 'price_action', label: '价格行为' },
  { value: 'trend', label: '趋势' },
]

export function PipelinePage() {
  const {
    rdLoopRuns,
    currentRDLoop,
    selectedRDLoop,
    rdLoopStatistics,
    rdLoopFactors,
    loading,
    error,
    refresh,
    startRDLoop,
    stopRDLoop,
    selectRDLoop,
    clearSelection,
  } = usePipeline()

  const [configDialogOpen, setConfigDialogOpen] = useState(false)
  const [detailsDialogOpen, setDetailsDialogOpen] = useState(false)
  const [starting, setStarting] = useState(false)

  // 配置表单状态
  const [config, setConfig] = useState<RDLoopConfigRequest>({
    max_iterations: 10,
    max_hypotheses_per_iteration: 5,
    target_core_factors: 20,
    ic_threshold: 0.03,
    ir_threshold: 0.5,
    novelty_threshold: 0.3,
    run_benchmark: true,
    enable_combination: false,
    focus_families: [],
  })

  const handleStartRDLoop = async () => {
    setStarting(true)
    try {
      await startRDLoop(config)
      setConfigDialogOpen(false)
    } catch (err) {
      console.error('Failed to start RD Loop:', err)
    } finally {
      setStarting(false)
    }
  }

  const handleViewDetails = async (runId: string) => {
    await selectRDLoop(runId)
    setDetailsDialogOpen(true)
  }

  const formatDate = (dateStr: string | undefined) => {
    if (!dateStr) return '-'
    return new Date(dateStr).toLocaleString('zh-CN')
  }

  if (loading && rdLoopRuns.length === 0) {
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
          <h1 className="text-2xl font-bold">Pipeline 管理</h1>
          <p className="text-muted-foreground">
            运行 RD Loop 进行因子挖掘和评估
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={refresh}>
            <RefreshCw className={cn('mr-2 h-4 w-4', loading && 'animate-spin')} />
            刷新
          </Button>
          <Button
            onClick={() => setConfigDialogOpen(true)}
            disabled={currentRDLoop?.isRunning}
          >
            <Play className="mr-2 h-4 w-4" />
            启动 RD Loop
          </Button>
        </div>
      </div>

      {/* 当前运行状态 */}
      {currentRDLoop && currentRDLoop.runId && (
        <Card className={cn(
          'border-2',
          currentRDLoop.isRunning ? 'border-blue-500' : 'border-muted'
        )}>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {currentRDLoop.isRunning ? (
                  <Loader2 className="h-5 w-5 animate-spin text-blue-500" />
                ) : (
                  <CheckCircle className="h-5 w-5 text-green-500" />
                )}
                <div>
                  <CardTitle className="text-lg">
                    {currentRDLoop.isRunning ? '正在运行' : '最近完成'}
                  </CardTitle>
                  <CardDescription className="font-mono text-xs">
                    {currentRDLoop.runId}
                  </CardDescription>
                </div>
              </div>
              {currentRDLoop.isRunning && (
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => stopRDLoop(currentRDLoop.runId!)}
                >
                  <Square className="mr-2 h-4 w-4" />
                  停止
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* 阶段进度 */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="flex items-center gap-2">
                  {(() => {
                    const phase = phaseConfig[currentRDLoop.phase]
                    const Icon = phase.icon
                    return (
                      <>
                        <Icon className="h-4 w-4" />
                        {phase.label}
                      </>
                    )
                  })()}
                </span>
                <span className="text-muted-foreground">
                  迭代 {currentRDLoop.iteration}
                </span>
              </div>
              <Progress
                value={phaseConfig[currentRDLoop.phase].progress}
                className="h-2"
              />
            </div>

            {/* 统计数据 */}
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <div className="text-2xl font-bold">{currentRDLoop.iteration}</div>
                <div className="text-xs text-muted-foreground">迭代次数</div>
              </div>
              <div>
                <div className="text-2xl font-bold">{currentRDLoop.totalHypothesesTested}</div>
                <div className="text-xs text-muted-foreground">已测试假设</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-green-600">{currentRDLoop.coreFactorsCount}</div>
                <div className="text-xs text-muted-foreground">核心因子</div>
              </div>
            </div>

            {/* 阶段指示器 */}
            <div className="flex items-center justify-between pt-2">
              {(['initialization', 'hypothesis_generation', 'factor_coding', 'evaluation', 'benchmark', 'selection'] as RDLoopPhase[]).map((phase, index) => {
                const config = phaseConfig[phase]
                const isActive = phase === currentRDLoop.phase
                const isPast = phaseConfig[currentRDLoop.phase].progress > config.progress

                return (
                  <div key={phase} className="flex items-center">
                    <div
                      className={cn(
                        'flex h-8 w-8 items-center justify-center rounded-full text-xs font-medium',
                        isActive && 'bg-blue-500 text-white',
                        isPast && 'bg-green-500 text-white',
                        !isActive && !isPast && 'bg-muted text-muted-foreground'
                      )}
                    >
                      {index + 1}
                    </div>
                    {index < 5 && (
                      <ChevronRight className={cn(
                        'h-4 w-4 mx-1',
                        isPast ? 'text-green-500' : 'text-muted-foreground'
                      )} />
                    )}
                  </div>
                )
              })}
            </div>
          </CardContent>
        </Card>
      )}

      {/* 运行历史 */}
      <Tabs defaultValue="runs">
        <TabsList>
          <TabsTrigger value="runs">运行历史</TabsTrigger>
          <TabsTrigger value="factors">发现的因子</TabsTrigger>
        </TabsList>

        <TabsContent value="runs" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>RD Loop 运行记录</CardTitle>
              <CardDescription>所有 RD Loop 运行的历史记录</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>运行 ID</TableHead>
                    <TableHead>状态</TableHead>
                    <TableHead>阶段</TableHead>
                    <TableHead>迭代</TableHead>
                    <TableHead>核心因子</TableHead>
                    <TableHead>创建时间</TableHead>
                    <TableHead className="text-right">操作</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {rdLoopRuns.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={7} className="text-center text-muted-foreground">
                        暂无运行记录
                      </TableCell>
                    </TableRow>
                  ) : (
                    rdLoopRuns.map((run) => {
                      const phase = phaseConfig[run.phase] || phaseConfig.idle
                      return (
                        <TableRow key={run.run_id}>
                          <TableCell className="font-mono text-xs">
                            {run.run_id.slice(0, 8)}...
                          </TableCell>
                          <TableCell>
                            <Badge
                              variant={
                                run.status === 'completed'
                                  ? 'outline'
                                  : run.status === 'running'
                                  ? 'default'
                                  : run.status === 'failed'
                                  ? 'destructive'
                                  : 'secondary'
                              }
                            >
                              {run.status === 'completed' && '完成'}
                              {run.status === 'running' && '运行中'}
                              {run.status === 'failed' && '失败'}
                              {run.status === 'starting' && '启动中'}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            <span className={cn(
                              'inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs',
                              phase.color,
                              'text-white'
                            )}>
                              {phase.label}
                            </span>
                          </TableCell>
                          <TableCell>{run.iteration}</TableCell>
                          <TableCell>
                            <span className="font-medium text-green-600">
                              {run.core_factors_count}
                            </span>
                          </TableCell>
                          <TableCell className="text-sm text-muted-foreground">
                            {formatDate(run.created_at)}
                          </TableCell>
                          <TableCell className="text-right">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleViewDetails(run.run_id)}
                            >
                              <Eye className="h-4 w-4" />
                            </Button>
                          </TableCell>
                        </TableRow>
                      )
                    })
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="factors" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>发现的核心因子</CardTitle>
              <CardDescription>
                {selectedRDLoop
                  ? `运行 ${selectedRDLoop.run_id.slice(0, 8)} 发现的因子`
                  : '选择一个运行记录查看发现的因子'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {rdLoopFactors.length === 0 ? (
                <p className="text-center text-muted-foreground py-8">
                  {selectedRDLoop ? '该运行暂无发现核心因子' : '请选择一个运行记录'}
                </p>
              ) : (
                <div className="space-y-4">
                  {rdLoopFactors.map((factor, index) => (
                    <div
                      key={index}
                      className="rounded-lg border p-4 space-y-3"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Zap className="h-4 w-4 text-amber-500" />
                          <span className="font-medium">{factor.name}</span>
                        </div>
                        <Badge variant="outline">{factor.family}</Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {factor.hypothesis}
                      </p>
                      <div className="flex gap-4 text-sm">
                        <span>IC: <strong>{(factor.metrics.ic * 100).toFixed(2)}%</strong></span>
                        <span>IR: <strong>{factor.metrics.ir?.toFixed(2) || '-'}</strong></span>
                        <span>Sharpe: <strong>{factor.metrics.sharpe?.toFixed(2) || '-'}</strong></span>
                      </div>
                      <details className="text-sm">
                        <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                          查看代码
                        </summary>
                        <pre className="mt-2 rounded bg-muted p-3 text-xs overflow-auto max-h-32">
                          {factor.code}
                        </pre>
                      </details>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* 启动配置对话框 */}
      <Dialog open={configDialogOpen} onOpenChange={setConfigDialogOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>启动 RD Loop</DialogTitle>
            <DialogDescription>
              配置研究开发循环参数
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>最大迭代次数</Label>
                <Input
                  type="number"
                  value={config.max_iterations}
                  onChange={(e) =>
                    setConfig({ ...config, max_iterations: parseInt(e.target.value) })
                  }
                  min={1}
                  max={100}
                />
              </div>
              <div className="space-y-2">
                <Label>每轮最大假设数</Label>
                <Input
                  type="number"
                  value={config.max_hypotheses_per_iteration}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      max_hypotheses_per_iteration: parseInt(e.target.value),
                    })
                  }
                  min={1}
                  max={20}
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>目标核心因子数</Label>
                <Input
                  type="number"
                  value={config.target_core_factors}
                  onChange={(e) =>
                    setConfig({ ...config, target_core_factors: parseInt(e.target.value) })
                  }
                  min={1}
                  max={100}
                />
              </div>
              <div className="space-y-2">
                <Label>IC 阈值</Label>
                <Input
                  type="number"
                  step="0.01"
                  value={config.ic_threshold}
                  onChange={(e) =>
                    setConfig({ ...config, ic_threshold: parseFloat(e.target.value) })
                  }
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>IR 阈值</Label>
                <Input
                  type="number"
                  step="0.1"
                  value={config.ir_threshold}
                  onChange={(e) =>
                    setConfig({ ...config, ir_threshold: parseFloat(e.target.value) })
                  }
                />
              </div>
              <div className="space-y-2">
                <Label>新颖性阈值</Label>
                <Input
                  type="number"
                  step="0.1"
                  value={config.novelty_threshold}
                  onChange={(e) =>
                    setConfig({ ...config, novelty_threshold: parseFloat(e.target.value) })
                  }
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label>聚焦因子类型</Label>
              <Select
                value={config.focus_families?.[0] || 'all'}
                onValueChange={(v) =>
                  setConfig({
                    ...config,
                    focus_families: v === 'all' ? [] : [v as HypothesisFamily],
                  })
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder="选择因子类型" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">全部类型</SelectItem>
                  {familyOptions.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value}>
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <Label>运行基准对比</Label>
                <p className="text-xs text-muted-foreground">与 Alpha-158 等基准比较</p>
              </div>
              <Switch
                checked={config.run_benchmark}
                onCheckedChange={(v) => setConfig({ ...config, run_benchmark: v })}
              />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <Label>启用因子组合</Label>
                <p className="text-xs text-muted-foreground">尝试组合因子生成复合因子</p>
              </div>
              <Switch
                checked={config.enable_combination}
                onCheckedChange={(v) => setConfig({ ...config, enable_combination: v })}
              />
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setConfigDialogOpen(false)}>
              取消
            </Button>
            <Button onClick={handleStartRDLoop} disabled={starting}>
              {starting ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Play className="mr-2 h-4 w-4" />
              )}
              启动
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* 详情对话框 */}
      <Dialog open={detailsDialogOpen} onOpenChange={(open) => {
        setDetailsDialogOpen(open)
        if (!open) clearSelection()
      }}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-auto">
          <DialogHeader>
            <DialogTitle>RD Loop 详情</DialogTitle>
          </DialogHeader>

          {selectedRDLoop && (
            <div className="space-y-6">
              {/* 基本信息 */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">运行 ID</p>
                  <p className="font-mono">{selectedRDLoop.run_id}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground">阶段</p>
                  <Badge className={phaseConfig[selectedRDLoop.phase].color}>
                    {phaseConfig[selectedRDLoop.phase].label}
                  </Badge>
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground">迭代次数</p>
                  <p className="text-xl font-bold">{selectedRDLoop.iteration}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground">已测试假设</p>
                  <p className="text-xl font-bold">{selectedRDLoop.total_hypotheses_tested}</p>
                </div>
              </div>

              {/* 迭代结果 */}
              {rdLoopStatistics?.iteration_results && rdLoopStatistics.iteration_results.length > 0 && (
                <div>
                  <h3 className="font-medium mb-3">迭代结果</h3>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>迭代</TableHead>
                        <TableHead>测试假设</TableHead>
                        <TableHead>通过因子</TableHead>
                        <TableHead>最佳 IC</TableHead>
                        <TableHead>最佳 Sharpe</TableHead>
                        <TableHead>耗时</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {rdLoopStatistics.iteration_results.map((result) => (
                        <TableRow key={result.iteration}>
                          <TableCell>{result.iteration}</TableCell>
                          <TableCell>{result.hypotheses_tested}</TableCell>
                          <TableCell className="text-green-600">{result.factors_passed}</TableCell>
                          <TableCell>{(result.best_ic * 100).toFixed(2)}%</TableCell>
                          <TableCell>{result.best_sharpe.toFixed(2)}</TableCell>
                          <TableCell>{result.duration_seconds.toFixed(1)}s</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}

              {/* 核心因子 */}
              <div>
                <h3 className="font-medium mb-3">
                  核心因子 ({selectedRDLoop.core_factors_count})
                </h3>
                {selectedRDLoop.core_factors.length === 0 ? (
                  <p className="text-sm text-muted-foreground">暂无核心因子</p>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {selectedRDLoop.core_factors.map((factor, i) => (
                      <Badge key={i} variant="secondary">
                        {factor}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}
