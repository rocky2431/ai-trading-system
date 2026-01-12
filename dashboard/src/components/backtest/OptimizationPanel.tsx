/**
 * OptimizationPanel - 回测参数优化管理面板
 */

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Select } from '@/components/ui/select'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { useOptimizations, useOptimizationDetail, useStrategies } from '@/hooks/useBacktest'
import type {
  OptimizationResponse,
  OptimizationConfig,
  BacktestConfig,
  OptimizationMetricName,
  SamplerType,
  PrunerType,
} from '@/api/backtest'
import {
  Settings2,
  Play,
  Loader2,
  Eye,
  Trash2,
  XCircle,
  CheckCircle2,
  Clock,
  StopCircle,
  TrendingUp,
  BarChart3,
  Target,
  Beaker,
  RefreshCw,
} from 'lucide-react'

function OptimizationStatusBadge({ status }: { status: OptimizationResponse['status'] }) {
  const config = {
    pending: { label: 'Pending', variant: 'secondary' as const, icon: Clock },
    running: { label: 'Running', variant: 'default' as const, icon: Loader2 },
    completed: { label: 'Completed', variant: 'outline' as const, icon: CheckCircle2 },
    failed: { label: 'Failed', variant: 'destructive' as const, icon: XCircle },
    cancelled: { label: 'Cancelled', variant: 'secondary' as const, icon: StopCircle },
  }[status]

  const Icon = config.icon
  return (
    <Badge variant={config.variant} className="gap-1">
      <Icon className={`h-3 w-3 ${status === 'running' ? 'animate-spin' : ''}`} />
      {config.label}
    </Badge>
  )
}

function ProgressBar({ value, total }: { value: number; total: number }) {
  const percent = total > 0 ? (value / total) * 100 : 0
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <span className="text-muted-foreground">Trial {value} / {total}</span>
        <span className="font-medium">{percent.toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-muted rounded-full overflow-hidden">
        <div
          className="h-full bg-primary transition-all duration-300"
          style={{ width: `${Math.min(percent, 100)}%` }}
        />
      </div>
    </div>
  )
}

function OptimizationCard({
  optimization,
  onView,
  onCancel,
  onDelete,
}: {
  optimization: OptimizationResponse
  onView: (id: string) => void
  onCancel: (id: string) => void
  onDelete: (id: string) => void
}) {
  const isActive = optimization.status === 'running' || optimization.status === 'pending'
  const canCancel = optimization.status === 'running' || optimization.status === 'pending'

  return (
    <Card className={isActive ? 'border-primary/50' : ''}>
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between">
          <div>
            <CardTitle className="text-base">{optimization.name || `Optimization ${optimization.id.slice(0, 8)}`}</CardTitle>
            <CardDescription className="text-xs mt-1">
              Strategy: {optimization.strategy_id.slice(0, 8)}...
            </CardDescription>
          </div>
          <OptimizationStatusBadge status={optimization.status} />
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Progress */}
        {isActive && (
          <ProgressBar value={optimization.current_trial} total={optimization.total_trials} />
        )}

        {/* Best Metrics */}
        {optimization.best_metrics && (
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-emerald-500" />
              <span className="text-muted-foreground">Return:</span>
              <span className={`font-medium ${optimization.best_metrics.total_return >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                {optimization.best_metrics.total_return >= 0 ? '+' : ''}{optimization.best_metrics.total_return.toFixed(1)}%
              </span>
            </div>
            <div className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-primary" />
              <span className="text-muted-foreground">Sharpe:</span>
              <span className="font-medium">{optimization.best_metrics.sharpe_ratio.toFixed(2)}</span>
            </div>
          </div>
        )}

        {/* Trial Info */}
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Beaker className="h-3 w-3" />
          <span>{optimization.total_trials} trials</span>
          {optimization.best_trial_id !== null && (
            <>
              <span>•</span>
              <span>Best: Trial #{optimization.best_trial_id}</span>
            </>
          )}
        </div>

        {/* Actions */}
        <div className="flex gap-2">
          <Button
            size="sm"
            className="flex-1"
            onClick={() => onView(optimization.id)}
            disabled={optimization.status !== 'completed'}
          >
            <Eye className="h-4 w-4 mr-1" />
            View Details
          </Button>
          {canCancel && (
            <Button
              size="sm"
              variant="outline"
              onClick={() => onCancel(optimization.id)}
            >
              <StopCircle className="h-4 w-4" />
            </Button>
          )}
          <Button
            size="sm"
            variant="destructive"
            onClick={() => onDelete(optimization.id)}
            disabled={isActive}
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

function OptimizationDetailView({ optimizationId, onClose }: { optimizationId: string; onClose: () => void }) {
  const { detail, loading } = useOptimizationDetail(optimizationId)

  if (loading || !detail) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </CardContent>
      </Card>
    )
  }

  const { optimization, param_importance } = detail

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">{optimization.name || 'Optimization Results'}</h2>
          <p className="text-muted-foreground">
            {optimization.total_trials} trials • Best: Trial #{optimization.best_trial_id}
          </p>
        </div>
        <Button variant="outline" onClick={onClose}>Close</Button>
      </div>

      {/* Best Params */}
      {optimization.best_params && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2">
              <Target className="h-4 w-4" />
              Best Parameters
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {Object.entries(optimization.best_params).map(([key, value]) => (
                <div key={key} className="space-y-1">
                  <div className="text-xs text-muted-foreground">{key}</div>
                  <div className="font-mono text-sm">{String(value)}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Best Metrics */}
      {optimization.best_metrics && (
        <div className="grid gap-4 grid-cols-2 sm:grid-cols-4">
          <Card>
            <CardContent className="pt-4">
              <div className="text-2xl font-bold text-emerald-500">
                {optimization.best_metrics.total_return >= 0 ? '+' : ''}{optimization.best_metrics.total_return.toFixed(1)}%
              </div>
              <div className="text-sm text-muted-foreground">Total Return</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4">
              <div className="text-2xl font-bold">{optimization.best_metrics.sharpe_ratio.toFixed(2)}</div>
              <div className="text-sm text-muted-foreground">Sharpe Ratio</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4">
              <div className="text-2xl font-bold text-rose-500">-{optimization.best_metrics.max_drawdown.toFixed(1)}%</div>
              <div className="text-sm text-muted-foreground">Max Drawdown</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4">
              <div className="text-2xl font-bold">{optimization.best_metrics.calmar_ratio.toFixed(2)}</div>
              <div className="text-sm text-muted-foreground">Calmar Ratio</div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Parameter Importance */}
      {param_importance && Object.keys(param_importance).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Parameter Importance</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {Object.entries(param_importance)
                .sort(([, a], [, b]) => b - a)
                .map(([param, importance]) => (
                  <div key={param} className="flex items-center gap-4">
                    <span className="text-sm w-32 truncate">{param}</span>
                    <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary"
                        style={{ width: `${Math.min(importance * 100, 100)}%` }}
                      />
                    </div>
                    <span className="text-sm font-medium w-12 text-right">{(importance * 100).toFixed(1)}%</span>
                  </div>
                ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Top Trials */}
      {optimization.top_trials && optimization.top_trials.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Top Trials</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-2 px-2">Rank</th>
                    <th className="text-left py-2 px-2">Trial</th>
                    <th className="text-right py-2 px-2">Return</th>
                    <th className="text-right py-2 px-2">Sharpe</th>
                    <th className="text-right py-2 px-2">Max DD</th>
                    <th className="text-right py-2 px-2">Duration</th>
                  </tr>
                </thead>
                <tbody>
                  {optimization.top_trials.slice(0, 10).map((trial) => (
                    <tr key={trial.trial_id} className="border-b last:border-0">
                      <td className="py-2 px-2">#{trial.rank}</td>
                      <td className="py-2 px-2">Trial {trial.trial_id}</td>
                      <td className={`py-2 px-2 text-right ${trial.metrics.total_return >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                        {trial.metrics.total_return >= 0 ? '+' : ''}{trial.metrics.total_return.toFixed(1)}%
                      </td>
                      <td className="py-2 px-2 text-right">{trial.metrics.sharpe_ratio.toFixed(2)}</td>
                      <td className="py-2 px-2 text-right text-rose-500">-{trial.metrics.max_drawdown.toFixed(1)}%</td>
                      <td className="py-2 px-2 text-right">{trial.duration_seconds.toFixed(1)}s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

const SAMPLER_OPTIONS = [
  { value: 'tpe', label: 'TPE (Tree-structured Parzen Estimator)' },
  { value: 'cmaes', label: 'CMA-ES (Covariance Matrix Adaptation)' },
  { value: 'random', label: 'Random Search' },
  { value: 'grid', label: 'Grid Search' },
]

const PRUNER_OPTIONS = [
  { value: 'median', label: 'Median Pruner' },
  { value: 'hyperband', label: 'Hyperband' },
  { value: 'percentile', label: 'Percentile Pruner' },
  { value: 'none', label: 'No Pruning' },
]

const METRIC_OPTIONS = [
  { value: 'sharpe', label: 'Sharpe Ratio' },
  { value: 'calmar', label: 'Calmar Ratio' },
  { value: 'total_return', label: 'Total Return' },
  { value: 'sortino', label: 'Sortino Ratio' },
  { value: 'ic', label: 'Information Coefficient' },
]

export function OptimizationPanel() {
  const { strategies } = useStrategies()
  const { optimizations, loading, creating, createOptimization, cancelOptimization, deleteOptimization, refetch } = useOptimizations()

  // Form state
  const [selectedStrategy, setSelectedStrategy] = useState('')
  const [optimizationName, setOptimizationName] = useState('')
  const [nTrials, setNTrials] = useState('100')
  const [sampler, setSampler] = useState<SamplerType>('tpe')
  const [pruner, setPruner] = useState<PrunerType>('median')
  const [metric, setMetric] = useState<OptimizationMetricName>('sharpe')
  const [startDate, setStartDate] = useState('2023-01-01')
  const [endDate, setEndDate] = useState('2024-01-01')
  const [initialCapital, setInitialCapital] = useState('1000000')

  // Detail view
  const [viewingOptimizationId, setViewingOptimizationId] = useState<string | null>(null)

  // Tab state
  const [activeTab, setActiveTab] = useState('create')

  const handleCreateOptimization = async () => {
    if (!selectedStrategy) return

    const backtestConfig: BacktestConfig = {
      start_date: startDate,
      end_date: endDate,
      initial_capital: parseFloat(initialCapital),
    }

    const optimizationConfig: OptimizationConfig = {
      n_trials: parseInt(nTrials, 10),
      metrics: [{ name: metric, direction: 'maximize' }],
      sampler,
      pruner,
    }

    await createOptimization({
      strategy_id: selectedStrategy,
      backtest_config: backtestConfig,
      optimization_config: optimizationConfig,
      name: optimizationName || undefined,
    })

    setOptimizationName('')
  }

  const runningOptimizations = optimizations.filter(o => o.status === 'running' || o.status === 'pending')
  const completedOptimizations = optimizations.filter(o => o.status === 'completed' || o.status === 'failed' || o.status === 'cancelled')

  if (viewingOptimizationId) {
    return (
      <OptimizationDetailView
        optimizationId={viewingOptimizationId}
        onClose={() => setViewingOptimizationId(null)}
      />
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Parameter Optimization</h2>
          <p className="text-muted-foreground text-sm">
            Use Optuna to find optimal strategy parameters
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={refetch}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="create">New Optimization</TabsTrigger>
          <TabsTrigger value="running">
            Running
            {runningOptimizations.length > 0 && (
              <Badge variant="secondary" className="ml-2">{runningOptimizations.length}</Badge>
            )}
          </TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
        </TabsList>

        {/* Create Tab */}
        <TabsContent value="create" className="space-y-6 mt-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings2 className="h-5 w-5" />
                New Optimization Task
              </CardTitle>
              <CardDescription>Configure Optuna hyperparameter optimization</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Basic Config */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Strategy *</Label>
                  <Select
                    options={[
                      { value: '', label: 'Select a strategy' },
                      ...strategies.map((s) => ({ value: s.id, label: s.name })),
                    ]}
                    value={selectedStrategy}
                    onChange={(e) => setSelectedStrategy(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Optimization Name</Label>
                  <Input
                    placeholder="Optional name"
                    value={optimizationName}
                    onChange={(e) => setOptimizationName(e.target.value)}
                  />
                </div>
              </div>

              {/* Backtest Config */}
              <div className="grid grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label>Start Date</Label>
                  <Input
                    type="text"
                    placeholder="YYYY-MM-DD"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label>End Date</Label>
                  <Input
                    type="text"
                    placeholder="YYYY-MM-DD"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Initial Capital</Label>
                  <Input
                    type="number"
                    value={initialCapital}
                    onChange={(e) => setInitialCapital(e.target.value)}
                  />
                </div>
              </div>

              {/* Optimization Config */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="space-y-2">
                  <Label>Trials</Label>
                  <Input
                    type="number"
                    min="10"
                    max="1000"
                    value={nTrials}
                    onChange={(e) => setNTrials(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Metric</Label>
                  <Select
                    options={METRIC_OPTIONS}
                    value={metric}
                    onChange={(e) => setMetric(e.target.value as OptimizationMetricName)}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Sampler</Label>
                  <Select
                    options={SAMPLER_OPTIONS}
                    value={sampler}
                    onChange={(e) => setSampler(e.target.value as SamplerType)}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Pruner</Label>
                  <Select
                    options={PRUNER_OPTIONS}
                    value={pruner}
                    onChange={(e) => setPruner(e.target.value as PrunerType)}
                  />
                </div>
              </div>

              <Button
                className="w-full"
                size="lg"
                onClick={handleCreateOptimization}
                disabled={creating || !selectedStrategy}
              >
                {creating ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Starting...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Start Optimization
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Running Tab */}
        <TabsContent value="running" className="mt-6">
          {loading ? (
            <div className="flex items-center justify-center h-48">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
          ) : runningOptimizations.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Clock className="h-12 w-12 text-muted-foreground mb-4" />
                <p className="text-lg font-medium">No running optimizations</p>
                <p className="text-sm text-muted-foreground">Start a new optimization to see progress here</p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {runningOptimizations.map((opt) => (
                <OptimizationCard
                  key={opt.id}
                  optimization={opt}
                  onView={setViewingOptimizationId}
                  onCancel={cancelOptimization}
                  onDelete={deleteOptimization}
                />
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
          ) : completedOptimizations.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <BarChart3 className="h-12 w-12 text-muted-foreground mb-4" />
                <p className="text-lg font-medium">No completed optimizations</p>
                <p className="text-sm text-muted-foreground">Completed optimizations will appear here</p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {completedOptimizations.map((opt) => (
                <OptimizationCard
                  key={opt.id}
                  optimization={opt}
                  onView={setViewingOptimizationId}
                  onCancel={cancelOptimization}
                  onDelete={deleteOptimization}
                />
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}
