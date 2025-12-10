/**
 * Backtest Center Page - 回测执行和结果分析
 */

import { useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Select } from '@/components/ui/select'
import { useBacktests, useBacktestStats, useStrategies, useBacktestDetail } from '@/hooks/useBacktest'
import type { BacktestResponse, BacktestConfig } from '@/api/backtest'
import {
  Play,
  BarChart3,
  TrendingUp,
  TrendingDown,
  Calendar,
  DollarSign,
  Loader2,
  RefreshCw,
  Eye,
  Trash2,
  Target,
  Activity,
  Clock,
  CheckCircle2,
  XCircle,
} from 'lucide-react'

function StatusBadge({ status }: { status: BacktestResponse['status'] }) {
  const config = {
    pending: { label: 'Pending', variant: 'secondary' as const, icon: Clock },
    running: { label: 'Running', variant: 'default' as const, icon: Loader2 },
    completed: { label: 'Completed', variant: 'outline' as const, icon: CheckCircle2 },
    failed: { label: 'Failed', variant: 'destructive' as const, icon: XCircle },
  }[status]

  const Icon = config.icon
  return (
    <Badge variant={config.variant} className="gap-1">
      <Icon className={`h-3 w-3 ${status === 'running' ? 'animate-spin' : ''}`} />
      {config.label}
    </Badge>
  )
}

function ProgressBar({ value }: { value: number }) {
  return (
    <div className="h-2 bg-muted rounded-full overflow-hidden">
      <div
        className="h-full bg-primary transition-all duration-300"
        style={{ width: `${Math.min(value, 100)}%` }}
      />
    </div>
  )
}

function BacktestCard({
  backtest,
  onView,
  onDelete,
}: {
  backtest: BacktestResponse
  onView: (id: string) => void
  onDelete: (id: string) => void
}) {
  const isActive = backtest.status === 'running' || backtest.status === 'pending'

  return (
    <Card className={isActive ? 'border-primary/50' : ''}>
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between">
          <div>
            <CardTitle className="text-base">{backtest.name}</CardTitle>
            <CardDescription className="text-xs mt-1">
              Strategy: {backtest.strategy_name}
            </CardDescription>
          </div>
          <StatusBadge status={backtest.status} />
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Progress */}
        {isActive && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Progress</span>
              <span className="font-medium">{backtest.progress.toFixed(1)}%</span>
            </div>
            <ProgressBar value={backtest.progress} />
          </div>
        )}

        {/* Metrics */}
        {backtest.metrics && (
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-emerald-500" />
              <span className="text-muted-foreground">Return:</span>
              <span className={`font-medium ${backtest.metrics.total_return >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                {backtest.metrics.total_return >= 0 ? '+' : ''}{backtest.metrics.total_return.toFixed(1)}%
              </span>
            </div>
            <div className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-primary" />
              <span className="text-muted-foreground">Sharpe:</span>
              <span className="font-medium">{backtest.metrics.sharpe_ratio.toFixed(2)}</span>
            </div>
            <div className="flex items-center gap-2">
              <TrendingDown className="h-4 w-4 text-rose-500" />
              <span className="text-muted-foreground">Max DD:</span>
              <span className="font-medium text-rose-500">-{backtest.metrics.max_drawdown.toFixed(1)}%</span>
            </div>
            <div className="flex items-center gap-2">
              <Target className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground">Win Rate:</span>
              <span className="font-medium">{backtest.metrics.win_rate.toFixed(1)}%</span>
            </div>
          </div>
        )}

        {/* Config Summary */}
        <div className="text-xs text-muted-foreground">
          {backtest.config.start_date} to {backtest.config.end_date}
        </div>

        {/* Actions */}
        <div className="flex gap-2">
          <Button
            size="sm"
            className="flex-1"
            onClick={() => onView(backtest.id)}
            disabled={backtest.status !== 'completed'}
          >
            <Eye className="h-4 w-4 mr-1" />
            View Details
          </Button>
          <Button
            size="sm"
            variant="destructive"
            onClick={() => onDelete(backtest.id)}
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

function BacktestDetailView({ backtestId, onClose }: { backtestId: string; onClose: () => void }) {
  const { detail, loading } = useBacktestDetail(backtestId)

  if (loading || !detail) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </CardContent>
      </Card>
    )
  }

  const { backtest, monthly_returns, factor_contributions } = detail

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">{backtest.name}</h2>
          <p className="text-muted-foreground">{backtest.strategy_name}</p>
        </div>
        <Button variant="outline" onClick={onClose}>Close</Button>
      </div>

      {/* Key Metrics */}
      {backtest.metrics && (
        <div className="grid gap-4 grid-cols-2 sm:grid-cols-4">
          <Card>
            <CardContent className="pt-4">
              <div className="text-2xl font-bold text-emerald-500">
                {backtest.metrics.total_return >= 0 ? '+' : ''}{backtest.metrics.total_return.toFixed(1)}%
              </div>
              <div className="text-sm text-muted-foreground">Total Return</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4">
              <div className="text-2xl font-bold">{backtest.metrics.sharpe_ratio.toFixed(2)}</div>
              <div className="text-sm text-muted-foreground">Sharpe Ratio</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4">
              <div className="text-2xl font-bold text-rose-500">-{backtest.metrics.max_drawdown.toFixed(1)}%</div>
              <div className="text-sm text-muted-foreground">Max Drawdown</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4">
              <div className="text-2xl font-bold">{backtest.metrics.trade_count}</div>
              <div className="text-sm text-muted-foreground">Total Trades</div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Monthly Returns */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Monthly Returns</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-12 gap-2">
            {Object.entries(monthly_returns).slice(-12).map(([month, ret]) => (
              <div key={month} className="text-center">
                <div className={`text-xs font-medium ${ret >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                  {ret >= 0 ? '+' : ''}{ret.toFixed(1)}%
                </div>
                <div className="text-xs text-muted-foreground">{month.slice(5)}</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Factor Contributions */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Factor Contributions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {Object.entries(factor_contributions).map(([factor, contribution]) => (
              <div key={factor} className="flex items-center gap-4">
                <span className="text-sm capitalize w-24">{factor}</span>
                <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary"
                    style={{ width: `${Math.min(contribution, 100)}%` }}
                  />
                </div>
                <span className="text-sm font-medium w-12 text-right">{contribution.toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export function BacktestCenterPage() {
  const [searchParams] = useSearchParams()
  const preselectedStrategy = searchParams.get('strategy')

  const { stats } = useBacktestStats()
  const { strategies } = useStrategies()
  const { backtests, loading, creating, createBacktest, deleteBacktest, refetch } = useBacktests()

  // Form state
  const [selectedStrategy, setSelectedStrategy] = useState(preselectedStrategy || '')
  const [backtestName, setBacktestName] = useState('')
  const [startDate, setStartDate] = useState('2023-01-01')
  const [endDate, setEndDate] = useState('2024-01-01')
  const [initialCapital, setInitialCapital] = useState('1000000')

  // Detail view
  const [viewingBacktestId, setViewingBacktestId] = useState<string | null>(null)

  const handleRunBacktest = async () => {
    if (!selectedStrategy) return

    const config: BacktestConfig = {
      start_date: startDate,
      end_date: endDate,
      initial_capital: parseFloat(initialCapital),
    }

    await createBacktest(selectedStrategy, config, backtestName || undefined)

    setBacktestName('')
  }

  const runningBacktests = backtests.filter((b) => b.status === 'running' || b.status === 'pending')
  const completedBacktests = backtests.filter((b) => b.status === 'completed' || b.status === 'failed')

  if (viewingBacktestId) {
    return (
      <BacktestDetailView
        backtestId={viewingBacktestId}
        onClose={() => setViewingBacktestId(null)}
      />
    )
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Backtest Center</h1>
          <p className="text-muted-foreground">
            Run backtests and analyze strategy performance
          </p>
        </div>
        <Button variant="outline" onClick={refetch}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Stats */}
      {stats && (
        <div className="grid gap-4 grid-cols-2 sm:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Backtests</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.total_backtests}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Running</CardTitle>
              <Loader2 className="h-4 w-4 text-primary animate-spin" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-primary">{stats.running_backtests}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Avg Sharpe</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.avg_sharpe.toFixed(2)}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Best Sharpe</CardTitle>
              <TrendingUp className="h-4 w-4 text-emerald-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-emerald-500">{stats.best_sharpe.toFixed(2)}</div>
            </CardContent>
          </Card>
        </div>
      )}

      <Tabs defaultValue="run">
        <TabsList>
          <TabsTrigger value="run">Run Backtest</TabsTrigger>
          <TabsTrigger value="active">
            Active
            {runningBacktests.length > 0 && (
              <Badge variant="secondary" className="ml-2">{runningBacktests.length}</Badge>
            )}
          </TabsTrigger>
          <TabsTrigger value="results">Results</TabsTrigger>
        </TabsList>

        {/* Run Backtest Tab */}
        <TabsContent value="run" className="space-y-6 mt-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Play className="h-5 w-5" />
                New Backtest
              </CardTitle>
              <CardDescription>Configure and run a backtest</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
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
                  <Label>Backtest Name</Label>
                  <Input
                    placeholder="Optional name"
                    value={backtestName}
                    onChange={(e) => setBacktestName(e.target.value)}
                  />
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label>Start Date</Label>
                  <Input
                    type="date"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label>End Date</Label>
                  <Input
                    type="date"
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

              <Button
                className="w-full"
                size="lg"
                onClick={handleRunBacktest}
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
                    Run Backtest
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Active Tab */}
        <TabsContent value="active" className="mt-6">
          {loading ? (
            <div className="flex items-center justify-center h-48">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
          ) : runningBacktests.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Clock className="h-12 w-12 text-muted-foreground mb-4" />
                <p className="text-lg font-medium">No active backtests</p>
                <p className="text-sm text-muted-foreground">Run a new backtest to see progress here</p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {runningBacktests.map((bt) => (
                <BacktestCard
                  key={bt.id}
                  backtest={bt}
                  onView={setViewingBacktestId}
                  onDelete={deleteBacktest}
                />
              ))}
            </div>
          )}
        </TabsContent>

        {/* Results Tab */}
        <TabsContent value="results" className="mt-6">
          {loading ? (
            <div className="flex items-center justify-center h-48">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
          ) : completedBacktests.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <BarChart3 className="h-12 w-12 text-muted-foreground mb-4" />
                <p className="text-lg font-medium">No completed backtests</p>
                <p className="text-sm text-muted-foreground">Completed backtests will appear here</p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {completedBacktests.map((bt) => (
                <BacktestCard
                  key={bt.id}
                  backtest={bt}
                  onView={setViewingBacktestId}
                  onDelete={deleteBacktest}
                />
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}
