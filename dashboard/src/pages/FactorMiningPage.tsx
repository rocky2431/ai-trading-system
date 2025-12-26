/**
 * Factor Mining Page - 因子挖掘任务管理
 */

import { useState, useMemo, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { useMiningTasks, useFactorLibraryStats } from '@/hooks/useMining'
import { useDataRanges } from '@/hooks/useData'
import type { MiningTaskStatus } from '@/api/factors'
import type { DataRangeInfo } from '@/api/data'
import {
  Pickaxe,
  Play,
  Pause,
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  TrendingUp,
  Target,
  Layers,
  BarChart3,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  Settings2,
  Database,
  GitCompare,
  Brain,
  Shield,
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
    <Card className={isActive ? 'border-blue-500/50' : ''}>
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
            <span className="text-gray-500 dark:text-gray-400">Progress</span>
            <span className="font-medium">{task.progress.toFixed(1)}%</span>
          </div>
          <ProgressBar value={task.progress} />
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-xl font-bold">{task.generated_count}</div>
            <div className="text-xs text-gray-500 dark:text-gray-400">Generated</div>
          </div>
          <div>
            <div className="text-xl font-bold text-emerald-500">{task.passed_count}</div>
            <div className="text-xs text-gray-500 dark:text-gray-400">Passed</div>
          </div>
          <div>
            <div className="text-xl font-bold text-rose-500">{task.failed_count}</div>
            <div className="text-xs text-gray-500 dark:text-gray-400">Failed</div>
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
          <div className="text-sm text-red-600 dark:text-red-400 bg-red-500/10 rounded p-2">
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
        <div className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
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
  const { stats } = useFactorLibraryStats()
  const { data: rangesData, loading: rangesLoading } = useDataRanges()

  // Tab state
  const [activeTab, setActiveTab] = useState('create')

  // Form state
  const [taskName, setTaskName] = useState('')
  const [taskDescription, setTaskDescription] = useState('')
  const [targetCount, setTargetCount] = useState(10)
  const [autoEvaluate, setAutoEvaluate] = useState(true)
  const [selectedFamilies, setSelectedFamilies] = useState<string[]>(['momentum', 'volatility'])

  // Data selection state
  const [selectedMarketType, setSelectedMarketType] = useState<'spot' | 'futures'>('spot')
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>([])
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h')

  // Get available symbols for selected market type
  const availableSymbols = useMemo(() => {
    if (!rangesData?.ranges) return []
    // Filter by market type and group by symbol
    const symbolMap = new Map<string, DataRangeInfo>()
    rangesData.ranges
      .filter(r => r.market_type === selectedMarketType && r.timeframe === selectedTimeframe)
      .forEach(r => {
        if (!symbolMap.has(r.symbol) || (r.start_date && r.end_date)) {
          symbolMap.set(r.symbol, r)
        }
      })
    return Array.from(symbolMap.values()).sort((a, b) => a.symbol.localeCompare(b.symbol))
  }, [rangesData, selectedMarketType, selectedTimeframe])

  // Calculate valid date range based on selected symbols
  const validDateRange = useMemo(() => {
    if (selectedSymbols.length === 0 || !rangesData?.ranges) {
      return { start: null, end: null }
    }
    const selectedRanges = rangesData.ranges.filter(
      r => selectedSymbols.includes(r.symbol) &&
           r.market_type === selectedMarketType &&
           r.timeframe === selectedTimeframe
    )
    if (selectedRanges.length === 0) return { start: null, end: null }

    // Find the intersection of all date ranges
    const starts = selectedRanges.map(r => r.start_date).filter(Boolean) as string[]
    const ends = selectedRanges.map(r => r.end_date).filter(Boolean) as string[]

    return {
      start: starts.length > 0 ? starts.reduce((a, b) => a > b ? a : b) : null,
      end: ends.length > 0 ? ends.reduce((a, b) => a < b ? a : b) : null,
    }
  }, [selectedSymbols, rangesData, selectedMarketType, selectedTimeframe])

  // Toggle symbol selection
  const toggleSymbol = (symbol: string) => {
    setSelectedSymbols(prev =>
      prev.includes(symbol)
        ? prev.filter(s => s !== symbol)
        : [...prev, symbol]
    )
  }

  // Advanced config expansion state
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Data Config
  const [dataConfig, setDataConfig] = useState({
    startDate: '',
    endDate: '',
    symbols: [] as string[],
    timeframes: ['1h', '4h', '1d'] as string[],
    trainRatio: 60,
    validRatio: 20,
    testRatio: 20,
  })

  // 当 validDateRange 变化时，自动同步到 dataConfig
  useEffect(() => {
    if (validDateRange.start && validDateRange.end) {
      setDataConfig(prev => ({
        ...prev,
        startDate: validDateRange.start!.slice(0, 10),
        endDate: validDateRange.end!.slice(0, 10),
      }))
    }
  }, [validDateRange.start, validDateRange.end])

  // 验证日期格式 YYYY-MM-DD
  const isValidDate = (dateStr: string) => {
    if (!dateStr) return true
    const regex = /^\d{4}-\d{2}-\d{2}$/
    if (!regex.test(dateStr)) return false
    const date = new Date(dateStr)
    return !isNaN(date.getTime())
  }

  // 计算 Split 总和是否为 100
  const splitTotal = dataConfig.trainRatio + dataConfig.validRatio + dataConfig.testRatio
  const isSplitValid = Math.abs(splitTotal - 100) < 0.01

  // Benchmark Config
  const [benchmarkConfig, setBenchmarkConfig] = useState({
    benchmarkSet: 'alpha158' as 'alpha101' | 'alpha158' | 'alpha360' | 'custom',
    correlationThreshold: 0.7,
    minIcImprovement: 0.02,
  })

  // Model Config - P0 FIX: Added mlSignalEnabled for explicit ML signal control
  const [modelConfig, setModelConfig] = useState({
    mlSignalEnabled: true, // Enable ML-based signal generation (not just simple Z-Score)
    modelType: 'lightgbm' as 'lightgbm' | 'xgboost' | 'catboost' | 'linear',
    optimizationMethod: 'bayesian' as 'bayesian' | 'genetic' | 'grid' | 'random',
    maxTrials: 100,
    earlyStoppingRounds: 20,
  })

  // Robustness Config - P1 FIX: Added enableRedundancyFilter for vector dedup control
  const [robustnessConfig, setRobustnessConfig] = useState({
    enableWalkForward: true,
    walkForwardWindows: 5,
    trainWindowDays: 180,
    testWindowDays: 30,
    enableDynamicThreshold: true,
    enableIcDecay: true,
    icDecayThreshold: 0.5,
    enableRedundancyFilter: true, // Vector dedup: prevent generating duplicate factors
    redundancyThreshold: 0.85,
  })

  const handleCreateTask = async () => {
    if (!taskName.trim() || selectedSymbols.length === 0) return

    // Build data config from selection
    const baseDataConfig = {
      symbols: selectedSymbols,
      timeframes: [selectedTimeframe],
      market_type: selectedMarketType,
      start_date: validDateRange.start || undefined,
      end_date: validDateRange.end || undefined,
    }

    // Build advanced config if expanded
    const advancedConfig = showAdvanced
      ? {
          data_config: {
            ...baseDataConfig,
            start_date: dataConfig.startDate || baseDataConfig.start_date,
            end_date: dataConfig.endDate || baseDataConfig.end_date,
            train_ratio: dataConfig.trainRatio / 100,
            valid_ratio: dataConfig.validRatio / 100,
            test_ratio: dataConfig.testRatio / 100,
          },
          benchmark_config: {
            benchmark_set: benchmarkConfig.benchmarkSet,
            correlation_threshold: benchmarkConfig.correlationThreshold,
            min_ic_improvement: benchmarkConfig.minIcImprovement,
          },
          ml_config: {
            // P0 FIX: Only include models when ML signal is enabled
            // Backend checks models[0] to determine if ML is enabled
            models: modelConfig.mlSignalEnabled ? [modelConfig.modelType] : [],
            optimization_method: modelConfig.optimizationMethod,
            max_trials: modelConfig.maxTrials,
            early_stopping_rounds: modelConfig.earlyStoppingRounds,
          },
          robustness_config: {
            enable_walk_forward: robustnessConfig.enableWalkForward,
            walk_forward_windows: robustnessConfig.walkForwardWindows,
            train_window_days: robustnessConfig.trainWindowDays,
            test_window_days: robustnessConfig.testWindowDays,
            enable_dynamic_threshold: robustnessConfig.enableDynamicThreshold,
            enable_ic_decay: robustnessConfig.enableIcDecay,
            ic_decay_threshold: robustnessConfig.icDecayThreshold,
            // P1 FIX: Vector dedup configuration
            use_redundancy_filter: robustnessConfig.enableRedundancyFilter,
            redundancy_threshold: robustnessConfig.redundancyThreshold,
          },
        }
      : {
          data_config: baseDataConfig,
        }

    await createTask({
      name: taskName,
      description: taskDescription,
      factor_families: selectedFamilies,
      target_count: targetCount,
      auto_evaluate: autoEvaluate,
      ...advancedConfig,
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
          <p className="text-gray-500 dark:text-gray-400">
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
              <Layers className="h-4 w-4 text-gray-500 dark:text-gray-400" />
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
              <TrendingUp className="h-4 w-4 text-gray-500 dark:text-gray-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.avg_sharpe.toFixed(2)}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Avg IC</CardTitle>
              <BarChart3 className="h-4 w-4 text-gray-500 dark:text-gray-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{(stats.avg_ic * 100).toFixed(1)}%</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Best Sharpe</CardTitle>
              <Target className="h-4 w-4 text-blue-600 dark:text-blue-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">{stats.best_sharpe.toFixed(2)}</div>
            </CardContent>
          </Card>
        </div>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab}>
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

              {/* Data Selection Section */}
              <div className="border rounded-lg p-4 space-y-4 bg-muted/30">
                <h4 className="font-medium flex items-center gap-2">
                  <Database className="h-4 w-4 text-blue-500" />
                  Data Selection *
                </h4>

                {/* Market Type */}
                <div className="space-y-2">
                  <Label>Market Type</Label>
                  <div className="flex gap-2">
                    <Button
                      type="button"
                      variant={selectedMarketType === 'spot' ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => {
                        setSelectedMarketType('spot')
                        setSelectedSymbols([])
                      }}
                    >
                      Spot
                    </Button>
                    <Button
                      type="button"
                      variant={selectedMarketType === 'futures' ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => {
                        setSelectedMarketType('futures')
                        setSelectedSymbols([])
                      }}
                    >
                      Futures
                    </Button>
                  </div>
                </div>

                {/* Timeframe */}
                <div className="space-y-2">
                  <Label>Timeframe</Label>
                  <div className="flex gap-2">
                    {['1m', '5m', '15m', '1h', '4h', '1d'].map(tf => (
                      <Button
                        key={tf}
                        type="button"
                        variant={selectedTimeframe === tf ? 'default' : 'outline'}
                        size="sm"
                        onClick={() => {
                          setSelectedTimeframe(tf)
                          setSelectedSymbols([])
                        }}
                      >
                        {tf}
                      </Button>
                    ))}
                  </div>
                </div>

                {/* Symbol Selection */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <Label>Select Symbols</Label>
                    <span className="text-xs text-muted-foreground">
                      {selectedSymbols.length} selected
                    </span>
                  </div>

                  {rangesLoading ? (
                    <div className="flex items-center justify-center p-4">
                      <Loader2 className="h-4 w-4 animate-spin mr-2" />
                      Loading available data...
                    </div>
                  ) : availableSymbols.length === 0 ? (
                    <div className="p-4 bg-amber-500/10 rounded text-sm text-amber-600">
                      No {selectedMarketType === 'spot' ? 'spot' : 'futures'} {selectedTimeframe} data found.
                      Please download data in Data Center first.
                    </div>
                  ) : (
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2 max-h-48 overflow-y-auto p-2 border rounded">
                      {availableSymbols.map(range => (
                        <Button
                          key={range.symbol}
                          type="button"
                          variant={selectedSymbols.includes(range.symbol) ? 'default' : 'outline'}
                          size="sm"
                          className="justify-start text-xs h-auto py-2"
                          onClick={() => toggleSymbol(range.symbol)}
                        >
                          <div className="flex flex-col items-start">
                            <span>{range.symbol}</span>
                            <span className="text-[10px] opacity-70">
                              {range.start_date?.slice(0, 10)} ~ {range.end_date?.slice(0, 10)}
                            </span>
                          </div>
                        </Button>
                      ))}
                    </div>
                  )}

                  {/* Quick select buttons */}
                  {availableSymbols.length > 0 && (
                    <div className="flex gap-2">
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => setSelectedSymbols(availableSymbols.map(s => s.symbol))}
                      >
                        Select All
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => setSelectedSymbols([])}
                      >
                        Clear
                      </Button>
                    </div>
                  )}
                </div>

                {/* Valid Date Range Info */}
                {selectedSymbols.length > 0 && validDateRange.start && validDateRange.end && (
                  <div className="p-3 bg-green-500/10 rounded text-sm">
                    <span className="text-green-600 font-medium">Available data range:</span>
                    <span className="ml-2">
                      {validDateRange.start.slice(0, 10)} ~ {validDateRange.end.slice(0, 10)}
                    </span>
                  </div>
                )}
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
                          selectedFamilies.includes(family.value) ? 'bg-primary-foreground' : family.color
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
                  value={targetCount}
                  onValueChange={setTargetCount}
                  min={1}
                  max={50}
                  step={1}
                />
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  The mining task will generate up to {targetCount} factors
                </p>
              </div>

              {/* Auto Evaluate */}
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Auto Evaluate</Label>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Automatically evaluate generated factors against thresholds
                  </p>
                </div>
                <Switch checked={autoEvaluate} onCheckedChange={setAutoEvaluate} />
              </div>

              {/* Advanced Configuration Toggle */}
              <div className="border-t pt-4">
                <Button
                  type="button"
                  variant="ghost"
                  className="w-full justify-between"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                >
                  <span className="flex items-center gap-2">
                    <Settings2 className="h-4 w-4" />
                    Advanced Configuration
                  </span>
                  {showAdvanced ? (
                    <ChevronUp className="h-4 w-4" />
                  ) : (
                    <ChevronDown className="h-4 w-4" />
                  )}
                </Button>
              </div>

              {/* Advanced Configuration Panel */}
              {showAdvanced && (
                <div className="space-y-6 border rounded-lg p-4 bg-muted/50">
                  {/* Data Config */}
                  <div className="space-y-4">
                    <h4 className="font-medium flex items-center gap-2">
                      <Database className="h-4 w-4 text-blue-500" />
                      Data Configuration
                    </h4>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="start-date">Start Date</Label>
                        <Input
                          id="start-date"
                          type="text"
                          placeholder="YYYY-MM-DD"
                          value={dataConfig.startDate}
                          onChange={(e) =>
                            setDataConfig({ ...dataConfig, startDate: e.target.value })
                          }
                          className={!isValidDate(dataConfig.startDate) ? 'border-red-500' : ''}
                        />
                        {!isValidDate(dataConfig.startDate) && (
                          <p className="text-xs text-red-500">Format: YYYY-MM-DD</p>
                        )}
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="end-date">End Date</Label>
                        <Input
                          id="end-date"
                          type="text"
                          placeholder="YYYY-MM-DD"
                          value={dataConfig.endDate}
                          onChange={(e) =>
                            setDataConfig({ ...dataConfig, endDate: e.target.value })
                          }
                          className={!isValidDate(dataConfig.endDate) ? 'border-red-500' : ''}
                        />
                        {!isValidDate(dataConfig.endDate) && (
                          <p className="text-xs text-red-500">Format: YYYY-MM-DD</p>
                        )}
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Label>Train / Valid / Test Split</Label>
                        {!isSplitValid && (
                          <span className="text-xs text-red-500">
                            Must equal 100% (current: {splitTotal}%)
                          </span>
                        )}
                      </div>
                      <div className="grid grid-cols-3 gap-4">
                        <div className="space-y-1">
                          <Label className="text-xs text-muted-foreground">Train (%)</Label>
                          <Input
                            type="number"
                            min={0}
                            max={100}
                            value={dataConfig.trainRatio}
                            onChange={(e) =>
                              setDataConfig({ ...dataConfig, trainRatio: Number(e.target.value) || 0 })
                            }
                            className={!isSplitValid ? 'border-amber-500' : ''}
                          />
                        </div>
                        <div className="space-y-1">
                          <Label className="text-xs text-muted-foreground">Valid (%)</Label>
                          <Input
                            type="number"
                            min={0}
                            max={100}
                            value={dataConfig.validRatio}
                            onChange={(e) =>
                              setDataConfig({ ...dataConfig, validRatio: Number(e.target.value) || 0 })
                            }
                            className={!isSplitValid ? 'border-amber-500' : ''}
                          />
                        </div>
                        <div className="space-y-1">
                          <Label className="text-xs text-muted-foreground">Test (%)</Label>
                          <Input
                            type="number"
                            min={0}
                            max={100}
                            value={dataConfig.testRatio}
                            onChange={(e) =>
                              setDataConfig({ ...dataConfig, testRatio: Number(e.target.value) || 0 })
                            }
                            className={!isSplitValid ? 'border-amber-500' : ''}
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Benchmark Config */}
                  <div className="space-y-4 border-t pt-4">
                    <h4 className="font-medium flex items-center gap-2">
                      <GitCompare className="h-4 w-4 text-purple-500" />
                      Benchmark Configuration
                    </h4>
                    <div className="space-y-2">
                      <Label>Benchmark Factor Set</Label>
                      <div className="flex flex-wrap gap-2">
                        {(['alpha101', 'alpha158', 'alpha360', 'custom'] as const).map((set) => (
                          <Button
                            key={set}
                            type="button"
                            variant={benchmarkConfig.benchmarkSet === set ? 'default' : 'outline'}
                            size="sm"
                            onClick={() =>
                              setBenchmarkConfig({ ...benchmarkConfig, benchmarkSet: set })
                            }
                          >
                            {set.toUpperCase()}
                          </Button>
                        ))}
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label>
                          Correlation Threshold: {benchmarkConfig.correlationThreshold.toFixed(2)}
                        </Label>
                        <Slider
                          value={benchmarkConfig.correlationThreshold * 100}
                          onValueChange={(v) =>
                            setBenchmarkConfig({ ...benchmarkConfig, correlationThreshold: v / 100 })
                          }
                          min={50}
                          max={95}
                          step={5}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label>
                          Min IC Improvement: {benchmarkConfig.minIcImprovement.toFixed(2)}
                        </Label>
                        <Slider
                          value={benchmarkConfig.minIcImprovement * 100}
                          onValueChange={(v) =>
                            setBenchmarkConfig({ ...benchmarkConfig, minIcImprovement: v / 100 })
                          }
                          min={1}
                          max={10}
                          step={1}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Model Config */}
                  <div className="space-y-4 border-t pt-4">
                    <div className="flex items-center justify-between">
                      <h4 className="font-medium flex items-center gap-2">
                        <Brain className="h-4 w-4 text-emerald-500" />
                        ML Signal Generation
                      </h4>
                      {/* P0 FIX: Explicit toggle for ML signal generation */}
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-muted-foreground">
                          {modelConfig.mlSignalEnabled ? 'LightGBM' : 'Z-Score (default)'}
                        </span>
                        <Switch
                          checked={modelConfig.mlSignalEnabled}
                          onCheckedChange={(checked) =>
                            setModelConfig({ ...modelConfig, mlSignalEnabled: checked })
                          }
                        />
                      </div>
                    </div>
                    {modelConfig.mlSignalEnabled && (
                      <p className="text-sm text-muted-foreground">
                        Use ML model to generate signals from factors (instead of simple Z-Score normalization)
                      </p>
                    )}
                    <div className={`grid grid-cols-2 gap-4 ${!modelConfig.mlSignalEnabled ? 'opacity-50 pointer-events-none' : ''}`}>
                      <div className="space-y-2">
                        <Label>ML Model</Label>
                        <div className="flex flex-wrap gap-2">
                          {(['lightgbm', 'xgboost', 'catboost', 'linear'] as const).map((model) => (
                            <Button
                              key={model}
                              type="button"
                              variant={modelConfig.modelType === model ? 'default' : 'outline'}
                              size="sm"
                              onClick={() => setModelConfig({ ...modelConfig, modelType: model })}
                            >
                              {model.charAt(0).toUpperCase() + model.slice(1)}
                            </Button>
                          ))}
                        </div>
                      </div>
                      <div className="space-y-2">
                        <Label>Optimization Method</Label>
                        <div className="flex flex-wrap gap-2">
                          {(['bayesian', 'genetic', 'grid', 'random'] as const).map((method) => (
                            <Button
                              key={method}
                              type="button"
                              variant={
                                modelConfig.optimizationMethod === method ? 'default' : 'outline'
                              }
                              size="sm"
                              onClick={() =>
                                setModelConfig({ ...modelConfig, optimizationMethod: method })
                              }
                            >
                              {method.charAt(0).toUpperCase() + method.slice(1)}
                            </Button>
                          ))}
                        </div>
                      </div>
                    </div>
                    <div className={`grid grid-cols-2 gap-4 ${!modelConfig.mlSignalEnabled ? 'opacity-50 pointer-events-none' : ''}`}>
                      <div className="space-y-2">
                        <Label>Max Trials: {modelConfig.maxTrials}</Label>
                        <Slider
                          value={modelConfig.maxTrials}
                          onValueChange={(v) => setModelConfig({ ...modelConfig, maxTrials: v })}
                          min={10}
                          max={500}
                          step={10}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label>Early Stopping: {modelConfig.earlyStoppingRounds} rounds</Label>
                        <Slider
                          value={modelConfig.earlyStoppingRounds}
                          onValueChange={(v) =>
                            setModelConfig({ ...modelConfig, earlyStoppingRounds: v })
                          }
                          min={5}
                          max={50}
                          step={5}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Robustness Config */}
                  <div className="space-y-4 border-t pt-4">
                    <h4 className="font-medium flex items-center gap-2">
                      <Shield className="h-4 w-4 text-amber-500" />
                      Robustness & Overfitting Prevention
                    </h4>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <Label>Walk-Forward Validation</Label>
                          <p className="text-xs text-gray-500 dark:text-gray-400">
                            Rolling window train/test splits
                          </p>
                        </div>
                        <Switch
                          checked={robustnessConfig.enableWalkForward}
                          onCheckedChange={(v) =>
                            setRobustnessConfig({ ...robustnessConfig, enableWalkForward: v })
                          }
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <div>
                          <Label>Dynamic Threshold</Label>
                          <p className="text-xs text-gray-500 dark:text-gray-400">
                            Adjust thresholds based on trials
                          </p>
                        </div>
                        <Switch
                          checked={robustnessConfig.enableDynamicThreshold}
                          onCheckedChange={(v) =>
                            setRobustnessConfig({ ...robustnessConfig, enableDynamicThreshold: v })
                          }
                        />
                      </div>
                    </div>
                    {robustnessConfig.enableWalkForward && (
                      <div className="grid grid-cols-3 gap-4 bg-card p-3 rounded">
                        <div className="space-y-2">
                          <Label>Windows: {robustnessConfig.walkForwardWindows}</Label>
                          <Slider
                            value={robustnessConfig.walkForwardWindows}
                            onValueChange={(v) =>
                              setRobustnessConfig({ ...robustnessConfig, walkForwardWindows: v })
                            }
                            min={3}
                            max={10}
                            step={1}
                          />
                        </div>
                        <div className="space-y-2">
                          <Label>Train: {robustnessConfig.trainWindowDays} days</Label>
                          <Slider
                            value={robustnessConfig.trainWindowDays}
                            onValueChange={(v) =>
                              setRobustnessConfig({ ...robustnessConfig, trainWindowDays: v })
                            }
                            min={60}
                            max={365}
                            step={30}
                          />
                        </div>
                        <div className="space-y-2">
                          <Label>Test: {robustnessConfig.testWindowDays} days</Label>
                          <Slider
                            value={robustnessConfig.testWindowDays}
                            onValueChange={(v) =>
                              setRobustnessConfig({ ...robustnessConfig, testWindowDays: v })
                            }
                            min={7}
                            max={90}
                            step={7}
                          />
                        </div>
                      </div>
                    )}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <Label>IC Decay Detection</Label>
                          <p className="text-xs text-gray-500 dark:text-gray-400">
                            Detect alpha decay over time
                          </p>
                        </div>
                        <Switch
                          checked={robustnessConfig.enableIcDecay}
                          onCheckedChange={(v) =>
                            setRobustnessConfig({ ...robustnessConfig, enableIcDecay: v })
                          }
                        />
                      </div>
                      {/* P1 FIX: Vector dedup toggle and threshold */}
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <div>
                            <Label>Vector Dedup (Redundancy Filter)</Label>
                            <p className="text-xs text-gray-500 dark:text-gray-400">
                              Prevent generating duplicate factors
                            </p>
                          </div>
                          <Switch
                            checked={robustnessConfig.enableRedundancyFilter}
                            onCheckedChange={(v) =>
                              setRobustnessConfig({ ...robustnessConfig, enableRedundancyFilter: v })
                            }
                          />
                        </div>
                        <div className={`space-y-2 ${!robustnessConfig.enableRedundancyFilter ? 'opacity-50 pointer-events-none' : ''}`}>
                          <Label>
                            Similarity Threshold: {robustnessConfig.redundancyThreshold.toFixed(2)}
                          </Label>
                          <Slider
                            value={robustnessConfig.redundancyThreshold * 100}
                            onValueChange={(v) =>
                              setRobustnessConfig({ ...robustnessConfig, redundancyThreshold: v / 100 })
                            }
                            min={70}
                            max={95}
                            step={5}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Submit */}
              <Button
                className="w-full"
                size="lg"
                onClick={handleCreateTask}
                disabled={creating || !taskName.trim() || selectedFamilies.length === 0 || selectedSymbols.length === 0}
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
              <Loader2 className="h-8 w-8 animate-spin text-blue-600 dark:text-blue-400" />
            </div>
          ) : runningTasks.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Pickaxe className="h-12 w-12 text-gray-500 dark:text-gray-400 mb-4" />
                <p className="text-lg font-medium">No active mining tasks</p>
                <p className="text-sm text-gray-500 dark:text-gray-400">
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
              <Loader2 className="h-8 w-8 animate-spin text-blue-600 dark:text-blue-400" />
            </div>
          ) : completedTasks.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Clock className="h-12 w-12 text-gray-500 dark:text-gray-400 mb-4" />
                <p className="text-lg font-medium">No completed tasks</p>
                <p className="text-sm text-gray-500 dark:text-gray-400">
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
