/**
 * Strategy Workshop Page - 策略创建和管理
 */

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { Slider } from '@/components/ui/slider'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Select } from '@/components/ui/select'
import { useStrategies } from '@/hooks/useBacktest'
import { useFactors } from '@/hooks/useFactors'
import type { StrategyResponse } from '@/api/backtest'
import {
  Wrench,
  Plus,
  Trash2,
  Loader2,
  Play,
  Layers,
  BarChart3,
  Settings2,
  RefreshCw,
  CheckCircle2,
  Clock,
  FileCode,
  TrendingUp,
  Activity,
  Shield,
  Zap,
} from 'lucide-react'
import {
  STRATEGY_TEMPLATES,
  CATEGORY_LABELS,
  RISK_LEVEL_LABELS,
  RISK_LEVEL_COLORS,
  type StrategyTemplate,
  type StrategyCategory,
} from '@/data/strategyTemplates'

const WEIGHTING_METHODS = [
  { value: 'equal', label: 'Equal Weight' },
  { value: 'ic_weighted', label: 'IC Weighted' },
  { value: 'optimization', label: 'Optimization Based' },
]

const REBALANCE_FREQUENCIES = [
  { value: 'daily', label: 'Daily' },
  { value: 'weekly', label: 'Weekly' },
  { value: 'monthly', label: 'Monthly' },
]

const UNIVERSE_OPTIONS = [
  { value: 'all', label: 'All Symbols' },
  { value: 'top100', label: 'Top 100 by Volume' },
  { value: 'custom', label: 'Custom Selection' },
]

const CATEGORY_ICONS: Record<StrategyCategory, React.ReactNode> = {
  momentum: <TrendingUp className="h-5 w-5" />,
  mean_reversion: <Activity className="h-5 w-5" />,
  multi_factor: <Layers className="h-5 w-5" />,
  crypto: <Zap className="h-5 w-5" />,
}

function TemplateCard({
  template,
  onSelect,
  isCreating,
}: {
  template: StrategyTemplate
  onSelect: (template: StrategyTemplate) => void
  isCreating: boolean
}) {
  return (
    <Card className="hover:border-primary/50 transition-colors h-full flex flex-col">
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2">
            {CATEGORY_ICONS[template.category]}
            <CardTitle className="text-base">{template.name}</CardTitle>
          </div>
          <Badge className={RISK_LEVEL_COLORS[template.riskLevel]}>
            {RISK_LEVEL_LABELS[template.riskLevel]}
          </Badge>
        </div>
        <Badge variant="outline" className="w-fit mt-1">
          {CATEGORY_LABELS[template.category]}
        </Badge>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col">
        <p className="text-sm text-muted-foreground mb-4 flex-1">
          {template.description}
        </p>

        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-muted-foreground">Factors:</span>
              <span className="ml-1 font-medium">{template.factors.length}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Rebalance:</span>
              <span className="ml-1 font-medium capitalize">{template.rebalanceFrequency}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Exp. Sharpe:</span>
              <span className="ml-1 font-medium">{template.expectedSharpe.toFixed(1)}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Exp. Return:</span>
              <span className="ml-1 font-medium">{(template.expectedAnnualReturn * 100).toFixed(0)}%</span>
            </div>
          </div>

          <div className="flex flex-wrap gap-1">
            {template.tags.slice(0, 3).map((tag) => (
              <Badge key={tag} variant="secondary" className="text-xs">
                {tag}
              </Badge>
            ))}
          </div>

          <Button
            size="sm"
            className="w-full mt-2"
            onClick={() => onSelect(template)}
            disabled={isCreating}
          >
            {isCreating ? (
              <>
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                Creating...
              </>
            ) : (
              <>
                <FileCode className="h-4 w-4 mr-1" />
                Use Template
              </>
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

function StrategyCard({
  strategy,
  onDelete,
  onBacktest,
}: {
  strategy: StrategyResponse
  onDelete: (id: string) => void
  onBacktest: (id: string) => void
}) {
  return (
    <Card className="hover:border-primary/50 transition-colors">
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between">
          <div>
            <CardTitle className="text-base">{strategy.name}</CardTitle>
            <CardDescription className="text-xs mt-1">
              {strategy.description || 'No description'}
            </CardDescription>
          </div>
          <Badge variant={strategy.status === 'active' ? 'default' : 'secondary'}>
            {strategy.status}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-muted-foreground">Factors:</span>
            <span className="ml-2 font-medium">{strategy.factor_ids.length}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Rebalance:</span>
            <span className="ml-2 font-medium capitalize">{strategy.rebalance_frequency}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Method:</span>
            <span className="ml-2 font-medium capitalize">{strategy.weighting_method.replace('_', ' ')}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Max Pos:</span>
            <span className="ml-2 font-medium">{strategy.max_positions}</span>
          </div>
        </div>

        <div className="flex gap-2">
          <Button size="sm" className="flex-1" onClick={() => onBacktest(strategy.id)}>
            <Play className="h-4 w-4 mr-1" />
            Backtest
          </Button>
          <Button
            size="sm"
            variant="destructive"
            onClick={() => onDelete(strategy.id)}
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>

        <div className="text-xs text-muted-foreground">
          Created: {new Date(strategy.created_at).toLocaleDateString()}
        </div>
      </CardContent>
    </Card>
  )
}

export function StrategyWorkshopPage() {
  const { strategies, loading, creating, createStrategy, deleteStrategy, refetch } = useStrategies()
  const { factors } = useFactors()

  // Form state
  const [strategyName, setStrategyName] = useState('')
  const [strategyDescription, setStrategyDescription] = useState('')
  const [selectedFactors, setSelectedFactors] = useState<string[]>([])
  const [weightingMethod, setWeightingMethod] = useState('equal')
  const [rebalanceFrequency, setRebalanceFrequency] = useState('daily')
  const [universe, setUniverse] = useState('all')
  const [longOnly, setLongOnly] = useState(false)
  const [maxPositions, setMaxPositions] = useState(20)

  // Template state
  const [templateCategory, setTemplateCategory] = useState<StrategyCategory | 'all'>('all')
  const [creatingFromTemplate, setCreatingFromTemplate] = useState(false)

  const filteredTemplates = templateCategory === 'all'
    ? STRATEGY_TEMPLATES
    : STRATEGY_TEMPLATES.filter((t) => t.category === templateCategory)

  const handleCreateFromTemplate = async (template: StrategyTemplate) => {
    setCreatingFromTemplate(true)
    try {
      await createStrategy({
        name: `${template.name} Strategy`,
        description: template.description,
        factor_ids: template.factors,
        weighting_method: template.weightingMethod,
        rebalance_frequency: template.rebalanceFrequency,
        universe: 'all',
        long_only: template.longOnly,
        max_positions: template.maxPositions,
      })
    } finally {
      setCreatingFromTemplate(false)
    }
  }

  const handleCreateStrategy = async () => {
    if (!strategyName.trim()) return

    await createStrategy({
      name: strategyName,
      description: strategyDescription,
      factor_ids: selectedFactors,
      weighting_method: weightingMethod,
      rebalance_frequency: rebalanceFrequency,
      universe,
      long_only: longOnly,
      max_positions: maxPositions,
    })

    // Reset form
    setStrategyName('')
    setStrategyDescription('')
    setSelectedFactors([])
    setWeightingMethod('equal')
    setRebalanceFrequency('daily')
    setUniverse('all')
    setLongOnly(false)
    setMaxPositions(20)
  }

  const toggleFactor = (factorId: string) => {
    setSelectedFactors((prev) =>
      prev.includes(factorId) ? prev.filter((f) => f !== factorId) : [...prev, factorId]
    )
  }

  const handleBacktest = (strategyId: string) => {
    // Navigate to backtest page with strategy
    window.location.href = `/backtest?strategy=${strategyId}`
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Strategy Workshop</h1>
          <p className="text-muted-foreground">
            Design and manage your trading strategies
          </p>
        </div>
        <Button variant="outline" onClick={refetch}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Stats Overview */}
      <div className="grid gap-4 grid-cols-2 sm:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Strategies</CardTitle>
            <Layers className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{strategies.length}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active</CardTitle>
            <CheckCircle2 className="h-4 w-4 text-emerald-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-emerald-500">
              {strategies.filter((s) => s.status === 'active').length}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Draft</CardTitle>
            <Clock className="h-4 w-4 text-amber-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-amber-500">
              {strategies.filter((s) => s.status === 'draft').length}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Available Factors</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{factors.length}</div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="templates">
        <TabsList>
          <TabsTrigger value="templates">
            <FileCode className="h-4 w-4 mr-1" />
            From Template
            <Badge variant="secondary" className="ml-2">
              {STRATEGY_TEMPLATES.length}
            </Badge>
          </TabsTrigger>
          <TabsTrigger value="create">Create Custom</TabsTrigger>
          <TabsTrigger value="manage">
            Manage Strategies
            {strategies.length > 0 && (
              <Badge variant="secondary" className="ml-2">
                {strategies.length}
              </Badge>
            )}
          </TabsTrigger>
        </TabsList>

        {/* From Template Tab */}
        <TabsContent value="templates" className="space-y-6 mt-6">
          {/* Category Filter */}
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">Category:</span>
            <div className="flex gap-1">
              <Button
                size="sm"
                variant={templateCategory === 'all' ? 'default' : 'outline'}
                onClick={() => setTemplateCategory('all')}
              >
                All
              </Button>
              {(Object.keys(CATEGORY_LABELS) as StrategyCategory[]).map((cat) => (
                <Button
                  key={cat}
                  size="sm"
                  variant={templateCategory === cat ? 'default' : 'outline'}
                  onClick={() => setTemplateCategory(cat)}
                  className="gap-1"
                >
                  {CATEGORY_ICONS[cat]}
                  {CATEGORY_LABELS[cat]}
                </Button>
              ))}
            </div>
          </div>

          {/* Template Grid */}
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {filteredTemplates.map((template) => (
              <TemplateCard
                key={template.id}
                template={template}
                onSelect={handleCreateFromTemplate}
                isCreating={creatingFromTemplate}
              />
            ))}
          </div>

          {filteredTemplates.length === 0 && (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Shield className="h-12 w-12 text-muted-foreground mb-4" />
                <p className="text-lg font-medium">No templates in this category</p>
                <p className="text-sm text-muted-foreground">
                  Try selecting a different category
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Create Strategy Tab */}
        <TabsContent value="create" className="space-y-6 mt-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Wrench className="h-5 w-5" />
                New Strategy
              </CardTitle>
              <CardDescription>Configure a new trading strategy</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Basic Info */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="strategy-name">Strategy Name *</Label>
                  <Input
                    id="strategy-name"
                    placeholder="e.g., Momentum Alpha v1"
                    value={strategyName}
                    onChange={(e) => setStrategyName(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="strategy-desc">Description</Label>
                  <Input
                    id="strategy-desc"
                    placeholder="Optional description"
                    value={strategyDescription}
                    onChange={(e) => setStrategyDescription(e.target.value)}
                  />
                </div>
              </div>

              {/* Factor Selection */}
              <div className="space-y-2">
                <Label>Select Factors ({selectedFactors.length} selected)</Label>
                <div className="max-h-48 overflow-y-auto border rounded-md p-3 grid grid-cols-3 gap-2">
                  {factors.length === 0 ? (
                    <p className="text-sm text-muted-foreground col-span-3">
                      No factors available. Create factors first.
                    </p>
                  ) : (
                    factors.map((factor) => (
                      <Button
                        key={factor.id}
                        type="button"
                        variant={selectedFactors.includes(factor.id) ? 'default' : 'outline'}
                        size="sm"
                        className="justify-start text-xs"
                        onClick={() => toggleFactor(factor.id)}
                      >
                        {factor.name}
                      </Button>
                    ))
                  )}
                </div>
              </div>

              {/* Configuration */}
              <div className="grid grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label>Weighting Method</Label>
                  <Select
                    options={WEIGHTING_METHODS}
                    value={weightingMethod}
                    onChange={(e) => setWeightingMethod(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Rebalance Frequency</Label>
                  <Select
                    options={REBALANCE_FREQUENCIES}
                    value={rebalanceFrequency}
                    onChange={(e) => setRebalanceFrequency(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Universe</Label>
                  <Select
                    options={UNIVERSE_OPTIONS}
                    value={universe}
                    onChange={(e) => setUniverse(e.target.value)}
                  />
                </div>
              </div>

              {/* Max Positions */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label>Max Positions</Label>
                  <span className="text-2xl font-bold">{maxPositions}</span>
                </div>
                <Slider
                  value={[maxPositions]}
                  onValueChange={(v) => setMaxPositions(v[0])}
                  min={1}
                  max={100}
                  step={1}
                />
              </div>

              {/* Long Only */}
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Long Only</Label>
                  <p className="text-sm text-muted-foreground">
                    Only take long positions (no shorting)
                  </p>
                </div>
                <Switch checked={longOnly} onCheckedChange={setLongOnly} />
              </div>

              {/* Submit */}
              <Button
                className="w-full"
                size="lg"
                onClick={handleCreateStrategy}
                disabled={creating || !strategyName.trim()}
              >
                {creating ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Creating...
                  </>
                ) : (
                  <>
                    <Plus className="h-4 w-4 mr-2" />
                    Create Strategy
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Manage Strategies Tab */}
        <TabsContent value="manage" className="mt-6">
          {loading ? (
            <div className="flex items-center justify-center h-48">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
          ) : strategies.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Settings2 className="h-12 w-12 text-muted-foreground mb-4" />
                <p className="text-lg font-medium">No strategies yet</p>
                <p className="text-sm text-muted-foreground">
                  Create your first strategy to get started
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {strategies.map((strategy) => (
                <StrategyCard
                  key={strategy.id}
                  strategy={strategy}
                  onDelete={deleteStrategy}
                  onBacktest={handleBacktest}
                />
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}
