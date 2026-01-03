/**
 * 因子详情面板组件
 * 展示因子的完整信息、代码和评估指标
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import type { Factor, FactorFamily, FactorStatus } from '@/types/factor'
import { X, Code, Activity, Clock, User, Play } from 'lucide-react'

interface FactorDetailPanelProps {
  factor: Factor
  onClose: () => void
}

const familyConfig: Record<FactorFamily, { label: string; className: string }> = {
  momentum: { label: 'Momentum', className: 'bg-blue-500/10 text-blue-500' },
  value: { label: 'Value', className: 'bg-green-500/10 text-green-500' },
  volatility: { label: 'Volatility', className: 'bg-orange-500/10 text-orange-500' },
  liquidity: { label: 'Liquidity', className: 'bg-purple-500/10 text-purple-500' },
  sentiment: { label: 'Sentiment', className: 'bg-pink-500/10 text-pink-500' },
  fundamental: { label: 'Fundamental', className: 'bg-cyan-500/10 text-cyan-500' },
}

// Status config aligned with backend: src/iqfmp/models/factor.py FactorStatus enum
const statusConfig: Record<FactorStatus, { label: string; variant: 'default' | 'secondary' | 'destructive' | 'success' | 'warning' }> = {
  candidate: { label: 'Candidate', variant: 'warning' },
  rejected: { label: 'Rejected', variant: 'destructive' },
  core: { label: 'Core', variant: 'success' },
  redundant: { label: 'Redundant', variant: 'secondary' },
}

function formatDate(dateString: string): string {
  return new Date(dateString).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function getMetricColor(value: number, thresholds: { good: number; warning: number }, higherIsBetter = true): string {
  if (higherIsBetter) {
    if (value >= thresholds.good) return 'text-emerald-500'
    if (value >= thresholds.warning) return 'text-amber-500'
    return 'text-red-500'
  } else {
    if (value <= thresholds.good) return 'text-emerald-500'
    if (value <= thresholds.warning) return 'text-amber-500'
    return 'text-red-500'
  }
}

export function FactorDetailPanel({ factor, onClose }: FactorDetailPanelProps) {
  const family = familyConfig[factor.family]
  const status = statusConfig[factor.status]
  const metrics = factor.latestMetrics

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="flex-row items-start justify-between space-y-0 pb-4">
        <div className="space-y-1">
          <CardTitle className="text-xl">{factor.name}</CardTitle>
          <CardDescription className="flex items-center gap-2">
            <span className={`text-xs px-2 py-0.5 rounded-full ${family.className}`}>
              {family.label}
            </span>
            <Badge variant={status.variant}>{status.label}</Badge>
          </CardDescription>
        </div>
        <Button variant="ghost" size="icon" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </CardHeader>

      <CardContent className="flex-1 overflow-auto space-y-6">
        {/* Description */}
        <div>
          <h4 className="text-sm font-medium mb-2">Description</h4>
          <p className="text-sm text-muted-foreground">{factor.description}</p>
        </div>

        {/* Code */}
        <div>
          <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
            <Code className="h-4 w-4" />
            Factor Code
          </h4>
          <pre className="text-xs bg-muted p-3 rounded-md overflow-x-auto">
            <code>{factor.code}</code>
          </pre>
        </div>

        {/* Metrics */}
        {metrics ? (
          <div>
            <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Evaluation Metrics
            </h4>
            <div className="grid grid-cols-2 gap-4">
              {/* IC */}
              <div className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">IC</span>
                  <span className={`font-medium ${getMetricColor(metrics.ic, { good: 0.04, warning: 0.02 })}`}>
                    {(metrics.ic * 100).toFixed(2)}%
                  </span>
                </div>
                <Progress value={Math.min(100, metrics.ic * 1000)} className="h-1.5" />
              </div>

              {/* ICIR */}
              <div className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">IC/IR</span>
                  <span className={`font-medium ${getMetricColor(metrics.icir, { good: 0.8, warning: 0.5 })}`}>
                    {metrics.icir.toFixed(2)}
                  </span>
                </div>
                <Progress value={Math.min(100, metrics.icir * 50)} className="h-1.5" />
              </div>

              {/* Sharpe */}
              <div className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Sharpe Ratio</span>
                  <span className={`font-medium ${getMetricColor(metrics.sharpe, { good: 1.5, warning: 1.0 })}`}>
                    {metrics.sharpe.toFixed(2)}
                  </span>
                </div>
                <Progress value={Math.min(100, metrics.sharpe * 33)} className="h-1.5" />
              </div>

              {/* Max Drawdown */}
              <div className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Max Drawdown</span>
                  <span className={`font-medium ${getMetricColor(metrics.maxDrawdown, { good: 10, warning: 20 }, false)}`}>
                    {metrics.maxDrawdown.toFixed(1)}%
                  </span>
                </div>
                <Progress value={metrics.maxDrawdown} className="h-1.5 [&>div]:bg-red-500" />
              </div>

              {/* Win Rate */}
              <div className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Win Rate</span>
                  <span className={`font-medium ${getMetricColor(metrics.winRate, { good: 55, warning: 50 })}`}>
                    {metrics.winRate.toFixed(1)}%
                  </span>
                </div>
                <Progress value={metrics.winRate} className="h-1.5" />
              </div>

              {/* Stability */}
              <div className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Stability</span>
                  <span className={`font-medium ${getMetricColor(metrics.stability, { good: 0.75, warning: 0.5 })}`}>
                    {(metrics.stability * 100).toFixed(0)}%
                  </span>
                </div>
                <Progress value={metrics.stability * 100} className="h-1.5" />
              </div>

              {/* Turnover */}
              <div className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Turnover</span>
                  <span className={`font-medium ${getMetricColor(metrics.turnover, { good: 30, warning: 50 }, false)}`}>
                    {metrics.turnover.toFixed(1)}%
                  </span>
                </div>
                <Progress value={Math.min(100, metrics.turnover)} className="h-1.5" />
              </div>

              {/* Evaluation Count */}
              <div className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Evaluations</span>
                  <span className="font-medium">{factor.evaluationCount}</span>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-6 text-muted-foreground">
            <Activity className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p>No evaluation data available</p>
            <Button variant="outline" size="sm" className="mt-2">
              <Play className="h-4 w-4 mr-2" />
              Run Evaluation
            </Button>
          </div>
        )}

        {/* Meta */}
        <div className="border-t pt-4">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="flex items-center gap-2 text-muted-foreground">
              <User className="h-4 w-4" />
              <span>{factor.authorName}</span>
            </div>
            <div className="flex items-center gap-2 text-muted-foreground">
              <Clock className="h-4 w-4" />
              <span>Created: {formatDate(factor.createdAt)}</span>
            </div>
          </div>
        </div>

        {/* Tags */}
        <div>
          <h4 className="text-sm font-medium mb-2">Tags</h4>
          <div className="flex flex-wrap gap-2">
            {factor.tags.map((tag) => (
              <Badge key={tag} variant="outline" className="text-xs">
                {tag}
              </Badge>
            ))}
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-2 pt-4 border-t">
          <Button variant="outline" className="flex-1">
            Edit Factor
          </Button>
          <Button className="flex-1">
            <Play className="h-4 w-4 mr-2" />
            Run Evaluation
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
