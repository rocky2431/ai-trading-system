/**
 * 动态阈值图表组件
 * 展示随试验次数增加的动态阈值变化
 *
 * P2 增强：添加 DSR 公式详情和配置参数
 */

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogBody,
} from '@/components/ui/dialog'
import type { ThresholdHistory, ThresholdDetails, ResearchStats } from '@/types/research'
import { TrendingUp, Target, Info, Calculator, BookOpen, Settings2 } from 'lucide-react'

interface ThresholdChartProps {
  history: ThresholdHistory[]
  stats: ResearchStats
  details?: ThresholdDetails
}

export function ThresholdChart({ history, stats, details }: ThresholdChartProps) {
  const [showDetails, setShowDetails] = useState(false)
  const maxThreshold = Math.max(...history.map(h => h.threshold))
  const minThreshold = Math.min(...history.map(h => h.threshold))
  const range = maxThreshold - minThreshold || 1

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              Dynamic Threshold
            </CardTitle>
            <CardDescription>
              Threshold increases with trial count to prevent overfitting
            </CardDescription>
          </div>
          {details && (
            <>
              <Button variant="outline" size="sm" onClick={() => setShowDetails(true)}>
                <Info className="h-4 w-4 mr-1" />
                Details
              </Button>
              <Dialog open={showDetails} onOpenChange={setShowDetails}>
                <DialogContent className="max-w-2xl">
                  <DialogHeader onClose={() => setShowDetails(false)}>
                    <DialogTitle className="flex items-center gap-2">
                      <Calculator className="h-5 w-5" />
                      Threshold Calculation Details
                    </DialogTitle>
                  </DialogHeader>
                  <div className="px-6 py-2 text-sm text-muted-foreground">
                    {details.formula.name} methodology
                  </div>

                  <DialogBody className="space-y-6">
                  {/* Formula Section */}
                  <div className="rounded-lg border p-4">
                    <h4 className="font-medium flex items-center gap-2 mb-3">
                      <BookOpen className="h-4 w-4" />
                      Mathematical Formula
                    </h4>
                    <div className="bg-muted p-3 rounded font-mono text-sm mb-3">
                      {details.formula.equation}
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {details.formula.description}
                    </p>
                    <div className="mt-3 text-xs text-muted-foreground italic">
                      Reference: {details.formula.reference}
                    </div>
                  </div>

                  {/* Components Breakdown */}
                  <div className="rounded-lg border p-4">
                    <h4 className="font-medium mb-3">Component Breakdown</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between items-center p-2 bg-muted/50 rounded">
                        <span className="text-muted-foreground">Expected Maximum:</span>
                        <code className="text-xs">{details.formula.components.expectedMax}</code>
                      </div>
                      <div className="flex justify-between items-center p-2 bg-muted/50 rounded">
                        <span className="text-muted-foreground">Confidence Multiplier:</span>
                        <code className="text-xs">{details.formula.components.confidenceMultiplier}</code>
                      </div>
                      <div className="flex justify-between items-center p-2 bg-muted/50 rounded">
                        <span className="text-muted-foreground">Adjustment Factor:</span>
                        <code className="text-xs">{details.formula.components.adjustment}</code>
                      </div>
                    </div>
                  </div>

                  {/* Configuration */}
                  <div className="rounded-lg border p-4">
                    <h4 className="font-medium flex items-center gap-2 mb-3">
                      <Settings2 className="h-4 w-4" />
                      Current Configuration
                    </h4>
                    <div className="grid grid-cols-3 gap-4">
                      <div className="text-center p-3 bg-blue-500/10 rounded-lg">
                        <div className="text-2xl font-bold text-blue-600">
                          {details.config.baseSharpeThreshold}
                        </div>
                        <div className="text-xs text-muted-foreground">Base Threshold</div>
                      </div>
                      <div className="text-center p-3 bg-green-500/10 rounded-lg">
                        <div className="text-2xl font-bold text-green-600">
                          {(details.config.confidenceLevel * 100).toFixed(0)}%
                        </div>
                        <div className="text-xs text-muted-foreground">Confidence Level</div>
                      </div>
                      <div className="text-center p-3 bg-purple-500/10 rounded-lg">
                        <div className="text-2xl font-bold text-purple-600">
                          {details.config.minTrialsForAdjustment}
                        </div>
                        <div className="text-xs text-muted-foreground">Min Trials</div>
                      </div>
                    </div>
                  </div>

                  {/* Current State */}
                  <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
                    <div>
                      <div className="text-sm text-muted-foreground">Current Threshold</div>
                      <div className="text-3xl font-bold">{details.currentThreshold.toFixed(3)}</div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-muted-foreground">Total Trials</div>
                      <div className="text-3xl font-bold">{details.nTrials}</div>
                    </div>
                  </div>
                </DialogBody>
              </DialogContent>
            </Dialog>
            </>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {/* Current Threshold */}
        <div className="flex items-center justify-between mb-4 p-3 rounded-lg bg-blue-600/10">
          <div>
            <div className="text-sm text-gray-500 dark:text-gray-400">Current Threshold</div>
            <div className="text-2xl font-bold text-blue-600">
              {stats.currentThreshold.toFixed(2)}
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-500 dark:text-gray-400">Total Trials</div>
            <div className="text-2xl font-bold">{stats.totalTrials}</div>
          </div>
        </div>

        {/* Chart */}
        <div className="h-40 relative">
          {/* Y-axis labels */}
          <div className="absolute left-0 top-0 bottom-0 w-10 flex flex-col justify-between text-xs text-gray-500 dark:text-gray-400">
            <span>{maxThreshold.toFixed(1)}</span>
            <span>{((maxThreshold + minThreshold) / 2).toFixed(1)}</span>
            <span>{minThreshold.toFixed(1)}</span>
          </div>

          {/* Chart area */}
          <div className="ml-12 h-full relative border-l border-b border-gray-200 dark:border-gray-700">
            {/* Grid lines */}
            <div className="absolute inset-0 flex flex-col justify-between pointer-events-none">
              {[0, 1, 2].map((i) => (
                <div key={i} className="border-t border-gray-200/50 dark:border-gray-700/50" />
              ))}
            </div>

            {/* Line chart */}
            <svg className="absolute inset-0 w-full h-full overflow-visible" viewBox="0 0 100 100" preserveAspectRatio="none">
              {/* Area fill */}
              <path
                d={`
                  M 0 100
                  ${history.map((point, i) => {
                    const x = history.length > 1 ? (i / (history.length - 1)) * 100 : 50
                    const y = 100 - ((point.threshold - minThreshold) / range) * 100
                    return `L ${x} ${y}`
                  }).join(' ')}
                  L 100 100
                  Z
                `}
                fill="url(#gradient)"
                opacity="0.3"
              />

              {/* Line */}
              <path
                d={`
                  M ${history.map((point, i) => {
                    const x = history.length > 1 ? (i / (history.length - 1)) * 100 : 50
                    const y = 100 - ((point.threshold - minThreshold) / range) * 100
                    return `${i === 0 ? '' : 'L '}${x} ${y}`
                  }).join(' ')}
                `}
                fill="none"
                stroke="#2563eb"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                vectorEffect="non-scaling-stroke"
              />

              {/* Points */}
              {history.map((point, i) => {
                const x = history.length > 1 ? (i / (history.length - 1)) * 100 : 50
                const y = 100 - ((point.threshold - minThreshold) / range) * 100
                return (
                  <circle
                    key={`${point.timestamp}-${i}`}
                    cx={x}
                    cy={y}
                    r="3"
                    fill="#2563eb"
                    className="cursor-pointer"
                    vectorEffect="non-scaling-stroke"
                  >
                    <title>
                      Trial #{point.trialCount}: {point.threshold.toFixed(2)}
                    </title>
                  </circle>
                )
              })}

              {/* Gradient definition */}
              <defs>
                <linearGradient id="gradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#2563eb" />
                  <stop offset="100%" stopColor="#2563eb" stopOpacity="0" />
                </linearGradient>
              </defs>
            </svg>
          </div>
        </div>

        {/* X-axis labels */}
        <div className="ml-12 flex justify-between mt-2 text-xs text-gray-500 dark:text-gray-400">
          <span>Trial #1</span>
          <span>Trial #{stats.totalTrials}</span>
        </div>

        {/* Formula explanation */}
        <div className="mt-4 p-3 rounded-lg bg-gray-100 dark:bg-gray-800 text-sm">
          <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 mb-1">
            <TrendingUp className="h-4 w-4" />
            {details?.formula.name || 'Deflated Sharpe Ratio Formula'}
          </div>
          <code className="text-xs">
            {details?.formula.equation || 'T = T₀ × (1 + E[max(Z₁,...,Zₙ)] × α × 0.15)'}
          </code>
          {details && (
            <div className="mt-2 text-xs text-muted-foreground">
              {details.formula.reference}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
