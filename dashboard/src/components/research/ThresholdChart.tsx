/**
 * 动态阈值图表组件
 * 展示随试验次数增加的动态阈值变化
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import type { ThresholdHistory, ResearchStats } from '@/types/research'
import { TrendingUp, Target } from 'lucide-react'

interface ThresholdChartProps {
  history: ThresholdHistory[]
  stats: ResearchStats
}

export function ThresholdChart({ history, stats }: ThresholdChartProps) {
  const maxThreshold = Math.max(...history.map(h => h.threshold))
  const minThreshold = Math.min(...history.map(h => h.threshold))
  const range = maxThreshold - minThreshold || 1

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Target className="h-5 w-5" />
          Dynamic Threshold
        </CardTitle>
        <CardDescription>
          Threshold increases with trial count to prevent overfitting
        </CardDescription>
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
            Deflated Sharpe Ratio Formula
          </div>
          <code className="text-xs">
            threshold = 1.0 + 0.015 × N (trials)
          </code>
        </div>
      </CardContent>
    </Card>
  )
}
