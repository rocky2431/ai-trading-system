/**
 * PnL 曲线图表组件
 * 展示权益曲线和盈亏变化
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import type { PnLDataPoint } from '@/types/trading'
import { TrendingUp } from 'lucide-react'

interface PnLChartProps {
  history: PnLDataPoint[]
}

function formatTime(timestamp: string): string {
  return new Date(timestamp).toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function PnLChart({ history }: PnLChartProps) {
  if (history.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Equity Curve
          </CardTitle>
          <CardDescription>No data available</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64 flex items-center justify-center text-muted-foreground">
            No PnL history available
          </div>
        </CardContent>
      </Card>
    )
  }

  const equities = history.map(h => h.equity)
  const minEquity = Math.min(...equities)
  const maxEquity = Math.max(...equities)
  const range = maxEquity - minEquity || 1

  // 计算 SVG 路径
  const width = 600
  const height = 200
  const padding = { top: 20, right: 20, bottom: 30, left: 60 }
  const chartWidth = width - padding.left - padding.right
  const chartHeight = height - padding.top - padding.bottom

  const points = history.map((point, i) => {
    const x = padding.left + (i / (history.length - 1)) * chartWidth
    const y = padding.top + chartHeight - ((point.equity - minEquity) / range) * chartHeight
    return { x, y, ...point }
  })

  const linePath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ')

  // 渐变区域路径
  const areaPath = `
    ${linePath}
    L ${points[points.length - 1].x} ${padding.top + chartHeight}
    L ${points[0].x} ${padding.top + chartHeight}
    Z
  `

  // 计算 Y 轴刻度
  const yTicks = 5
  const yTickValues = Array.from({ length: yTicks }, (_, i) =>
    minEquity + (range * i) / (yTicks - 1)
  )

  // 当前值
  const currentEquity = history[history.length - 1].equity
  const firstEquity = history[0].equity
  const change = currentEquity - firstEquity
  const changePercent = (change / firstEquity) * 100

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Equity Curve
            </CardTitle>
            <CardDescription>24-hour performance</CardDescription>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold">
              ${currentEquity.toLocaleString('en-US', { minimumFractionDigits: 2 })}
            </div>
            <div className={`text-sm ${change >= 0 ? 'text-emerald-500' : 'text-red-500'}`}>
              {change >= 0 ? '+' : ''}{change.toFixed(2)} ({changePercent.toFixed(2)}%)
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto">
          <defs>
            <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={change >= 0 ? '#10b981' : '#ef4444'} stopOpacity="0.3" />
              <stop offset="100%" stopColor={change >= 0 ? '#10b981' : '#ef4444'} stopOpacity="0" />
            </linearGradient>
          </defs>

          {/* Y 轴网格线和标签 */}
          {yTickValues.map((value, i) => {
            const y = padding.top + chartHeight - ((value - minEquity) / range) * chartHeight
            return (
              <g key={i}>
                <line
                  x1={padding.left}
                  y1={y}
                  x2={width - padding.right}
                  y2={y}
                  stroke="currentColor"
                  strokeOpacity="0.1"
                />
                <text
                  x={padding.left - 8}
                  y={y}
                  textAnchor="end"
                  dominantBaseline="middle"
                  className="text-xs fill-muted-foreground"
                >
                  {(value / 1000).toFixed(1)}k
                </text>
              </g>
            )
          })}

          {/* X 轴标签 */}
          {[0, Math.floor(history.length / 2), history.length - 1].map((i) => (
            <text
              key={i}
              x={points[i].x}
              y={height - 8}
              textAnchor="middle"
              className="text-xs fill-muted-foreground"
            >
              {formatTime(history[i].timestamp)}
            </text>
          ))}

          {/* 渐变区域 */}
          <path d={areaPath} fill="url(#equityGradient)" />

          {/* 曲线 */}
          <path
            d={linePath}
            fill="none"
            stroke={change >= 0 ? '#10b981' : '#ef4444'}
            strokeWidth="2"
          />

          {/* 当前点 */}
          <circle
            cx={points[points.length - 1].x}
            cy={points[points.length - 1].y}
            r="4"
            fill={change >= 0 ? '#10b981' : '#ef4444'}
          />
        </svg>

        {/* 图例 */}
        <div className="flex items-center justify-center gap-6 mt-4 text-xs text-muted-foreground">
          <div className="flex items-center gap-1">
            <div className="w-3 h-0.5 bg-emerald-500" />
            <span>Realized PnL: ${history[history.length - 1].realizedPnl.toFixed(2)}</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-emerald-500/30 rounded" />
            <span>Unrealized PnL: ${history[history.length - 1].unrealizedPnl.toFixed(2)}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
