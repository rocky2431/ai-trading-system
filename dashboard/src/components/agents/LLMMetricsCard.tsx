/**
 * LLM 指标卡片组件
 * 展示 LLM 延迟、成功率、Token 使用量和费用
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import type { LLMMetrics } from '@/types/agent'
import { Brain, Clock, CheckCircle, Coins, Zap } from 'lucide-react'

interface LLMMetricsCardProps {
  metrics: LLMMetrics
}

function formatNumber(num: number): string {
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
  return num.toString()
}

function formatLatency(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`
  return `${ms}ms`
}

export function LLMMetricsCard({ metrics }: LLMMetricsCardProps) {
  const successRateColor = metrics.successRate >= 99 ? 'text-emerald-500' :
                           metrics.successRate >= 95 ? 'text-amber-500' :
                           'text-red-500'

  const latencyColor = metrics.avgLatency <= 500 ? 'text-emerald-500' :
                       metrics.avgLatency <= 1000 ? 'text-amber-500' :
                       'text-red-500'

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5" />
          LLM Metrics
        </CardTitle>
        <CardDescription>
          {metrics.provider} / {metrics.model}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Success Rate */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-muted-foreground" />
                <span>Success Rate</span>
              </div>
              <span className={`font-semibold ${successRateColor}`}>
                {metrics.successRate.toFixed(1)}%
              </span>
            </div>
            <Progress value={metrics.successRate} className="h-2" />
          </div>

          {/* Latency */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-2">
                <Clock className="h-4 w-4 text-muted-foreground" />
                <span>Avg Latency</span>
              </div>
              <span className={`font-semibold ${latencyColor}`}>
                {formatLatency(metrics.avgLatency)}
              </span>
            </div>
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>P95: {formatLatency(metrics.p95Latency)}</span>
              <span>P99: {formatLatency(metrics.p99Latency)}</span>
            </div>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-2 gap-4 pt-2">
            <div className="space-y-1">
              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                <Zap className="h-3 w-3" />
                Total Requests
              </div>
              <div className="text-lg font-semibold">
                {formatNumber(metrics.totalRequests)}
              </div>
            </div>
            <div className="space-y-1">
              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                <Coins className="h-3 w-3" />
                Tokens Used
              </div>
              <div className="text-lg font-semibold">
                {formatNumber(metrics.tokensUsed)}
              </div>
            </div>
          </div>

          {/* Cost */}
          <div className="flex items-center justify-between pt-2 border-t">
            <span className="text-sm text-muted-foreground">Est. Cost (24h)</span>
            <span className="text-lg font-semibold text-primary">
              ${metrics.costEstimate.toFixed(2)}
            </span>
          </div>

          {/* Request Sparkline (Simple) */}
          <div className="space-y-2">
            <div className="text-xs text-muted-foreground">
              Requests (Last Hour)
            </div>
            <div className="flex items-end gap-0.5 h-8">
              {metrics.lastHourRequests.map((count, index) => {
                const maxCount = Math.max(...metrics.lastHourRequests)
                const height = maxCount > 0 ? (count / maxCount) * 100 : 0
                return (
                  <div
                    key={index}
                    className="flex-1 bg-primary/60 rounded-t-sm transition-all hover:bg-primary"
                    style={{ height: `${height}%` }}
                    title={`${count} requests`}
                  />
                )
              })}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
