/**
 * 风控状态卡片组件
 * 展示当前风险等级和各项风控指标
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import type { RiskMetrics } from '@/types/trading'
import { Shield, AlertTriangle, AlertCircle, XCircle, CheckCircle } from 'lucide-react'

interface RiskStatusCardProps {
  risk: RiskMetrics
}

const riskConfig: Record<RiskMetrics['level'], {
  label: string
  variant: 'success' | 'warning' | 'destructive' | 'default'
  color: string
  bgColor: string
  Icon: typeof CheckCircle
}> = {
  normal: {
    label: 'Normal',
    variant: 'success',
    color: 'text-emerald-500',
    bgColor: 'bg-emerald-500/10',
    Icon: CheckCircle,
  },
  warning: {
    label: 'Warning',
    variant: 'warning',
    color: 'text-amber-500',
    bgColor: 'bg-amber-500/10',
    Icon: AlertTriangle,
  },
  danger: {
    label: 'Danger',
    variant: 'destructive',
    color: 'text-orange-500',
    bgColor: 'bg-orange-500/10',
    Icon: AlertCircle,
  },
  critical: {
    label: 'Critical',
    variant: 'destructive',
    color: 'text-red-500',
    bgColor: 'bg-red-500/10',
    Icon: XCircle,
  },
}

function getProgressColor(value: number, thresholds: [number, number, number]): string {
  if (value >= thresholds[2]) return '[&>div]:bg-red-500'
  if (value >= thresholds[1]) return '[&>div]:bg-orange-500'
  if (value >= thresholds[0]) return '[&>div]:bg-amber-500'
  return '[&>div]:bg-emerald-500'
}

// Alert severity style configuration
const alertStyles: Record<string, { bg: string; badge: 'destructive' | 'warning' }> = {
  critical: { bg: 'bg-red-500/10 text-red-500', badge: 'destructive' },
  danger: { bg: 'bg-orange-500/10 text-orange-500', badge: 'destructive' },
  warning: { bg: 'bg-amber-500/10 text-amber-500', badge: 'warning' },
}

function getAlertStyle(severity: string) {
  return alertStyles[severity] || alertStyles.warning
}

export function RiskStatusCard({ risk }: RiskStatusCardProps) {
  const config = riskConfig[risk.level]
  const StatusIcon = config.Icon

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Shield className="h-5 w-5" />
          Risk Control Status
        </CardTitle>
        <CardDescription>
          Real-time risk monitoring and alerts
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* 总体风险等级 */}
        <div className={`p-4 rounded-lg ${config.bgColor}`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <StatusIcon className={`h-8 w-8 ${config.color}`} />
              <div>
                <div className="text-sm text-muted-foreground">Risk Level</div>
                <div className={`text-xl font-bold ${config.color}`}>
                  {config.label}
                </div>
              </div>
            </div>
            <Badge variant={config.variant} className="text-sm">
              Active
            </Badge>
          </div>
        </div>

        {/* 风控指标 */}
        <div className="space-y-4">
          {/* 保证金使用率 */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Margin Usage</span>
              <span className="font-medium">{risk.marginUsagePercent.toFixed(1)}%</span>
            </div>
            <Progress
              value={risk.marginUsagePercent}
              className={`h-2 ${getProgressColor(risk.marginUsagePercent, [40, 60, 80])}`}
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>0%</span>
              <span className="text-amber-500">60%</span>
              <span className="text-red-500">80%</span>
              <span>100%</span>
            </div>
          </div>

          {/* 当前回撤 */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Current Drawdown</span>
              <span className="font-medium">{risk.currentDrawdownPercent.toFixed(2)}%</span>
            </div>
            <Progress
              value={(risk.currentDrawdownPercent / 20) * 100}
              className={`h-2 ${getProgressColor(risk.currentDrawdownPercent, [5, 10, 15])}`}
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>0%</span>
              <span className="text-amber-500">10%</span>
              <span className="text-red-500">20%</span>
            </div>
          </div>

          {/* 最大回撤 */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Max Drawdown</span>
              <span className="font-medium">{risk.maxDrawdownPercent.toFixed(2)}%</span>
            </div>
            <Progress
              value={(risk.maxDrawdownPercent / 20) * 100}
              className={`h-2 ${getProgressColor(risk.maxDrawdownPercent, [5, 10, 15])}`}
            />
          </div>

          {/* 持仓集中度 */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Position Concentration</span>
              <span className="font-medium">{risk.positionConcentration.toFixed(1)}%</span>
            </div>
            <Progress
              value={risk.positionConcentration}
              className={`h-2 ${getProgressColor(risk.positionConcentration, [50, 70, 85])}`}
            />
          </div>
        </div>

        {/* 告警列表 */}
        {risk.alerts.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-amber-500" />
              Active Alerts ({risk.alerts.length})
            </h4>
            <div className="space-y-2">
              {risk.alerts.map((alert) => {
                const style = getAlertStyle(alert.severity)
                return (
                  <div
                    key={alert.id}
                    className={`p-3 rounded-lg text-sm ${style.bg}`}
                  >
                    <div className="flex items-start justify-between">
                      <span>{alert.message}</span>
                      <Badge variant={style.badge} className="text-xs ml-2">
                        {alert.severity}
                      </Badge>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
