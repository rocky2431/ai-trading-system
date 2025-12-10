/**
 * 过拟合风险卡片组件
 * 展示当前研究的过拟合风险评估
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import type { OverfittingRisk } from '@/types/research'
import { ShieldAlert, AlertTriangle, Info } from 'lucide-react'

interface OverfittingRiskCardProps {
  risk: OverfittingRisk
}

const riskConfig: Record<OverfittingRisk['level'], {
  label: string
  variant: 'success' | 'warning' | 'destructive' | 'default'
  color: string
  bgColor: string
}> = {
  low: {
    label: 'Low Risk',
    variant: 'success',
    color: 'text-emerald-500',
    bgColor: 'bg-emerald-500/10',
  },
  medium: {
    label: 'Medium Risk',
    variant: 'warning',
    color: 'text-amber-500',
    bgColor: 'bg-amber-500/10',
  },
  high: {
    label: 'High Risk',
    variant: 'destructive',
    color: 'text-orange-500',
    bgColor: 'bg-orange-500/10',
  },
  critical: {
    label: 'Critical Risk',
    variant: 'destructive',
    color: 'text-red-500',
    bgColor: 'bg-red-500/10',
  },
}

function getRiskColor(score: number): string {
  if (score < 30) return '[&>div]:bg-emerald-500'
  if (score < 50) return '[&>div]:bg-amber-500'
  if (score < 70) return '[&>div]:bg-orange-500'
  return '[&>div]:bg-red-500'
}

export function OverfittingRiskCard({ risk }: OverfittingRiskCardProps) {
  const config = riskConfig[risk.level]

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ShieldAlert className="h-5 w-5" />
          Overfitting Risk Assessment
        </CardTitle>
        <CardDescription>
          Analysis of potential overfitting in research trials
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Overall Risk Score */}
        <div className={`p-4 rounded-lg ${config.bgColor}`}>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Overall Risk Score</span>
            <Badge variant={config.variant}>{config.label}</Badge>
          </div>
          <div className="flex items-center gap-4">
            <div className={`text-4xl font-bold ${config.color}`}>
              {risk.score.toFixed(0)}
            </div>
            <div className="flex-1">
              <Progress
                value={risk.score}
                className={`h-3 ${getRiskColor(risk.score)}`}
              />
            </div>
          </div>
        </div>

        {/* Risk Factors */}
        <div className="space-y-4">
          <h4 className="text-sm font-medium flex items-center gap-2">
            <AlertTriangle className="h-4 w-4" />
            Risk Factors
          </h4>

          {/* Multiple Testing Penalty */}
          <div className="space-y-1">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Multiple Testing Penalty</span>
              <span className="font-medium">
                {risk.factors.multipleTestingPenalty.toFixed(0)}%
              </span>
            </div>
            <Progress
              value={risk.factors.multipleTestingPenalty}
              className={`h-2 ${getRiskColor(risk.factors.multipleTestingPenalty)}`}
            />
          </div>

          {/* Data Snooping Risk */}
          <div className="space-y-1">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Data Snooping Risk</span>
              <span className="font-medium">
                {risk.factors.dataSnopingRisk.toFixed(0)}%
              </span>
            </div>
            <Progress
              value={risk.factors.dataSnopingRisk}
              className={`h-2 ${getRiskColor(risk.factors.dataSnopingRisk)}`}
            />
          </div>

          {/* Parameter Sensitivity */}
          <div className="space-y-1">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Parameter Sensitivity</span>
              <span className="font-medium">
                {risk.factors.parameterSensitivity.toFixed(0)}%
              </span>
            </div>
            <Progress
              value={risk.factors.parameterSensitivity}
              className={`h-2 ${getRiskColor(risk.factors.parameterSensitivity)}`}
            />
          </div>

          {/* Out-of-Sample Degradation */}
          <div className="space-y-1">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Out-of-Sample Degradation</span>
              <span className="font-medium">
                {risk.factors.outOfSampleDegradation.toFixed(0)}%
              </span>
            </div>
            <Progress
              value={risk.factors.outOfSampleDegradation}
              className={`h-2 ${getRiskColor(risk.factors.outOfSampleDegradation)}`}
            />
          </div>
        </div>

        {/* Recommendation */}
        <div className="p-3 rounded-lg bg-muted/50">
          <div className="flex items-start gap-2">
            <Info className="h-4 w-4 mt-0.5 text-muted-foreground" />
            <div>
              <div className="text-sm font-medium mb-1">Recommendation</div>
              <p className="text-sm text-muted-foreground">
                {risk.recommendation}
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
