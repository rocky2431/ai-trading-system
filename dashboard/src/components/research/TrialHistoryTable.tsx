/**
 * 试验历史列表组件
 * 展示所有研究试验的历史记录
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { Trial, TrialStatus } from '@/types/research'
import { CheckCircle, XCircle, AlertCircle, Clock, Beaker } from 'lucide-react'

interface TrialHistoryTableProps {
  trials: Trial[]
  limit?: number
}

const statusConfig: Record<TrialStatus, { label: string; variant: 'success' | 'destructive' | 'warning'; icon: typeof CheckCircle }> = {
  passed: { label: 'Passed', variant: 'success', icon: CheckCircle },
  failed: { label: 'Failed', variant: 'destructive', icon: XCircle },
  inconclusive: { label: 'Inconclusive', variant: 'warning', icon: AlertCircle },
}

function formatDate(dateString: string): string {
  return new Date(dateString).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}

export function TrialHistoryTable({ trials, limit }: TrialHistoryTableProps) {
  const displayTrials = limit ? trials.slice(0, limit) : trials

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Beaker className="h-5 w-5" />
          Trial History
        </CardTitle>
        <CardDescription>
          {trials.length} total trials recorded
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {displayTrials.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              No trials recorded yet
            </div>
          ) : (
            displayTrials.map((trial) => {
              const status = statusConfig[trial.status]
              const StatusIcon = status.icon

              return (
                <div
                  key={trial.id}
                  className="flex items-center gap-4 p-3 rounded-lg bg-muted/50 hover:bg-muted transition-colors"
                >
                  {/* Status Icon */}
                  <StatusIcon
                    className={`h-5 w-5 shrink-0 ${
                      trial.status === 'passed' ? 'text-emerald-500' :
                      trial.status === 'failed' ? 'text-red-500' :
                      'text-amber-500'
                    }`}
                  />

                  {/* Trial Info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm">
                        #{trial.trialNumber}
                      </span>
                      <span className="text-sm truncate">{trial.factorName}</span>
                    </div>
                    <p className="text-xs text-muted-foreground truncate">
                      {trial.hypothesis}
                    </p>
                  </div>

                  {/* Metrics */}
                  <div className="hidden md:flex items-center gap-4 text-sm">
                    <div className="text-center w-16">
                      <div className="font-medium">
                        {trial.adjustedSharpe.toFixed(2)}
                      </div>
                      <div className="text-xs text-muted-foreground">Adj. SR</div>
                    </div>
                    <div className="text-center w-16">
                      <div className="font-medium">
                        {trial.threshold.toFixed(2)}
                      </div>
                      <div className="text-xs text-muted-foreground">Thresh</div>
                    </div>
                    <div className="text-center w-20">
                      <div className="font-medium">
                        {(trial.metrics.ic * 100).toFixed(2)}%
                      </div>
                      <div className="text-xs text-muted-foreground">IC</div>
                    </div>
                  </div>

                  {/* Status & Date */}
                  <div className="flex items-center gap-3">
                    <Badge variant={status.variant} className="text-xs">
                      {status.label}
                    </Badge>
                    <div className="hidden sm:flex items-center gap-1 text-xs text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      {formatDate(trial.createdAt)}
                    </div>
                  </div>
                </div>
              )
            })
          )}
        </div>

        {limit && trials.length > limit && (
          <div className="mt-4 text-center">
            <span className="text-sm text-muted-foreground">
              Showing {limit} of {trials.length} trials
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
