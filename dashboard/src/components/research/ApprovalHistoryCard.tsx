/**
 * 审批历史卡片组件
 * 展示因子代码的人工审批记录
 *
 * P2 增强：连接 HumanReview 系统显示审批历史
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { ApprovalRecord } from '@/types/research'
import {
  CheckCircle,
  XCircle,
  Clock,
  AlertCircle,
  Shield,
  User,
  Calendar,
  MessageSquare,
} from 'lucide-react'

interface ApprovalHistoryCardProps {
  records: ApprovalRecord[]
  limit?: number
}

const statusConfig: Record<
  ApprovalRecord['status'],
  { label: string; variant: 'success' | 'destructive' | 'warning' | 'secondary'; icon: typeof CheckCircle }
> = {
  approved: { label: 'Approved', variant: 'success', icon: CheckCircle },
  rejected: { label: 'Rejected', variant: 'destructive', icon: XCircle },
  pending: { label: 'Pending', variant: 'warning', icon: Clock },
  timeout: { label: 'Timeout', variant: 'secondary', icon: AlertCircle },
}

function formatDate(dateString: string | null): string {
  if (!dateString) return 'N/A'
  return new Date(dateString).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function formatRelativeTime(dateString: string | null): string {
  if (!dateString) return ''
  const date = new Date(dateString)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)

  if (diffMins < 60) return `${diffMins}m ago`
  if (diffHours < 24) return `${diffHours}h ago`
  return `${diffDays}d ago`
}

export function ApprovalHistoryCard({ records, limit = 10 }: ApprovalHistoryCardProps) {
  const displayRecords = records.slice(0, limit)

  // Calculate stats
  const approved = records.filter(r => r.status === 'approved').length
  const rejected = records.filter(r => r.status === 'rejected').length
  const pending = records.filter(r => r.status === 'pending').length
  const timeout = records.filter(r => r.status === 'timeout').length

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Shield className="h-5 w-5" />
          Human Review History
        </CardTitle>
        <CardDescription>
          Factor code approval decisions from the security review queue
        </CardDescription>
      </CardHeader>
      <CardContent>
        {/* Stats Summary */}
        <div className="grid grid-cols-4 gap-2 mb-4">
          <div className="text-center p-2 rounded-lg bg-emerald-500/10">
            <div className="text-lg font-bold text-emerald-600">{approved}</div>
            <div className="text-xs text-muted-foreground">Approved</div>
          </div>
          <div className="text-center p-2 rounded-lg bg-red-500/10">
            <div className="text-lg font-bold text-red-600">{rejected}</div>
            <div className="text-xs text-muted-foreground">Rejected</div>
          </div>
          <div className="text-center p-2 rounded-lg bg-amber-500/10">
            <div className="text-lg font-bold text-amber-600">{pending}</div>
            <div className="text-xs text-muted-foreground">Pending</div>
          </div>
          <div className="text-center p-2 rounded-lg bg-gray-500/10">
            <div className="text-lg font-bold text-gray-600">{timeout}</div>
            <div className="text-xs text-muted-foreground">Timeout</div>
          </div>
        </div>

        {/* Records List */}
        <div className="h-[280px] overflow-y-auto">
          <div className="space-y-2">
            {displayRecords.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <Shield className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>No review records yet</p>
                <p className="text-xs">Factor code reviews will appear here</p>
              </div>
            ) : (
              displayRecords.map((record) => {
                const status = statusConfig[record.status]
                const StatusIcon = status.icon

                return (
                  <div
                    key={record.requestId}
                    className="flex items-start gap-3 p-3 rounded-lg bg-muted/50 hover:bg-muted transition-colors"
                  >
                    {/* Status Icon */}
                    <StatusIcon
                      className={`h-5 w-5 shrink-0 mt-0.5 ${
                        record.status === 'approved' ? 'text-emerald-500' :
                        record.status === 'rejected' ? 'text-red-500' :
                        record.status === 'pending' ? 'text-amber-500' :
                        'text-gray-500'
                      }`}
                    />

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-medium text-sm truncate">
                          {record.factorName}
                        </span>
                        <Badge variant={status.variant} className="text-xs shrink-0">
                          {status.label}
                        </Badge>
                      </div>

                      {/* Reviewer and Reason */}
                      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                        {record.reviewer && (
                          <span className="flex items-center gap-1">
                            <User className="h-3 w-3" />
                            {record.reviewer}
                          </span>
                        )}
                        {record.decidedAt && (
                          <span className="flex items-center gap-1">
                            <Calendar className="h-3 w-3" />
                            {formatRelativeTime(record.decidedAt)}
                          </span>
                        )}
                      </div>

                      {/* Reason if rejected */}
                      {record.reason && (
                        <div className="mt-1.5 flex items-start gap-1 text-xs text-muted-foreground">
                          <MessageSquare className="h-3 w-3 mt-0.5 shrink-0" />
                          <span className="line-clamp-2">{record.reason}</span>
                        </div>
                      )}
                    </div>

                    {/* Time */}
                    <div className="text-xs text-muted-foreground shrink-0">
                      {formatDate(record.decidedAt || record.createdAt)}
                    </div>
                  </div>
                )
              })
            )}
          </div>
        </div>

        {records.length > limit && (
          <div className="mt-4 text-center">
            <span className="text-sm text-muted-foreground">
              Showing {limit} of {records.length} records
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
