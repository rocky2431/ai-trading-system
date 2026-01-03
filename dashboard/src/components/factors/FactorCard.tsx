/**
 * 因子卡片组件
 * 展示单个因子的概要信息和关键指标
 */

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { Factor, FactorFamily, FactorStatus } from '@/types/factor'
import { Clock, User, Tag } from 'lucide-react'

interface FactorCardProps {
  factor: Factor
  onClick?: () => void
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
  })
}

export function FactorCard({ factor, onClick }: FactorCardProps) {
  const family = familyConfig[factor.family]
  const status = statusConfig[factor.status]

  return (
    <Card
      className="cursor-pointer transition-all hover:shadow-md hover:border-primary/50"
      onClick={onClick}
    >
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="text-base font-semibold line-clamp-1">
            {factor.name}
          </CardTitle>
          <Badge variant={status.variant} className="text-xs shrink-0">
            {status.label}
          </Badge>
        </div>
        <div className="flex items-center gap-2 mt-1">
          <span className={`text-xs px-2 py-0.5 rounded-full ${family.className}`}>
            {family.label}
          </span>
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground line-clamp-2 mb-3">
          {factor.description}
        </p>

        {/* Metrics */}
        {factor.latestMetrics ? (
          <div className="grid grid-cols-3 gap-2 mb-3">
            <div className="text-center">
              <div className="text-lg font-semibold text-primary">
                {(factor.latestMetrics.ic * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-muted-foreground">IC</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-semibold">
                {factor.latestMetrics.sharpe.toFixed(2)}
              </div>
              <div className="text-xs text-muted-foreground">Sharpe</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-semibold">
                {(factor.latestMetrics.stability * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-muted-foreground">Stability</div>
            </div>
          </div>
        ) : (
          <div className="text-center py-3 text-sm text-muted-foreground">
            No evaluation data
          </div>
        )}

        {/* Meta Info */}
        <div className="flex items-center justify-between text-xs text-muted-foreground border-t pt-2">
          <div className="flex items-center gap-1">
            <User className="h-3 w-3" />
            <span>{factor.authorName}</span>
          </div>
          <div className="flex items-center gap-1">
            <Clock className="h-3 w-3" />
            <span>{formatDate(factor.createdAt)}</span>
          </div>
        </div>

        {/* Tags */}
        {factor.tags.length > 0 && (
          <div className="flex items-center gap-1 mt-2 flex-wrap">
            <Tag className="h-3 w-3 text-muted-foreground" />
            {factor.tags.slice(0, 3).map((tag) => (
              <span
                key={tag}
                className="text-xs px-1.5 py-0.5 bg-muted rounded"
              >
                {tag}
              </span>
            ))}
            {factor.tags.length > 3 && (
              <span className="text-xs text-muted-foreground">
                +{factor.tags.length - 3}
              </span>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
