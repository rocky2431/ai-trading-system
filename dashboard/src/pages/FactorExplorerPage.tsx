/**
 * 因子浏览器页面
 * 展示因子库列表、筛选、搜索和详情
 */

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { FactorCard } from '@/components/factors/FactorCard'
import { FactorDetailPanel } from '@/components/factors/FactorDetailPanel'
import { useFactors } from '@/hooks/useFactors'
import type { Factor, FactorFamily, FactorStatus } from '@/types/factor'
import {
  Search,
  Filter,
  Plus,
  Loader2,
  LayoutGrid,
  List,
  TrendingUp,
  CheckCircle,
  Clock,
  XCircle,
} from 'lucide-react'

const familyOptions = [
  { value: 'all', label: 'All Families' },
  { value: 'momentum', label: 'Momentum' },
  { value: 'value', label: 'Value' },
  { value: 'volatility', label: 'Volatility' },
  { value: 'liquidity', label: 'Liquidity' },
  { value: 'sentiment', label: 'Sentiment' },
  { value: 'fundamental', label: 'Fundamental' },
]

const statusOptions = [
  { value: 'all', label: 'All Status' },
  { value: 'approved', label: 'Approved' },
  { value: 'evaluating', label: 'Evaluating' },
  { value: 'draft', label: 'Draft' },
  { value: 'rejected', label: 'Rejected' },
  { value: 'archived', label: 'Archived' },
]

const sortOptions = [
  { value: 'createdAt', label: 'Created Date' },
  { value: 'name', label: 'Name' },
  { value: 'ic', label: 'IC' },
  { value: 'sharpe', label: 'Sharpe Ratio' },
  { value: 'stability', label: 'Stability' },
]

export function FactorExplorerPage() {
  const { factors, filter, setFilter, loading, stats } = useFactors()
  const [selectedFactor, setSelectedFactor] = useState<Factor | null>(null)
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-muted-foreground">Loading factors...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Factor Explorer</h1>
          <p className="text-muted-foreground">
            Browse, search, and manage your factor library
          </p>
        </div>
        <Button>
          <Plus className="h-4 w-4 mr-2" />
          New Factor
        </Button>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 grid-cols-2 sm:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Factors</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.total}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Approved</CardTitle>
            <CheckCircle className="h-4 w-4 text-emerald-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-emerald-500">
              {stats.byStatus.approved}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Evaluating</CardTitle>
            <Clock className="h-4 w-4 text-amber-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-amber-500">
              {stats.byStatus.evaluating}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Draft</CardTitle>
            <XCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.byStatus.draft}</div>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search factors by name, description, or tags..."
            className="pl-9"
            value={filter.search || ''}
            onChange={(e) => setFilter({ ...filter, search: e.target.value })}
          />
        </div>
        <div className="flex gap-2">
          <Select
            options={familyOptions}
            value={filter.family || 'all'}
            onChange={(e) =>
              setFilter({ ...filter, family: e.target.value as FactorFamily | 'all' })
            }
            className="w-36"
          />
          <Select
            options={statusOptions}
            value={filter.status || 'all'}
            onChange={(e) =>
              setFilter({ ...filter, status: e.target.value as FactorStatus | 'all' })
            }
            className="w-32"
          />
          <Select
            options={sortOptions}
            value={filter.sortBy || 'createdAt'}
            onChange={(e) =>
              setFilter({
                ...filter,
                sortBy: e.target.value as typeof filter.sortBy,
              })
            }
            className="w-36"
          />
          <Button
            variant="outline"
            size="icon"
            onClick={() =>
              setFilter({
                ...filter,
                sortOrder: filter.sortOrder === 'asc' ? 'desc' : 'asc',
              })
            }
          >
            {filter.sortOrder === 'asc' ? '↑' : '↓'}
          </Button>
          <div className="border-l pl-2 flex gap-1">
            <Button
              variant={viewMode === 'grid' ? 'default' : 'ghost'}
              size="icon"
              onClick={() => setViewMode('grid')}
            >
              <LayoutGrid className="h-4 w-4" />
            </Button>
            <Button
              variant={viewMode === 'list' ? 'default' : 'ghost'}
              size="icon"
              onClick={() => setViewMode('list')}
            >
              <List className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Results count */}
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">
          Showing {factors.length} of {stats.total} factors
        </p>
        {(filter.family !== 'all' || filter.status !== 'all' || filter.search) && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() =>
              setFilter({
                family: 'all',
                status: 'all',
                search: '',
                sortBy: 'createdAt',
                sortOrder: 'desc',
              })
            }
          >
            Clear filters
          </Button>
        )}
      </div>

      {/* Main Content */}
      <div className="flex gap-6">
        {/* Factor Grid/List */}
        <div className={selectedFactor ? 'flex-1' : 'w-full'}>
          {factors.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Filter className="h-12 w-12 text-muted-foreground mb-4" />
                <p className="text-lg font-medium">No factors found</p>
                <p className="text-sm text-muted-foreground">
                  Try adjusting your filters or create a new factor
                </p>
                <Button className="mt-4">
                  <Plus className="h-4 w-4 mr-2" />
                  Create Factor
                </Button>
              </CardContent>
            </Card>
          ) : viewMode === 'grid' ? (
            <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
              {factors.map((factor) => (
                <FactorCard
                  key={factor.id}
                  factor={factor}
                  onClick={() => setSelectedFactor(factor)}
                />
              ))}
            </div>
          ) : (
            <div className="space-y-2">
              {factors.map((factor) => (
                <Card
                  key={factor.id}
                  className="cursor-pointer hover:border-primary/50 transition-colors"
                  onClick={() => setSelectedFactor(factor)}
                >
                  <CardContent className="flex items-center gap-4 py-3">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-medium truncate">{factor.name}</span>
                        <Badge variant="outline" className="text-xs shrink-0">
                          {factor.family}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground truncate">
                        {factor.description}
                      </p>
                    </div>
                    {factor.latestMetrics && (
                      <div className="flex items-center gap-6 text-sm">
                        <div className="text-center">
                          <div className="font-medium">
                            {(factor.latestMetrics.ic * 100).toFixed(1)}%
                          </div>
                          <div className="text-xs text-muted-foreground">IC</div>
                        </div>
                        <div className="text-center">
                          <div className="font-medium">
                            {factor.latestMetrics.sharpe.toFixed(2)}
                          </div>
                          <div className="text-xs text-muted-foreground">Sharpe</div>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>

        {/* Detail Panel */}
        {selectedFactor && (
          <div className="w-96 shrink-0">
            <FactorDetailPanel
              factor={selectedFactor}
              onClose={() => setSelectedFactor(null)}
            />
          </div>
        )}
      </div>
    </div>
  )
}
