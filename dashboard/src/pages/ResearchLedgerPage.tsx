/**
 * Research Ledger 页面
 * 展示研究试验历史、动态阈值和过拟合风险
 *
 * P2 增强：添加阈值详情和审批记录
 */

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { TrialHistoryTable } from '@/components/research/TrialHistoryTable'
import { ThresholdChart } from '@/components/research/ThresholdChart'
import { OverfittingRiskCard } from '@/components/research/OverfittingRiskCard'
import { ApprovalHistoryCard } from '@/components/research/ApprovalHistoryCard'
import { useResearchLedger } from '@/hooks/useResearchLedger'
import {
  Beaker,
  CheckCircle,
  XCircle,
  AlertCircle,
  TrendingUp,
  Percent,
  Loader2,
  Shield,
  History,
} from 'lucide-react'

export function ResearchLedgerPage() {
  const { data, loading, error } = useResearchLedger()

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-muted-foreground">Loading research ledger...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <Card className="max-w-md">
          <CardHeader>
            <CardTitle className="text-destructive">Error</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">{error.message}</p>
          </CardContent>
        </Card>
      </div>
    )
  }

  const { trials, stats, thresholdHistory, thresholdDetails, overfittingRisk, approvalRecords } = data

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Research Ledger</h1>
        <p className="text-muted-foreground">
          Track research trials and prevent overfitting with dynamic thresholds
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 grid-cols-2 lg:grid-cols-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Trials</CardTitle>
            <Beaker className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.totalTrials}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Passed</CardTitle>
            <CheckCircle className="h-4 w-4 text-emerald-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-emerald-500">
              {stats.passedTrials}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Failed</CardTitle>
            <XCircle className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-500">
              {stats.failedTrials}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Inconclusive</CardTitle>
            <AlertCircle className="h-4 w-4 text-amber-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-amber-500">
              {stats.inconclusiveTrials}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Pass Rate</CardTitle>
            <Percent className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {stats.passRate.toFixed(1)}%
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Sharpe</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {stats.averageSharpe.toFixed(2)}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Grid */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Threshold Chart with Details */}
        <ThresholdChart history={thresholdHistory} stats={stats} details={thresholdDetails} />

        {/* Overfitting Risk */}
        <OverfittingRiskCard risk={overfittingRisk} />
      </div>

      {/* Monthly Stats */}
      <Card>
        <CardHeader>
          <CardTitle>Trials by Month</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-32 flex items-end gap-2">
            {stats.trialsByMonth.map((month) => {
              const maxCount = Math.max(...stats.trialsByMonth.map(m => m.count))
              const height = maxCount > 0 ? (month.count / maxCount) * 100 : 0
              const passRate = month.count > 0 ? (month.passed / month.count) * 100 : 0

              return (
                <div
                  key={month.month}
                  className="flex-1 flex flex-col items-center gap-1"
                >
                  <div
                    className="w-full bg-muted rounded-t relative overflow-hidden"
                    style={{ height: `${height}%`, minHeight: '4px' }}
                  >
                    <div
                      className="absolute bottom-0 w-full bg-emerald-500"
                      style={{ height: `${passRate}%` }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground">
                    {month.month.split('-')[1]}
                  </span>
                </div>
              )
            })}
          </div>
          <div className="flex items-center justify-center gap-4 mt-4 text-xs text-muted-foreground">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-muted rounded" />
              <span>Total Trials</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-emerald-500 rounded" />
              <span>Passed</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Best & Worst Trials */}
      {(stats.bestTrial || stats.worstTrial) && (
        <div className="grid gap-4 md:grid-cols-2">
          {stats.bestTrial && (
            <Card className="border-emerald-500/50">
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-emerald-500" />
                  Best Trial
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="font-medium">{stats.bestTrial.factorName}</div>
                  <p className="text-sm text-muted-foreground">
                    {stats.bestTrial.hypothesis}
                  </p>
                  <div className="flex gap-4 text-sm">
                    <span>Adj. Sharpe: <strong>{stats.bestTrial.adjustedSharpe.toFixed(2)}</strong></span>
                    <span>IC: <strong>{(stats.bestTrial.metrics.ic * 100).toFixed(2)}%</strong></span>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {stats.worstTrial && (
            <Card className="border-red-500/50">
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <XCircle className="h-4 w-4 text-red-500" />
                  Worst Trial
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="font-medium">{stats.worstTrial.factorName}</div>
                  <p className="text-sm text-muted-foreground">
                    {stats.worstTrial.hypothesis}
                  </p>
                  <div className="flex gap-4 text-sm">
                    <span>Adj. Sharpe: <strong>{stats.worstTrial.adjustedSharpe.toFixed(2)}</strong></span>
                    <span>IC: <strong>{(stats.worstTrial.metrics.ic * 100).toFixed(2)}%</strong></span>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Trial History with Tabs */}
      <Tabs defaultValue="trials" className="space-y-4">
        <TabsList>
          <TabsTrigger value="trials" className="flex items-center gap-2">
            <History className="h-4 w-4" />
            Trial History
          </TabsTrigger>
          <TabsTrigger value="approvals" className="flex items-center gap-2">
            <Shield className="h-4 w-4" />
            Review History
            {approvalRecords.length > 0 && (
              <span className="ml-1 px-1.5 py-0.5 text-xs bg-primary/10 rounded-full">
                {approvalRecords.length}
              </span>
            )}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="trials">
          <TrialHistoryTable trials={trials} limit={15} />
        </TabsContent>

        <TabsContent value="approvals">
          <ApprovalHistoryCard records={approvalRecords} limit={20} />
        </TabsContent>
      </Tabs>
    </div>
  )
}
