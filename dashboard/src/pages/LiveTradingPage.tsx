/**
 * 实盘监控页面
 * 展示实时持仓、PnL、风控状态和紧急平仓功能
 */

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { PositionsTable } from '@/components/trading/PositionsTable'
import { PnLChart } from '@/components/trading/PnLChart'
import { RiskStatusCard } from '@/components/trading/RiskStatusCard'
import { EmergencyCloseButton } from '@/components/trading/EmergencyCloseButton'
import { useLiveTrading } from '@/hooks/useLiveTrading'
import {
  Wallet,
  DollarSign,
  TrendingUp,
  TrendingDown,
  Activity,
  Wifi,
  WifiOff,
} from 'lucide-react'

function formatCurrency(value: number): string {
  return value.toLocaleString('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
  })
}

function formatTime(timestamp: string): string {
  return new Date(timestamp).toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

export function LiveTradingPage() {
  const {
    account,
    positions,
    pnlHistory,
    risk,
    isConnected,
    lastUpdated,
    isClosingAll,
    closeAllPositions,
    closePosition,
  } = useLiveTrading()

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Live Trading</h1>
          <p className="text-muted-foreground">
            Monitor positions and manage risk in real-time
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge
            variant={isConnected ? 'success' : 'destructive'}
            className="flex items-center gap-1"
          >
            {isConnected ? (
              <>
                <Wifi className="h-3 w-3" />
                Connected
              </>
            ) : (
              <>
                <WifiOff className="h-3 w-3" />
                Disconnected
              </>
            )}
          </Badge>
          <span className="text-xs text-muted-foreground">
            Last update: {formatTime(lastUpdated)}
          </span>
        </div>
      </div>

      {/* Account Summary Cards */}
      <div className="grid gap-4 grid-cols-2 lg:grid-cols-5">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Equity</CardTitle>
            <Wallet className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatCurrency(account.totalEquity)}
            </div>
            <p className="text-xs text-muted-foreground">
              Available: {formatCurrency(account.availableBalance)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Unrealized PnL</CardTitle>
            {account.unrealizedPnl >= 0 ? (
              <TrendingUp className="h-4 w-4 text-emerald-500" />
            ) : (
              <TrendingDown className="h-4 w-4 text-red-500" />
            )}
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${
              account.unrealizedPnl >= 0 ? 'text-emerald-500' : 'text-red-500'
            }`}>
              {account.unrealizedPnl >= 0 ? '+' : ''}
              {formatCurrency(account.unrealizedPnl)}
            </div>
            <p className="text-xs text-muted-foreground">
              Across {positions.length} positions
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Realized PnL</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${
              account.realizedPnl >= 0 ? 'text-emerald-500' : 'text-red-500'
            }`}>
              {account.realizedPnl >= 0 ? '+' : ''}
              {formatCurrency(account.realizedPnl)}
            </div>
            <p className="text-xs text-muted-foreground">
              Lifetime realized
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Today's PnL</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${
              account.todayPnl >= 0 ? 'text-emerald-500' : 'text-red-500'
            }`}>
              {account.todayPnl >= 0 ? '+' : ''}
              {formatCurrency(account.todayPnl)}
            </div>
            <p className={`text-xs ${
              account.todayPnlPercent >= 0 ? 'text-emerald-500' : 'text-red-500'
            }`}>
              {account.todayPnlPercent >= 0 ? '+' : ''}
              {account.todayPnlPercent.toFixed(2)}% today
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Margin Used</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatCurrency(account.marginUsed)}
            </div>
            <p className="text-xs text-muted-foreground">
              {risk.marginUsagePercent.toFixed(1)}% of equity
            </p>
          </CardContent>
        </Card>
      </div>

      {/* PnL Chart */}
      <PnLChart history={pnlHistory} />

      {/* Main Content Grid */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Positions Table - 2 columns */}
        <div className="lg:col-span-2">
          <PositionsTable
            positions={positions}
            onClosePosition={closePosition}
          />
        </div>

        {/* Risk & Emergency - 1 column */}
        <div className="space-y-6">
          <RiskStatusCard risk={risk} />
          <EmergencyCloseButton
            positionCount={positions.length}
            totalUnrealizedPnl={account.unrealizedPnl}
            isClosing={isClosingAll}
            onCloseAll={closeAllPositions}
          />
        </div>
      </div>
    </div>
  )
}
