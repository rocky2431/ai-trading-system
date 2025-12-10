/**
 * 持仓列表组件
 * 展示当前所有持仓信息
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { Position } from '@/types/trading'
import { TrendingUp, TrendingDown, X } from 'lucide-react'

interface PositionsTableProps {
  positions: Position[]
  onClosePosition: (id: string) => void
}

function formatNumber(value: number, decimals: number = 2): string {
  return value.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

export function PositionsTable({ positions, onClosePosition }: PositionsTableProps) {
  if (positions.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Open Positions</CardTitle>
          <CardDescription>No open positions</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            No active positions at this time
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Open Positions</CardTitle>
        <CardDescription>
          {positions.length} active position{positions.length !== 1 ? 's' : ''}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b text-left text-sm text-muted-foreground">
                <th className="pb-3 font-medium">Symbol</th>
                <th className="pb-3 font-medium">Side</th>
                <th className="pb-3 font-medium text-right">Size</th>
                <th className="pb-3 font-medium text-right">Entry</th>
                <th className="pb-3 font-medium text-right">Mark</th>
                <th className="pb-3 font-medium text-right">PnL</th>
                <th className="pb-3 font-medium text-right">Liq. Price</th>
                <th className="pb-3 font-medium text-right">Action</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((position) => (
                <tr key={position.id} className="border-b last:border-0">
                  <td className="py-3">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{position.symbol}</span>
                      <Badge variant="secondary" className="text-xs">
                        {position.leverage}x
                      </Badge>
                    </div>
                  </td>
                  <td className="py-3">
                    <Badge
                      variant={position.side === 'long' ? 'success' : 'destructive'}
                      className="flex items-center gap-1 w-fit"
                    >
                      {position.side === 'long' ? (
                        <TrendingUp className="h-3 w-3" />
                      ) : (
                        <TrendingDown className="h-3 w-3" />
                      )}
                      {position.side.toUpperCase()}
                    </Badge>
                  </td>
                  <td className="py-3 text-right font-mono">
                    {formatNumber(position.size, 4)}
                  </td>
                  <td className="py-3 text-right font-mono">
                    ${formatNumber(position.entryPrice)}
                  </td>
                  <td className="py-3 text-right font-mono">
                    ${formatNumber(position.markPrice)}
                  </td>
                  <td className="py-3 text-right">
                    <div className={`font-mono ${
                      position.unrealizedPnl >= 0 ? 'text-emerald-500' : 'text-red-500'
                    }`}>
                      <div className="font-medium">
                        {position.unrealizedPnl >= 0 ? '+' : ''}
                        ${formatNumber(position.unrealizedPnl)}
                      </div>
                      <div className="text-xs">
                        {position.unrealizedPnlPercent >= 0 ? '+' : ''}
                        {formatNumber(position.unrealizedPnlPercent)}%
                      </div>
                    </div>
                  </td>
                  <td className="py-3 text-right font-mono text-orange-500">
                    ${formatNumber(position.liquidationPrice)}
                  </td>
                  <td className="py-3 text-right">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-8 w-8 p-0 hover:bg-red-500/10 hover:text-red-500"
                      onClick={() => onClosePosition(position.id)}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  )
}
