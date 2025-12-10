/**
 * 紧急平仓按钮组件
 * 一键平仓所有持仓，带确认对话框
 */

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { AlertOctagon, Loader2, X, AlertTriangle } from 'lucide-react'

interface EmergencyCloseButtonProps {
  positionCount: number
  totalUnrealizedPnl: number
  isClosing: boolean
  onCloseAll: () => void
}

export function EmergencyCloseButton({
  positionCount,
  totalUnrealizedPnl,
  isClosing,
  onCloseAll,
}: EmergencyCloseButtonProps) {
  const [showConfirm, setShowConfirm] = useState(false)

  const handleConfirm = () => {
    onCloseAll()
    setShowConfirm(false)
  }

  return (
    <>
      <Card className="border-red-500/50">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-red-500">
            <AlertOctagon className="h-5 w-5" />
            Emergency Close
          </CardTitle>
          <CardDescription>
            Close all positions immediately at market price
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="p-3 rounded-lg bg-muted/50">
              <div className="text-muted-foreground">Open Positions</div>
              <div className="text-xl font-bold">{positionCount}</div>
            </div>
            <div className="p-3 rounded-lg bg-muted/50">
              <div className="text-muted-foreground">Unrealized PnL</div>
              <div className={`text-xl font-bold ${
                totalUnrealizedPnl >= 0 ? 'text-emerald-500' : 'text-red-500'
              }`}>
                {totalUnrealizedPnl >= 0 ? '+' : ''}${totalUnrealizedPnl.toFixed(2)}
              </div>
            </div>
          </div>

          <Button
            variant="destructive"
            size="lg"
            className="w-full"
            disabled={positionCount === 0 || isClosing}
            onClick={() => setShowConfirm(true)}
          >
            {isClosing ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Closing All Positions...
              </>
            ) : (
              <>
                <AlertOctagon className="mr-2 h-4 w-4" />
                Close All Positions
              </>
            )}
          </Button>

          <p className="text-xs text-muted-foreground text-center">
            This action will immediately close all open positions at current market prices.
            This cannot be undone.
          </p>
        </CardContent>
      </Card>

      {/* 确认对话框 */}
      {showConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          {/* 背景遮罩 */}
          <div
            className="absolute inset-0 bg-background/80 backdrop-blur-sm"
            onClick={() => setShowConfirm(false)}
          />

          {/* 对话框内容 */}
          <div className="relative z-10 w-full max-w-md mx-4 p-6 bg-card border rounded-lg shadow-lg">
            <button
              className="absolute top-4 right-4 text-muted-foreground hover:text-foreground"
              onClick={() => setShowConfirm(false)}
            >
              <X className="h-4 w-4" />
            </button>

            <div className="flex flex-col items-center text-center space-y-4">
              <div className="p-3 rounded-full bg-red-500/10">
                <AlertTriangle className="h-8 w-8 text-red-500" />
              </div>

              <div>
                <h3 className="text-lg font-semibold">Confirm Emergency Close</h3>
                <p className="text-sm text-muted-foreground mt-2">
                  You are about to close <strong>{positionCount} position{positionCount !== 1 ? 's' : ''}</strong> at
                  market price. This action cannot be undone.
                </p>
              </div>

              <div className="w-full p-4 rounded-lg bg-muted/50">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Positions to close:</span>
                  <span className="font-medium">{positionCount}</span>
                </div>
                <div className="flex justify-between text-sm mt-2">
                  <span className="text-muted-foreground">Unrealized PnL:</span>
                  <span className={`font-medium ${
                    totalUnrealizedPnl >= 0 ? 'text-emerald-500' : 'text-red-500'
                  }`}>
                    {totalUnrealizedPnl >= 0 ? '+' : ''}${totalUnrealizedPnl.toFixed(2)}
                  </span>
                </div>
              </div>

              <div className="flex gap-3 w-full">
                <Button
                  variant="outline"
                  className="flex-1"
                  onClick={() => setShowConfirm(false)}
                >
                  Cancel
                </Button>
                <Button
                  variant="destructive"
                  className="flex-1"
                  onClick={handleConfirm}
                >
                  Confirm Close All
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
