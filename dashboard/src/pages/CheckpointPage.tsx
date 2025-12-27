/**
 * Checkpoint 管理页面
 *
 * 提供 LangGraph 检查点的浏览、查看和恢复功能：
 * - 线程列表：查看所有 pipeline 执行线程
 * - 检查点列表：查看线程的所有检查点
 * - 状态查看：查看检查点详细状态
 * - 一键恢复：恢复到指定检查点
 */

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogBody,
} from '@/components/ui/dialog'
import { useCheckpoints } from '@/hooks/useConfig'
import type { CheckpointInfo } from '@/api/config'
import {
  GitBranch,
  Clock,
  Loader2,
  RotateCcw,
  Eye,
  ChevronRight,
  FileCode,
  Lightbulb,
  BarChart3,
  Layers,
  AlertCircle,
  CheckCircle,
  RefreshCw,
} from 'lucide-react'

// Phase badge colors
const phaseColors: Record<string, string> = {
  hypothesis: 'bg-purple-500/10 text-purple-500',
  generation: 'bg-blue-500/10 text-blue-500',
  evaluation: 'bg-amber-500/10 text-amber-500',
  assembly: 'bg-green-500/10 text-green-500',
  backtest: 'bg-cyan-500/10 text-cyan-500',
  complete: 'bg-emerald-500/10 text-emerald-500',
}

// Format date
function formatDate(dateStr: string): string {
  const date = new Date(dateStr)
  return date.toLocaleString('zh-CN', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

// Format relative time
function formatRelativeTime(dateStr: string): string {
  const date = new Date(dateStr)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMins / 60)
  const diffDays = Math.floor(diffHours / 24)

  if (diffMins < 1) return 'Just now'
  if (diffMins < 60) return `${diffMins}m ago`
  if (diffHours < 24) return `${diffHours}h ago`
  return `${diffDays}d ago`
}

export function CheckpointPage() {
  const {
    threads,
    selectedThread,
    checkpoints,
    checkpointState,
    loading,
    error,
    restoring,
    fetchThreads,
    fetchCheckpoints,
    fetchCheckpointState,
    restoreCheckpoint,
  } = useCheckpoints()

  const [selectedCheckpoint, setSelectedCheckpoint] = useState<CheckpointInfo | null>(null)
  const [stateDialogOpen, setStateDialogOpen] = useState(false)
  const [restoreDialogOpen, setRestoreDialogOpen] = useState(false)
  const [restoreResult, setRestoreResult] = useState<{ success: boolean; message: string } | null>(null)

  // Handle thread selection
  const handleThreadSelect = async (threadId: string) => {
    setSelectedCheckpoint(null)
    await fetchCheckpoints(threadId)
  }

  // Handle checkpoint selection
  const handleCheckpointSelect = async (checkpoint: CheckpointInfo) => {
    setSelectedCheckpoint(checkpoint)
    await fetchCheckpointState(checkpoint.thread_id, checkpoint.checkpoint_id)
    setStateDialogOpen(true)
  }

  // Handle restore
  const handleRestore = async () => {
    if (!selectedCheckpoint) return

    const result = await restoreCheckpoint(
      selectedCheckpoint.thread_id,
      selectedCheckpoint.checkpoint_id
    )
    setRestoreResult(result)
    setRestoreDialogOpen(false)

    // Refresh threads after restore
    if (result.success) {
      await fetchThreads()
    }
  }

  // Error state
  if (error) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <Card className="max-w-md">
          <CardHeader>
            <CardTitle className="text-destructive flex items-center gap-2">
              <AlertCircle className="h-5 w-5" />
              Error
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">{error}</p>
            <Button onClick={() => fetchThreads()} className="mt-4">
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Checkpoint Manager</h1>
          <p className="text-muted-foreground">
            Browse and restore LangGraph pipeline checkpoints
          </p>
        </div>
        <Button variant="outline" onClick={() => fetchThreads()} disabled={loading}>
          <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Thread List */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <GitBranch className="h-5 w-5" />
              Pipeline Threads
            </CardTitle>
            <CardDescription>
              {threads.length} threads found
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[500px] overflow-y-auto">
              {loading && threads.length === 0 ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                </div>
              ) : threads.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <GitBranch className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No threads found</p>
                  <p className="text-sm">Run a pipeline to create checkpoints</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {threads.map((thread) => (
                    <div
                      key={thread.thread_id}
                      className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                        selectedThread === thread.thread_id
                          ? 'border-primary bg-primary/5'
                          : 'border-border hover:border-primary/50 hover:bg-muted/50'
                      }`}
                      onClick={() => handleThreadSelect(thread.thread_id)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex-1 min-w-0">
                          <p className="font-medium truncate">
                            {thread.name || thread.thread_id.slice(0, 8)}
                          </p>
                          <div className="flex items-center gap-2 text-xs text-muted-foreground mt-1">
                            <Clock className="h-3 w-3" />
                            <span>{formatRelativeTime(thread.last_updated)}</span>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge variant="secondary" className="text-xs">
                            {thread.checkpoint_count} checkpoints
                          </Badge>
                          {thread.current_phase && (
                            <Badge className={phaseColors[thread.current_phase] || 'bg-muted'}>
                              {thread.current_phase}
                            </Badge>
                          )}
                          <ChevronRight className="h-4 w-4 text-muted-foreground" />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Checkpoint List */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Layers className="h-5 w-5" />
              Checkpoints
            </CardTitle>
            <CardDescription>
              {selectedThread
                ? `Checkpoints for thread ${selectedThread.slice(0, 8)}...`
                : 'Select a thread to view checkpoints'}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[500px] overflow-y-auto">
              {!selectedThread ? (
                <div className="text-center py-12 text-muted-foreground">
                  <Layers className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>Select a thread from the left panel</p>
                </div>
              ) : loading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                </div>
              ) : !checkpoints || checkpoints.checkpoints.length === 0 ? (
                <div className="text-center py-12 text-muted-foreground">
                  <AlertCircle className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No checkpoints in this thread</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {checkpoints.checkpoints.map((checkpoint, index) => (
                    <div
                      key={checkpoint.checkpoint_id}
                      className="p-4 rounded-lg border border-border hover:border-primary/50 transition-colors"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-3">
                            <div className="flex items-center justify-center w-8 h-8 rounded-full bg-muted text-sm font-medium">
                              {checkpoints.checkpoints.length - index}
                            </div>
                            <div>
                              <div className="flex items-center gap-2">
                                <Badge className={phaseColors[checkpoint.phase] || 'bg-muted'}>
                                  {checkpoint.phase}
                                </Badge>
                                <span className="text-xs text-muted-foreground">
                                  {checkpoint.checkpoint_id.slice(0, 12)}...
                                </span>
                              </div>
                              <div className="flex items-center gap-2 text-xs text-muted-foreground mt-1">
                                <Clock className="h-3 w-3" />
                                <span>{formatDate(checkpoint.created_at)}</span>
                              </div>
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleCheckpointSelect(checkpoint)}
                          >
                            <Eye className="h-4 w-4 mr-1" />
                            View State
                          </Button>
                          <Button
                            variant="default"
                            size="sm"
                            onClick={() => {
                              setSelectedCheckpoint(checkpoint)
                              setRestoreDialogOpen(true)
                            }}
                          >
                            <RotateCcw className="h-4 w-4 mr-1" />
                            Restore
                          </Button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* State Detail Dialog */}
      <Dialog open={stateDialogOpen} onOpenChange={setStateDialogOpen}>
        <DialogContent className="max-w-3xl">
          <DialogHeader onClose={() => setStateDialogOpen(false)}>
            <DialogTitle>Checkpoint State</DialogTitle>
          </DialogHeader>

          {selectedCheckpoint && (
            <div className="px-6 py-2 text-sm text-muted-foreground">
              Phase: <Badge className={phaseColors[selectedCheckpoint.phase] || 'bg-muted'}>
                {selectedCheckpoint.phase}
              </Badge>
              {' | '}
              Created: {formatDate(selectedCheckpoint.created_at)}
            </div>
          )}

          <DialogBody className="max-h-[60vh] overflow-y-auto">
            {loading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-6 w-6 animate-spin" />
              </div>
            ) : checkpointState ? (
              <div className="space-y-4">
                {/* Hypothesis */}
                {checkpointState.hypothesis && (
                  <Card>
                    <CardHeader className="py-3">
                      <CardTitle className="text-sm flex items-center gap-2">
                        <Lightbulb className="h-4 w-4 text-amber-500" />
                        Hypothesis
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="py-3">
                      <p className="text-sm text-muted-foreground">
                        {checkpointState.hypothesis}
                      </p>
                    </CardContent>
                  </Card>
                )}

                {/* Factors */}
                {checkpointState.factors && checkpointState.factors.length > 0 && (
                  <Card>
                    <CardHeader className="py-3">
                      <CardTitle className="text-sm flex items-center gap-2">
                        <FileCode className="h-4 w-4 text-blue-500" />
                        Generated Factors ({checkpointState.factors.length})
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="py-3">
                      <div className="space-y-2">
                        {checkpointState.factors.slice(0, 5).map((factor, i) => (
                          <div key={i} className="text-sm p-2 bg-muted rounded">
                            <span className="font-medium">
                              {(factor as Record<string, unknown>).name as string || `Factor ${i + 1}`}
                            </span>
                          </div>
                        ))}
                        {checkpointState.factors.length > 5 && (
                          <p className="text-xs text-muted-foreground">
                            ... and {checkpointState.factors.length - 5} more factors
                          </p>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Evaluation Results */}
                {checkpointState.evaluation_results && Object.keys(checkpointState.evaluation_results).length > 0 && (
                  <Card>
                    <CardHeader className="py-3">
                      <CardTitle className="text-sm flex items-center gap-2">
                        <BarChart3 className="h-4 w-4 text-green-500" />
                        Evaluation Results
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="py-3">
                      <pre className="text-xs bg-muted p-3 rounded overflow-auto max-h-40">
                        {JSON.stringify(checkpointState.evaluation_results, null, 2)}
                      </pre>
                    </CardContent>
                  </Card>
                )}

                {/* Strategy */}
                {checkpointState.strategy && (
                  <Card>
                    <CardHeader className="py-3">
                      <CardTitle className="text-sm flex items-center gap-2">
                        <Layers className="h-4 w-4 text-purple-500" />
                        Strategy
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="py-3">
                      <pre className="text-xs bg-muted p-3 rounded overflow-auto max-h-40">
                        {JSON.stringify(checkpointState.strategy, null, 2)}
                      </pre>
                    </CardContent>
                  </Card>
                )}

                {/* Backtest Results */}
                {checkpointState.backtest_results && (
                  <Card>
                    <CardHeader className="py-3">
                      <CardTitle className="text-sm flex items-center gap-2">
                        <BarChart3 className="h-4 w-4 text-cyan-500" />
                        Backtest Results
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="py-3">
                      <pre className="text-xs bg-muted p-3 rounded overflow-auto max-h-40">
                        {JSON.stringify(checkpointState.backtest_results, null, 2)}
                      </pre>
                    </CardContent>
                  </Card>
                )}
              </div>
            ) : (
              <div className="text-center py-12 text-muted-foreground">
                <AlertCircle className="h-8 w-8 mx-auto mb-2" />
                <p>Failed to load checkpoint state</p>
              </div>
            )}
          </DialogBody>

          <DialogFooter>
            <Button variant="outline" onClick={() => setStateDialogOpen(false)}>
              Close
            </Button>
            <Button
              onClick={() => {
                setStateDialogOpen(false)
                setRestoreDialogOpen(true)
              }}
            >
              <RotateCcw className="h-4 w-4 mr-2" />
              Restore This Checkpoint
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Restore Confirmation Dialog */}
      <Dialog open={restoreDialogOpen} onOpenChange={setRestoreDialogOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader onClose={() => setRestoreDialogOpen(false)}>
            <DialogTitle>Restore Checkpoint?</DialogTitle>
          </DialogHeader>

          <DialogBody>
            <p className="text-sm text-muted-foreground">
              This will restore the pipeline to the state at checkpoint{' '}
              <code className="text-xs bg-muted px-1 py-0.5 rounded">
                {selectedCheckpoint?.checkpoint_id.slice(0, 12)}...
              </code>
              {selectedCheckpoint && (
                <span>
                  {' '}(Phase: <strong>{selectedCheckpoint.phase}</strong>)
                </span>
              )}
            </p>
            <p className="text-sm text-muted-foreground mt-3">
              The pipeline will resume from this point. Any progress after this checkpoint will be lost.
            </p>
          </DialogBody>

          <DialogFooter>
            <Button variant="outline" onClick={() => setRestoreDialogOpen(false)} disabled={restoring}>
              Cancel
            </Button>
            <Button onClick={handleRestore} disabled={restoring}>
              {restoring ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Restoring...
                </>
              ) : (
                <>
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Restore
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Restore Result Toast */}
      {restoreResult && (
        <div
          className={`fixed bottom-4 right-4 p-4 rounded-lg shadow-lg flex items-center gap-3 ${
            restoreResult.success
              ? 'bg-emerald-500 text-white'
              : 'bg-destructive text-destructive-foreground'
          }`}
        >
          {restoreResult.success ? (
            <CheckCircle className="h-5 w-5" />
          ) : (
            <AlertCircle className="h-5 w-5" />
          )}
          <span>{restoreResult.message}</span>
          <Button
            variant="ghost"
            size="sm"
            className="ml-2 text-white hover:text-white/80"
            onClick={() => setRestoreResult(null)}
          >
            Dismiss
          </Button>
        </div>
      )}
    </div>
  )
}
