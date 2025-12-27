/**
 * Review Queue 页面
 * 人工审核队列 - 审批 LLM 生成的代码
 */

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Textarea } from '@/components/ui/textarea'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useReviewQueue } from '@/hooks/useReviewQueue'
import {
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle,
  Eye,
  Loader2,
  RefreshCw,
  Timer,
  ListChecks,
  History,
  Settings,
} from 'lucide-react'
import { ReviewRequest, ReviewDecision } from '@/api/review'

function formatTimeAgo(dateString: string): string {
  const date = new Date(dateString)
  const now = new Date()
  const seconds = Math.floor((now.getTime() - date.getTime()) / 1000)

  if (seconds < 60) return `${seconds}s ago`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  return `${Math.floor(seconds / 86400)}d ago`
}

function StatusBadge({ status }: { status: string }) {
  switch (status) {
    case 'pending':
      return <Badge variant="outline" className="bg-amber-50 text-amber-700 border-amber-200"><Clock className="h-3 w-3 mr-1" />Pending</Badge>
    case 'approved':
      return <Badge variant="outline" className="bg-emerald-50 text-emerald-700 border-emerald-200"><CheckCircle className="h-3 w-3 mr-1" />Approved</Badge>
    case 'rejected':
      return <Badge variant="outline" className="bg-red-50 text-red-700 border-red-200"><XCircle className="h-3 w-3 mr-1" />Rejected</Badge>
    case 'timeout':
      return <Badge variant="outline" className="bg-gray-50 text-gray-700 border-gray-200"><Timer className="h-3 w-3 mr-1" />Timeout</Badge>
    default:
      return <Badge variant="outline">{status}</Badge>
  }
}

export function ReviewQueuePage() {
  const {
    pendingRequests,
    decisionHistory,
    stats,
    config,
    pendingPage,
    setPendingPage,
    pendingTotal,
    historyPage,
    setHistoryPage,
    historyTotal,
    loading,
    error,
    refresh,
    approve,
    reject,
    processTimeouts,
  } = useReviewQueue()

  // 审核对话框状态
  const [selectedRequest, setSelectedRequest] = useState<ReviewRequest | null>(null)
  const [rejectReason, setRejectReason] = useState('')
  const [isApproving, setIsApproving] = useState(false)
  const [isRejecting, setIsRejecting] = useState(false)

  // 详情对话框状态
  const [viewingRequest, setViewingRequest] = useState<ReviewRequest | null>(null)

  const handleApprove = async () => {
    if (!selectedRequest) return
    setIsApproving(true)
    try {
      await approve(selectedRequest.request_id)
      setSelectedRequest(null)
    } finally {
      setIsApproving(false)
    }
  }

  const handleReject = async () => {
    if (!selectedRequest || !rejectReason.trim()) return
    setIsRejecting(true)
    try {
      await reject(selectedRequest.request_id, rejectReason)
      setSelectedRequest(null)
      setRejectReason('')
    } finally {
      setIsRejecting(false)
    }
  }

  if (loading && !stats) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-muted-foreground">Loading review queue...</p>
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
            <Button onClick={refresh} className="mt-4">
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
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Review Queue</h1>
          <p className="text-muted-foreground">
            Review and approve LLM-generated code before execution
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={processTimeouts}>
            <Timer className="h-4 w-4 mr-2" />
            Process Timeouts
          </Button>
          <Button variant="outline" onClick={refresh}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid gap-4 grid-cols-2 lg:grid-cols-5">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Pending</CardTitle>
              <Clock className="h-4 w-4 text-amber-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-amber-500">{stats.pending_count}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Approved</CardTitle>
              <CheckCircle className="h-4 w-4 text-emerald-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-emerald-500">{stats.approved_count}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Rejected</CardTitle>
              <XCircle className="h-4 w-4 text-red-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-500">{stats.rejected_count}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Timeout</CardTitle>
              <Timer className="h-4 w-4 text-gray-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-gray-500">{stats.timeout_count}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Avg Review Time</CardTitle>
              <AlertCircle className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {stats.average_review_time_seconds
                  ? `${Math.floor(stats.average_review_time_seconds / 60)}m`
                  : '-'}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Main Content Tabs */}
      <Tabs defaultValue="pending" className="space-y-4">
        <TabsList>
          <TabsTrigger value="pending">
            <ListChecks className="h-4 w-4 mr-2" />
            Pending ({stats?.pending_count || 0})
          </TabsTrigger>
          <TabsTrigger value="history">
            <History className="h-4 w-4 mr-2" />
            History
          </TabsTrigger>
          <TabsTrigger value="config">
            <Settings className="h-4 w-4 mr-2" />
            Configuration
          </TabsTrigger>
        </TabsList>

        {/* Pending Requests Tab */}
        <TabsContent value="pending">
          <Card>
            <CardHeader>
              <CardTitle>Pending Review Requests</CardTitle>
              <CardDescription>
                Click on a request to review and approve or reject
              </CardDescription>
            </CardHeader>
            <CardContent>
              {pendingRequests.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <CheckCircle className="h-12 w-12 mx-auto mb-4 text-emerald-500" />
                  <p>No pending reviews</p>
                  <p className="text-sm">All caught up!</p>
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Factor</TableHead>
                      <TableHead>Summary</TableHead>
                      <TableHead>Priority</TableHead>
                      <TableHead>Created</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {pendingRequests.map((request) => (
                      <TableRow key={request.request_id}>
                        <TableCell className="font-medium">{request.factor_name}</TableCell>
                        <TableCell className="max-w-md truncate">{request.code_summary}</TableCell>
                        <TableCell>
                          <Badge variant={request.priority > 5 ? 'destructive' : 'outline'}>
                            P{request.priority}
                          </Badge>
                        </TableCell>
                        <TableCell>{formatTimeAgo(request.created_at)}</TableCell>
                        <TableCell>
                          <StatusBadge status={request.status} />
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex justify-end gap-2">
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => setViewingRequest(request)}
                            >
                              <Eye className="h-4 w-4" />
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              className="text-emerald-600"
                              onClick={() => setSelectedRequest(request)}
                            >
                              Review
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}

              {/* Pagination */}
              {pendingTotal > 10 && (
                <div className="flex justify-center gap-2 mt-4">
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={pendingPage === 1}
                    onClick={() => setPendingPage(pendingPage - 1)}
                  >
                    Previous
                  </Button>
                  <span className="py-2 px-4 text-sm">
                    Page {pendingPage} of {Math.ceil(pendingTotal / 10)}
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={pendingPage * 10 >= pendingTotal}
                    onClick={() => setPendingPage(pendingPage + 1)}
                  >
                    Next
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* History Tab */}
        <TabsContent value="history">
          <Card>
            <CardHeader>
              <CardTitle>Decision History</CardTitle>
              <CardDescription>
                Past review decisions and outcomes
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Request ID</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Reviewer</TableHead>
                    <TableHead>Reason</TableHead>
                    <TableHead>Decided</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {decisionHistory.map((decision) => (
                    <TableRow key={decision.request_id}>
                      <TableCell className="font-mono text-sm">
                        {decision.request_id.slice(0, 8)}...
                      </TableCell>
                      <TableCell>
                        <StatusBadge status={decision.status} />
                      </TableCell>
                      <TableCell>{decision.reviewer || '-'}</TableCell>
                      <TableCell className="max-w-md truncate">
                        {decision.reason || '-'}
                      </TableCell>
                      <TableCell>{formatTimeAgo(decision.decided_at)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>

              {/* Pagination */}
              {historyTotal > 10 && (
                <div className="flex justify-center gap-2 mt-4">
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={historyPage === 1}
                    onClick={() => setHistoryPage(historyPage - 1)}
                  >
                    Previous
                  </Button>
                  <span className="py-2 px-4 text-sm">
                    Page {historyPage} of {Math.ceil(historyTotal / 10)}
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={historyPage * 10 >= historyTotal}
                    onClick={() => setHistoryPage(historyPage + 1)}
                  >
                    Next
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Configuration Tab */}
        <TabsContent value="config">
          <Card>
            <CardHeader>
              <CardTitle>Review Configuration</CardTitle>
              <CardDescription>
                Configure review queue settings
              </CardDescription>
            </CardHeader>
            <CardContent>
              {config && (
                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <label className="text-sm font-medium">Timeout (seconds)</label>
                      <p className="text-2xl font-bold">{config.timeout_seconds}</p>
                      <p className="text-sm text-muted-foreground">
                        {Math.floor(config.timeout_seconds / 60)} minutes
                      </p>
                    </div>
                    <div>
                      <label className="text-sm font-medium">Max Queue Size</label>
                      <p className="text-2xl font-bold">{config.max_queue_size}</p>
                    </div>
                    <div>
                      <label className="text-sm font-medium">Auto-reject on Timeout</label>
                      <p className="text-2xl font-bold">
                        {config.auto_reject_on_timeout ? 'Yes' : 'No'}
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Review Dialog */}
      <Dialog open={!!selectedRequest} onOpenChange={() => setSelectedRequest(null)}>
        <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Review: {selectedRequest?.factor_name}</DialogTitle>
            <DialogDescription>
              Review the generated code and approve or reject
            </DialogDescription>
          </DialogHeader>

          {selectedRequest && (
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium">Summary</label>
                <p className="text-muted-foreground">{selectedRequest.code_summary}</p>
              </div>

              <div>
                <label className="text-sm font-medium">Code</label>
                <pre className="bg-muted p-4 rounded-md overflow-x-auto text-sm">
                  <code>{selectedRequest.code}</code>
                </pre>
              </div>

              <div>
                <label className="text-sm font-medium">Rejection Reason (required for reject)</label>
                <Textarea
                  value={rejectReason}
                  onChange={(e) => setRejectReason(e.target.value)}
                  placeholder="Enter reason for rejection..."
                  className="mt-1"
                />
              </div>
            </div>
          )}

          <DialogFooter>
            <Button variant="outline" onClick={() => setSelectedRequest(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleReject}
              disabled={isRejecting || !rejectReason.trim()}
            >
              {isRejecting && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              <XCircle className="h-4 w-4 mr-2" />
              Reject
            </Button>
            <Button onClick={handleApprove} disabled={isApproving}>
              {isApproving && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              <CheckCircle className="h-4 w-4 mr-2" />
              Approve
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* View Details Dialog */}
      <Dialog open={!!viewingRequest} onOpenChange={() => setViewingRequest(null)}>
        <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Details: {viewingRequest?.factor_name}</DialogTitle>
          </DialogHeader>

          {viewingRequest && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium">Request ID</label>
                  <p className="font-mono text-sm">{viewingRequest.request_id}</p>
                </div>
                <div>
                  <label className="text-sm font-medium">Priority</label>
                  <p>P{viewingRequest.priority}</p>
                </div>
                <div>
                  <label className="text-sm font-medium">Created</label>
                  <p>{new Date(viewingRequest.created_at).toLocaleString()}</p>
                </div>
                <div>
                  <label className="text-sm font-medium">Status</label>
                  <StatusBadge status={viewingRequest.status} />
                </div>
              </div>

              <div>
                <label className="text-sm font-medium">Summary</label>
                <p className="text-muted-foreground">{viewingRequest.code_summary}</p>
              </div>

              <div>
                <label className="text-sm font-medium">Code</label>
                <pre className="bg-muted p-4 rounded-md overflow-x-auto text-sm">
                  <code>{viewingRequest.code}</code>
                </pre>
              </div>

              {viewingRequest.metadata && Object.keys(viewingRequest.metadata).length > 0 && (
                <div>
                  <label className="text-sm font-medium">Metadata</label>
                  <pre className="bg-muted p-4 rounded-md overflow-x-auto text-sm">
                    <code>{JSON.stringify(viewingRequest.metadata, null, 2)}</code>
                  </pre>
                </div>
              )}
            </div>
          )}

          <DialogFooter>
            <Button variant="outline" onClick={() => setViewingRequest(null)}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
