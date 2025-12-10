/**
 * Data Center Page - 数据管理中心
 */

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Label } from '@/components/ui/label'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { useDataStatus, useDataOptions, useSymbols, useDownloads, useDataRanges } from '@/hooks/useData'
import {
  Database,
  Plus,
  Trash2,
  Download,
  RefreshCw,
  CheckCircle,
  XCircle,
  Loader2,
  Clock,
  HardDrive,
  BarChart3,
  Calendar,
  X,
  Play,
} from 'lucide-react'

export function DataCenterPage() {
  const [activeTab, setActiveTab] = useState('overview')

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
          <Database className="h-8 w-8" />
          Data Center
        </h1>
        <p className="text-muted-foreground">
          Manage market data, download historical data, and monitor data status
        </p>
      </div>

      <StatusOverview />

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">
            <BarChart3 className="h-4 w-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="symbols">
            <HardDrive className="h-4 w-4 mr-2" />
            Symbols
          </TabsTrigger>
          <TabsTrigger value="downloads">
            <Download className="h-4 w-4 mr-2" />
            Downloads
          </TabsTrigger>
          <TabsTrigger value="ranges">
            <Calendar className="h-4 w-4 mr-2" />
            Data Ranges
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <DataOverviewSection />
        </TabsContent>

        <TabsContent value="symbols">
          <SymbolManagementSection />
        </TabsContent>

        <TabsContent value="downloads">
          <DownloadManagementSection />
        </TabsContent>

        <TabsContent value="ranges">
          <DataRangesSection />
        </TabsContent>
      </Tabs>
    </div>
  )
}

// ============== Status Overview ==============

function StatusOverview() {
  const { status, loading, refetch } = useDataStatus()

  if (loading) {
    return (
      <Card>
        <CardContent className="py-6">
          <div className="flex items-center justify-center">
            <Loader2 className="h-6 w-6 animate-spin mr-2" />
            Loading status...
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center gap-2">
            {status?.database.connected ? (
              <CheckCircle className="h-5 w-5 text-green-500" />
            ) : (
              <XCircle className="h-5 w-5 text-red-500" />
            )}
            <div>
              <p className="text-sm font-medium">Database</p>
              <p className="text-xs text-muted-foreground">
                {status?.database.connected ? 'Connected' : 'Disconnected'}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center gap-2">
            <HardDrive className="h-5 w-5 text-blue-500" />
            <div>
              <p className="text-sm font-medium">{status?.overview.total_symbols || 0} Symbols</p>
              <p className="text-xs text-muted-foreground">
                {status?.overview.total_rows?.toLocaleString() || 0} rows
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center gap-2">
            <Database className="h-5 w-5 text-purple-500" />
            <div>
              <p className="text-sm font-medium">{status?.database.total_size_mb?.toFixed(1) || 0} MB</p>
              <p className="text-xs text-muted-foreground">
                {status?.database.hypertables_enabled ? 'TimescaleDB' : 'PostgreSQL'}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center gap-2">
            <Download className="h-5 w-5 text-orange-500" />
            <div>
              <p className="text-sm font-medium">{status?.active_downloads || 0} Active</p>
              <p className="text-xs text-muted-foreground">Downloads</p>
            </div>
          </div>
          <Button variant="ghost" size="sm" className="mt-2" onClick={refetch}>
            <RefreshCw className="h-4 w-4" />
          </Button>
        </CardContent>
      </Card>
    </div>
  )
}

// ============== Data Overview Section ==============

function DataOverviewSection() {
  const { status, loading } = useDataStatus()

  if (loading) {
    return <LoadingCard title="Data Overview" />
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Data Overview</CardTitle>
        <CardDescription>Summary of available market data</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 bg-muted rounded-lg text-center">
            <p className="text-2xl font-bold">{status?.overview.total_symbols || 0}</p>
            <p className="text-sm text-muted-foreground">Total Symbols</p>
          </div>
          <div className="p-4 bg-muted rounded-lg text-center">
            <p className="text-2xl font-bold">{status?.overview.total_rows?.toLocaleString() || 0}</p>
            <p className="text-sm text-muted-foreground">Total Rows</p>
          </div>
          <div className="p-4 bg-muted rounded-lg text-center">
            <p className="text-2xl font-bold">{status?.database.total_size_mb?.toFixed(1) || 0}</p>
            <p className="text-sm text-muted-foreground">Size (MB)</p>
          </div>
          <div className="p-4 bg-muted rounded-lg text-center">
            <p className="text-2xl font-bold">{status?.active_downloads || 0}</p>
            <p className="text-sm text-muted-foreground">Active Downloads</p>
          </div>
        </div>

        {status?.overview.oldest_data && status?.overview.newest_data && (
          <div className="p-4 border rounded-lg">
            <h4 className="font-medium mb-2">Data Range</h4>
            <div className="flex items-center gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">From:</span>{' '}
                {new Date(status.overview.oldest_data).toLocaleDateString()}
              </div>
              <div>
                <span className="text-muted-foreground">To:</span>{' '}
                {new Date(status.overview.newest_data).toLocaleDateString()}
              </div>
            </div>
          </div>
        )}

        <div className="p-4 border rounded-lg">
          <h4 className="font-medium mb-2">Database Info</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-muted-foreground">Status:</span>{' '}
              <Badge variant={status?.database.connected ? 'default' : 'destructive'}>
                {status?.database.connected ? 'Connected' : 'Disconnected'}
              </Badge>
            </div>
            <div>
              <span className="text-muted-foreground">Type:</span>{' '}
              {status?.database.hypertables_enabled ? 'TimescaleDB' : 'PostgreSQL'}
            </div>
            {status?.database.version && (
              <div className="col-span-2">
                <span className="text-muted-foreground">Version:</span>{' '}
                {status.database.version.split(' ')[0]}
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

// ============== Symbol Management Section ==============

function SymbolManagementSection() {
  const { data, loading, adding, addSymbol, removeSymbol, refetch } = useSymbols()
  const { options } = useDataOptions()

  const [newSymbol, setNewSymbol] = useState('')
  const [exchange, setExchange] = useState('binance')

  const handleAdd = async () => {
    if (!newSymbol.trim()) return
    const result = await addSymbol({ symbol: newSymbol.trim(), exchange })
    if (result.success) {
      setNewSymbol('')
    }
  }

  const exchangeOptions = options?.exchanges.map(e => ({ value: e.id, label: e.name })) || [
    { value: 'binance', label: 'Binance' }
  ]

  if (loading) {
    return <LoadingCard title="Symbol Management" />
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Symbol Management</CardTitle>
            <CardDescription>Add and manage trading symbols</CardDescription>
          </div>
          <Button variant="ghost" size="sm" onClick={refetch}>
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Add Symbol */}
        <div className="flex gap-4">
          <div className="flex-1">
            <Input
              placeholder="Enter symbol (e.g., BTC/USDT)"
              value={newSymbol}
              onChange={(e) => setNewSymbol(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
            />
          </div>
          <Select
            options={exchangeOptions}
            value={exchange}
            onChange={(e) => setExchange(e.target.value)}
            className="w-40"
          />
          <Button onClick={handleAdd} disabled={adding || !newSymbol.trim()}>
            {adding ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
            Add
          </Button>
        </div>

        {/* Symbol List */}
        <div className="border rounded-lg">
          <div className="grid grid-cols-6 gap-4 p-3 bg-muted font-medium text-sm">
            <div>Symbol</div>
            <div>Exchange</div>
            <div>Timeframes</div>
            <div>Data Range</div>
            <div>Rows</div>
            <div></div>
          </div>
          {data?.symbols.length === 0 ? (
            <div className="p-8 text-center text-muted-foreground">
              No symbols configured. Add your first symbol above.
            </div>
          ) : (
            data?.symbols.map((symbol) => (
              <div
                key={symbol.symbol}
                className="grid grid-cols-6 gap-4 p-3 border-t items-center"
              >
                <div className="font-medium">{symbol.symbol}</div>
                <div className="text-sm">{symbol.exchange}</div>
                <div className="flex flex-wrap gap-1">
                  {symbol.has_1m && <Badge variant="outline">1m</Badge>}
                  {symbol.has_5m && <Badge variant="outline">5m</Badge>}
                  {symbol.has_15m && <Badge variant="outline">15m</Badge>}
                  {symbol.has_1h && <Badge variant="outline">1h</Badge>}
                  {symbol.has_4h && <Badge variant="outline">4h</Badge>}
                  {symbol.has_1d && <Badge variant="outline">1d</Badge>}
                </div>
                <div className="text-sm text-muted-foreground">
                  {symbol.data_start && symbol.data_end ? (
                    <>
                      {new Date(symbol.data_start).toLocaleDateString()} -{' '}
                      {new Date(symbol.data_end).toLocaleDateString()}
                    </>
                  ) : (
                    'No data'
                  )}
                </div>
                <div className="text-sm">{symbol.total_rows.toLocaleString()}</div>
                <div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => removeSymbol(symbol.symbol)}
                  >
                    <Trash2 className="h-4 w-4 text-red-500" />
                  </Button>
                </div>
              </div>
            ))
          )}
        </div>

        <p className="text-sm text-muted-foreground">
          Total: {data?.total || 0} symbols
        </p>
      </CardContent>
    </Card>
  )
}

// ============== Download Management Section ==============

function DownloadManagementSection() {
  const { data, loading, starting, startDownload, cancelDownload, refetch } = useDownloads()
  const { options } = useDataOptions()
  const { data: symbolsData } = useSymbols()

  const [symbol, setSymbol] = useState('')
  const [timeframe, setTimeframe] = useState('1h')
  const [exchange, setExchange] = useState('binance')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')

  const handleStart = async () => {
    if (!symbol || !startDate) return
    await startDownload({
      symbol,
      timeframe,
      exchange,
      start_date: startDate,
      end_date: endDate || undefined,
    })
  }

  const symbolOptions = [
    { value: '', label: 'Select symbol...' },
    ...(symbolsData?.symbols.map(s => ({ value: s.symbol, label: s.symbol })) || [])
  ]

  const timeframeOptions = options?.timeframes.map(t => ({
    value: t.id,
    label: t.name
  })) || [
    { value: '1m', label: '1 Minute' },
    { value: '5m', label: '5 Minutes' },
    { value: '15m', label: '15 Minutes' },
    { value: '1h', label: '1 Hour' },
    { value: '4h', label: '4 Hours' },
    { value: '1d', label: '1 Day' },
  ]

  if (loading) {
    return <LoadingCard title="Download Management" />
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Download Management</CardTitle>
            <CardDescription>Download historical market data</CardDescription>
          </div>
          <Button variant="ghost" size="sm" onClick={refetch}>
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* New Download Form */}
        <div className="p-4 border rounded-lg space-y-4">
          <h4 className="font-medium">Start New Download</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="space-y-2">
              <Label>Symbol</Label>
              <Select
                options={symbolOptions}
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Timeframe</Label>
              <Select
                options={timeframeOptions}
                value={timeframe}
                onChange={(e) => setTimeframe(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Start Date</Label>
              <Input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>End Date (optional)</Label>
              <Input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
              />
            </div>
          </div>
          <Button onClick={handleStart} disabled={starting || !symbol || !startDate}>
            {starting ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Play className="h-4 w-4 mr-2" />}
            Start Download
          </Button>
        </div>

        {/* Download Tasks */}
        <div className="space-y-3">
          <h4 className="font-medium">Download Tasks</h4>
          {data?.tasks.length === 0 ? (
            <div className="p-8 text-center text-muted-foreground border rounded-lg">
              No download tasks. Start a new download above.
            </div>
          ) : (
            data?.tasks.map((task) => (
              <div
                key={task.id}
                className="p-4 border rounded-lg space-y-2"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="font-medium">{task.symbol}</span>
                    <Badge variant="outline">{task.timeframe}</Badge>
                    <Badge variant={getStatusVariant(task.status)}>
                      {task.status}
                    </Badge>
                  </div>
                  {(task.status === 'pending' || task.status === 'running') && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => cancelDownload(task.id)}
                    >
                      <X className="h-4 w-4 text-red-500" />
                    </Button>
                  )}
                </div>
                <div className="text-sm text-muted-foreground">
                  {new Date(task.start_date).toLocaleDateString()} -{' '}
                  {new Date(task.end_date).toLocaleDateString()}
                </div>
                {task.status === 'running' && (
                  <div className="space-y-1">
                    <Progress value={task.progress} />
                    <p className="text-xs text-muted-foreground">
                      {task.progress.toFixed(1)}% - {task.rows_downloaded.toLocaleString()} rows
                    </p>
                  </div>
                )}
                {task.error_message && (
                  <p className="text-sm text-red-500">{task.error_message}</p>
                )}
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  )
}

function getStatusVariant(status: string): 'default' | 'secondary' | 'destructive' | 'outline' {
  switch (status) {
    case 'completed':
      return 'default'
    case 'running':
      return 'secondary'
    case 'failed':
      return 'destructive'
    default:
      return 'outline'
  }
}

// ============== Data Ranges Section ==============

function DataRangesSection() {
  const { data, loading, refetch } = useDataRanges()

  if (loading) {
    return <LoadingCard title="Data Ranges" />
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Data Ranges</CardTitle>
            <CardDescription>Data availability by symbol and timeframe</CardDescription>
          </div>
          <Button variant="ghost" size="sm" onClick={refetch}>
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {data?.ranges.length === 0 ? (
          <div className="p-8 text-center text-muted-foreground">
            No data available. Download some data first.
          </div>
        ) : (
          <div className="border rounded-lg">
            <div className="grid grid-cols-5 gap-4 p-3 bg-muted font-medium text-sm">
              <div>Symbol</div>
              <div>Timeframe</div>
              <div>Start Date</div>
              <div>End Date</div>
              <div>Rows</div>
            </div>
            {data?.ranges.map((range, i) => (
              <div
                key={`${range.symbol}-${range.timeframe}-${i}`}
                className="grid grid-cols-5 gap-4 p-3 border-t items-center text-sm"
              >
                <div className="font-medium">{range.symbol}</div>
                <div>
                  <Badge variant="outline">{range.timeframe}</Badge>
                </div>
                <div>
                  {range.start_date ? new Date(range.start_date).toLocaleDateString() : '-'}
                </div>
                <div>
                  {range.end_date ? new Date(range.end_date).toLocaleDateString() : '-'}
                </div>
                <div>{range.total_rows.toLocaleString()}</div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// ============== Shared Components ==============

function LoadingCard({ title }: { title: string }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 animate-spin mr-2" />
          Loading...
        </div>
      </CardContent>
    </Card>
  )
}
