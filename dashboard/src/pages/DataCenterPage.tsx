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
import { useDataStatus, useDownloads, useDataRanges, useBinanceSymbols, useExtendedDataOptions } from '@/hooks/useData'
import { SymbolSelect } from '@/components/ui/symbol-select'
import { DataTypeSelect } from '@/components/ui/data-type-select'
import {
  Database,
  Download,
  RefreshCw,
  CheckCircle,
  XCircle,
  Loader2,
  BarChart3,
  Calendar,
  X,
  Play,
  HardDrive,
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
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="overview">
            <BarChart3 className="h-4 w-4 mr-2" />
            Overview
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

// ============== Download Management Section ==============

function DownloadManagementSection() {
  const { data, loading, starting, startDownload, cancelDownload, refetch } = useDownloads()
  const { options: extendedOptions, loading: optionsLoading } = useExtendedDataOptions()
  const { data: binanceSymbols, loading: symbolsLoading } = useBinanceSymbols({ limit: 200 })

  const [symbol, setSymbol] = useState('')
  const [marketType, setMarketType] = useState('spot')
  const [timeframe, setTimeframe] = useState('1h')
  const [dataTypes, setDataTypes] = useState<string[]>(['ohlcv'])
  const [exchange, setExchange] = useState('binance')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')

  const handleStart = async () => {
    if (!symbol || !startDate || dataTypes.length === 0) return
    // Format symbol from BTCUSDT to BTC/USDT
    const selectedSymbol = binanceSymbols?.symbols.find(s => s.symbol === symbol)
    const formattedSymbol = selectedSymbol
      ? `${selectedSymbol.base_asset}/${selectedSymbol.quote_asset}`
      : symbol

    // Start download for each selected data type
    for (const dt of dataTypes) {
      await startDownload({
        symbol: formattedSymbol,
        timeframe,
        exchange,
        data_type: dt,
        market_type: marketType,
        start_date: startDate,
        end_date: endDate || undefined,
      })
    }
  }

  const marketTypeOptions = extendedOptions?.market_types?.map(m => ({
    value: m.id,
    label: m.name
  })) || [
    { value: 'spot', label: '现货 (Spot)' },
    { value: 'futures', label: '合约 (USDT-M Futures)' },
  ]

  const timeframeOptions = extendedOptions?.timeframes.map(t => ({
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

  // Filter data types based on market type
  const allDataTypes = extendedOptions?.data_types || []
  const dataTypeOptions = marketType === 'futures'
    ? allDataTypes  // Futures: show all types
    : allDataTypes.filter(dt => !dt.requires_futures)  // Spot: hide futures-only types

  if (loading) {
    return <LoadingCard title="Download Management" />
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Download Management</CardTitle>
            <CardDescription>Download historical market data from Binance</CardDescription>
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

          {/* Row 1: Market Type, Symbol, Timeframe */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="space-y-2">
              <Label>Market Type</Label>
              <Select
                options={marketTypeOptions}
                value={marketType}
                onChange={(e) => {
                  setMarketType(e.target.value)
                  // Reset data types when switching market type
                  setDataTypes(['ohlcv'])
                }}
              />
            </div>
            <div className="space-y-2">
              <Label>Symbol (Top 200 by Volume)</Label>
              <SymbolSelect
                symbols={binanceSymbols?.symbols || []}
                value={symbol}
                onChange={setSymbol}
                loading={symbolsLoading}
                placeholder="Search or select symbol..."
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
                type="text"
                placeholder="YYYY-MM-DD"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
              />
            </div>
          </div>

          {/* Row 2: End Date */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="space-y-2">
              <Label>End Date (optional, defaults to now)</Label>
              <Input
                type="text"
                placeholder="YYYY-MM-DD"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
              />
            </div>
          </div>

          {/* Row 2: Data Types Multi-select */}
          <div className="space-y-2">
            <Label>Data Types (select multiple)</Label>
            <DataTypeSelect
              options={dataTypeOptions}
              value={dataTypes}
              onChange={setDataTypes}
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
            />
            {dataTypes.length === 0 && (
              <p className="text-sm text-orange-500">Please select at least one data type</p>
            )}
          </div>

          <Button onClick={handleStart} disabled={starting || !symbol || !startDate || dataTypes.length === 0}>
            {starting ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Play className="h-4 w-4 mr-2" />}
            Start Download ({dataTypes.length} data type{dataTypes.length !== 1 ? 's' : ''})
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
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-medium">{task.symbol}</span>
                    <Badge variant="outline">{task.timeframe}</Badge>
                    <Badge variant="secondary">{task.data_type || 'ohlcv'}</Badge>
                    <Badge variant="outline" className="text-xs">
                      {task.market_type === 'futures' ? '合约' : '现货'}
                    </Badge>
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
          <div className="border rounded-lg overflow-x-auto">
            <div className="grid grid-cols-7 gap-4 p-3 bg-muted font-medium text-sm min-w-[800px]">
              <div>Symbol</div>
              <div>Market</div>
              <div>Data Type</div>
              <div>Timeframe</div>
              <div>Start Date</div>
              <div>End Date</div>
              <div>Rows</div>
            </div>
            {data?.ranges.map((range, i) => (
              <div
                key={`${range.symbol}-${range.timeframe}-${range.market_type}-${i}`}
                className="grid grid-cols-7 gap-4 p-3 border-t items-center text-sm min-w-[800px]"
              >
                <div className="font-medium">{range.symbol}</div>
                <div>
                  <Badge variant="outline" className="text-xs">
                    {range.market_type === 'futures' ? '合约' : '现货'}
                  </Badge>
                </div>
                <div>
                  <Badge variant="secondary">{(range.data_type || 'ohlcv').toUpperCase()}</Badge>
                </div>
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
