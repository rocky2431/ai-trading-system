/**
 * Data API - 数据管理 API
 */

import { api } from './client'

// ============== Types ==============

export interface DatabaseStatus {
  connected: boolean
  version: string | null
  hypertables_enabled: boolean
  total_size_mb: number
}

export interface DataOverview {
  total_symbols: number
  total_rows: number
  data_size_mb: number
  oldest_data: string | null
  newest_data: string | null
}

export interface DataStatusResponse {
  database: DatabaseStatus
  overview: DataOverview
  active_downloads: number
}

export interface SymbolInfo {
  symbol: string
  exchange: string
  base_asset: string
  quote_asset: string
  is_active: boolean
  has_1m: boolean
  has_5m: boolean
  has_15m: boolean
  has_1h: boolean
  has_4h: boolean
  has_1d: boolean
  data_start: string | null
  data_end: string | null
  total_rows: number
  created_at: string | null
  updated_at: string | null
}

export interface SymbolListResponse {
  symbols: SymbolInfo[]
  total: number
}

export interface AddSymbolRequest {
  symbol: string
  exchange?: string
}

export interface AddSymbolResponse {
  success: boolean
  message: string
  symbol: SymbolInfo | null
}

export interface DownloadTaskStatus {
  id: string
  symbol: string
  timeframe: string
  exchange: string
  start_date: string
  end_date: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  rows_downloaded: number
  error_message: string | null
  created_at: string | null
  started_at: string | null
  completed_at: string | null
}

export interface DownloadTaskListResponse {
  tasks: DownloadTaskStatus[]
  total: number
}

export interface StartDownloadRequest {
  symbol: string
  timeframe: string
  exchange?: string
  start_date: string
  end_date?: string
}

export interface StartDownloadResponse {
  success: boolean
  message: string
  task_id: string | null
}

export interface CancelDownloadResponse {
  success: boolean
  message: string
}

export interface OHLCVBar {
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface OHLCVDataResponse {
  symbol: string
  timeframe: string
  data: OHLCVBar[]
  total_rows: number
}

export interface DataRangeInfo {
  symbol: string
  timeframe: string
  start_date: string | null
  end_date: string | null
  total_rows: number
}

export interface DataRangeResponse {
  ranges: DataRangeInfo[]
}

export interface ExchangeOption {
  id: string
  name: string
  supported: boolean
}

export interface TimeframeOption {
  id: string
  name: string
  minutes: number
}

export interface DataOptionsResponse {
  exchanges: ExchangeOption[]
  timeframes: TimeframeOption[]
}

// ============== API ==============

export const dataApi = {
  // Status
  getStatus: () => api.get<DataStatusResponse>('/data/status'),
  getOptions: () => api.get<DataOptionsResponse>('/data/options'),

  // Symbols
  listSymbols: (params?: { exchange?: string; page?: number; page_size?: number }) =>
    api.get<SymbolListResponse>('/data/symbols', { params }),
  addSymbol: (data: AddSymbolRequest) =>
    api.post<AddSymbolResponse>('/data/symbols', data),
  removeSymbol: (symbol: string) =>
    api.delete<AddSymbolResponse>(`/data/symbols/${encodeURIComponent(symbol)}`),

  // Downloads
  listDownloads: (params?: { status?: string; limit?: number }) =>
    api.get<DownloadTaskListResponse>('/data/downloads', { params }),
  startDownload: (data: StartDownloadRequest) =>
    api.post<StartDownloadResponse>('/data/downloads', data),
  getDownload: (taskId: string) =>
    api.get<DownloadTaskStatus>(`/data/downloads/${taskId}`),
  cancelDownload: (taskId: string) =>
    api.delete<CancelDownloadResponse>(`/data/downloads/${taskId}`),

  // OHLCV Data
  getOHLCV: (symbol: string, params: { timeframe: string; start_date: string; end_date?: string; limit?: number }) =>
    api.get<OHLCVDataResponse>(`/data/ohlcv/${encodeURIComponent(symbol)}`, { params }),

  // Data Ranges
  getRanges: (params?: { symbol?: string; timeframe?: string }) =>
    api.get<DataRangeResponse>('/data/ranges', { params }),
}
