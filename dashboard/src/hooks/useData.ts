/**
 * Data API Hooks
 */

import { useState, useEffect, useCallback } from 'react'
import {
  dataApi,
  type DataStatusResponse,
  type SymbolListResponse,
  type DownloadTaskListResponse,
  type DataOptionsResponse,
  type DataRangeResponse,
  type AddSymbolRequest,
  type StartDownloadRequest,
  type BinanceSymbolListResponse,
  type ExtendedDataOptionsResponse,
} from '@/api/data'

// ============== Status Hook ==============

export function useDataStatus() {
  const [status, setStatus] = useState<DataStatusResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchStatus = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await dataApi.getStatus()
      setStatus(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data status')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchStatus()
  }, [fetchStatus])

  return { status, loading, error, refetch: fetchStatus }
}

// ============== Options Hook ==============

export function useDataOptions() {
  const [options, setOptions] = useState<DataOptionsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchOptions = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await dataApi.getOptions()
      setOptions(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data options')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchOptions()
  }, [fetchOptions])

  return { options, loading, error, refetch: fetchOptions }
}

// ============== Symbols Hook ==============

export function useSymbols(params?: { exchange?: string; page?: number; page_size?: number }) {
  const [data, setData] = useState<SymbolListResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [adding, setAdding] = useState(false)

  const fetchSymbols = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const result = await dataApi.listSymbols(params)
      setData(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch symbols')
    } finally {
      setLoading(false)
    }
  }, [params?.exchange, params?.page, params?.page_size])

  const addSymbol = useCallback(async (request: AddSymbolRequest) => {
    try {
      setAdding(true)
      setError(null)
      const result = await dataApi.addSymbol(request)
      if (result.success) {
        await fetchSymbols()
      }
      return result
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to add symbol'
      setError(message)
      return { success: false, message, symbol: null }
    } finally {
      setAdding(false)
    }
  }, [fetchSymbols])

  const removeSymbol = useCallback(async (symbol: string) => {
    try {
      setError(null)
      const result = await dataApi.removeSymbol(symbol)
      if (result.success) {
        await fetchSymbols()
      }
      return result
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to remove symbol'
      setError(message)
      return { success: false, message, symbol: null }
    }
  }, [fetchSymbols])

  useEffect(() => {
    fetchSymbols()
  }, [fetchSymbols])

  return { data, loading, error, adding, addSymbol, removeSymbol, refetch: fetchSymbols }
}

// ============== Downloads Hook ==============

export function useDownloads(params?: { status?: string; limit?: number }) {
  const [data, setData] = useState<DownloadTaskListResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)

  const fetchDownloads = useCallback(async () => {
    try {
      // Don't set loading to true for poll updates to avoid UI flicker
      if (!data) {
        setLoading(true)
      }
      setError(null)
      const result = await dataApi.listDownloads(params)
      setData(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch downloads')
    } finally {
      setLoading(false)
    }
  }, [params?.status, params?.limit, data])

  const startDownload = useCallback(async (request: StartDownloadRequest) => {
    try {
      setStarting(true)
      setError(null)
      const result = await dataApi.startDownload(request)
      if (result.success) {
        await fetchDownloads()
      }
      return result
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to start download'
      setError(message)
      return { success: false, message, task_id: null }
    } finally {
      setStarting(false)
    }
  }, [fetchDownloads])

  const cancelDownload = useCallback(async (taskId: string) => {
    try {
      setError(null)
      const result = await dataApi.cancelDownload(taskId)
      if (result.success) {
        await fetchDownloads()
      }
      return result
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to cancel download'
      setError(message)
      return { success: false, message }
    }
  }, [fetchDownloads])

  // Initial fetch
  useEffect(() => {
    fetchDownloads()
  }, []) // Only run once on mount

  // Auto-poll when there are active (pending/running) tasks
  useEffect(() => {
    const hasActiveTasks = data?.tasks.some(
      task => task.status === 'pending' || task.status === 'running'
    )

    if (!hasActiveTasks) {
      return
    }

    // Poll every 2 seconds when there are active tasks
    const interval = setInterval(() => {
      fetchDownloads()
    }, 2000)

    return () => clearInterval(interval)
  }, [data?.tasks, fetchDownloads])

  return { data, loading, error, starting, startDownload, cancelDownload, refetch: fetchDownloads }
}

// ============== Data Ranges Hook ==============

export function useDataRanges(params?: { symbol?: string; timeframe?: string }) {
  const [data, setData] = useState<DataRangeResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchRanges = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const result = await dataApi.getRanges(params)
      setData(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data ranges')
    } finally {
      setLoading(false)
    }
  }, [params?.symbol, params?.timeframe])

  useEffect(() => {
    fetchRanges()
  }, [fetchRanges])

  return { data, loading, error, refetch: fetchRanges }
}

// ============== Binance Symbols Hook ==============

export function useBinanceSymbols(options?: { quote_asset?: string; limit?: number }) {
  const [data, setData] = useState<BinanceSymbolListResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const quoteAsset = options?.quote_asset ?? 'USDT'
  const limit = options?.limit ?? 200

  const fetchSymbols = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const result = await dataApi.getBinanceSymbols({ quote_asset: quoteAsset, limit })
      setData(result)
    } catch (err) {
      console.error('Failed to fetch Binance symbols:', err)
      setError(err instanceof Error ? err.message : 'Failed to fetch Binance symbols')
    } finally {
      setLoading(false)
    }
  }, [quoteAsset, limit])

  useEffect(() => {
    fetchSymbols()
  }, [fetchSymbols])

  return { data, loading, error, refetch: fetchSymbols }
}

// ============== Extended Options Hook ==============

export function useExtendedDataOptions() {
  const [options, setOptions] = useState<ExtendedDataOptionsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchOptions = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const result = await dataApi.getExtendedOptions()
      setOptions(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch extended options')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchOptions()
  }, [fetchOptions])

  return { options, loading, error, refetch: fetchOptions }
}
