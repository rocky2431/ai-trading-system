/**
 * Symbol Select - 带搜索功能的交易对选择器
 */

import * as React from "react"
import { useState, useEffect, useRef, useMemo } from "react"
import { cn } from "@/lib/utils"
import { ChevronDown, Search, Loader2, X } from "lucide-react"
import { Input } from "./input"
import type { BinanceSymbolInfo } from "@/api/data"

export interface SymbolSelectProps {
  symbols: BinanceSymbolInfo[]
  value: string
  onChange: (symbol: string) => void
  loading?: boolean
  placeholder?: string
  className?: string
  disabled?: boolean
}

/**
 * Format volume to human readable string
 */
function formatVolume(volume: number): string {
  if (volume >= 1e9) return `$${(volume / 1e9).toFixed(1)}B`
  if (volume >= 1e6) return `$${(volume / 1e6).toFixed(1)}M`
  if (volume >= 1e3) return `$${(volume / 1e3).toFixed(1)}K`
  return `$${volume.toFixed(0)}`
}

export function SymbolSelect({
  symbols,
  value,
  onChange,
  loading = false,
  placeholder = "Select symbol...",
  className,
  disabled = false,
}: SymbolSelectProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [search, setSearch] = useState("")
  const containerRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Filter symbols based on search
  const filteredSymbols = useMemo(() => {
    if (!search) return symbols
    const searchUpper = search.toUpperCase()
    return symbols.filter(
      (s) =>
        s.base_asset.includes(searchUpper) ||
        s.symbol.includes(searchUpper)
    )
  }, [symbols, search])

  // Get selected symbol info
  const selectedSymbol = useMemo(
    () => symbols.find((s) => s.symbol === value),
    [symbols, value]
  )

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false)
      }
    }
    document.addEventListener("mousedown", handleClickOutside)
    return () => document.removeEventListener("mousedown", handleClickOutside)
  }, [])

  // Focus search input when opening
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus()
    }
  }, [isOpen])

  const handleSelect = (symbol: string) => {
    onChange(symbol)
    setIsOpen(false)
    setSearch("")
  }

  const handleClear = (e: React.MouseEvent) => {
    e.stopPropagation()
    onChange("")
    setSearch("")
  }

  return (
    <div ref={containerRef} className={cn("relative", className)}>
      {/* Trigger button */}
      <button
        type="button"
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        className={cn(
          "flex h-9 w-full items-center justify-between rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors",
          "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
          "disabled:cursor-not-allowed disabled:opacity-50",
          isOpen && "ring-1 ring-ring"
        )}
      >
        <span className={cn(!value && "text-muted-foreground")}>
          {loading ? (
            <span className="flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading...
            </span>
          ) : selectedSymbol ? (
            <span className="flex items-center gap-2">
              <span className="font-medium">{selectedSymbol.base_asset}</span>
              <span className="text-muted-foreground text-xs">
                #{selectedSymbol.rank}
              </span>
            </span>
          ) : (
            placeholder
          )}
        </span>
        <div className="flex items-center gap-1">
          {value && !disabled && (
            <X
              className="h-4 w-4 opacity-50 hover:opacity-100 cursor-pointer"
              onClick={handleClear}
            />
          )}
          <ChevronDown
            className={cn(
              "h-4 w-4 opacity-50 transition-transform",
              isOpen && "rotate-180"
            )}
          />
        </div>
      </button>

      {/* Dropdown */}
      {isOpen && (
        <div className="absolute z-50 mt-1 w-full rounded-md border bg-white dark:bg-zinc-900 shadow-lg">
          {/* Search input */}
          <div className="p-2 border-b bg-white dark:bg-zinc-900">
            <div className="relative">
              <Search className="absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                ref={inputRef}
                type="text"
                placeholder="Search symbol..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-8 h-8"
              />
            </div>
          </div>

          {/* Symbol list */}
          <div className="max-h-[300px] overflow-y-auto">
            {filteredSymbols.length === 0 ? (
              <div className="p-4 text-center text-sm text-muted-foreground">
                No symbols found
              </div>
            ) : (
              filteredSymbols.map((symbol) => (
                <button
                  key={symbol.symbol}
                  type="button"
                  onClick={() => handleSelect(symbol.symbol)}
                  className={cn(
                    "flex w-full items-center justify-between px-3 py-2 text-sm hover:bg-accent",
                    value === symbol.symbol && "bg-accent"
                  )}
                >
                  <div className="flex items-center gap-3">
                    <span className="w-8 text-xs text-muted-foreground">
                      #{symbol.rank}
                    </span>
                    <div className="flex flex-col items-start">
                      <span className="font-medium">{symbol.base_asset}</span>
                      <span className="text-xs text-muted-foreground">
                        {symbol.symbol}
                      </span>
                    </div>
                  </div>
                  <span className="text-xs text-muted-foreground">
                    {formatVolume(symbol.volume_24h_usd)}
                  </span>
                </button>
              ))
            )}
          </div>

          {/* Footer info */}
          <div className="p-2 border-t text-xs text-muted-foreground text-center bg-white dark:bg-zinc-900">
            {filteredSymbols.length} symbols (sorted by 24h volume)
          </div>
        </div>
      )}
    </div>
  )
}
