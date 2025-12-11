/**
 * Data Type Multi-Select - 数据类型多选组件
 */

import * as React from "react"
import { cn } from "@/lib/utils"
import { Check } from "lucide-react"
import type { DataTypeOption } from "@/api/data"

export interface DataTypeSelectProps {
  options: DataTypeOption[]
  value: string[]
  onChange: (value: string[]) => void
  className?: string
}

export function DataTypeSelect({
  options,
  value,
  onChange,
  className,
}: DataTypeSelectProps) {
  const toggleOption = (optionId: string, supported: boolean) => {
    // Don't allow toggling unsupported options
    if (!supported) return

    if (value.includes(optionId)) {
      onChange(value.filter(v => v !== optionId))
    } else {
      onChange([...value, optionId])
    }
  }

  return (
    <div className={cn("grid gap-2", className)}>
      {options.map((option) => {
        const isSelected = value.includes(option.id)
        const isSupported = option.supported !== false // Default to true if not specified

        return (
          <label
            key={option.id}
            onClick={() => toggleOption(option.id, isSupported)}
            className={cn(
              "flex items-start gap-3 p-3 rounded-lg border transition-colors bg-white dark:bg-gray-900",
              isSupported
                ? "cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                : "cursor-not-allowed opacity-60",
              isSelected && isSupported
                ? "border-blue-600 bg-blue-50 dark:bg-blue-900/20"
                : "border-gray-300 dark:border-gray-600"
            )}
          >
            <div
              className={cn(
                "flex h-5 w-5 shrink-0 items-center justify-center rounded border mt-0.5",
                isSelected && isSupported
                  ? "border-blue-600 bg-blue-600 text-white"
                  : "border-gray-300 dark:border-gray-600",
                !isSupported && "bg-gray-100 dark:bg-gray-800"
              )}
            >
              {isSelected && isSupported && <Check className="h-3 w-3" />}
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className={cn(
                  "font-medium text-sm",
                  !isSupported && "text-gray-500 dark:text-gray-400"
                )}>{option.name}</span>
                {option.requires_futures && (
                  <span className="text-xs px-1.5 py-0.5 rounded bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400">
                    Futures
                  </span>
                )}
                {!isSupported && (
                  <span className="text-xs px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400">
                    Coming Soon
                  </span>
                )}
              </div>
              <p className={cn(
                "text-xs mt-0.5",
                isSupported ? "text-gray-500 dark:text-gray-400" : "text-gray-400 dark:text-gray-500"
              )}>
                {option.description}
              </p>
              {option.min_interval && option.min_interval !== '1m' && (
                <p className={cn(
                  "text-xs mt-0.5",
                  isSupported ? "text-blue-600 dark:text-blue-400" : "text-blue-400/60 dark:text-blue-500/60"
                )}>
                  Min interval: {option.min_interval}
                </p>
              )}
            </div>
          </label>
        )
      })}
    </div>
  )
}
