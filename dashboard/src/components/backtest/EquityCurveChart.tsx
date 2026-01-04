/**
 * EquityCurveChart - Interactive equity curve visualization with drawdown overlay
 *
 * Features:
 * - Dual-axis: Equity curve (left) + Drawdown (right)
 * - Interactive tooltip with detailed metrics
 * - Benchmark comparison line (optional)
 * - Responsive design
 */

import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { useMemo } from "react";

export interface EquityDataPoint {
  date: string;
  equity: number;
  drawdown: number;
  benchmark?: number;
}

export interface EquityCurveChartProps {
  data: EquityDataPoint[];
  showBenchmark?: boolean;
  benchmarkLabel?: string;
  height?: number;
  className?: string;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{
    name: string;
    value: number;
    color: string;
    dataKey: string;
  }>;
  label?: string;
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload || payload.length === 0) {
    return null;
  }

  const formatValue = (value: number, key: string): string => {
    if (key === "drawdown") {
      return `${(value * 100).toFixed(2)}%`;
    }
    return value.toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  };

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 shadow-xl">
      <p className="text-gray-400 text-sm mb-2">{label}</p>
      {payload.map((entry) => (
        <div key={entry.dataKey} className="flex items-center gap-2 text-sm">
          <div
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: entry.color }}
          />
          <span className="text-gray-300">{entry.name}:</span>
          <span className="text-white font-medium">
            {formatValue(entry.value, entry.dataKey)}
          </span>
        </div>
      ))}
    </div>
  );
}

export function EquityCurveChart({
  data,
  showBenchmark = false,
  benchmarkLabel = "Benchmark",
  height = 400,
  className = "",
}: EquityCurveChartProps) {
  // Calculate normalized equity for display
  const chartData = useMemo(() => {
    if (data.length === 0) return [];

    const initialEquity = data[0].equity;
    return data.map((point) => ({
      ...point,
      normalizedEquity: (point.equity / initialEquity - 1) * 100,
      normalizedBenchmark: point.benchmark
        ? (point.benchmark / (data[0].benchmark || 1) - 1) * 100
        : undefined,
      drawdownPct: point.drawdown * 100,
    }));
  }, [data]);

  // Calculate statistics
  const stats = useMemo(() => {
    if (data.length === 0) return null;

    const returns = chartData.map((d) => d.normalizedEquity);
    const drawdowns = data.map((d) => d.drawdown);

    return {
      totalReturn: returns[returns.length - 1] || 0,
      maxDrawdown: Math.min(...drawdowns) * 100,
      currentDrawdown: (drawdowns[drawdowns.length - 1] || 0) * 100,
    };
  }, [data, chartData]);

  if (data.length === 0) {
    return (
      <div
        className={`flex items-center justify-center h-[${height}px] bg-gray-800 rounded-lg ${className}`}
      >
        <p className="text-gray-400">No equity data available</p>
      </div>
    );
  }

  return (
    <div className={`bg-gray-800 rounded-lg p-4 ${className}`}>
      {/* Header with stats */}
      {stats && (
        <div className="flex gap-6 mb-4">
          <div>
            <span className="text-gray-400 text-sm">Total Return</span>
            <p
              className={`text-lg font-semibold ${stats.totalReturn >= 0 ? "text-green-400" : "text-red-400"}`}
            >
              {stats.totalReturn >= 0 ? "+" : ""}
              {stats.totalReturn.toFixed(2)}%
            </p>
          </div>
          <div>
            <span className="text-gray-400 text-sm">Max Drawdown</span>
            <p className="text-lg font-semibold text-red-400">
              {stats.maxDrawdown.toFixed(2)}%
            </p>
          </div>
          <div>
            <span className="text-gray-400 text-sm">Current Drawdown</span>
            <p
              className={`text-lg font-semibold ${stats.currentDrawdown < -5 ? "text-red-400" : "text-yellow-400"}`}
            >
              {stats.currentDrawdown.toFixed(2)}%
            </p>
          </div>
        </div>
      )}

      {/* Chart */}
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart
          data={chartData}
          margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="date"
            tick={{ fill: "#9CA3AF", fontSize: 12 }}
            tickLine={{ stroke: "#4B5563" }}
            axisLine={{ stroke: "#4B5563" }}
          />
          <YAxis
            yAxisId="left"
            tick={{ fill: "#9CA3AF", fontSize: 12 }}
            tickLine={{ stroke: "#4B5563" }}
            axisLine={{ stroke: "#4B5563" }}
            tickFormatter={(value) => `${value.toFixed(0)}%`}
            label={{
              value: "Return %",
              angle: -90,
              position: "insideLeft",
              fill: "#9CA3AF",
            }}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            tick={{ fill: "#9CA3AF", fontSize: 12 }}
            tickLine={{ stroke: "#4B5563" }}
            axisLine={{ stroke: "#4B5563" }}
            tickFormatter={(value) => `${value.toFixed(0)}%`}
            label={{
              value: "Drawdown %",
              angle: 90,
              position: "insideRight",
              fill: "#9CA3AF",
            }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ paddingTop: "20px" }}
            formatter={(value) => (
              <span className="text-gray-300">{value}</span>
            )}
          />
          <ReferenceLine
            yAxisId="left"
            y={0}
            stroke="#6B7280"
            strokeDasharray="3 3"
          />

          {/* Drawdown area (background) */}
          <Area
            yAxisId="right"
            type="monotone"
            dataKey="drawdownPct"
            name="Drawdown"
            fill="#EF444433"
            stroke="#EF4444"
            strokeWidth={1}
          />

          {/* Equity curve */}
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="normalizedEquity"
            name="Strategy"
            stroke="#10B981"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: "#10B981" }}
          />

          {/* Benchmark line (optional) */}
          {showBenchmark && (
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="normalizedBenchmark"
              name={benchmarkLabel}
              stroke="#6366F1"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              activeDot={{ r: 4, fill: "#6366F1" }}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export default EquityCurveChart;
