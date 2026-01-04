/**
 * MonthlyReturnsHeatmap - Monthly returns visualization in a calendar grid format
 *
 * Features:
 * - Year x Month grid layout (12 columns)
 * - Color-coded returns (green positive, red negative)
 * - Hover tooltips with exact values
 * - Year totals column
 * - Responsive design
 */

import { useMemo, useState } from "react";

export interface MonthlyReturn {
  year: number;
  month: number; // 1-12
  return: number; // decimal, e.g., 0.05 = 5%
}

export interface MonthlyReturnsHeatmapProps {
  data: MonthlyReturn[];
  className?: string;
}

const MONTH_LABELS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

function getReturnColor(value: number): string {
  if (value >= 0.1) return "bg-green-600";
  if (value >= 0.05) return "bg-green-500";
  if (value >= 0.02) return "bg-green-400";
  if (value > 0) return "bg-green-300";
  if (value === 0 || isNaN(value)) return "bg-gray-600";
  if (value > -0.02) return "bg-red-300";
  if (value > -0.05) return "bg-red-400";
  if (value > -0.1) return "bg-red-500";
  return "bg-red-600";
}

function getTextColor(value: number): string {
  if (Math.abs(value) >= 0.02) return "text-white";
  return "text-gray-900";
}

function formatReturn(value: number): string {
  if (isNaN(value)) return "-";
  const pct = value * 100;
  const sign = pct >= 0 ? "+" : "";
  return `${sign}${pct.toFixed(1)}%`;
}

interface TooltipState {
  visible: boolean;
  x: number;
  y: number;
  content: string;
}

export function MonthlyReturnsHeatmap({
  data,
  className = "",
}: MonthlyReturnsHeatmapProps) {
  const [tooltip, setTooltip] = useState<TooltipState>({
    visible: false,
    x: 0,
    y: 0,
    content: "",
  });

  // Organize data by year and month
  const { gridData, years, yearlyTotals, stats } = useMemo(() => {
    const byYearMonth: Record<number, Record<number, number>> = {};
    const yearSet = new Set<number>();

    data.forEach((d) => {
      yearSet.add(d.year);
      if (!byYearMonth[d.year]) {
        byYearMonth[d.year] = {};
      }
      byYearMonth[d.year][d.month] = d.return;
    });

    const sortedYears = Array.from(yearSet).sort((a, b) => a - b);

    // Calculate yearly totals (compound returns)
    const yearlyTotals: Record<number, number> = {};
    sortedYears.forEach((year) => {
      const monthlyReturns = Object.values(byYearMonth[year] || {});
      if (monthlyReturns.length > 0) {
        // Compound the monthly returns
        const compounded =
          monthlyReturns.reduce((acc, r) => acc * (1 + r), 1) - 1;
        yearlyTotals[year] = compounded;
      } else {
        yearlyTotals[year] = NaN;
      }
    });

    // Calculate overall stats
    const allReturns = data.map((d) => d.return);
    const avgMonthly =
      allReturns.reduce((a, b) => a + b, 0) / allReturns.length;
    const positiveMonths = allReturns.filter((r) => r > 0).length;
    const winRate = (positiveMonths / allReturns.length) * 100;
    const bestMonth = Math.max(...allReturns);
    const worstMonth = Math.min(...allReturns);

    return {
      gridData: byYearMonth,
      years: sortedYears,
      yearlyTotals,
      stats: {
        avgMonthly,
        winRate,
        bestMonth,
        worstMonth,
        totalMonths: allReturns.length,
      },
    };
  }, [data]);

  const handleMouseEnter = (
    e: React.MouseEvent,
    year: number,
    month: number,
    value: number
  ) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setTooltip({
      visible: true,
      x: rect.left + rect.width / 2,
      y: rect.top - 10,
      content: `${MONTH_LABELS[month - 1]} ${year}: ${formatReturn(value)}`,
    });
  };

  const handleMouseLeave = () => {
    setTooltip((prev) => ({ ...prev, visible: false }));
  };

  if (data.length === 0) {
    return (
      <div
        className={`flex items-center justify-center h-64 bg-gray-800 rounded-lg ${className}`}
      >
        <p className="text-gray-400">No monthly returns data available</p>
      </div>
    );
  }

  return (
    <div className={`bg-gray-800 rounded-lg p-4 ${className}`}>
      {/* Stats header */}
      <div className="flex gap-6 mb-4 text-sm">
        <div>
          <span className="text-gray-400">Avg Monthly</span>
          <p
            className={`font-semibold ${stats.avgMonthly >= 0 ? "text-green-400" : "text-red-400"}`}
          >
            {formatReturn(stats.avgMonthly)}
          </p>
        </div>
        <div>
          <span className="text-gray-400">Win Rate</span>
          <p className="font-semibold text-white">
            {stats.winRate.toFixed(1)}%
          </p>
        </div>
        <div>
          <span className="text-gray-400">Best Month</span>
          <p className="font-semibold text-green-400">
            {formatReturn(stats.bestMonth)}
          </p>
        </div>
        <div>
          <span className="text-gray-400">Worst Month</span>
          <p className="font-semibold text-red-400">
            {formatReturn(stats.worstMonth)}
          </p>
        </div>
      </div>

      {/* Heatmap grid */}
      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr>
              <th className="p-2 text-left text-gray-400 text-sm font-medium">
                Year
              </th>
              {MONTH_LABELS.map((month) => (
                <th
                  key={month}
                  className="p-2 text-center text-gray-400 text-sm font-medium w-14"
                >
                  {month}
                </th>
              ))}
              <th className="p-2 text-center text-gray-400 text-sm font-medium w-16">
                Year
              </th>
            </tr>
          </thead>
          <tbody>
            {years.map((year) => {
              const yearData = gridData[year] || {};
              const yearTotal = yearlyTotals[year] ?? NaN;

              return (
                <tr key={year}>
                  <td className="p-2 text-gray-300 text-sm font-medium">
                    {year}
                  </td>
                  {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].map((month) => {
                    const value = yearData[month];
                    const hasValue = value !== undefined;

                    return (
                      <td key={month} className="p-1">
                        <div
                          className={`
                            w-12 h-8 rounded flex items-center justify-center cursor-default
                            text-xs font-medium transition-all duration-150
                            ${hasValue ? getReturnColor(value) : "bg-gray-700"}
                            ${hasValue ? getTextColor(value) : "text-gray-500"}
                            ${hasValue ? "hover:ring-2 hover:ring-white/30" : ""}
                          `}
                          onMouseEnter={
                            hasValue
                              ? (e) => handleMouseEnter(e, year, month, value)
                              : undefined
                          }
                          onMouseLeave={hasValue ? handleMouseLeave : undefined}
                        >
                          {hasValue ? formatReturn(value) : "-"}
                        </div>
                      </td>
                    );
                  })}
                  <td className="p-1">
                    <div
                      className={`
                        w-14 h-8 rounded flex items-center justify-center
                        text-xs font-semibold
                        ${!isNaN(yearTotal) ? getReturnColor(yearTotal) : "bg-gray-700"}
                        ${!isNaN(yearTotal) ? getTextColor(yearTotal) : "text-gray-500"}
                      `}
                    >
                      {formatReturn(yearTotal)}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Color legend - matches getReturnColor thresholds */}
      <div className="flex items-center gap-2 mt-4 text-xs text-gray-400">
        <span>Returns:</span>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 bg-red-600 rounded" />
          <span>&lt;-10%</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 bg-red-500 rounded" />
          <span>-10% to -5%</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 bg-red-400 rounded" />
          <span>-5% to -2%</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 bg-gray-600 rounded" />
          <span>~0%</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 bg-green-400 rounded" />
          <span>+2% to +5%</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 bg-green-500 rounded" />
          <span>+5% to +10%</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 bg-green-600 rounded" />
          <span>&gt;+10%</span>
        </div>
      </div>

      {/* Tooltip */}
      {tooltip.visible && (
        <div
          className="fixed z-50 bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-sm text-white shadow-xl pointer-events-none"
          style={{
            left: tooltip.x,
            top: tooltip.y,
            transform: "translate(-50%, -100%)",
          }}
        >
          {tooltip.content}
        </div>
      )}
    </div>
  );
}

export default MonthlyReturnsHeatmap;
