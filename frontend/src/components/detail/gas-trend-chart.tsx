// src/components/detail/gas-trend-chart.tsx
"use client";

import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DgaRow } from "@/types/dga";
import { cn, formatDate } from "@/lib/utils";

type RangeKey = "1M" | "3M" | "6M" | "1Y" | "YTD";

const RANGE_DAYS: Record<RangeKey, number | null> = {
  "1M": 30,
  "3M": 90,
  "6M": 182,
  "1Y": 365,
  YTD: null,
};

type SeriesKey =
  | "h2"
  | "ch4"
  | "c2h2"
  | "c2h4"
  | "co"
  | "co2"
  | "tdcg";

const SERIES: {
  key: SeriesKey;
  label: string;
  color: string;
  dash?: string;
}[] = [
  { key: "c2h2", label: "C2H2", color: "#c62828" },
  { key: "c2h4", label: "C2H4", color: "#a8571c", dash: "5 3" },
  { key: "h2", label: "H2", color: "#184843" },
  { key: "ch4", label: "CH4", color: "#6c5ce7", dash: "2 3" },
  { key: "co", label: "CO", color: "#0f7ea8" },
  { key: "co2", label: "CO2", color: "#7a5c00", dash: "7 3" },
  { key: "tdcg", label: "TDCG", color: "#9a4a1f", dash: "1 4" },
];

function finiteOrZero(value: unknown): number {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function tdcgForRow(row: DgaRow): number {
  const direct = Number(row.tdcg);

  if (Number.isFinite(direct) && direct >= 0) {
    return direct;
  }

  const combustible: number[] = [
    row.h2,
    row.ch4,
    row.c2h6,
    row.c2h4,
    row.c2h2,
    row.co,
  ].map(finiteOrZero);

  return combustible.reduce(
    (sum: number, value: number) => sum + value,
    0
  );
}

export function GasTrendChart({ rows }: { rows: DgaRow[] }) {
  const [range, setRange] = useState<RangeKey>("6M");
  const [hidden, setHidden] = useState<Set<string>>(new Set());

  const sortedRows = useMemo(
    () =>
      [...rows].sort(
        (a, b) =>
          new Date(a.sample_day).getTime() -
          new Date(b.sample_day).getTime()
      ),
    [rows]
  );

  const data = useMemo(() => {
    if (sortedRows.length === 0) {
      return [];
    }

    const latest = new Date(
      sortedRows[sortedRows.length - 1].sample_day
    ).getTime();

    let filtered = sortedRows;

    if (range === "YTD") {
      const year = new Date(latest).getUTCFullYear();

      filtered = sortedRows.filter(
        (row) =>
          new Date(row.sample_day).getUTCFullYear() === year
      );
    } else {
      const days = RANGE_DAYS[range];

      if (days == null) {
        filtered = sortedRows;
      } else {
        const cutoff = latest - days * 86400000;

        filtered = sortedRows.filter(
          (row) => new Date(row.sample_day).getTime() >= cutoff
        );
      }
    }

    return filtered.map((row) => ({
      day: row.sample_day,
      h2: finiteOrZero(row.h2),
      ch4: finiteOrZero(row.ch4),
      c2h2: finiteOrZero(row.c2h2),
      c2h4: finiteOrZero(row.c2h4),
      co: finiteOrZero(row.co),
      co2: finiteOrZero(row.co2),
      tdcg: tdcgForRow(row),
    }));
  }, [sortedRows, range]);

  const singlePoint = data.length === 1;

  function toggleSeries(key: string) {
    setHidden((previous) => {
      const next = new Set(previous);

      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }

      return next;
    });
  }

  return (
    <div>
      <div className="mb-3 flex flex-wrap justify-end gap-1">
        {(Object.keys(RANGE_DAYS) as RangeKey[]).map((rangeKey) => (
          <button
            key={rangeKey}
            type="button"
            onClick={() => setRange(rangeKey)}
            className={cn(
              "cursor-pointer rounded-md px-2.5 py-1 text-xs font-semibold transition-colors",
              range === rangeKey
                ? "bg-teal-800 text-white"
                : "bg-cream-100 text-teal-600 hover:bg-teal-100"
            )}
          >
            {rangeKey}
          </button>
        ))}
      </div>

      {data.length === 0 ? (
        <p className="py-14 text-center text-sm text-teal-400">
          No samples in this range.
        </p>
      ) : (
        <>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              data={data}
              margin={{ top: 8, right: 16, left: -16, bottom: 0 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="#e8e5d9"
                vertical={false}
              />

              <XAxis
                dataKey="day"
                tickFormatter={(value) => formatDate(String(value))}
                tick={{ fontSize: 11, fill: "#4f8f83" }}
                axisLine={{ stroke: "#d9d5c4" }}
                tickLine={false}
              />

              <YAxis
                tick={{ fontSize: 12, fill: "#4f8f83" }}
                axisLine={false}
                tickLine={false}
              />

              <Tooltip
                labelFormatter={(value) =>
                  formatDate(String(value))
                }
                contentStyle={{
                  borderRadius: 10,
                  borderColor: "#d9d5c4",
                  fontSize: 12,
                }}
              />

              <Legend
                onClick={(entry) =>
                  toggleSeries(String(entry.dataKey))
                }
                wrapperStyle={{
                  fontSize: 12,
                  cursor: "pointer",
                }}
                formatter={(value, entry) => (
                  <span
                    style={{
                      opacity: hidden.has(String(entry.dataKey))
                        ? 0.35
                        : 1,
                    }}
                  >
                    {value}
                  </span>
                )}
              />

              {SERIES.map((series) => (
                <Line
                  key={series.key}
                  type="monotone"
                  dataKey={series.key}
                  stroke={series.color}
                  strokeWidth={2}
                  strokeDasharray={series.dash}
                  dot={{
                    r: singlePoint ? 6 : 3,
                    fill: series.color,
                  }}
                  activeDot={{ r: singlePoint ? 8 : 5 }}
                  name={series.label}
                  hide={hidden.has(series.key)}
                  isAnimationActive={!singlePoint}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>

          {singlePoint && (
            <p className="mt-1 text-center text-[11px] text-teal-400">
              Only one record in this range ({formatDate(data[0].day)}) — a
              trend line needs at least 2 samples. Try a wider range if this
              transformer has older history.
            </p>
          )}

          <p className="mt-1 text-center text-[11px] text-teal-400">
            Click a legend item to show/hide that gas.
          </p>
        </>
      )}
    </div>
  );
}