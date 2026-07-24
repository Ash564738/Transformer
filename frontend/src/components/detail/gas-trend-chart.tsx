"use client";

import { useMemo, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { DgaRow } from "@/types/dga";
import { cn, formatDate } from "@/lib/utils";

type RangeKey = "1M" | "3M" | "6M" | "1Y" | "YTD";
const RANGE_DAYS: Record<RangeKey, number | null> = { "1M": 30, "3M": 90, "6M": 182, "1Y": 365, YTD: null };

// Matches the six values shown on the Gas Indicator Cards above this chart,
// so "what's elevated right now" (cards) and "how did it get there" (this
// chart) always cover the same gases.
const SERIES: { key: "h2" | "ch4" | "c2h2" | "c2h4" | "co" | "tdcg"; label: string; color: string; dash?: string }[] = [
  { key: "c2h2", label: "C2H2", color: "#c62828" },
  { key: "c2h4", label: "C2H4", color: "#a8571c", dash: "5 3" },
  { key: "h2", label: "H2", color: "#184843" },
  { key: "ch4", label: "CH4", color: "#6c5ce7", dash: "2 3" },
  { key: "co", label: "CO", color: "#0f7ea8" },
  { key: "tdcg", label: "TDCG", color: "#9a4a1f", dash: "1 4" },
];

export function GasTrendChart({ rows }: { rows: DgaRow[] }) {
  const [range, setRange] = useState<RangeKey>("6M");
  const [hidden, setHidden] = useState<Set<string>>(new Set());

  const data = useMemo(() => {
    if (rows.length === 0) return [];
    const latest = new Date(rows[rows.length - 1].sample_day).getTime();
    let filtered = rows;
    if (range === "YTD") {
      const year = new Date(latest).getUTCFullYear();
      filtered = rows.filter((r) => new Date(r.sample_day).getUTCFullYear() === year);
    } else {
      const days = RANGE_DAYS[range]!;
      const cutoff = latest - days * 86400000;
      filtered = rows.filter((r) => new Date(r.sample_day).getTime() >= cutoff);
    }
    return filtered.map((r) => ({
      day: r.sample_day,
      h2: Number(r.h2 ?? 0),
      ch4: Number(r.ch4 ?? 0),
      c2h2: Number(r.c2h2 ?? 0),
      c2h4: Number(r.c2h4 ?? 0),
      co: Number(r.co ?? 0),
      tdcg: Number(r.tdcg ?? 0),
    }));
  }, [rows, range]);

  const singlePoint = data.length === 1;

  function toggleSeries(key: string) {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  return (
    <div>
      <div className="mb-3 flex justify-end gap-1">
        {(Object.keys(RANGE_DAYS) as RangeKey[]).map((r) => (
          <button
            key={r}
            onClick={() => setRange(r)}
            className={cn(
              "rounded-md px-2.5 py-1 text-xs font-semibold transition-colors cursor-pointer",
              range === r ? "bg-teal-800 text-white" : "bg-cream-100 text-teal-600 hover:bg-teal-100"
            )}
          >
            {r}
          </button>
        ))}
      </div>
      {data.length === 0 ? (
        <p className="py-14 text-center text-sm text-teal-400">No samples in this range.</p>
      ) : (
        <>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={data} margin={{ top: 8, right: 16, left: -16, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e8e5d9" vertical={false} />
              <XAxis
                dataKey="day"
                tickFormatter={(v) => formatDate(v)}
                tick={{ fontSize: 11, fill: "#4f8f83" }}
                axisLine={{ stroke: "#d9d5c4" }}
                tickLine={false}
              />
              <YAxis tick={{ fontSize: 12, fill: "#4f8f83" }} axisLine={false} tickLine={false} />
              <Tooltip
                labelFormatter={(v) => formatDate(v as string)}
                contentStyle={{ borderRadius: 10, borderColor: "#d9d5c4", fontSize: 12 }}
              />
              <Legend
                onClick={(e) => toggleSeries(String(e.dataKey))}
                wrapperStyle={{ fontSize: 12, cursor: "pointer" }}
                formatter={(value, entry) => (
                  <span style={{ opacity: hidden.has(String(entry.dataKey)) ? 0.35 : 1 }}>{value}</span>
                )}
              />
              {SERIES.map((s) => (
                <Line
                  key={s.key}
                  type="monotone"
                  dataKey={s.key}
                  stroke={s.color}
                  strokeWidth={2}
                  strokeDasharray={s.dash}
                  dot={{ r: singlePoint ? 6 : 3, fill: s.color }}
                  activeDot={{ r: singlePoint ? 8 : 5 }}
                  name={s.label}
                  hide={hidden.has(s.key)}
                  isAnimationActive={!singlePoint}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
          {singlePoint && (
            <p className="mt-1 text-center text-[11px] text-teal-400">
              Only one record in this range ({formatDate(data[0].day)}) — a trend line needs at least 2 samples.
              Try a wider range if this transformer has older history.
            </p>
          )}
          <p className="mt-1 text-center text-[11px] text-teal-400">Click a legend item to show/hide that gas.</p>
        </>
      )}
    </div>
  );
}
