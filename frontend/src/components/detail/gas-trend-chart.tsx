"use client";

import { useMemo, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { DgaRow } from "@/types/dga";
import { cn, formatDate } from "@/lib/utils";

type RangeKey = "1M" | "3M" | "6M" | "1Y" | "YTD";
const RANGE_DAYS: Record<RangeKey, number | null> = { "1M": 30, "3M": 90, "6M": 182, "1Y": 365, YTD: null };

export function GasTrendChart({ rows }: { rows: DgaRow[] }) {
  const [range, setRange] = useState<RangeKey>("6M");

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
      C2H2: Number(r.c2h2 ?? 0),
      C2H4: Number(r.c2h4 ?? 0),
      H2: Number(r.h2 ?? 0),
    }));
  }, [rows, range]);

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
        <ResponsiveContainer width="100%" height={260}>
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
            <Tooltip labelFormatter={(v) => formatDate(v as string)} contentStyle={{ borderRadius: 10, borderColor: "#d9d5c4", fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="C2H2" stroke="#c62828" strokeWidth={2.5} dot={{ r: 3 }} name="C2H2 (Critical)" />
            <Line type="monotone" dataKey="C2H4" stroke="#a8571c" strokeWidth={2} strokeDasharray="5 3" dot={{ r: 3 }} name="C2H4" />
            <Line type="monotone" dataKey="H2" stroke="#184843" strokeWidth={2} dot={{ r: 3 }} name="H2" />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
