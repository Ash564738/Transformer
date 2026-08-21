// src/components/overview/top-trend-chart.tsx
"use client";

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

import type { DgaPayload } from "@/types/dga";
import { formatDate } from "@/lib/utils";

const LINE_COLORS = [
  "#c62828",
  "#a8571c",
  "#a16a07",
  "#316f64",
  "#7db0a6",
];

export function TopTrendChart({
  payload,
}: {
  payload: DgaPayload;
}) {
  const topIds = [...payload.transformer_summary]
    .sort(
      (a, b) =>
        (a.maintenance_rank ?? a.rank) -
        (b.maintenance_rank ?? b.rank)
    )
    .slice(0, 5)
    .map((summary) => summary.transformer_id);

  const dayMap = new Map<
    string,
    Record<string, number | string | null>
  >();

  for (const transformerId of topIds) {
    const series =
      payload.transformer_timeseries[transformerId] ?? [];

    for (const point of series) {
      const day = point["Sample Day"];

      if (!dayMap.has(day)) {
        dayMap.set(day, { day });
      }

      dayMap.get(day)![transformerId] =
        point.ieee_status ??
        point.pred_ensemble ??
        0;
    }
  }

  const data = Array.from(dayMap.values()).sort(
    (a, b) =>
      new Date(a.day as string).getTime() -
      new Date(b.day as string).getTime()
  );

  if (data.length === 0) {
    return (
      <p className="py-10 text-center text-sm text-teal-400">
        No trend data available.
      </p>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart
        data={data}
        margin={{
          top: 8,
          right: 16,
          left: -16,
          bottom: 0,
        }}
      >
        <CartesianGrid
          strokeDasharray="3 3"
          stroke="#e8e5d9"
          vertical={false}
        />

        <XAxis
          dataKey="day"
          tickFormatter={(value) =>
            formatDate(String(value))
          }
          tick={{
            fontSize: 11,
            fill: "#4f8f83",
          }}
          axisLine={{
            stroke: "#d9d5c4",
          }}
          tickLine={false}
        />

        <YAxis
          domain={[0, 3]}
          ticks={[0, 1, 2, 3]}
          tickFormatter={(value) => {
            const numeric = Number(value);

            if (numeric === 0) {
              return "—";
            }

            return `S${numeric}`;
          }}
          tick={{
            fontSize: 12,
            fill: "#4f8f83",
          }}
          axisLine={false}
          tickLine={false}
          allowDecimals={false}
        />

        <Tooltip
          labelFormatter={(value) =>
            formatDate(String(value))
          }
          formatter={(value) => {
            const numeric = Number(value ?? 0);

            return [
              numeric > 0 ? `S${numeric}` : "Insufficient",
              "IEEE Status",
            ];
          }}
          contentStyle={{
            borderRadius: 10,
            borderColor: "#d9d5c4",
            fontSize: 12,
          }}
        />

        <Legend
          wrapperStyle={{
            fontSize: 12,
          }}
        />

        {topIds.map((id, index) => (
          <Line
            key={id}
            type="monotone"
            dataKey={id}
            stroke={LINE_COLORS[index % LINE_COLORS.length]}
            strokeWidth={2}
            dot={{ r: 3 }}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}