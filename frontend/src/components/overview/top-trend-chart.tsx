// src/components/overview/top-trend-chart.tsx
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
import type { DgaPayload } from "@/types/dga";
import { formatDate, formatNumber } from "@/lib/utils";

const LINE_COLORS = [
  "#c62828",
  "#a8571c",
  "#a16a07",
  "#316f64",
  "#7db0a6",
  "#6b4f9a",
  "#276678",
  "#8b5e34",
  "#5f7a61",
  "#ad3f63",
];

type ChartMetric = "status" | "evidence";
type ChartValue = number | null;
type TrendRow = { day: string; [transformerId: string]: ChartValue | string };

export function TopTrendChart({ payload }: { payload: DgaPayload }) {
  const [topN, setTopN] = useState(5);
  const [metric, setMetric] = useState<ChartMetric>("evidence");

  const topIds = useMemo(
    () =>
      [...payload.transformer_summary]
        .sort(
          (a, b) =>
            (a.maintenance_rank ?? a.rank) -
            (b.maintenance_rank ?? b.rank)
        )
        .slice(0, topN)
        .map((summary) => summary.transformer_id),
    [payload.transformer_summary, topN]
  );

  const { data, pointCounts } = useMemo(() => {
    const dayMap = new Map<string, TrendRow>();

    for (const transformerId of topIds) {
      const series = payload.transformer_timeseries[transformerId] ?? [];
      for (const point of series) {
        const day = String(point["Sample Day"] ?? "").trim();
        if (!day) continue;

        if (!dayMap.has(day)) dayMap.set(day, { day });

        const raw =
          metric === "status"
            ? point.ieee_status
            : point.continuous_evidence_ratio;

        const numeric = Number(raw);
        dayMap.get(day)![transformerId] = Number.isFinite(numeric)
          ? numeric
          : null;
      }
    }

    const chartData = Array.from(dayMap.values()).sort(
      (a, b) =>
        new Date(String(a.day)).getTime() -
        new Date(String(b.day)).getTime()
    );

    const counts = topIds.map((id) => ({
      id,
      count: chartData.filter((row) => Number.isFinite(Number(row[id]))).length,
    }));

    return { data: chartData, pointCounts: counts };
  }, [metric, payload.transformer_timeseries, topIds]);

  if (data.length === 0) {
    return (
      <p className="py-10 text-center text-sm text-teal-400">
        No trend data available.
      </p>
    );
  }

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div className="space-y-1 text-[11px] text-teal-400">
          {metric === "status" ? (
            <div>
              Y-axis: discrete IEEE DGA Status. Equal S3 points can still have
              materially different gas evidence.
            </div>
          ) : (
            <div>
              Y-axis: per-sample DGA evidence ratio. This is threshold evidence,
              not a weighted severity or health score.
            </div>
          )}
          <div>
            Lines connect available samples across date gaps; the gap itself does
            not imply a missing status.
          </div>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-xs font-semibold text-teal-600">
            Metric
          </label>
          <select
            value={metric}
            onChange={(event) => setMetric(event.target.value as ChartMetric)}
            className="h-8 rounded-md border border-teal-200 bg-white px-2 text-xs text-teal-800"
          >
            <option value="evidence">DGA evidence ratio</option>
            <option value="status">IEEE Status</option>
          </select>

          <label className="text-xs font-semibold text-teal-600">
            Top
          </label>
          <select
            value={topN}
            onChange={(event) => setTopN(Number(event.target.value))}
            className="h-8 rounded-md border border-teal-200 bg-white px-2 text-xs text-teal-800"
          >
            {[3, 5, 8, 10, 15, 20].map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={320}>
        <LineChart
          data={data}
          margin={{ top: 8, right: 20, left: 0, bottom: 8 }}
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
            minTickGap={24}
          />
          <YAxis
            domain={metric === "status" ? [1, 3] : [0, "auto"]}
            ticks={metric === "status" ? [1, 2, 3] : undefined}
            tickFormatter={(value) =>
              metric === "status"
                ? `S${Number(value)}`
                : `${formatNumber(Number(value), 1)}×`
            }
            tick={{ fontSize: 12, fill: "#4f8f83" }}
            axisLine={false}
            tickLine={false}
            allowDecimals={metric !== "status"}
          />
          <Tooltip
            labelFormatter={(value) => formatDate(String(value))}
            formatter={(value, name) => {
              const numeric = Number(value);
              if (!Number.isFinite(numeric)) return ["—", String(name ?? "Transformer")];
              return [
                metric === "status" ? `S${numeric}` : `${formatNumber(numeric, 2)}×`,
                String(name ?? "Transformer"),
              ];
            }}
            contentStyle={{
              borderRadius: 10,
              borderColor: "#d9d5c4",
              fontSize: 12,
            }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />

          {topIds.map((id, index) => {
            const sampleCount = pointCounts[index]?.count ?? 0;
            return (
              <Line
                key={id}
                type={metric === "status" ? "monotone" : "monotone"}
                dataKey={id}
                name={`${id}${sampleCount === 1 ? " · 1 sample" : ""}`}
                stroke={LINE_COLORS[index % LINE_COLORS.length]}
                strokeWidth={2.25}
                dot={{ r: 3 }}
                activeDot={{ r: 5 }}
                connectNulls
                isAnimationActive={false}
              />
            );
          })}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}