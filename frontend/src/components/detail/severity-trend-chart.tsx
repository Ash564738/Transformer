"use client";

import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DgaRow } from "@/types/dga";
import { classifyScore, nativeToStatus, scoreToRisk, STATUS_STYLES } from "@/lib/severity";
import { formatDate } from "@/lib/utils";

/** Each transformer's own severity trajectory over time — one point per
 * sample record, not the raw gas concentrations. This is what lets an
 * engineer see whether a unit's condition is trending toward Critical,
 * holding steady, or recovering after maintenance. */
export function SeverityTrendChart({ rows }: { rows: DgaRow[] }) {
  const data = useMemo(
    () =>
      rows
        .filter((r) => typeof r.severity_score === "number")
        .map((r) => {
          const risk = scoreToRisk(r.severity_score!);
          const status = nativeToStatus(r.severity_label ?? classifyScore(r.severity_score!));
          return {
            day: r.sample_day,
            risk,
            status,
            fault: r.consensus_fault ?? "UNCERTAIN",
          };
        }),
    [rows]
  );

  if (data.length === 0) {
    return <p className="py-14 text-center text-sm text-teal-400">No severity history available.</p>;
  }

  if (data.length === 1) {
    const only = data[0];
    const style = STATUS_STYLES[only.status];
    return (
      <div className="flex flex-col items-center justify-center gap-2 py-12 text-center">
        <div className={`rounded-full px-4 py-2 text-2xl font-extrabold ${style.bg} ${style.text}`}>
          {only.risk}
        </div>
        <p className="text-xs text-teal-500">
          Only one record for this transformer ({formatDate(only.day)}) — a trend needs at least 2 samples.
        </p>
      </div>
    );
  }

  return (
    <div>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} margin={{ top: 8, right: 16, left: -16, bottom: 0 }}>
          <ReferenceArea y1={0} y2={30} fill="#2f9e6f" fillOpacity={0.06} />
          <ReferenceArea y1={30} y2={60} fill="#c9922b" fillOpacity={0.06} />
          <ReferenceArea y1={60} y2={89} fill="#b5562b" fillOpacity={0.06} />
          <ReferenceArea y1={89} y2={100} fill="#c62828" fillOpacity={0.07} />
          <CartesianGrid strokeDasharray="3 3" stroke="#e8e5d9" vertical={false} />
          <XAxis
            dataKey="day"
            tickFormatter={(v) => formatDate(v)}
            tick={{ fontSize: 11, fill: "#4f8f83" }}
            axisLine={{ stroke: "#d9d5c4" }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fontSize: 12, fill: "#4f8f83" }}
            axisLine={false}
            tickLine={false}
            width={32}
          />
          <Tooltip
            labelFormatter={(v) => formatDate(v as string)}
            formatter={(value, _name, entry) => {
              const p = entry.payload as (typeof data)[number];
              return [`${value} · ${p.status} · ${p.fault}`, "Severity"];
            }}
            contentStyle={{ borderRadius: 10, borderColor: "#d9d5c4", fontSize: 12 }}
          />
          <Line
            type="monotone"
            dataKey="risk"
            stroke="#184843"
            strokeWidth={2.5}
            dot={{ r: 3.5, fill: "#184843" }}
            activeDot={{ r: 5 }}
            name="Severity"
          />
        </LineChart>
      </ResponsiveContainer>
      <div className="mt-1 flex flex-wrap justify-center gap-3 text-[11px] text-teal-500">
        {(["Normal", "Watch", "High", "Critical"] as const).map((s) => (
          <span key={s} className="flex items-center gap-1">
            <span className={`h-2 w-2 rounded-full ${STATUS_STYLES[s].dot}`} />
            {s}
          </span>
        ))}
      </div>
    </div>
  );
}
