"use client";

import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { TransformerSummary } from "@/types/dga";
import { STATUS_ORDER, scoreToStatus } from "@/lib/severity";

const COLORS: Record<string, string> = {
  Normal: "#1f7a4d",
  Watch: "#a16a07",
  High: "#9a4a1f",
  Critical: "#c62828",
};

export function SeverityDistributionChart({ summaries }: { summaries: TransformerSummary[] }) {
  const counts = { Normal: 0, Watch: 0, High: 0, Critical: 0 };
  for (const s of summaries) counts[scoreToStatus(s.latest_score)]++;
  const data = STATUS_ORDER.map((status) => ({
    status: status === "High" ? "High Risk" : status,
    count: counts[status],
    fill: COLORS[status],
  }));

  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e8e5d9" vertical={false} />
        <XAxis dataKey="status" tick={{ fontSize: 12, fill: "#4f8f83" }} axisLine={{ stroke: "#d9d5c4" }} tickLine={false} />
        <YAxis tick={{ fontSize: 12, fill: "#4f8f83" }} axisLine={false} tickLine={false} allowDecimals={false} />
        <Tooltip
          cursor={{ fill: "rgba(15,47,44,0.04)" }}
          contentStyle={{ borderRadius: 10, borderColor: "#d9d5c4", fontSize: 12 }}
        />
        <Bar dataKey="count" radius={[6, 6, 0, 0]} maxBarSize={64}>
          {data.map((d) => (
            <Cell key={d.status} fill={d.fill} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
