// src/components/overview/severity-distribution-chart.tsx
"use client";

import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { RiskStatus, TransformerSummary } from "@/types/dga";
import { STATUS_HEX, STATUS_ORDER, statusFromSummary } from "@/lib/severity";

export function SeverityDistributionChart({ summaries }: { summaries: TransformerSummary[] }) {
  const counts: Record<RiskStatus, number> = {
    Normal: 0,
    Watch: 0,
    High: 0,
    "Insufficient data": 0,
  };

  for (const summary of summaries) counts[statusFromSummary(summary)] += 1;

  const data = STATUS_ORDER.map((status) => ({
    status: status === "High" ? "Status 3 · High" : status === "Watch" ? "Status 2 · Watch" : status === "Normal" ? "Status 1 · Normal" : "Insufficient data",
    count: counts[status],
    fill: STATUS_HEX[status],
  }));

  return (
    <div>
      <div className="mb-2 text-[11px] text-teal-400">
        Transformer-level distribution of the latest IEEE DGA status. Insufficient-data cases are shown even when the count is zero.
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={data} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e8e5d9" vertical={false} />
          <XAxis dataKey="status" tick={{ fontSize: 11, fill: "#4f8f83" }} axisLine={{ stroke: "#d9d5c4" }} tickLine={false} />
          <YAxis tick={{ fontSize: 12, fill: "#4f8f83" }} axisLine={false} tickLine={false} allowDecimals={false} />
          <Tooltip cursor={{ fill: "rgba(15,47,44,0.04)" }} contentStyle={{ borderRadius: 10, borderColor: "#d9d5c4", fontSize: 12 }} />
          <Bar dataKey="count" name="Transformers" radius={[6, 6, 0, 0]} maxBarSize={64}>
            {data.map((item) => <Cell key={item.status} fill={item.fill} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}