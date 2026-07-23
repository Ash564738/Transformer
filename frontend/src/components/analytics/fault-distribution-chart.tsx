"use client";

import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { TransformerSummary } from "@/types/dga";

const PALETTE = ["#184843", "#316f64", "#4f8f83", "#c96f28", "#e08a3c", "#854318", "#7db0a6", "#a8571c"];

export function FaultDistributionChart({ summaries }: { summaries: TransformerSummary[] }) {
  const counts = new Map<string, number>();
  for (const s of summaries) {
    const key = s.fault_type || "UNCERTAIN";
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  const data = Array.from(counts.entries())
    .map(([fault, count]) => ({ fault, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 8);

  if (data.length === 0) {
    return <p className="py-10 text-center text-sm text-teal-400">No fault data available.</p>;
  }

  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={data} layout="vertical" margin={{ top: 8, right: 16, left: 8, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e8e5d9" horizontal={false} />
        <XAxis type="number" tick={{ fontSize: 12, fill: "#4f8f83" }} axisLine={false} tickLine={false} allowDecimals={false} />
        <YAxis
          type="category"
          dataKey="fault"
          width={90}
          tick={{ fontSize: 12, fill: "#184843", fontWeight: 600 }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip cursor={{ fill: "rgba(15,47,44,0.04)" }} contentStyle={{ borderRadius: 10, borderColor: "#d9d5c4", fontSize: 12 }} />
        <Bar dataKey="count" radius={[0, 6, 6, 0]} maxBarSize={20}>
          {data.map((d, i) => (
            <Cell key={d.fault} fill={PALETTE[i % PALETTE.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
