// src/components/overview/fault-distribution-chart.tsx
"use client";

import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { DgaRow, TransformerSummary } from "@/types/dga";

const PALETTE = ["#184843", "#316f64", "#4f8f83", "#c96f28", "#e08a3c", "#854318", "#7db0a6", "#a8571c"];

type Basis = "sample" | "transformer";

export function FaultDistributionChart({ summaries, rows }: { summaries: TransformerSummary[]; rows?: DgaRow[] }) {
  const [basis, setBasis] = useState<Basis>("sample");
  const [topN, setTopN] = useState(8);

  const sourceRows = rows && rows.length > 0 ? rows : summaries;

  const data = useMemo(() => {
    const counts = new Map<string, number>();

    if (basis === "sample") {
      for (const item of sourceRows) {
        const finalFault = "final_fault" in item ? item.final_fault : undefined;
        const consensusFault = "consensus_fault" in item ? item.consensus_fault : undefined;
        const fallbackFault = "fault_type" in item ? item.fault_type : undefined;
        const rawKey =
          typeof finalFault === "string" && finalFault !== "ABSTAIN" ? finalFault :
          typeof consensusFault === "string" && consensusFault !== "ABSTAIN" ? consensusFault :
          typeof fallbackFault === "string" && fallbackFault ? fallbackFault : "ABSTAIN";
        const key = String(rawKey).trim() || "ABSTAIN";
        counts.set(key, (counts.get(key) ?? 0) + 1);
      }
    } else {
      for (const item of summaries) {
        const key = String(item.fault_type || "ABSTAIN").trim() || "ABSTAIN";
        counts.set(key, (counts.get(key) ?? 0) + 1);
      }
    }

    return Array.from(counts.entries())
      .map(([fault, count]) => ({ fault, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, topN);
  }, [basis, sourceRows, summaries, topN]);

  if (data.length === 0) {
    return <p className="py-10 text-center text-sm text-teal-400">No fault data available.</p>;
  }

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div className="text-[11px] text-teal-400">
          Distribution basis: <span className="font-semibold text-teal-600">{basis === "sample" ? "sample-level diagnoses" : "transformer-level latest diagnoses"}</span>. This is not a fault-prevalence estimate for the real fleet.
        </div>

        <div className="flex items-center gap-2 text-xs text-teal-600">
          <select value={basis} onChange={(event) => setBasis(event.target.value as Basis)} className="h-8 rounded-md border border-teal-200 bg-white px-2 text-xs text-teal-800">
            <option value="sample">By sample</option>
            <option value="transformer">By transformer</option>
          </select>
          <select value={topN} onChange={(event) => setTopN(Number(event.target.value))} className="h-8 rounded-md border border-teal-200 bg-white px-2 text-xs text-teal-800">
            {[5, 8, 10, 15].map((value) => <option key={value} value={value}>Top {value}</option>)}
          </select>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} layout="vertical" margin={{ top: 8, right: 16, left: 8, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e8e5d9" horizontal={false} />
          <XAxis type="number" tick={{ fontSize: 12, fill: "#4f8f83" }} axisLine={false} tickLine={false} allowDecimals={false} />
          <YAxis type="category" dataKey="fault" width={100} tick={{ fontSize: 12, fill: "#184843", fontWeight: 600 }} axisLine={false} tickLine={false} />
          <Tooltip cursor={{ fill: "rgba(15,47,44,0.04)" }} contentStyle={{ borderRadius: 10, borderColor: "#d9d5c4", fontSize: 12 }} />
          <Bar dataKey="count" name={basis === "sample" ? "Samples" : "Transformers"} radius={[0, 6, 6, 0]} maxBarSize={20}>
            {data.map((item, index) => <Cell key={item.fault} fill={PALETTE[index % PALETTE.length]} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}