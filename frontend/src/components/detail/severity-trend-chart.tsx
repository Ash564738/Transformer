// src/components/detail/severity-trend-chart.tsx
"use client";

import { useMemo } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { DgaRow, RiskStatus } from "@/types/dga";
import { ieeeStatusToRiskStatus, STATUS_HEX, STATUS_STYLES } from "@/lib/severity";
import { formatDate } from "@/lib/utils";

interface SeverityPoint {
  day: string;
  risk: number;
  status: RiskStatus;
  fault: string;
}

function StatusDot(props: { cx?: number; cy?: number; payload?: SeverityPoint; singlePoint?: boolean }) {
  const { cx, cy, payload, singlePoint } = props;
  if (cx == null || cy == null || !payload) return null;
  return <circle cx={cx} cy={cy} r={singlePoint ? 7 : 4} fill={STATUS_HEX[payload.status]} stroke="white" strokeWidth={singlePoint ? 2.5 : 1.5} />;
}

const STATUS_TICK_FORMATTER = (value: number) => {
  switch (value) {
    case 1: return "S1 · Normal";
    case 2: return "S2 · Watch";
    case 3: return "S3 · High";
    default: return "—";
  }
};

export function SeverityTrendChart({ rows }: { rows: DgaRow[] }) {
  const data = useMemo(() => rows
    .filter((r) => [1, 2, 3].includes(Number(r.ieee_dga_status)))
    .map((r) => {
      const risk = Number(r.ieee_dga_status);
      return {
        day: r.sample_day,
        risk,
        status: ieeeStatusToRiskStatus(risk),
        fault: r.final_fault ?? r.consensus_fault ?? "ABSTAIN",
      };
    })
    .sort((a, b) => new Date(a.day).getTime() - new Date(b.day).getTime()), [rows]);

  if (data.length === 0) {
    return <p className="py-14 text-center text-sm text-teal-400">No IEEE Status history available.</p>;
  }

  const singlePoint = data.length === 1;

  return (
    <div>
      <div className="mb-2 text-[11px] text-teal-400">
        This chart shows the ordinal IEEE status over time. It does not convert the history into a weighted severity score.
      </div>

      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} margin={{ top: 8, right: 16, left: -16, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e8e5d9" vertical={false} />
          <XAxis dataKey="day" tickFormatter={(v) => formatDate(v)} tick={{ fontSize: 11, fill: "#4f8f83" }} axisLine={{ stroke: "#d9d5c4" }} tickLine={false} />
          <YAxis type="number" domain={[1, 3]} ticks={[1, 2, 3]} tickFormatter={STATUS_TICK_FORMATTER} tick={{ fontSize: 11, fill: "#4f8f83" }} axisLine={false} tickLine={false} width={92} allowDecimals={false} />
          <Tooltip
            labelFormatter={(v) => formatDate(v as string)}
            formatter={(value) => [`S${Number(value)}`, "IEEE Status"]}
            contentStyle={{ borderRadius: 10, borderColor: "#d9d5c4", fontSize: 12 }}
          />
          <Line
            type="monotone"
            dataKey="risk"
            stroke="#184843"
            strokeWidth={2.5}
            dot={(props: { cx?: number; cy?: number; payload?: SeverityPoint; index?: number }) => (
              <StatusDot key={props.index} cx={props.cx} cy={props.cy} payload={props.payload} singlePoint={singlePoint} />
            )}
            activeDot={{ r: singlePoint ? 9 : 6, stroke: "white", strokeWidth: 2 }}
            name="IEEE Status"
            isAnimationActive={!singlePoint}
          />
        </LineChart>
      </ResponsiveContainer>

      {singlePoint && <p className="mt-1 text-center text-[11px] text-teal-400">Only one record is available in this range; no temporal trend can be inferred.</p>}

      <div className="mt-1 flex flex-wrap justify-center gap-3 text-[11px] text-teal-500">
        {(["Normal", "Watch", "High", "Insufficient data"] as const).map((s) => (
          <span key={s} className="flex items-center gap-1">
            <span className={`h-2 w-2 rounded-full ${STATUS_STYLES[s].dot}`} />
            {s}
          </span>
        ))}
      </div>
    </div>
  );
}