"use client";

import type { TransformerSummary } from "@/types/dga";
import type { RiskStatus } from "@/types/dga";
import { STATUS_HEX } from "@/lib/severity";
import { stationOf } from "@/lib/transformer-helpers";

export type StationStatusRecord = Pick<TransformerSummary, "loc" | "transformer_id"> & {
  transformer_overall_severity_level?: number | string | null;
  ieee_status?: number | string | null;
};

function statusForRecord(summary: StationStatusRecord): RiskStatus {
  const status = Number(summary.transformer_overall_severity_level ?? summary.ieee_status ?? 0);
  return status === 3 ? "High" : status === 2 ? "Watch" : status === 1 ? "Normal" : "Insufficient data";
}

export function StationStatusHeatmap({ summaries }: { summaries: StationStatusRecord[] }) {
  const uniqueSummaries = Array.from(
    new Map(summaries.map((summary) => [summary.transformer_id, summary])).values()
  );
  const stations = Array.from(new Set(uniqueSummaries.map((summary) => stationOf(summary as TransformerSummary)))).sort();
  const counts = new Map<string, { total: number; high: number; watch: number; normal: number }>();
  for (const summary of uniqueSummaries) {
    const station = stationOf(summary as TransformerSummary);
    const current = counts.get(station) ?? { total: 0, high: 0, watch: 0, normal: 0 };
    current.total += 1;
    if (statusForRecord(summary) === "High") current.high += 1;
    if (statusForRecord(summary) === "Watch") current.watch += 1;
    if (statusForRecord(summary) === "Normal") current.normal += 1;
    counts.set(station, current);
  }
  return (
    <div className="overflow-x-auto">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2 text-[11px] text-teal-500">
        <span>Darker cells indicate a higher share of transformers in that condition.</span>
        <span className="flex gap-3">
          <span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-red-500" />Status 3</span>
          <span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-amber-400" />Status 2</span>
          <span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-emerald-500" />Status 1</span>
        </span>
      </div>
      <table className="min-w-full border-separate border-spacing-y-1 text-left text-xs">
        <thead><tr className="text-[10px] uppercase tracking-wider text-teal-500"><th className="px-3 py-2">Station</th><th className="px-3 py-2">Fleet</th><th className="px-2 py-2">Status 3</th><th className="px-2 py-2">Status 2</th><th className="px-2 py-2">Status 1</th></tr></thead>
        <tbody>
          {stations.map((station) => {
            const value = counts.get(station)!;
            const cell = (count: number, color: string) => ({
              backgroundColor: `color-mix(in srgb, ${color} ${Math.max(10, Math.round(count / value.total * 100))}%, white)`,
            });
            return <tr key={station} className="border-b border-teal-50">
              <td className="rounded-l-lg bg-teal-50 px-3 py-3 font-semibold text-teal-800">{station}</td>
              <td className="bg-slate-50 px-3 py-3 font-semibold text-slate-600">{value.total}</td>
              <td className="px-2 py-3 text-center font-semibold text-red-950" style={cell(value.high, STATUS_HEX.High)}>{value.high}<span className="ml-1 text-[10px] font-normal opacity-70">({Math.round(value.high / value.total * 100)}%)</span></td>
              <td className="px-2 py-3 text-center font-semibold text-amber-950" style={cell(value.watch, STATUS_HEX.Watch)}>{value.watch}<span className="ml-1 text-[10px] font-normal opacity-70">({Math.round(value.watch / value.total * 100)}%)</span></td>
              <td className="rounded-r-lg px-2 py-3 text-center font-semibold text-emerald-950" style={cell(value.normal, STATUS_HEX.Normal)}>{value.normal}<span className="ml-1 text-[10px] font-normal opacity-70">({Math.round(value.normal / value.total * 100)}%)</span></td>
            </tr>;
          })}
        </tbody>
      </table>
    </div>
  );
}
