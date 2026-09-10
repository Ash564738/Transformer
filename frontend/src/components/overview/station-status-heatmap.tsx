"use client";

import type { TransformerSummary } from "@/types/dga";
import { STATUS_HEX, statusFromSummary } from "@/lib/severity";
import { stationOf } from "@/lib/transformer-helpers";

export function StationStatusHeatmap({ summaries }: { summaries: TransformerSummary[] }) {
  const stations = Array.from(new Set(summaries.map(stationOf))).sort();
  const counts = new Map<string, { total: number; high: number; watch: number }>();
  for (const summary of summaries) {
    const station = stationOf(summary);
    const current = counts.get(station) ?? { total: 0, high: 0, watch: 0 };
    current.total += 1;
    if (statusFromSummary(summary) === "High") current.high += 1;
    if (statusFromSummary(summary) === "Watch") current.watch += 1;
    counts.set(station, current);
  }
  return (
    <div className="overflow-x-auto">
      <div className="mb-3 text-[11px] text-teal-400">
        Station heatmap: darker cells indicate more transformers in the condition class.
      </div>
      <table className="min-w-full text-left text-xs">
        <thead><tr className="border-b border-teal-100 text-teal-500"><th className="px-3 py-2">Station</th><th className="px-3 py-2">Transformers</th><th className="px-3 py-2">Status 3</th><th className="px-3 py-2">Status 2</th></tr></thead>
        <tbody>
          {stations.map((station) => {
            const value = counts.get(station)!;
            return <tr key={station} className="border-b border-teal-50">
              <td className="px-3 py-2 font-semibold text-teal-800">{station}</td>
              <td className="px-3 py-2 text-teal-600">{value.total}</td>
              <td className="px-3 py-2 font-semibold" style={{ backgroundColor: `color-mix(in srgb, ${STATUS_HEX.High} ${Math.max(12, Math.round(value.high / value.total * 100))}%, white)` }}>{value.high}</td>
              <td className="px-3 py-2 font-semibold" style={{ backgroundColor: `color-mix(in srgb, ${STATUS_HEX.Watch} ${Math.max(12, Math.round(value.watch / value.total * 100))}%, white)` }}>{value.watch}</td>
            </tr>;
          })}
        </tbody>
      </table>
    </div>
  );
}
