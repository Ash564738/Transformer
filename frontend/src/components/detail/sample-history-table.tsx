"use client";

import { useState } from "react";
import type { DgaRow } from "@/types/dga";
import { StatusBadge } from "@/components/ui/badge";
import { nativeToStatus } from "@/lib/severity";
import { formatDate, formatNumber } from "@/lib/utils";

export function SampleHistoryTable({ rows }: { rows: DgaRow[] }) {
  const [expanded, setExpanded] = useState(false);
  // Most recent first.
  const ordered = [...rows].reverse();
  const visible = expanded ? ordered : ordered.slice(0, 5);

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[760px] text-left text-sm">
          <thead>
            <tr className="border-b border-cream-300 text-xs font-semibold uppercase tracking-wide text-teal-400">
              <th className="px-3 py-2">Sample Day</th>
              <th className="px-3 py-2">H₂</th>
              <th className="px-3 py-2">CH₄</th>
              <th className="px-3 py-2">C₂H₂</th>
              <th className="px-3 py-2">C₂H₄</th>
              <th className="px-3 py-2">CO</th>
              <th className="px-3 py-2">TDCG</th>
              <th className="px-3 py-2">Status</th>
              <th className="px-3 py-2">Consensus Fault</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((r, i) => (
              <tr key={`${r.sample_day}-${i}`} className="border-b border-cream-200 last:border-0 hover:bg-cream-50">
                <td className="px-3 py-2 font-medium text-teal-800">{formatDate(r.sample_day)}</td>
                <td className="px-3 py-2 text-teal-600">{formatNumber(Number(r.h2 ?? 0), 1)}</td>
                <td className="px-3 py-2 text-teal-600">{formatNumber(Number(r.ch4 ?? 0), 1)}</td>
                <td className="px-3 py-2 text-teal-600">{formatNumber(Number(r.c2h2 ?? 0), 1)}</td>
                <td className="px-3 py-2 text-teal-600">{formatNumber(Number(r.c2h4 ?? 0), 1)}</td>
                <td className="px-3 py-2 text-teal-600">{formatNumber(Number(r.co ?? 0), 1)}</td>
                <td className="px-3 py-2 text-teal-600">{formatNumber(Number(r.tdcg ?? 0), 1)}</td>
                <td className="px-3 py-2">
                  {r.severity_label ? <StatusBadge status={nativeToStatus(r.severity_label)} /> : "—"}
                </td>
                <td className="px-3 py-2 font-mono text-xs text-teal-700">{r.consensus_fault ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {ordered.length > 5 && (
        <button
          onClick={() => setExpanded((v) => !v)}
          className="mt-3 text-xs font-semibold text-copper-600 hover:underline cursor-pointer"
        >
          {expanded ? "Show fewer records" : `Show all ${ordered.length} records`}
        </button>
      )}
    </div>
  );
}
