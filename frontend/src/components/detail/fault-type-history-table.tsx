"use client";

import type { DgaRow } from "@/types/dga";
import { classifyScore, nativeToStatus, STATUS_STYLES } from "@/lib/severity";
import { formatDate } from "@/lib/utils";

const METHOD_COLUMNS: { key: keyof DgaRow; label: string }[] = [
  { key: "keygas_fault", label: "Key Gas" },
  { key: "iec_fault", label: "IEC 60599" },
  { key: "rogers_fault", label: "Rogers" },
  { key: "doernenburg_fault", label: "Doernenburg" },
  { key: "duval_triangle_fault", label: "Duval Triangle" },
  { key: "fault_p1", label: "Pentagon 1" },
  { key: "duval_pentagon_fault", label: "Pentagon 2" },
];

/** One row per DGA sample (not just the latest): a transformer's fault type
 * is not fixed — each traditional method can disagree at each sampling
 * point, and gas levels sometimes fall below every method's threshold (an
 * ABSTAIN, shown as "—") even when the unit is genuinely degrading. Seeing
 * the full history side-by-side is what lets an engineer tell "consistently
 * flagged as D2" apart from "flickered between MIXED and UNCERTAIN because
 * gas concentrations were borderline." */
export function FaultTypeHistoryTable({ rows }: { rows: DgaRow[] }) {
  if (rows.length === 0) {
    return <p className="text-sm text-teal-400">No sample history available.</p>;
  }
  // Most recent first.
  const sorted = [...rows].sort((a, b) => new Date(b.sample_day).getTime() - new Date(a.sample_day).getTime());

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-cream-300 text-xs font-semibold uppercase tracking-wide text-teal-400">
            <th className="sticky left-0 bg-white py-2 pr-3">Sample day</th>
            <th className="py-2 pr-3">Severity</th>
            <th className="py-2 pr-3">Consensus</th>
            {METHOD_COLUMNS.map((c) => (
              <th key={String(c.key)} className="py-2 pr-3 whitespace-nowrap">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, i) => {
            const status =
              typeof row.severity_score === "number"
                ? nativeToStatus(row.severity_label ?? classifyScore(row.severity_score))
                : undefined;
            const style = status ? STATUS_STYLES[status] : undefined;
            const consensus = row.consensus_fault ?? "UNCERTAIN";
            const mixedComponents = (row.mixed_components as string[] | undefined) ?? [];
            return (
              <tr key={`${row.sample_day}-${i}`} className="border-b border-cream-200 last:border-0">
                <td className="sticky left-0 bg-white py-2 pr-3 font-medium text-teal-800 whitespace-nowrap">
                  {formatDate(row.sample_day)}
                </td>
                <td className="py-2 pr-3">
                  {style ? (
                    <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${style.bg} ${style.text}`}>
                      {status}
                    </span>
                  ) : (
                    "—"
                  )}
                </td>
                <td className="py-2 pr-3 font-mono font-semibold text-teal-900">
                  {consensus === "MIXED" && mixedComponents.length > 0
                    ? `MIXED (${mixedComponents.join("+")})`
                    : consensus}
                </td>
                {METHOD_COLUMNS.map((c) => {
                  const value = row[c.key] as string | undefined;
                  const isAbstain = !value || value === "UNCERTAIN" || value === "-1";
                  return (
                    <td
                      key={String(c.key)}
                      className={`py-2 pr-3 font-mono text-xs whitespace-nowrap ${
                        isAbstain ? "text-teal-300" : "text-teal-700"
                      }`}
                    >
                      {isAbstain ? "— abstain" : value}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
