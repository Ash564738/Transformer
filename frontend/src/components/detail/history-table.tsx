// src/components/detail/history-table.tsx
"use client";

import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp, ArrowUpDown, Search } from "lucide-react";
import type { DgaRow } from "@/types/dga";
import { StatusBadge } from "@/components/ui/badge";
import { ieeeStatusToRiskStatus, STATUS_STYLES } from "@/lib/severity";
import { formatDate, formatNumber } from "@/lib/utils";

const METHOD_COLUMNS: { key: keyof DgaRow; label: string }[] = [
  { key: "keygas_fault", label: "Key Gas" },
  { key: "iec_fault", label: "IEC 60599" },
  { key: "rogers_fault", label: "Rogers" },
  { key: "doernenburg_fault", label: "Doernenburg" },
  { key: "duval_triangle_fault", label: "Duval Triangle" },
  { key: "duval_pentagon_p1_fault", label: "Pentagon 1" },
  { key: "duval_pentagon_p2_fault", label: "Pentagon 2" },
];

const GAS_COLUMNS: { key: "h2" | "ch4" | "c2h2" | "c2h4" | "co" | "tdcg"; label: string }[] = [
  { key: "h2", label: "H₂" },
  { key: "ch4", label: "CH₄" },
  { key: "c2h2", label: "C₂H₂" },
  { key: "c2h4", label: "C₂H₄" },
  { key: "co", label: "CO" },
  { key: "tdcg", label: "TDCG" },
];

type SortKey = "sample_day" | "risk" | "evidence" | (typeof GAS_COLUMNS)[number]["key"] | "consensus_fault";
type SortDir = "asc" | "desc";

export function HistoryTable({ rows }: { rows: DgaRow[] }) {
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("sample_day");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const enriched = useMemo(
    () => rows.map((row) => ({ row, status: ieeeStatusToRiskStatus(row.ieee_dga_status) })),
    [rows]
  );

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    let list = enriched;

    if (q) {
      list = list.filter(({ row, status }) => [
        formatDate(row.sample_day),
        status,
        row.final_fault,
        row.consensus_fault,
        row.fault_criticality_class,
        row.ieee_dga_status_reason,
        ...METHOD_COLUMNS.map((column) => String(row[column.key] ?? "")),
      ].join(" ").toLowerCase().includes(q));
    }

    return [...list].sort((a, b) => {
      let cmp = 0;
      if (sortKey === "sample_day") cmp = new Date(a.row.sample_day).getTime() - new Date(b.row.sample_day).getTime();
      else if (sortKey === "risk") cmp = (a.row.ieee_dga_status ?? -1) - (b.row.ieee_dga_status ?? -1);
      else if (sortKey === "evidence") cmp = Number(a.row.ieee_max_status3_standardized_exceedance ?? 1) - Number(b.row.ieee_max_status3_standardized_exceedance ?? 1);
      else if (sortKey === "consensus_fault") cmp = String(a.row.final_fault ?? a.row.consensus_fault ?? "").localeCompare(String(b.row.final_fault ?? b.row.consensus_fault ?? ""));
      else cmp = Number(a.row[sortKey] ?? 0) - Number(b.row[sortKey] ?? 0);
      return sortDir === "asc" ? cmp : -cmp;
    });
  }, [enriched, query, sortKey, sortDir]);

  function toggleSort(key: SortKey) {
    if (sortKey === key) setSortDir((dir) => (dir === "asc" ? "desc" : "asc"));
    else {
      setSortKey(key);
      setSortDir("desc");
    }
  }

  const insightCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const { status } of enriched) counts[status] = (counts[status] ?? 0) + 1;
    return counts;
  }, [enriched]);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 rounded-xl border border-cream-300 bg-cream-50 px-3 py-2.5 text-xs">
        <span className="font-semibold text-teal-700">{enriched.length} record{enriched.length === 1 ? "" : "s"}</span>
        {(["Normal", "Watch", "High", "Insufficient data"] as const).map((severity) =>
          insightCounts[severity] ? (
            <span key={severity} className={`rounded-full px-2 py-0.5 font-semibold ${STATUS_STYLES[severity].bg} ${STATUS_STYLES[severity].text}`}>
              {insightCounts[severity]} {severity}
            </span>
          ) : null
        )}
      </div>

      <div className="relative">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-teal-400" />
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search by date, IEEE status, fault, or evidence…"
          className="w-full rounded-lg border border-teal-200 bg-white py-2 pl-8 pr-3 text-xs text-teal-900 outline-none focus:border-teal-500"
        />
      </div>

      <div className="max-h-[420px] overflow-auto rounded-lg border border-cream-200">
        <table className="w-full min-w-[1320px] text-left text-sm">
          <thead className="sticky top-0 z-10 bg-white">
            <tr className="border-b border-cream-300 text-xs font-semibold uppercase tracking-wide text-teal-400">
              <Header label="Sample Day" column="sample_day" sortKey={sortKey} sortDir={sortDir} onSort={toggleSort} />
              <Header label="IEEE Status" column="risk" sortKey={sortKey} sortDir={sortDir} onSort={toggleSort} />
              <Header label="S3 Evidence" column="evidence" sortKey={sortKey} sortDir={sortDir} onSort={toggleSort} />
              {GAS_COLUMNS.map((column) => <Header key={column.key} label={column.label} column={column.key} sortKey={sortKey} sortDir={sortDir} onSort={toggleSort} />)}
              <Header label="Fault" column="consensus_fault" sortKey={sortKey} sortDir={sortDir} onSort={toggleSort} />
              <th className="px-3 py-2">Fault context</th>
              {METHOD_COLUMNS.map((column) => <th key={String(column.key)} className="whitespace-nowrap px-3 py-2">{column.label}</th>)}
            </tr>
          </thead>
          <tbody>
            {filtered.map(({ row, status }, index) => {
              const consensus = row.final_fault ?? row.consensus_fault ?? "ABSTAIN";
              return (
                <tr key={`${row.transformer_id}-${row.sample_day}-${index}`} className="border-b border-cream-200 last:border-0 hover:bg-cream-50">
                  <td className="sticky left-0 z-10 whitespace-nowrap bg-white px-3 py-2 font-medium text-teal-800">{formatDate(row.sample_day)}</td>
                  <td className="px-3 py-2"><StatusBadge status={status} /></td>
                  <td className="px-3 py-2 font-mono text-teal-700">{row.ieee_max_status3_standardized_exceedance != null ? `${formatNumber(row.ieee_max_status3_standardized_exceedance, 2)}×` : "—"}</td>
                  {GAS_COLUMNS.map((column) => <td key={column.key} className="px-3 py-2 text-teal-600">{formatNumber(Number(row[column.key] ?? 0), 1)}</td>)}
                  <td className="whitespace-nowrap px-3 py-2 font-mono font-semibold text-teal-900">{consensus}</td>
                  <td className="whitespace-nowrap px-3 py-2 text-[11px] uppercase tracking-wide text-teal-500">{row.fault_criticality_class ?? "UNKNOWN"}</td>
                  {METHOD_COLUMNS.map((column) => {
                    const value = row[column.key] as string | undefined;
                    const abstain = !value || value === "ABSTAIN" || value === "-1";
                    return <td key={String(column.key)} className={`whitespace-nowrap px-3 py-2 font-mono text-xs ${abstain ? "text-teal-300" : "text-teal-700"}`}>{abstain ? "— abstain" : value}</td>;
                  })}
                </tr>
              );
            })}
            {filtered.length === 0 && <tr><td colSpan={3 + GAS_COLUMNS.length + METHOD_COLUMNS.length + 2} className="px-3 py-8 text-center text-teal-400">No records match “{query}”.</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Header({ label, column, sortKey, sortDir, onSort }: { label: string; column: SortKey; sortKey: SortKey; sortDir: SortDir; onSort: (column: SortKey) => void }) {
  const active = sortKey === column;
  return (
    <th className="px-3 py-2">
      <button type="button" onClick={() => onSort(column)} className="flex items-center gap-1 whitespace-nowrap hover:text-teal-800">
        {label}
        {!active ? <ArrowUpDown className="h-3 w-3 text-teal-300" /> : sortDir === "asc" ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />}
      </button>
    </th>
  );
}