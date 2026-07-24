"use client";

import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp, ArrowUpDown, Search } from "lucide-react";
import type { DgaRow } from "@/types/dga";
import { StatusBadge } from "@/components/ui/badge";
import { classifyScore, nativeToStatus, scoreToRisk, STATUS_STYLES } from "@/lib/severity";
import { formatDate, formatNumber } from "@/lib/utils";

const METHOD_COLUMNS: { key: keyof DgaRow; label: string }[] = [
  { key: "keygas_fault", label: "Key Gas" },
  { key: "iec_fault", label: "IEC 60599" },
  { key: "rogers_fault", label: "Rogers" },
  { key: "doernenburg_fault", label: "Doernenburg" },
  { key: "duval_triangle_fault", label: "Duval Triangle" },
  { key: "fault_p1", label: "Pentagon 1" },
  { key: "duval_pentagon_fault", label: "Pentagon 2" },
];

const GAS_COLUMNS: { key: "h2" | "ch4" | "c2h2" | "c2h4" | "co" | "tdcg"; label: string }[] = [
  { key: "h2", label: "H₂" },
  { key: "ch4", label: "CH₄" },
  { key: "c2h2", label: "C₂H₂" },
  { key: "c2h4", label: "C₂H₄" },
  { key: "co", label: "CO" },
  { key: "tdcg", label: "TDCG" },
];

type SortKey = "sample_day" | "risk" | (typeof GAS_COLUMNS)[number]["key"] | "consensus_fault";
type SortDir = "asc" | "desc";

function SortIcon({ active, dir }: { active: boolean; dir: SortDir }) {
  if (!active) return <ArrowUpDown className="h-3 w-3 text-teal-300" />;
  return dir === "asc" ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />;
}

function Th({
  column,
  label,
  className,
  sortKey,
  sortDir,
  onSort,
}: {
  column: SortKey;
  label: string;
  className?: string;
  sortKey: SortKey;
  sortDir: SortDir;
  onSort: (column: SortKey) => void;
}) {
  return (
    <th className={className}>
      <button
        onClick={() => onSort(column)}
        className="flex items-center gap-1 whitespace-nowrap text-left cursor-pointer hover:text-teal-800"
      >
        {label}
        <SortIcon active={sortKey === column} dir={sortDir} />
      </button>
    </th>
  );
}

/** One row per DGA sample: gas readings, severity status, consensus fault,
 * and every traditional method's individual vote for that record — replaces
 * what used to be two separate tables (Fault Type History + Sample History),
 * since they were both "one row per record" views of the same data. Sortable
 * by clicking a header, filterable by the search box, with a summary bar
 * above showing which fault type/status recurs most for this transformer. */
export function HistoryTable({ rows }: { rows: DgaRow[] }) {
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("sample_day");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const enriched = useMemo(
    () =>
      rows.map((r) => {
        const score = typeof r.severity_score === "number" ? r.severity_score : null;
        const risk = score !== null ? scoreToRisk(score) : null;
        const status = score !== null ? nativeToStatus(r.severity_label ?? classifyScore(score)) : null;
        return { row: r, risk, status };
      }),
    [rows]
  );

  const insights = useMemo(() => {
    const total = enriched.length;
    const statusCounts: Record<string, number> = {};
    const faultCounts: Record<string, number> = {};
    for (const { row, status } of enriched) {
      if (status) statusCounts[status] = (statusCounts[status] ?? 0) + 1;
      const f = row.consensus_fault ?? "UNCERTAIN";
      faultCounts[f] = (faultCounts[f] ?? 0) + 1;
    }
    const topFault = Object.entries(faultCounts).sort((a, b) => b[1] - a[1])[0];
    return { total, statusCounts, topFault };
  }, [enriched]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    let list = enriched;
    if (q) {
      list = list.filter(({ row, status }) => {
        const haystack = [
          formatDate(row.sample_day),
          status ?? "",
          row.consensus_fault ?? "",
          ...METHOD_COLUMNS.map((c) => String(row[c.key] ?? "")),
        ]
          .join(" ")
          .toLowerCase();
        return haystack.includes(q);
      });
    }
    const sorted = [...list].sort((a, b) => {
      let cmp = 0;
      if (sortKey === "sample_day") {
        cmp = new Date(a.row.sample_day).getTime() - new Date(b.row.sample_day).getTime();
      } else if (sortKey === "risk") {
        cmp = (a.risk ?? -1) - (b.risk ?? -1);
      } else if (sortKey === "consensus_fault") {
        cmp = String(a.row.consensus_fault ?? "").localeCompare(String(b.row.consensus_fault ?? ""));
      } else {
        cmp = Number(a.row[sortKey] ?? 0) - Number(b.row[sortKey] ?? 0);
      }
      return sortDir === "asc" ? cmp : -cmp;
    });
    return sorted;
  }, [enriched, query, sortKey, sortDir]);

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  }

  return (
    <div className="space-y-3">
      {/* Insights */}
      <div className="flex flex-wrap items-center gap-2 rounded-xl border border-cream-300 bg-cream-50 px-3 py-2.5 text-xs">
        <span className="font-semibold text-teal-700">{insights.total} record{insights.total === 1 ? "" : "s"}</span>
        {(["Normal", "Watch", "High", "Critical"] as const).map((s) =>
          insights.statusCounts[s] ? (
            <span
              key={s}
              className={`rounded-full px-2 py-0.5 font-semibold ${STATUS_STYLES[s].bg} ${STATUS_STYLES[s].text}`}
            >
              {insights.statusCounts[s]} {s}
            </span>
          ) : null
        )}
        {insights.topFault && insights.topFault[1] > 1 && (
          <span className="ml-auto text-teal-500">
            Most frequent fault:{" "}
            <span className="font-mono font-semibold text-teal-800">{insights.topFault[0]}</span> (
            {insights.topFault[1]}/{insights.total})
          </span>
        )}
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-teal-400" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search by date, status, or fault type…"
          className="w-full rounded-lg border border-teal-200 bg-white py-2 pl-8 pr-3 text-xs text-teal-900 outline-none focus:border-teal-500"
        />
      </div>

      <div className="max-h-[420px] overflow-auto rounded-lg border border-cream-200">
        <table className="w-full min-w-[1180px] text-left text-sm">
          <thead className="sticky top-0 z-10 bg-white">
            <tr className="border-b border-cream-300 text-xs font-semibold uppercase tracking-wide text-teal-400">
              <Th
                column="sample_day"
                label="Sample Day"
                className="sticky left-0 z-20 bg-white px-3 py-2"
                sortKey={sortKey}
                sortDir={sortDir}
                onSort={toggleSort}
              />
              <Th column="risk" label="Status" className="px-3 py-2" sortKey={sortKey} sortDir={sortDir} onSort={toggleSort} />
              {GAS_COLUMNS.map((c) => (
                <Th
                  key={c.key}
                  column={c.key}
                  label={c.label}
                  className="px-3 py-2"
                  sortKey={sortKey}
                  sortDir={sortDir}
                  onSort={toggleSort}
                />
              ))}
              <Th
                column="consensus_fault"
                label="Consensus"
                className="px-3 py-2"
                sortKey={sortKey}
                sortDir={sortDir}
                onSort={toggleSort}
              />
              {METHOD_COLUMNS.map((c) => (
                <th key={String(c.key)} className="whitespace-nowrap px-3 py-2">
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map(({ row, status }, i) => {
              const mixedComponents = (row.mixed_components as string[] | undefined) ?? [];
              const consensus = row.consensus_fault ?? "UNCERTAIN";
              return (
                <tr key={`${row.sample_day}-${i}`} className="border-b border-cream-200 last:border-0 hover:bg-cream-50">
                  <td className="sticky left-0 z-10 bg-white px-3 py-2 font-medium text-teal-800 whitespace-nowrap">
                    {formatDate(row.sample_day)}
                  </td>
                  <td className="px-3 py-2">{status ? <StatusBadge status={status} /> : "—"}</td>
                  {GAS_COLUMNS.map((c) => (
                    <td key={c.key} className="px-3 py-2 text-teal-600">
                      {formatNumber(Number(row[c.key] ?? 0), 1)}
                    </td>
                  ))}
                  <td className="px-3 py-2 font-mono font-semibold text-teal-900 whitespace-nowrap">
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
                        className={`px-3 py-2 font-mono text-xs whitespace-nowrap ${
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
            {filtered.length === 0 && (
              <tr>
                <td colSpan={2 + GAS_COLUMNS.length + 1 + METHOD_COLUMNS.length} className="px-3 py-8 text-center text-teal-400">
                  No records match “{query}”.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
