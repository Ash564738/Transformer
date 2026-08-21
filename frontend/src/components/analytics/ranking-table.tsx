// src/components/analytics/ranking-table.tsx
"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { Search, ArrowUpDown, ArrowUp, ArrowDown } from "lucide-react";
import type { DgaPayload, MaintenancePriority } from "@/types/dga";
import { StatusBadge } from "@/components/ui/badge";
import {
  MAINTENANCE_PRIORITY_STYLES,
  normalizeMaintenancePriority,
  maintenancePriorityLabel,
  statusFromSummary,
  STATUS_STYLES,
} from "@/lib/severity";
import { getStations, latestRowFor, stationOf, topGasLabel } from "@/lib/transformer-helpers";
import { formatDate, formatNumber } from "@/lib/utils";

type SortColumn = "rank" | "id" | "station" | "priority" | "status" | "fault" | "date" | "evidence";
type SortDirection = "asc" | "desc";

interface RowData {
  summary: DgaPayload["transformer_summary"][number];
  row: ReturnType<typeof latestRowFor>;
}

function PriorityBadge({ priority }: { priority: MaintenancePriority | string }) {
  const normalized = normalizeMaintenancePriority(priority);
  const style = MAINTENANCE_PRIORITY_STYLES[normalized];
  return (
    <span className={`inline-flex rounded-full border px-2 py-0.5 text-[11px] font-semibold ${style.bg} ${style.border} ${style.text}`}>
      {maintenancePriorityLabel(normalized)}
    </span>
  );
}

export function RankingTable({ payload, limit }: { payload: DgaPayload; limit?: number }) {
  const [query, setQuery] = useState("");
  const [station, setStation] = useState("All Stations");
  const [faultFilter, setFaultFilter] = useState("All");
  const [priorityFilter, setPriorityFilter] = useState("All");
  const [sortColumn, setSortColumn] = useState<SortColumn>("rank");
  const [sortDirection, setSortDirection] = useState<SortDirection>("asc");

  const stations = useMemo(() => ["All Stations", ...getStations(payload)], [payload]);
  const faultTypes = useMemo(() => {
    const types = new Set(payload.transformer_summary.map((s) => s.fault_type).filter(Boolean));
    return ["All", ...Array.from(types).sort()];
  }, [payload]);

  const allRows = useMemo<RowData[]>(
    () => payload.transformer_summary.map((summary) => ({ summary, row: latestRowFor(payload, summary.transformer_id) })),
    [payload]
  );

  const filteredRows = useMemo(() => {
    let list = allRows;
    const q = query.trim().toLowerCase();
    if (q) {
      list = list.filter(({ summary }) =>
        [summary.transformer_id, stationOf(summary), summary.fault_type, summary.maintenance_priority]
          .join(" ").toLowerCase().includes(q)
      );
    }
    if (station !== "All Stations") list = list.filter(({ summary }) => stationOf(summary) === station);
    if (faultFilter !== "All") list = list.filter(({ summary }) => summary.fault_type === faultFilter);
    if (priorityFilter !== "All") list = list.filter(({ summary }) => summary.maintenance_priority === priorityFilter);
    return list;
  }, [allRows, query, station, faultFilter, priorityFilter]);

  const sortedRows = useMemo(() => {
    const list = [...filteredRows];
    const dir = sortDirection === "asc" ? 1 : -1;
    const priorityOrder: MaintenancePriority[] = ["CRITICAL", "HIGH_RISK", "WATCH", "NORMAL", "DATA_REVIEW"];
    const statusOrder = ["Insufficient data", "Normal", "Watch", "High"];

    return list.sort((a, b) => {
      switch (sortColumn) {
        case "rank": return ((a.summary.maintenance_rank ?? a.summary.rank) - (b.summary.maintenance_rank ?? b.summary.rank)) * dir;
        case "id": return a.summary.transformer_id.localeCompare(b.summary.transformer_id) * dir;
        case "station": return stationOf(a.summary).localeCompare(stationOf(b.summary)) * dir;
        case "priority": return (priorityOrder.indexOf(a.summary.maintenance_priority as MaintenancePriority) - priorityOrder.indexOf(b.summary.maintenance_priority as MaintenancePriority)) * dir;
        case "status": return (statusOrder.indexOf(statusFromSummary(a.summary)) - statusOrder.indexOf(statusFromSummary(b.summary))) * dir;
        case "fault": return (a.summary.fault_type || "").localeCompare(b.summary.fault_type || "") * dir;
        case "date": return (new Date(a.summary.latest_sample_day).getTime() - new Date(b.summary.latest_sample_day).getTime()) * dir;
        case "evidence": return (Number(a.summary.current_status3_standardized_exceedance ?? 1) - Number(b.summary.current_status3_standardized_exceedance ?? 1)) * dir;
        default: return 0;
      }
    });
  }, [filteredRows, sortColumn, sortDirection]);

  const displayRows = limit ? sortedRows.slice(0, limit) : sortedRows;

  function handleSort(column: SortColumn) {
    if (sortColumn === column) setSortDirection((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortColumn(column);
      setSortDirection(column === "rank" || column === "priority" ? "asc" : "desc");
    }
  }

  function SortIcon({ column }: { column: SortColumn }) {
    if (column !== sortColumn) return <ArrowUpDown className="h-3 w-3 text-teal-300" />;
    return sortDirection === "asc" ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />;
  }

  return (
    <div className="space-y-4">
      {!limit && (
        <div className="flex flex-wrap items-end justify-end gap-3">
          <div className="relative min-w-[220px] max-w-xs flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-teal-300" />
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search transformer, station, fault…" className="h-10 w-full rounded-lg border border-teal-200 bg-white pl-9 pr-3 text-sm text-teal-900 outline-none focus:border-teal-500" />
          </div>
          <select value={station} onChange={(e) => setStation(e.target.value)} className="h-10 rounded-lg border border-teal-200 bg-white px-3 text-sm text-teal-800">
            {stations.map((s) => <option key={s}>{s}</option>)}
          </select>
          <select value={faultFilter} onChange={(e) => setFaultFilter(e.target.value)} className="h-10 rounded-lg border border-teal-200 bg-white px-3 text-sm text-teal-800">
            {faultTypes.map((ft) => <option key={ft}>{ft === "All" ? "All Faults" : ft}</option>)}
          </select>
          <select value={priorityFilter} onChange={(e) => setPriorityFilter(e.target.value)} className="h-10 rounded-lg border border-teal-200 bg-white px-3 text-sm text-teal-800">
            <option value="All">All Priorities</option>
            <option value="CRITICAL">Critical</option>
            <option value="HIGH_RISK">High Risk</option>
            <option value="WATCH">Watch</option>
            <option value="NORMAL">Normal</option>
            <option value="DATA_REVIEW">Data Review</option>
          </select>
          <span className="text-xs text-teal-500">({filteredRows.length}/{allRows.length})</span>
        </div>
      )}

      <div className="card-surface overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[1200px] text-left text-sm">
            <thead>
              <tr className="border-b border-cream-300 text-xs font-semibold uppercase tracking-wide text-teal-400">
                {[
                  ["rank", "#"], ["id", "Transformer"], ["station", "Station"], ["priority", "Priority"], ["status", "IEEE Status"], ["fault", "Fault"], ["evidence", "Status-3 evidence"], ["date", "Last Test"],
                ].map(([key, label]) => (
                  <th key={key} className="px-4 py-3">
                    <button type="button" onClick={() => handleSort(key as SortColumn)} className="inline-flex items-center gap-1 hover:text-teal-700">
                      {label}<SortIcon column={key as SortColumn} />
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {displayRows.map(({ summary, row }, i) => {
                const ieeeStatus = statusFromSummary(summary);
                const isCritical = summary.maintenance_priority === "CRITICAL" || summary.critical_front === true;
                return (
                  <tr key={summary.transformer_id} className={`border-b border-cream-200 last:border-0 hover:bg-cream-50 ${isCritical ? "bg-red-50/40" : ""}`}>
                    <td className="px-4 py-3 font-mono font-semibold text-teal-700">{summary.maintenance_rank ?? summary.rank}</td>
                    <td className="px-4 py-3">
                      <Link href={`/transformer/${encodeURIComponent(summary.transformer_id)}`} className="font-bold text-teal-900 hover:text-copper-600 hover:underline">
                        {summary.transformer_id}
                      </Link>
                    </td>
                    <td className="px-4 py-3 text-teal-600">{stationOf(summary)}</td>
                    <td className="px-4 py-3"><PriorityBadge priority={summary.maintenance_priority} /></td>
                    <td className="px-4 py-3"><StatusBadge status={ieeeStatus} /></td>
                    <td className="px-4 py-3">
                      <div className="font-medium text-teal-800">{summary.fault_type || "—"}</div>
                      {summary.fault_criticality_class && <div className="mt-0.5 text-[10px] uppercase tracking-wide text-teal-400">{summary.fault_criticality_class}</div>}
                    </td>
                    <td className="px-4 py-3 font-mono text-teal-700">
                      {summary.current_status3_standardized_exceedance != null ? `${formatNumber(summary.current_status3_standardized_exceedance, 2)}×` : "—"}
                    </td>
                    <td className="px-4 py-3 text-teal-600">{formatDate(summary.latest_sample_day)}</td>
                  </tr>
                );
              })}
              {displayRows.length === 0 && <tr><td colSpan={8} className="px-4 py-10 text-center text-sm text-teal-400">No transformers match the current filters.</td></tr>}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export function RiskLegend() {
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-teal-500">
      <span className="font-semibold text-teal-700">IEEE DGA Status:</span>
      <LegendItem colorClass="bg-status-normal" label="Status 1 · Normal" />
      <LegendItem colorClass="bg-status-watch" label="Status 2 · Watch" />
      <LegendItem colorClass="bg-status-high" label="Status 3 · High" />
      <LegendItem colorClass="bg-red-700" label="Critical queue" />
    </div>
  );
}

function LegendItem({ colorClass, label }: { colorClass: string; label: string }) {
  return <span className="flex items-center gap-1.5"><span className={`h-2 w-2 rounded-full ${colorClass}`} />{label}</span>;
}
