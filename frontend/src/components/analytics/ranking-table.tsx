// src/components/analytics/ranking-table.tsx
"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { Search, ArrowUpDown, ArrowUp, ArrowDown } from "lucide-react";

import type { DgaPayload, DgaRow, MaintenancePriority } from "@/types/dga";
import { StatusBadge } from "@/components/ui/badge";
import {
  normalizeMaintenancePriority,
  maintenancePriorityLabel,
  statusFromSummary,
  MAINTENANCE_PRIORITY_STYLES,
} from "@/lib/severity";
import { getStations, stationOf } from "@/lib/transformer-helpers";
import { formatDate, formatNumber } from "@/lib/utils";

type SortColumn = "rank" | "id" | "station" | "status" | "fault" | "evidence" | "topGas" | "date";
type SortDirection = "asc" | "desc";

type GasMetric = { label: string; value: number };

const GAS_FIELDS: Array<{ key: keyof DgaRow; label: string }> = [
  { key: "h2", label: "H2" },
  { key: "ch4", label: "CH4" },
  { key: "c2h6", label: "C2H6" },
  { key: "c2h4", label: "C2H4" },
  { key: "c2h2", label: "C2H2" },
  { key: "co", label: "CO" },
  { key: "co2", label: "CO2" },
];

function PriorityBadge({ priority }: { priority: MaintenancePriority | string }) {
  const normalized = normalizeMaintenancePriority(priority);
  const style = MAINTENANCE_PRIORITY_STYLES[normalized];

  return (
    <span className={`inline-flex rounded-full border px-2 py-0.5 text-[11px] font-semibold ${style.bg} ${style.border} ${style.text}`}>
      {maintenancePriorityLabel(normalized)}
    </span>
  );
}

function latestRowMap(rows: DgaRow[]) {
  const map = new Map<string, DgaRow>();

  for (const row of rows) {
    const current = map.get(row.transformer_id);
    if (!current || new Date(row.sample_day).getTime() > new Date(current.sample_day).getTime()) {
      map.set(row.transformer_id, row);
    }
  }

  return map;
}

function topGas(row: DgaRow | undefined): GasMetric | null {
  if (!row) return null;

  const candidates = GAS_FIELDS
    .map(({ key, label }) => ({ label, value: Number(row[key]) }))
    .filter(({ value }) => Number.isFinite(value));

  if (candidates.length === 0) return null;

  return candidates.reduce((best, item) => (item.value > best.value ? item : best));
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
    const types = new Set(
      payload.transformer_summary
        .map((summary) => summary.fault_type)
        .filter(Boolean)
    );
    return ["All", ...Array.from(types).sort()];
  }, [payload]);

  const latestRows = useMemo(() => latestRowMap(payload.rows), [payload.rows]);
  const allRows = payload.transformer_summary;

  const filteredRows = useMemo(() => {
    let list = allRows;
    const q = query.trim().toLowerCase();

    if (q) {
      list = list.filter((summary) => [
        summary.transformer_id,
        stationOf(summary),
        summary.fault_type,
        summary.fault_group,
        summary.maintenance_priority,
        summary.recommended_action,
      ].join(" ").toLowerCase().includes(q));
    }

    if (station !== "All Stations") {
      list = list.filter((summary) => stationOf(summary) === station);
    }

    if (faultFilter !== "All") {
      list = list.filter((summary) => summary.fault_type === faultFilter);
    }

    if (priorityFilter !== "All") {
      list = list.filter((summary) => normalizeMaintenancePriority(summary.maintenance_priority) === priorityFilter);
    }

    return list;
  }, [allRows, query, station, faultFilter, priorityFilter]);

  const sortedRows = useMemo(() => {
    const list = [...filteredRows];
    const dir = sortDirection === "asc" ? 1 : -1;
    const statusOrder = ["Insufficient data", "Normal", "Watch", "High"];

    return list.sort((a, b) => {
      switch (sortColumn) {
        case "rank":
          return (((a.maintenance_rank ?? a.rank) - (b.maintenance_rank ?? b.rank)) * dir);
        case "id":
          return a.transformer_id.localeCompare(b.transformer_id) * dir;
        case "station":
          return stationOf(a).localeCompare(stationOf(b)) * dir;
        case "status":
          return ((statusOrder.indexOf(statusFromSummary(a)) - statusOrder.indexOf(statusFromSummary(b))) * dir);
        case "fault":
          return ((a.fault_type || "").localeCompare(b.fault_type || "") * dir);
        case "evidence":
          return ((Number(a.current_status3_standardized_exceedance ?? 0) - Number(b.current_status3_standardized_exceedance ?? 0)) * dir);
        case "topGas":
          return ((Number(topGas(latestRows.get(a.transformer_id))?.value ?? -Infinity) - Number(topGas(latestRows.get(b.transformer_id))?.value ?? -Infinity)) * dir);
        case "date":
          return ((new Date(a.latest_sample_day).getTime() - new Date(b.latest_sample_day).getTime()) * dir);
        default:
          return 0;
      }
    });
  }, [filteredRows, sortColumn, sortDirection, latestRows]);

  const displayRows = limit ? sortedRows.slice(0, limit) : sortedRows;

  function handleSort(column: SortColumn) {
    if (sortColumn === column) {
      setSortDirection((direction) => direction === "asc" ? "desc" : "asc");
      return;
    }
    setSortColumn(column);
    setSortDirection(column === "rank" || column === "status" ? "asc" : "desc");
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
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search transformer, station, fault…"
              className="h-10 w-full rounded-lg border border-teal-200 bg-white pl-9 pr-3 text-sm text-teal-900 outline-none focus:border-teal-500"
            />
          </div>

          <select value={station} onChange={(event) => setStation(event.target.value)} className="h-10 rounded-lg border border-teal-200 bg-white px-3 text-sm text-teal-800">
            {stations.map((item) => <option key={item}>{item}</option>)}
          </select>

          <select value={faultFilter} onChange={(event) => setFaultFilter(event.target.value)} className="h-10 rounded-lg border border-teal-200 bg-white px-3 text-sm text-teal-800">
            {faultTypes.map((fault) => <option key={fault} value={fault}>{fault === "All" ? "All Faults" : fault}</option>)}
          </select>

          <select value={priorityFilter} onChange={(event) => setPriorityFilter(event.target.value)} className="h-10 rounded-lg border border-teal-200 bg-white px-3 text-sm text-teal-800">
            <option value="All">All Priorities</option>
            <option value="HIGH_RISK">High Risk</option>
            <option value="WATCH">Watch</option>
            <option value="NORMAL">Normal</option>
            <option value="DATA_REVIEW">Data Review</option>
          </select>

          <span className="text-xs text-teal-500">({filteredRows.length}/{allRows.length})</span>
        </div>
      )}

      <div className="rounded-lg border border-teal-100 bg-teal-50 px-3 py-2 text-[11px] leading-5 text-teal-700">
        Ranking is a relative fleet ordering based on explicit lexicographic evidence. It is not a weighted health score. Top Gas shows the highest measured combustible gas at the latest sample.
      </div>

      <div className="card-surface overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[1280px] text-left text-sm">
            <thead>
              <tr className="border-b border-cream-300 text-xs font-semibold uppercase tracking-wide text-teal-400">
                {[
                  ["rank", "#"],
                  ["id", "Transformer"],
                  ["station", "Station"],
                  ["status", "IEEE Status"],
                  ["fault", "Fault"],
                  ["evidence", "S3 evidence"],
                  ["topGas", "Top Gas"],
                  ["date", "Last Test"],
                ].map(([key, label]) => (
                  <th key={key} className="px-4 py-3">
                    <button type="button" onClick={() => handleSort(key as SortColumn)} className="inline-flex items-center gap-1 hover:text-teal-700">
                      {label}
                      <SortIcon column={key as SortColumn} />
                    </button>
                  </th>
                ))}
              </tr>
            </thead>

            <tbody>
              {displayRows.map((summary) => {
                const ieeeStatus = statusFromSummary(summary);
                const isHighRisk = normalizeMaintenancePriority(summary.maintenance_priority) === "HIGH_RISK";
                const gas = topGas(latestRows.get(summary.transformer_id));

                return (
                  <tr key={summary.transformer_id} className={`border-b border-cream-200 last:border-0 hover:bg-cream-50 ${isHighRisk ? "bg-orange-50/40" : ""}`}>
                    <td className="px-4 py-3 font-mono font-semibold text-teal-700">{summary.maintenance_rank ?? summary.rank}</td>
                    <td className="px-4 py-3">
                      <Link href={`/transformer/${encodeURIComponent(summary.transformer_id)}`} className="font-bold text-teal-900 hover:text-copper-600 hover:underline">
                        {summary.transformer_id}
                      </Link>
                    </td>
                    <td className="px-4 py-3 text-teal-600">{stationOf(summary)}</td>
                    <td className="px-4 py-3"><StatusBadge status={ieeeStatus} /></td>
                    <td className="px-4 py-3">
                      <div className="font-medium text-teal-800">{summary.fault_type || "—"}</div>
                      {summary.fault_group && <div className="mt-0.5 text-[10px] text-teal-400">{summary.fault_group}</div>}
                    </td>
                    <td className="px-4 py-3 font-mono text-teal-700">
                      {summary.current_status3_standardized_exceedance != null
                        ? `${formatNumber(summary.current_status3_standardized_exceedance, 2)}×`
                        : "—"}
                    </td>
                    <td className="px-4 py-3">
                      {gas ? (
                        <div>
                          <div className="font-mono font-semibold text-teal-800">{gas.label}</div>
                          <div className="text-[10px] text-teal-400">{formatNumber(gas.value, 1)} ppm</div>
                        </div>
                      ) : <span className="text-teal-400">—</span>}
                    </td>
                    <td className="px-4 py-3 text-teal-600">{formatDate(summary.latest_sample_day)}</td>
                  </tr>
                );
              })}

              {displayRows.length === 0 && (
                <tr><td colSpan={8} className="px-4 py-10 text-center text-sm text-teal-400">No transformers match the current filters.</td></tr>
              )}
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
      <LegendItem colorClass="bg-status-high" label="Status 3 · High / maintenance HIGH_RISK" />
      <LegendItem colorClass="bg-slate-400" label="Insufficient data" />
    </div>
  );
}

function LegendItem({ colorClass, label }: { colorClass: string; label: string }) {
  return <span className="flex items-center gap-1.5"><span className={`h-2 w-2 rounded-full ${colorClass}`} />{label}</span>;
}