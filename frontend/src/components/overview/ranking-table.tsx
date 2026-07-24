"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { Search, ArrowUpDown, ArrowUp, ArrowDown } from "lucide-react";
import type { DgaPayload } from "@/types/dga";
import { StatusBadge } from "@/components/ui/badge";
import { scoreToRisk, scoreToStatus, STATUS_STYLES } from "@/lib/severity";
import { getStations, latestRowFor, stationOf, topGasLabel } from "@/lib/transformer-helpers";
import { formatDate } from "@/lib/utils";

type SortColumn = "id" | "station" | "score" | "status" | "date" | "gas";
type SortDirection = "asc" | "desc";

interface RowData {
  summary: DgaPayload["transformer_summary"][number];
  row: any;
}

export function RankingTable({ payload, limit }: { payload: DgaPayload; limit?: number }) {
  const [query, setQuery] = useState("");
  const [station, setStation] = useState("All Stations");
  const [sortColumn, setSortColumn] = useState<SortColumn>("score");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");

  const stations = useMemo(() => ["All Stations", ...getStations(payload)], [payload]);

  const allRows = useMemo<RowData[]>(() => {
    return payload.transformer_summary.map((s) => ({
      summary: s,
      row: latestRowFor(payload, s.transformer_id),
    }));
  }, [payload]);

  const filteredRows = useMemo(() => {
    let list = allRows;
    if (query.trim()) {
      const q = query.trim().toLowerCase();
      list = list.filter((r) => r.summary.transformer_id.toLowerCase().includes(q));
    }
    if (station !== "All Stations") {
      list = list.filter((r) => stationOf(r.summary) === station);
    }
    return list;
  }, [allRows, query, station]);

  const sortedRows = useMemo(() => {
    const list = [...filteredRows];
    const dir = sortDirection === "asc" ? 1 : -1;

    const compare = (a: RowData, b: RowData) => {
      switch (sortColumn) {
        case "id":
          return a.summary.transformer_id.localeCompare(b.summary.transformer_id) * dir;
        case "station":
          return (stationOf(a.summary) || "").localeCompare(stationOf(b.summary) || "") * dir;
        case "score":
          return (a.summary.latest_score - b.summary.latest_score) * dir;
        case "status": {
          const statusOrder: Record<string, number> = { NORMAL: 0, WATCH: 1, HIGH: 2, CRITICAL: 3 };
          const sa = scoreToStatus(a.summary.latest_score);
          const sb = scoreToStatus(b.summary.latest_score);
          return ((statusOrder[sa] ?? 0) - (statusOrder[sb] ?? 0)) * dir;
        }
        case "date":
          return (new Date(a.summary.latest_sample_day).getTime() - new Date(b.summary.latest_sample_day).getTime()) * dir;
        case "gas":
          return (topGasLabel(a.row, scoreToStatus(a.summary.latest_score)) || "").localeCompare(
            topGasLabel(b.row, scoreToStatus(b.summary.latest_score)) || ""
          ) * dir;
        default:
          return 0;
      }
    };

    return list.sort(compare);
  }, [filteredRows, sortColumn, sortDirection]);

  const displayRows = limit ? sortedRows.slice(0, limit) : sortedRows;

  const handleSort = (column: SortColumn) => {
    if (sortColumn === column) {
      setSortDirection((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortColumn(column);
      setSortDirection("asc");
    }
  };

  const renderSortIcon = (column: SortColumn) => {
    if (sortColumn !== column) {
      return <ArrowUpDown className="h-3 w-3 text-teal-300 group-hover:text-teal-500" />;
    }
    return sortDirection === "asc" ? (
      <ArrowUp className="h-3 w-3 text-teal-700" />
    ) : (
      <ArrowDown className="h-3 w-3 text-teal-700" />
    );
  };

  return (
    <div className="space-y-4">
      {/* Bộ lọc bên ngoài: Search + Station */}
      {!limit && (
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-end">
          <div className="relative flex-1 sm:max-w-xs">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-teal-300" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search by transformer code…"
              className="h-10 w-full rounded-lg border border-teal-200 bg-white pl-9 pr-3 text-sm text-teal-900 outline-none focus:border-teal-500"
            />
          </div>
          <select
            value={station}
            onChange={(e) => setStation(e.target.value)}
            className="h-10 rounded-lg border border-teal-200 bg-white px-3 text-sm text-teal-800 outline-none focus:border-teal-500"
          >
            {stations.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </div>
      )}

      <div className="card-surface overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px] text-left text-sm">
            <thead>
              <tr className="border-b border-cream-300 text-xs font-semibold uppercase tracking-wide text-teal-400">
                <th className="px-4 py-3">#</th>
                <th className="px-4 py-3">
                  <button
                    onClick={() => handleSort("id")}
                    className="group inline-flex items-center gap-1 hover:text-teal-700 transition-colors"
                  >
                    Transformer Code
                    {renderSortIcon("id")}
                  </button>
                </th>
                <th className="px-4 py-3">
                  <button
                    onClick={() => handleSort("station")}
                    className="group inline-flex items-center gap-1 hover:text-teal-700 transition-colors"
                  >
                    Station
                    {renderSortIcon("station")}
                  </button>
                </th>
                <th className="px-4 py-3">
                  <button
                    onClick={() => handleSort("score")}
                    className="group inline-flex items-center gap-1 hover:text-teal-700 transition-colors"
                  >
                    Risk Score
                    {renderSortIcon("score")}
                  </button>
                </th>
                <th className="px-4 py-3">
                  <button
                    onClick={() => handleSort("status")}
                    className="group inline-flex items-center gap-1 hover:text-teal-700 transition-colors"
                  >
                    Status
                    {renderSortIcon("status")}
                  </button>
                </th>
                <th className="px-4 py-3">
                  <button
                    onClick={() => handleSort("date")}
                    className="group inline-flex items-center gap-1 hover:text-teal-700 transition-colors"
                  >
                    Last Test Date
                    {renderSortIcon("date")}
                  </button>
                </th>
                <th className="px-4 py-3">
                  <button
                    onClick={() => handleSort("gas")}
                    className="group inline-flex items-center gap-1 hover:text-teal-700 transition-colors"
                  >
                    Top Gas
                    {renderSortIcon("gas")}
                  </button>
                </th>
              </tr>
            </thead>
            <tbody>
              {displayRows.map(({ summary, row }, i) => {
                const status = scoreToStatus(summary.latest_score);
                const risk = scoreToRisk(summary.latest_score);
                const style = STATUS_STYLES[status];
                return (
                  <tr key={summary.transformer_id} className="border-b border-cream-200 last:border-0 hover:bg-cream-50">
                    <td className="px-4 py-3 text-teal-400">{i + 1}</td>
                    <td className="px-4 py-3">
                      <Link
                        href={`/transformer/${encodeURIComponent(summary.transformer_id)}`}
                        className="font-bold text-teal-900 hover:text-copper-600 hover:underline"
                      >
                        {summary.transformer_id}
                      </Link>
                    </td>
                    <td className="px-4 py-3 text-teal-600">{stationOf(summary)}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 w-24 overflow-hidden rounded-full bg-cream-200">
                          <div className={`h-full rounded-full ${style.bar}`} style={{ width: `${risk}%` }} />
                        </div>
                        <span className={`text-sm font-bold ${style.text}`}>{risk}</span>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <StatusBadge status={status} />
                    </td>
                    <td className="px-4 py-3 text-teal-600">{formatDate(summary.latest_sample_day)}</td>
                    <td className="px-4 py-3 text-teal-600">{topGasLabel(row, status)}</td>
                  </tr>
                );
              })}
              {displayRows.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-4 py-10 text-center text-sm text-teal-400">
                    No transformers match the current filters.
                  </td>
                </tr>
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
      <span className="font-semibold text-teal-700">Risk Legend:</span>
      <LegendItem colorClass="bg-status-normal" label="0–30 (Normal)" />
      <LegendItem colorClass="bg-status-watch" label="31–60 (Watch)" />
      <LegendItem colorClass="bg-status-high" label="61–89 (High)" />
      <LegendItem colorClass="bg-status-critical" label="90–100 (Critical)" />
    </div>
  );
}

function LegendItem({ colorClass, label }: { colorClass: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className={`h-2 w-2 rounded-full ${colorClass}`} />
      {label}
    </span>
  );
}