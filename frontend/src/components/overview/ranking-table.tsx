"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { Search } from "lucide-react";
import type { DgaPayload } from "@/types/dga";
import { StatusBadge } from "@/components/ui/badge";
import { scoreToRisk, scoreToStatus, STATUS_STYLES } from "@/lib/severity";
import { getStations, latestRowFor, stationOf, topGasLabel } from "@/lib/transformer-helpers";
import { formatDate } from "@/lib/utils";

type SortMode = "risk-desc" | "risk-asc" | "id-asc" | "recent";

export function RankingTable({ payload, limit }: { payload: DgaPayload; limit?: number }) {
  const [query, setQuery] = useState("");
  const [station, setStation] = useState("All Stations");
  const [sort, setSort] = useState<SortMode>("risk-desc");

  const stations = useMemo(() => ["All Stations", ...getStations(payload)], [payload]);

  const rows = useMemo(() => {
    let list = payload.transformer_summary.map((s) => ({
      summary: s,
      row: latestRowFor(payload, s.transformer_id),
    }));

    if (query.trim()) {
      const q = query.trim().toLowerCase();
      list = list.filter((r) => r.summary.transformer_id.toLowerCase().includes(q));
    }
    if (station !== "All Stations") {
      list = list.filter((r) => stationOf(r.summary) === station);
    }
    switch (sort) {
      case "risk-desc":
        list.sort((a, b) => b.summary.latest_score - a.summary.latest_score);
        break;
      case "risk-asc":
        list.sort((a, b) => a.summary.latest_score - b.summary.latest_score);
        break;
      case "id-asc":
        list.sort((a, b) => a.summary.transformer_id.localeCompare(b.summary.transformer_id));
        break;
      case "recent":
        list.sort(
          (a, b) => new Date(b.summary.latest_sample_day).getTime() - new Date(a.summary.latest_sample_day).getTime()
        );
        break;
    }
    return limit ? list.slice(0, limit) : list;
  }, [payload, query, station, sort, limit]);

  return (
    <div className="space-y-4">
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
          <select
            value={sort}
            onChange={(e) => setSort(e.target.value as SortMode)}
            className="h-10 rounded-lg border border-teal-200 bg-white px-3 text-sm text-teal-800 outline-none focus:border-teal-500"
          >
            <option value="risk-desc">Highest risk first</option>
            <option value="risk-asc">Lowest risk first</option>
            <option value="id-asc">Transformer code (A–Z)</option>
            <option value="recent">Most recently tested</option>
          </select>
        </div>
      )}

      <div className="card-surface overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px] text-left text-sm">
            <thead>
              <tr className="border-b border-cream-300 text-xs font-semibold uppercase tracking-wide text-teal-400">
                <th className="px-4 py-3 font-semibold">#</th>
                <th className="px-4 py-3 font-semibold">Transformer Code</th>
                <th className="px-4 py-3 font-semibold">Station</th>
                <th className="px-4 py-3 font-semibold">Risk Score</th>
                <th className="px-4 py-3 font-semibold">Status</th>
                <th className="px-4 py-3 font-semibold">Last Test Date</th>
                <th className="px-4 py-3 font-semibold">Top Gas</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(({ summary, row }, i) => {
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
              {rows.length === 0 && (
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
