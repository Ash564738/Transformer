"use client";

import { useMemo, useState } from "react";
import { Search } from "lucide-react";
import { useDashboardStore } from "@/store/use-dashboard-store";
import { EmptyState } from "@/components/layout/empty-state";
import { TransformerCard } from "@/components/fleet/transformer-card";
import { STATUS_ORDER, scoreToStatus } from "@/lib/severity";
import { getStations, stationOf } from "@/lib/transformer-helpers";
import { cn } from "@/lib/utils";
import type { RiskStatus } from "@/types/dga";

export default function FleetPage() {
  const payload = useDashboardStore((s) => s.payload);
  const [query, setQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<RiskStatus | "All">("All");

  const stations = useMemo(() => (payload ? getStations(payload) : []), [payload]);

  const grouped = useMemo(() => {
    if (!payload) return [];
    let list = payload.transformer_summary;
    if (query.trim()) {
      const q = query.trim().toLowerCase();
      list = list.filter((s) => s.transformer_id.toLowerCase().includes(q));
    }
    if (statusFilter !== "All") {
      list = list.filter((s) => scoreToStatus(s.latest_score) === statusFilter);
    }
    const map = new Map<string, typeof list>();
    for (const s of list) {
      const station = stationOf(s);
      if (!map.has(station)) map.set(station, []);
      map.get(station)!.push(s);
    }
    return stations
      .filter((st) => map.has(st))
      .map((st) => ({ station: st, items: map.get(st)!.sort((a, b) => b.latest_score - a.latest_score) }));
  }, [payload, query, statusFilter, stations]);

  if (!payload) {
    return (
      <EmptyState
        title="No fleet data yet"
        subtitle="Upload a DGA dataset to browse the transformer directory."
      />
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-extrabold tracking-tight text-teal-900">Fleet Directory</h1>
        <p className="mt-1 text-sm text-teal-500">
          {payload.dataset_summary.total_transformers} transformers across {stations.length} station
          {stations.length === 1 ? "" : "s"}.
        </p>
      </div>

      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="relative sm:max-w-xs">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-teal-300" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search transformer code…"
            className="h-10 w-full rounded-lg border border-teal-200 bg-white pl-9 pr-3 text-sm text-teal-900 outline-none focus:border-teal-500"
          />
        </div>
        <div className="flex flex-wrap gap-1.5">
          <FilterChip active={statusFilter === "All"} onClick={() => setStatusFilter("All")} label="All" />
          {STATUS_ORDER.map((status) => (
            <FilterChip
              key={status}
              active={statusFilter === status}
              onClick={() => setStatusFilter(status)}
              label={status === "High" ? "High Risk" : status}
            />
          ))}
        </div>
      </div>

      {grouped.length === 0 && (
        <p className="rounded-xl border border-dashed border-teal-200 bg-white py-10 text-center text-sm text-teal-400">
          No transformers match the current filters.
        </p>
      )}

      <div className="space-y-8">
        {grouped.map(({ station, items }) => (
          <section key={station}>
            <div className="mb-3 flex items-baseline gap-2">
              <h2 className="text-sm font-bold uppercase tracking-wide text-teal-600">{station}</h2>
              <span className="text-xs text-teal-400">{items.length} unit{items.length === 1 ? "" : "s"}</span>
            </div>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {items.map((s) => (
                <TransformerCard key={s.transformer_id} summary={s} />
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}

function FilterChip({ active, onClick, label }: { active: boolean; onClick: () => void; label: string }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "rounded-full border px-3 py-1.5 text-xs font-semibold transition-colors cursor-pointer",
        active ? "border-teal-800 bg-teal-800 text-white" : "border-teal-200 bg-white text-teal-600 hover:bg-teal-50"
      )}
    >
      {label}
    </button>
  );
}
