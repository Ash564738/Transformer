"use client";

import { useDashboardStore } from "@/store/use-dashboard-store";
import { StatCards } from "@/components/overview/stat-cards";
import { RankingTable, RiskLegend } from "@/components/overview/ranking-table";
import { EmptyState } from "@/components/layout/empty-state";

export default function AnalyticsPage() {
  const payload = useDashboardStore((s) => s.payload);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-extrabold tracking-tight text-teal-900">Fleet Overview</h1>
        <p className="mt-1 text-sm text-teal-500">Real-time status of all monitored transformers.</p>
      </div>

      {!payload ? (
        <EmptyState
          title="No data loaded yet"
          subtitle="Upload a DGA dataset from the settings panel to explore the dashboard."
        />
      ) : (
        <>
          <StatCards summaries={payload.transformer_summary} />
          <RankingTable payload={payload} />
          <RiskLegend />
        </>
      )}
    </div>
  );
}
