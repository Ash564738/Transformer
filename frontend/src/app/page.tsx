// src/app/page.tsx
"use client";

import { useDashboardStore } from "@/store/use-dashboard-store";
import { EmptyState } from "@/components/layout/empty-state";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { SeverityDistributionChart } from "@/components/overview/severity-distribution-chart";
import { FaultDistributionChart } from "@/components/overview/fault-distribution-chart";
import { TopTrendChart } from "@/components/overview/top-trend-chart";
import { formatNumber } from "@/lib/utils";
import { statusFromSummary } from "@/lib/severity";

export default function OverviewPage() {
  const payload = useDashboardStore((s) => s.payload);

  if (!payload) {
    return (
      <EmptyState
        title="No data to analyze yet"
        subtitle="Upload a DGA dataset to see fleet-wide analytics."
      />
    );
  }

  const confidenceValues = payload.rows
    .map((row) => Number(row.diagnostic_confidence))
    .filter((value) => Number.isFinite(value));

  const avgConfidence =
    confidenceValues.length > 0
      ? confidenceValues.reduce((sum, value) => sum + value, 0) /
        confidenceValues.length
      : null;

  const status3Count = payload.transformer_summary.filter(
    (summary) => statusFromSummary(summary) === "High"
  ).length;

  const firstPriorityId =
    payload.dataset_summary.first_priority_transformer_id ??
    payload.transformer_summary[0]?.transformer_id ??
    null;

  const firstPriorityRank =
    payload.dataset_summary.first_priority_rank ??
    payload.transformer_summary[0]?.maintenance_rank ??
    payload.transformer_summary[0]?.rank ??
    null;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-extrabold tracking-tight text-teal-900">
          Overview
        </h1>
        <p className="mt-1 text-sm text-teal-500">
          Fleet-wide IEEE DGA condition status, continuous gas evidence,
          diagnostic inference, and relative maintenance ordering.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:gap-4 md:grid-cols-5">
        <MetricCard label="Transformers" value={payload.dataset_summary.total_transformers} />
        <MetricCard label="Total samples" value={payload.dataset_summary.total_rows} />
        <MetricCard label="IEEE Status 3" value={status3Count} />
        <MetricCard label="First fleet rank" value={firstPriorityRank == null ? "—" : `#${firstPriorityRank}`} />
        <MetricCard
          label="Avg. diagnostic confidence"
          value={avgConfidence == null ? "—" : `${formatNumber(avgConfidence, 0)}%`}
        />
      </div>

      <div className="rounded-xl border border-teal-100 bg-teal-50 px-4 py-3 text-xs leading-5 text-teal-700">
        IEEE Status and fleet maintenance ordering are separate. Status 3 is a
        discrete IEEE condition class. Maintenance rank is an unweighted
        lexicographic evidence order; continuous DGA evidence is threshold
        evidence, not a health score.
        {firstPriorityId && (
          <span className="ml-1 font-semibold">
            Current first-ranked transformer: {firstPriorityId}.
          </span>
        )}
      </div>

      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle>Fleet DGA Trend</CardTitle>
          </CardHeader>
          <CardContent>
            <TopTrendChart payload={payload} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>IEEE DGA Status Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <SeverityDistributionChart summaries={payload.transformer_summary} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Fault Type Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <FaultDistributionChart summaries={payload.transformer_summary} rows={payload.rows} />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="card-surface px-5 py-4">
      <div className="text-xs font-semibold uppercase tracking-wide text-teal-400">{label}</div>
      <div className="mt-1 text-2xl font-extrabold text-teal-900">{value}</div>
    </div>
  );
}