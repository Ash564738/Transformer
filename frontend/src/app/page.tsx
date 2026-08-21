// src/app/page.tsx
"use client";

import { useDashboardStore } from "@/store/use-dashboard-store";
import { EmptyState } from "@/components/layout/empty-state";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { SeverityDistributionChart } from "@/components/overview/severity-distribution-chart";
import { FaultDistributionChart } from "@/components/overview/fault-distribution-chart";
import { TopTrendChart } from "@/components/overview/top-trend-chart";
import { formatNumber } from "@/lib/utils";

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

  const highRiskQueue =
    payload.dataset_summary.high_risk_transformer_count ??
    payload.dataset_summary.maintenance_priority_counts?.HIGH_RISK ??
    payload.transformer_summary.filter(
      (summary) =>
        String(summary.maintenance_priority).toUpperCase() === "HIGH_RISK"
    ).length;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-extrabold tracking-tight text-teal-900">
          Overview
        </h1>
        <p className="mt-1 text-sm text-teal-500">
          Fleet-wide view of IEEE DGA condition status, maintenance ranking,
          fault diagnostics, and temporal history.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:gap-4 md:grid-cols-4">
        <MetricCard
          label="Transformers"
          value={payload.dataset_summary.total_transformers}
        />
        <MetricCard
          label="Total samples"
          value={payload.dataset_summary.total_rows}
        />
        <MetricCard label="Maintenance high risk" value={highRiskQueue} />
        <MetricCard
          label="Avg. diagnostic confidence"
          value={
            avgConfidence == null
              ? "—"
              : `${formatNumber(avgConfidence, 0)}%`
          }
        />
      </div>

      <div className="rounded-xl border border-teal-100 bg-teal-50 px-4 py-3 text-xs leading-5 text-teal-700">
        IEEE Status and maintenance priority are separate axes. The ranking is
        produced by an explicit lexicographic evidence policy; the displayed
        priority percentile is not a weighted health score.
      </div>

      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle>IEEE Status Trend</CardTitle>
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
            <FaultDistributionChart
              summaries={payload.transformer_summary}
              rows={payload.rows}
            />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) {
  return (
    <div className="card-surface px-5 py-4">
      <div className="text-xs font-semibold uppercase tracking-wide text-teal-400">
        {label}
      </div>
      <div className="mt-1 text-2xl font-extrabold text-teal-900">
        {value}
      </div>
    </div>
  );
}