"use client";

import { useDashboardStore } from "@/store/use-dashboard-store";
import { EmptyState } from "@/components/layout/empty-state";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { SeverityDistributionChart } from "@/components/analytics/severity-distribution-chart";
import { FaultDistributionChart } from "@/components/analytics/fault-distribution-chart";
import { TopTrendChart } from "@/components/analytics/top-trend-chart";
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

  const avgScore =
    payload.transformer_summary.reduce((sum, s) => sum + s.latest_score, 0) /
    Math.max(1, payload.transformer_summary.length);
  const avgConfidence =
    payload.rows.reduce((sum, r) => sum + (Number(r.diagnostic_confidence) || 0), 0) /
    Math.max(1, payload.rows.length);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-extrabold tracking-tight text-teal-900">Overview</h1>
        <p className="mt-1 text-sm text-teal-500">Fleet-wide trends across severity, fault type, and time.</p>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:gap-4 md:grid-cols-4">
        <MetricCard label="Transformers" value={payload.dataset_summary.total_transformers} />
        <MetricCard label="Total samples" value={payload.dataset_summary.total_rows} />
        <MetricCard label="Avg. risk score" value={formatNumber(avgScore, 2)} />
        <MetricCard label="Avg. diagnostic confidence" value={`${formatNumber(avgConfidence, 0)}%`} />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Trend of Top 5 Highest-Risk Transformers</CardTitle>
          </CardHeader>
          <CardContent>
            <TopTrendChart payload={payload} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Severity Distribution</CardTitle>
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
            <FaultDistributionChart summaries={payload.transformer_summary} />
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