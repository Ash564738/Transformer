"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { useDashboardStore } from "@/store/use-dashboard-store";
import { EmptyState } from "@/components/layout/empty-state";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { DetailHeader } from "@/components/detail/detail-header";
import { GasTrendChart } from "@/components/detail/gas-trend-chart";
import { SeverityTrendChart } from "@/components/detail/severity-trend-chart";
import { DiagnosticSwitcher } from "@/components/detail/diagnostic-switcher";
import { WhyFlagged } from "@/components/detail/why-flagged";
import { RankingBreakdownPanel } from "@/components/detail/ranking-breakdown-panel";
import { TechnicalNotes } from "@/components/detail/technical-notes";
import { GasIndicatorCards } from "@/components/detail/gas-indicator-cards";
import { HistoryTable } from "@/components/detail/history-table";
import { ConfirmInspectionDialog } from "@/components/detail/confirm-inspection-dialog";
import { latestRowFor, rowsForTransformer } from "@/lib/transformer-helpers";
import { CheckCircle2 } from "lucide-react";

export default function TransformerDetailPage() {
  const params = useParams<{ id: string }>();
  const transformerId = decodeURIComponent(params.id);
  const payload = useDashboardStore((s) => s.payload);
  const setSelectedTransformer = useDashboardStore((s) => s.setSelectedTransformer);
  const [confirmOpen, setConfirmOpen] = useState(false);

  useEffect(() => {
    setSelectedTransformer(transformerId);
    return () => setSelectedTransformer(null);
  }, [transformerId, setSelectedTransformer]);

  if (!payload) {
    return (
      <EmptyState
        title="No data loaded"
        subtitle="Upload a DGA dataset to view transformer details."
      />
    );
  }

  const summary = payload.transformer_summary.find((s) => s.transformer_id === transformerId);
  const row = latestRowFor(payload, transformerId);
  const rows = rowsForTransformer(payload, transformerId);

  if (!summary || !row) {
    return (
      <EmptyState
        title={`Transformer "${transformerId}" not found`}
        subtitle="It may not exist in the currently loaded dataset."
      />
    );
  }

  return (
    <div className="space-y-6 print:space-y-4">
      <DetailHeader summary={summary} />

      <GasIndicatorCards row={row} />

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Severity Trend Over Time</CardTitle>
          </CardHeader>
          <CardContent>
            <SeverityTrendChart rows={rows} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Gas Trend Over Time</CardTitle>
          </CardHeader>
          <CardContent>
            <GasTrendChart rows={rows} />
          </CardContent>
        </Card>

        <Card className="print:hidden">
          <CardHeader>
            <CardTitle>Traditional Diagnostic Methods</CardTitle>
          </CardHeader>
          <CardContent>
            <DiagnosticSwitcher row={row} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Why is this transformer flagged?</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <WhyFlagged row={row} />
            {summary.ranking_breakdown && <RankingBreakdownPanel breakdown={summary.ranking_breakdown} />}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Technical Notes</CardTitle>
          </CardHeader>
          <CardContent>
            <TechnicalNotes rows={rows} />
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Sample &amp; Fault History ({rows.length} record{rows.length === 1 ? "" : "s"})</CardTitle>
        </CardHeader>
        <CardContent>
          <HistoryTable rows={rows} />
        </CardContent>
      </Card>

      <div className="no-print flex justify-end">
        <Button onClick={() => setConfirmOpen(true)}>
          <CheckCircle2 className="h-4 w-4" /> Confirm Field Inspection Result
        </Button>
      </div>

      <ConfirmInspectionDialog
        transformerId={transformerId}
        suggestedFaultType={row.consensus_fault ?? ""}
        open={confirmOpen}
        onClose={() => setConfirmOpen(false)}
      />
    </div>
  );
}
