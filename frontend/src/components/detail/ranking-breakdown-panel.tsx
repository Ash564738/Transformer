// src/components/detail/ranking-breakdown-panel.tsx
import { Minus, TrendingDown, TrendingUp } from "lucide-react";
import type { RankingBreakdown } from "@/types/dga";
import { formatNumber } from "@/lib/utils";

export function RankingBreakdownPanel({ breakdown }: { breakdown: RankingBreakdown }) {
  const statusSlope = breakdown.trend_slope ?? null;
  const evidenceSlope = breakdown.evidence_trend_slope ?? null;
  const slope = evidenceSlope ?? statusSlope;

  const TrendIcon =
    slope == null || slope === 0 ? Minus : slope > 0 ? TrendingUp : TrendingDown;
  const trendLabel =
    slope == null || slope === 0 ? "Stable / insufficient trend" : slope > 0 ? "Worsening" : "Improving";

  return (
    <div className="rounded-xl border border-cream-300 bg-white p-3">
      <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-teal-400">
        Maintenance ranking evidence
      </div>

      <div className="space-y-1.5">
        <Row label="Current IEEE status" value={`S${breakdown.current_status}`} />
        {breakdown.maintenance_rank != null && <Row label="Fleet rank" value={`#${breakdown.maintenance_rank}`} />}
        {breakdown.current_status3_standardized_exceedance != null && (
          <Row label="Latest Status-3 evidence" value={`${formatNumber(breakdown.current_status3_standardized_exceedance, 2)}×`} />
        )}
        {breakdown.current_continuous_evidence_ratio != null && (
          <Row label="Latest continuous DGA evidence" value={`${formatNumber(breakdown.current_continuous_evidence_ratio, 2)}×`} />
        )}
        {breakdown.current_continuous_evidence_basis && (
          <Row label="Evidence basis" value={breakdown.current_continuous_evidence_basis} />
        )}
        {breakdown.current_standard_trigger_count != null && (
          <Row label="IEEE trigger tables" value={String(breakdown.current_standard_trigger_count)} />
        )}
        {breakdown.history_max_status_before_current != null && (
          <Row label="Historical max status" value={`S${breakdown.history_max_status_before_current}`} />
        )}
        {breakdown.historical_max_continuous_evidence_ratio != null && (
          <Row label="Historical max continuous evidence" value={`${formatNumber(breakdown.historical_max_continuous_evidence_ratio, 2)}×`} />
        )}
        {breakdown.historical_max_standardized_exceedance != null && (
          <Row label="Historical max standardized evidence" value={`${formatNumber(breakdown.historical_max_standardized_exceedance, 2)}×`} />
        )}
        {breakdown.history_abnormal_record_ratio != null && (
          <Row label="Historical abnormal-record ratio" value={`${formatNumber(breakdown.history_abnormal_record_ratio * 100, 1)}%`} />
        )}
        {breakdown.history_critical_record_ratio != null && (
          <Row label="Historical Status-3 ratio" value={`${formatNumber(breakdown.history_critical_record_ratio * 100, 1)}%`} />
        )}
        {breakdown.history_fault_recurrence_rate != null && (
          <Row label="Fault recurrence rate" value={`${formatNumber(breakdown.history_fault_recurrence_rate * 100, 1)}%`} />
        )}

        <div className="flex items-center justify-between text-xs">
          <span className="flex items-center gap-1 text-teal-600">
            <TrendIcon className="h-3 w-3" />
            DGA trend
            <span className="text-teal-400">({trendLabel})</span>
          </span>
          <span className="font-mono font-semibold text-teal-900">
            {slope == null ? "—" : formatNumber(slope, 4)}
          </span>
        </div>
      </div>

      <div className="mt-2 border-t border-cream-200 pt-2 text-xs leading-5 text-teal-500">
        IEEE condition status is discrete. Fleet rank is a separate unweighted
        lexicographic ordering. Continuous DGA evidence is shown as threshold
        evidence and is not converted into a weighted health score.
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 text-xs">
      <span className="text-teal-600">{label}</span>
      <span className="max-w-[60%] text-right font-mono font-semibold text-teal-900">{value}</span>
    </div>
  );
}