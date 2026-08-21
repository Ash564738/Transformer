// src/components/detail/ranking-breakdown-panel.tsx
import {
  Minus,
  TrendingDown,
  TrendingUp,
} from "lucide-react";

import type { RankingBreakdown } from "@/types/dga";
import { formatNumber } from "@/lib/utils";

export function RankingBreakdownPanel({
  breakdown,
}: {
  breakdown: RankingBreakdown;
}) {
  const slope = breakdown.trend_slope ?? 0;

  const TrendIcon =
    slope > 0
      ? TrendingUp
      : slope < 0
        ? TrendingDown
        : Minus;

  const trendLabel =
    slope > 0
      ? "Worsening"
      : slope < 0
        ? "Improving"
        : "Stable";

  return (
    <div className="rounded-xl border border-cream-300 bg-white p-3">
      <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-teal-400">
        Maintenance ranking evidence
      </div>

      <div className="space-y-1.5">
        <Row
          label="Current IEEE status"
          value={`S${breakdown.current_status}`}
        />

        {breakdown.maintenance_priority && (
          <Row
            label="Maintenance priority"
            value={breakdown.maintenance_priority}
          />
        )}

        {breakdown.maintenance_rank != null && (
          <Row
            label="Maintenance rank"
            value={`#${breakdown.maintenance_rank}`}
          />
        )}

        {breakdown.current_status3_standardized_exceedance !=
          null && (
          <Row
            label="Status-3 standardized exceedance"
            value={`${formatNumber(
              breakdown.current_status3_standardized_exceedance,
              2
            )}×`}
          />
        )}

        {breakdown.current_delta_exceedance !=
          null && (
          <Row
            label="Current delta exceedance"
            value={String(
              breakdown.current_delta_exceedance
            )}
          />
        )}

        {breakdown.current_standard_trigger_count !=
          null && (
          <Row
            label="Standard trigger count"
            value={String(
              breakdown.current_standard_trigger_count
            )}
          />
        )}

        {breakdown.pareto_dominance_count !=
          null && (
          <Row
            label="Dominance count"
            value={String(
              breakdown.pareto_dominance_count
            )}
          />
        )}

        {breakdown.history_max_status_before_current !=
          null && (
          <Row
            label="Historical max status before current"
            value={`S${breakdown.history_max_status_before_current}`}
          />
        )}

        {breakdown.historical_max_standardized_exceedance !=
          null && (
          <Row
            label="Historical max exceedance"
            value={`${formatNumber(
              breakdown.historical_max_standardized_exceedance,
              2
            )}×`}
          />
        )}

        {breakdown.history_abnormal_record_ratio !=
          null && (
          <Row
            label="Historical abnormal-record ratio"
            value={`${formatNumber(
              breakdown.history_abnormal_record_ratio * 100,
              1
            )}%`}
          />
        )}

        {breakdown.history_critical_record_ratio !=
          null && (
          <Row
            label="Historical Status-3 ratio"
            value={`${formatNumber(
              breakdown.history_critical_record_ratio * 100,
              1
            )}%`}
          />
        )}

        {breakdown.history_fault_recurrence_rate !=
          null && (
          <Row
            label="Fault recurrence rate"
            value={`${formatNumber(
              breakdown.history_fault_recurrence_rate * 100,
              1
            )}%`}
          />
        )}

        {breakdown.trend_slope !== undefined && (
          <div className="flex items-center justify-between text-xs">
            <span className="flex items-center gap-1 text-teal-600">
              <TrendIcon className="h-3 w-3" />

              Trend

              <span className="text-teal-400">
                ({trendLabel})
              </span>
            </span>

            <span className="font-mono font-semibold text-teal-900">
              {breakdown.trend_slope !== null
                ? formatNumber(
                    breakdown.trend_slope,
                    3
                  )
                : "—"}
            </span>
          </div>
        )}
      </div>

      <div className="mt-2 border-t border-cream-200 pt-2 text-xs text-teal-500">
        IEEE status and maintenance priority are separate
        concepts, but the current backend maps Status 3 to
        HIGH_RISK. The ranking itself uses explicit
        lexicographic evidence and no weighted health score.
      </div>
    </div>
  );
}

function Row({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center justify-between gap-3 text-xs">
      <span className="text-teal-600">
        {label}
      </span>

      <span className="text-right font-mono font-semibold text-teal-900">
        {value}
      </span>
    </div>
  );
}