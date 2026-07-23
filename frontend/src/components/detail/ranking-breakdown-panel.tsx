import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import type { RankingBreakdown } from "@/types/dga";
import { formatNumber } from "@/lib/utils";

/** Distinct from the per-record "Severity score breakdown" in WhyFlagged:
 * that one explains a single sample. This explains the transformer's
 * OVERALL fleet-ranking score (backend/ranking.py final_score) — current
 * condition weighted most heavily, but blended with EWM-weighted history,
 * trend direction, and how often this specific unit has hit Critical
 * before, so a transformer with one bad reading doesn't outrank one with a
 * long history of repeated faults. */
export function RankingBreakdownPanel({ breakdown }: { breakdown: RankingBreakdown }) {
  const {
    weights,
    current_severity,
    current_contribution,
    historical_severity,
    historical_contribution,
    trend_bonus,
    trend_contribution,
    critical_history_count,
    critical_history_contribution,
    diagnostic_confidence,
    confidence_contribution,
    persistence_bonus_factor,
    fault_persistence,
    days_since_last_critical,
  } = breakdown;

  const baseScore =
    current_contribution + historical_contribution + trend_contribution + critical_history_contribution + confidence_contribution;
  const persistenceMultiplier = 1 + persistence_bonus_factor * fault_persistence;
  const finalScore = baseScore * persistenceMultiplier;

  const TrendIcon = trend_bonus > 0 ? TrendingUp : trend_bonus < 0 ? TrendingDown : Minus;

  const rows = [
    {
      label: "Current severity",
      hint: `latest sample, ×${weights.current}`,
      raw: current_severity,
      weighted: current_contribution,
    },
    {
      label: "Historical severity",
      hint: `recency-weighted average, ×${weights.history}`,
      raw: historical_severity,
      weighted: historical_contribution,
    },
    {
      label: "Trend",
      hint: `${trend_bonus > 0 ? "worsening" : trend_bonus < 0 ? "improving" : "stable"}, ×${weights.trend}`,
      raw: trend_bonus,
      weighted: trend_contribution,
      icon: TrendIcon,
    },
    {
      label: "Critical-event history",
      hint: `${critical_history_count} past critical sample(s), ×${weights.critical_history}`,
      raw: Math.min(critical_history_count, 3),
      weighted: critical_history_contribution,
    },
    {
      label: "Diagnostic confidence",
      hint: `${formatNumber(diagnostic_confidence, 0)}% confidence, ×${weights.confidence}`,
      raw: diagnostic_confidence >= 50 ? 1 : 0,
      weighted: confidence_contribution,
    },
  ];

  return (
    <div className="rounded-xl border border-cream-300 bg-white p-3">
      <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-teal-400">
        Fleet ranking score breakdown
      </div>
      <div className="space-y-1.5">
        {rows.map((r) => {
          const Icon = r.icon;
          return (
            <div key={r.label} className="flex items-center justify-between text-xs">
              <span className="flex items-center gap-1 text-teal-600">
                {Icon && <Icon className="h-3 w-3" />}
                {r.label} <span className="text-teal-400">({r.hint})</span>
              </span>
              <span className="font-mono font-semibold text-teal-900">
                {formatNumber(r.raw, 2)} → {formatNumber(r.weighted, 2)}
              </span>
            </div>
          );
        })}
      </div>
      <div className="mt-2 flex items-center justify-between border-t border-cream-200 pt-2 text-xs">
        <span className="text-teal-600">
          Fault persistence bonus <span className="text-teal-400">(×{formatNumber(persistence_bonus_factor, 2)})</span>
        </span>
        <span className="font-mono font-semibold text-teal-900">
          {formatNumber(fault_persistence, 2)} → ×{formatNumber(persistenceMultiplier, 3)}
        </span>
      </div>
      <div className="mt-2 flex items-center justify-between border-t border-cream-200 pt-2 text-sm">
        <span className="font-semibold text-teal-800">Overall ranking score</span>
        <span className="font-mono font-extrabold text-teal-900">{formatNumber(finalScore, 2)}</span>
      </div>
      {days_since_last_critical !== null && (
        <p className="mt-2 text-[11px] text-teal-400">
          Last hit Critical {formatNumber(days_since_last_critical, 0)} day(s) ago.
        </p>
      )}
    </div>
  );
}
