import { Minus, TrendingDown, TrendingUp } from "lucide-react";
import type { RankingBreakdown } from "@/types/dga";
import { formatNumber } from "@/lib/utils";

export function RankingBreakdownPanel({ breakdown }: { breakdown: RankingBreakdown }) {
  const { final_score, severity_score, trend_slope } = breakdown;

  const slope = trend_slope ?? 0;
  const TrendIcon = slope > 0 ? TrendingUp : slope < 0 ? TrendingDown : Minus;
  const trendLabel = slope > 0 ? "Worsening" : slope < 0 ? "Improving" : "Stable";

  return (
    <div className="rounded-xl border border-cream-300 bg-white p-3">
      <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-teal-400">
        Fleet ranking score breakdown
      </div>
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-xs">
          <span className="text-teal-600">Current severity score</span>
          <span className="font-mono font-semibold text-teal-900">
            {formatNumber(severity_score, 2)}
          </span>
        </div>
        <div className="flex items-center justify-between text-xs">
          <span className="flex items-center gap-1 text-teal-600">
            <TrendIcon className="h-3 w-3" /> Trend slope
            <span className="text-teal-400">({trendLabel})</span>
          </span>
          <span className="font-mono font-semibold text-teal-900">
            {trend_slope !== null ? formatNumber(trend_slope, 3) : "—"}
          </span>
        </div>
      </div>
      <div className="mt-2 flex items-center justify-between border-t border-cream-200 pt-2 text-sm">
        <span className="font-semibold text-teal-800">Overall ranking score</span>
        <span className="font-mono font-extrabold text-teal-900">
          {formatNumber(final_score, 2)}
        </span>
      </div>
    </div>
  );
}