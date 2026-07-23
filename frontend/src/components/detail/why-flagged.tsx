import { HelpCircle } from "lucide-react";
import type { DgaRow } from "@/types/dga";
import { formatNumber } from "@/lib/utils";

function buildInterpretation(row: DgaRow): string {
  const parts: string[] = [];
  const c2h2 = Number(row.c2h2 ?? 0);
  const tcg = Number(row.tdcg ?? 0);
  const rocC2H2 = Number(row.c2h2_rate_per_day ?? 0);
  const rocTcg = Number(row.tdcg_rate_per_day ?? 0);
  const c2h2C2h4 = Number(row.iec_r1_c2h2_c2h4 ?? row.r2_c2h2_c2h4 ?? 0);
  const co2Co = Number(row.ratio_co2_co ?? 0);
  const co = Number(row.co ?? 0);

  if (c2h2 > 10) parts.push(`C₂H₂ elevated (${c2h2.toFixed(1)} ppm)`);
  if (c2h2C2h4 > 0.1) parts.push(`C₂H₂/C₂H₄ ratio high (${c2h2C2h4.toFixed(2)})`);
  if (rocC2H2 > 0.5) parts.push(`C₂H₂ rising fast (${rocC2H2.toFixed(2)} ppm/day)`);
  if (rocTcg > 5) parts.push(`TCG increasing rapidly (${rocTcg.toFixed(1)} ppm/day)`);
  if (tcg > 720) parts.push(`TCG very high (${tcg.toFixed(0)} ppm)`);
  if (co > 300 && co2Co < 7) parts.push(`CO elevated (${co.toFixed(0)} ppm) with low CO₂/CO (${co2Co.toFixed(1)})`);

  if (parts.length === 0) return "No strong diagnostic signal detected.";
  return `The rapid increase in ${parts.join(", ")} strongly suggests active degradation. Immediate physical inspection is recommended.`;
}

// Weights mirror backend/config.py SEVERITY_WEIGHTS — shown here only as
// labels; the actual component scores below (severity_gas_score, etc.) are
// the real values the backend computed and returned in the row, not
// recomputed on the frontend.
const SCORE_COMPONENTS: { key: keyof DgaRow; label: string; weight: number }[] = [
  { key: "severity_gas_score", label: "Gas concentration", weight: 1.0 },
  { key: "severity_trend_score", label: "Trend / rate-of-change", weight: 1.2 },
  { key: "severity_fault_score", label: "Consensus fault severity", weight: 1.0 },
  { key: "severity_aging_score", label: "Cellulose ageing (CO, CO₂/CO)", weight: 0.8 },
];

function ScoreBreakdown({ row }: { row: DgaRow }) {
  const hasBreakdown = SCORE_COMPONENTS.some((c) => row[c.key] !== undefined);
  if (!hasBreakdown) return null;

  return (
    <div className="rounded-xl border border-cream-300 bg-white p-3">
      <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-teal-400">
        Severity score breakdown
      </div>
      <div className="space-y-1.5">
        {SCORE_COMPONENTS.map((c) => {
          const raw = Number(row[c.key] ?? 0);
          const weighted = raw * c.weight;
          return (
            <div key={c.key} className="flex items-center justify-between text-xs">
              <span className="text-teal-600">
                {c.label} <span className="text-teal-400">(×{c.weight})</span>
              </span>
              <span className="font-mono font-semibold text-teal-900">
                {formatNumber(raw, 1)} → {formatNumber(weighted, 2)}
              </span>
            </div>
          );
        })}
      </div>
      <div className="mt-2 flex items-center justify-between border-t border-cream-200 pt-2 text-sm">
        <span className="font-semibold text-teal-800">Total severity score</span>
        <span className="font-mono font-extrabold text-teal-900">{formatNumber(Number(row.severity_score ?? 0), 2)}</span>
      </div>
    </div>
  );
}

export function WhyFlagged({ row }: { row: DgaRow }) {
  const interpretation = buildInterpretation(row);
  const c2h2Rate = Number(row.c2h2_rate_per_day ?? 0);
  const confidence = Number(row.diagnostic_confidence ?? 0);
  const confidenceLabel = confidence >= 75 ? "High" : confidence >= 50 ? "Moderate" : "Low";

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-2">
        <HelpCircle className="mt-0.5 h-4 w-4 shrink-0 text-copper-500" />
        <p className="text-sm leading-relaxed text-teal-700">{interpretation}</p>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-xl bg-status-critical-soft px-4 py-3">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-status-critical">
            C₂H₂ generation rate
          </div>
          <div className="mt-1 text-lg font-extrabold text-status-critical">
            {formatNumber(c2h2Rate, 2)} <span className="text-xs font-semibold">ppm/day</span>
          </div>
        </div>
        <div className="rounded-xl bg-cream-100 px-4 py-3">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-teal-500">Data confidence</div>
          <div className="mt-1 text-lg font-extrabold text-teal-900">
            {confidenceLabel} <span className="text-xs font-semibold text-teal-500">({confidence.toFixed(0)}%)</span>
          </div>
        </div>
      </div>
      <ScoreBreakdown row={row} />
    </div>
  );
}
