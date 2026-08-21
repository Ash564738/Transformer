// src/components/detail/technical-notes.tsx
"use client";

import type { DgaRow } from "@/types/dga";
import { formatDate, formatNumber } from "@/lib/utils";
import { ieeeStatusToRiskStatus } from "@/lib/severity";

function list(value: unknown): string[] {
  if (Array.isArray(value)) return value.map(String).filter(Boolean);
  if (typeof value === "string" && value.trim()) {
    try {
      const parsed = JSON.parse(value);
      if (Array.isArray(parsed)) return parsed.map(String).filter(Boolean);
    } catch {
      // Fall back to comma-separated text.
    }
    return value.split(",").map((v) => v.trim()).filter(Boolean);
  }
  return [];
}

function latestFirst(rows: DgaRow[]): DgaRow {
  return [...rows].sort(
    (a, b) => new Date(b.sample_day).getTime() - new Date(a.sample_day).getTime()
  )[0];
}

export function TechnicalNotes({ rows }: { rows: DgaRow[] }) {
  if (rows.length === 0) {
    return <p className="text-sm text-teal-400">No technical notes available.</p>;
  }

  const latest = latestFirst(rows);
  const status = ieeeStatusToRiskStatus(latest.ieee_dga_status);

  const table1 = list(latest.ieee_table1_exceeding_gases);
  const table2 = list(latest.ieee_table2_exceeding_gases);
  const table3 = list(latest.ieee_table3_exceeding_gases);
  const table4 = list(latest.ieee_table4_exceeding_gases);

  const evidenceRatio = Number(latest.ieee_continuous_evidence_ratio);
  const hasEvidenceRatio = Number.isFinite(evidenceRatio);
  const evidenceBasis = String(latest.ieee_continuous_evidence_basis ?? "");

  const rates = latest.ieee_gas_rate_ppm_per_year ?? {};
  const deltas = latest.ieee_delta ?? {};

  const activeRates = Object.entries(rates).filter(
    ([, value]) => typeof value === "number" && Number.isFinite(value) && value > 0
  );

  const activeDeltas = Object.entries(deltas).filter(
    ([, value]) => typeof value === "number" && Number.isFinite(value) && value !== 0
  );

  return (
    <div className="space-y-3">
      <div className="rounded-xl border border-cream-300 bg-white p-3">
        <div className="text-[11px] font-semibold uppercase tracking-wide text-teal-400">
          IEEE screening result
        </div>
        <div className="mt-1 text-sm font-bold text-teal-900">{status}</div>
        <div className="mt-1 text-xs text-teal-500">Latest sample: {formatDate(latest.sample_day)}</div>
      </div>

      {hasEvidenceRatio && (
        <div className="rounded-xl border border-orange-200 bg-orange-50 px-3 py-2">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-orange-800">
            Latest-sample continuous DGA evidence
          </div>
          <div className="mt-1 font-mono text-lg font-bold text-orange-900">
            {formatNumber(evidenceRatio, 2)}×
          </div>
          <div className="mt-1 text-[11px] leading-4 text-orange-800/80">
            Basis: {evidenceBasis || "IEEE threshold ratio"}. This is a per-sample evidence ratio, not a transformer-wide severity score.
          </div>
        </div>
      )}

      {latest.ieee_max_status3_standardized_exceedance != null && (
        <div className="rounded-xl border border-red-200 bg-red-50 px-3 py-2">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-red-700">
            Latest-sample strongest Status-3 evidence
          </div>
          <div className="mt-1 font-mono text-lg font-bold text-red-800">
            {formatNumber(latest.ieee_max_status3_standardized_exceedance, 2)}×
          </div>
        </div>
      )}

      <div className="rounded-xl border border-cream-300 bg-white p-3">
        <div className="text-[11px] font-semibold uppercase tracking-wide text-teal-400">
          IEEE evidence tables
        </div>
        <div className="mt-2 space-y-2 text-xs">
          <div>Table 1: <span className="font-semibold">{table1.length ? table1.join(", ") : "not exceeded"}</span></div>
          <div>Table 2: <span className="font-semibold">{table2.length ? table2.join(", ") : "not exceeded"}</span></div>
          <div>Table 3: <span className="font-semibold">{table3.length ? table3.join(", ") : "not exceeded"}</span></div>
          <div>Table 4: <span className="font-semibold">{table4.length ? table4.join(", ") : "not exceeded"}</span></div>
        </div>
      </div>

      {activeDeltas.length > 0 && (
        <div className="rounded-xl border border-cream-300 bg-white p-3">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-teal-400">
            Gas change vs previous sample
          </div>
          <div className="mt-2 space-y-1 text-xs">
            {activeDeltas.map(([gas, value]) => (
              <div key={gas} className="flex justify-between">
                <span>{gas.toUpperCase()}</span>
                <span className="font-mono font-semibold">{formatNumber(value, 1)} ppm</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {activeRates.length > 0 && (
        <div className="rounded-xl border border-cream-300 bg-white p-3">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-teal-400">Gas rate</div>
          <div className="mt-2 space-y-1 text-xs">
            {activeRates.map(([gas, value]) => (
              <div key={gas} className="flex justify-between">
                <span>{gas.toUpperCase()}</span>
                <span className="font-mono font-semibold">{formatNumber(value, 1)} ppm/year</span>
              </div>
            ))}
          </div>
          {latest.ieee_rate_span_months != null && (
            <div className="mt-2 text-[11px] text-teal-400">
              Rate window: {formatNumber(latest.ieee_rate_span_months, 1)} months
            </div>
          )}
        </div>
      )}

      <div className="rounded-xl border border-cream-300 bg-white p-3">
        <div className="text-[11px] font-semibold uppercase tracking-wide text-teal-400">IEEE context</div>
        <div className="mt-2 space-y-1 text-xs">
          <div className="flex justify-between">
            <span>O₂/N₂ ratio</span>
            <span className="font-mono font-semibold">{latest.ieee_o2_n2_ratio != null ? formatNumber(latest.ieee_o2_n2_ratio, 3) : "—"}</span>
          </div>
          <div className="flex justify-between">
            <span>O₂/N₂ section</span>
            <span className="font-mono font-semibold">{latest.ieee_o2_n2_section ?? "—"}</span>
          </div>
          <div className="flex justify-between">
            <span>Age bucket</span>
            <span className="font-mono font-semibold">{latest.ieee_age_bucket ?? "unknown"}</span>
          </div>
        </div>
      </div>
    </div>
  );
}