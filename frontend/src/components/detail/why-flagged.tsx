// src/components/detail/why-flagged.tsx
"use client";

import { AlertTriangle, CheckCircle2, Info } from "lucide-react";
import type { DgaRow } from "@/types/dga";
import { formatNumber } from "@/lib/utils";

function asNumber(value: unknown): number | null {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function asString(value: unknown): string {
  return value == null ? "" : String(value);
}

function asArray(value: unknown): string[] {
  if (Array.isArray(value)) return value.map(String).filter(Boolean);
  if (typeof value !== "string" || !value.trim()) return [];
  const text = value.trim();
  try {
    const parsed = JSON.parse(text);
    if (Array.isArray(parsed)) return parsed.map(String).filter(Boolean);
  } catch {
    // Fall through to comma-separated parsing.
  }
  return text.split(",").map((item) => item.trim()).filter(Boolean);
}

function ieeeStatusLabel(status?: number | null): string {
  switch (Number(status)) {
    case 1: return "Status 1";
    case 2: return "Status 2";
    case 3: return "Status 3";
    default: return "Insufficient data";
  }
}

function faultExplanation(fault?: string): string {
  switch (String(fault ?? "").trim().toUpperCase()) {
    case "T1": return "Low-temperature thermal fault.";
    case "T2": return "Medium-temperature thermal fault.";
    case "T3": return "High-temperature thermal fault.";
    case "T3_H": return "High-temperature thermal fault, oil-dominant interpretation.";
    case "D1": return "Low-energy electrical discharge.";
    case "D2": return "High-energy electrical discharge / arcing.";
    case "PD": return "Partial discharge.";
    case "DT": return "Combined discharge and thermal fault.";
    case "C": return "Cellulose carbonization.";
    case "S": return "Stray gassing.";
    case "NORMAL": return "No significant fault indicated.";
    case "ABSTAIN": return "No sufficiently supported fine-grained fault diagnosis.";
    default: return "Fault diagnosis returned by the backend.";
  }
}

function diagnosticMethods(row: DgaRow) {
  const methods: Array<{ key: keyof DgaRow; label: string }> = [
    { key: "keygas_fault", label: "Key Gas" },
    { key: "iec_fault", label: "IEC 60599" },
    { key: "rogers_fault", label: "Rogers" },
    { key: "doernenburg_fault", label: "Doernenburg" },
    { key: "duval_triangle_fault", label: "Duval Triangle" },
    { key: "duval_pentagon_p1_fault", label: "Duval Pentagon 1" },
    { key: "duval_pentagon_p2_fault", label: "Duval Pentagon 2" },
  ];

  return methods.map((method) => {
    const value = asString(row[method.key]);
    const active = value.length > 0 && value.toUpperCase() !== "ABSTAIN" && value !== "-1";
    return { label: method.label, value: active ? value : "ABSTAIN", active };
  });
}

function MethodAgreement({ row }: { row: DgaRow }) {
  const methods = diagnosticMethods(row);
  const active = methods.filter((method) => method.active);
  const counts = new Map<string, number>();
  for (const method of active) {
    const normalized = method.value.trim().toUpperCase();
    counts.set(normalized, (counts.get(normalized) ?? 0) + 1);
  }
  const sorted = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
  const dominant = sorted[0]?.[0] ?? "ABSTAIN";
  const dominantCount = sorted[0]?.[1] ?? 0;

  return (
    <div className="rounded-xl border border-cream-300 bg-white p-4">
      <div className="mb-3">
        <div className="text-[11px] font-semibold uppercase tracking-wide text-teal-400">Traditional diagnostic evidence</div>
        <div className="mt-1 text-sm text-teal-700">{active.length} of {methods.length} diagnostic methods returned an active diagnosis.</div>
      </div>

      <div className="grid gap-2 sm:grid-cols-2">
        {methods.map((method) => (
          <div key={method.label} className="flex items-center justify-between rounded-lg border border-cream-200 bg-cream-50 px-3 py-2">
            <span className="text-xs text-teal-600">{method.label}</span>
            <span className={`font-mono text-xs font-semibold ${method.active ? "text-teal-900" : "text-teal-300"}`}>{method.value}</span>
          </div>
        ))}
      </div>

      {dominant !== "ABSTAIN" && (
        <div className="mt-3 flex items-start gap-2 rounded-lg border border-teal-200 bg-teal-50 px-3 py-2">
          <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-teal-700" />
          <div className="text-xs leading-5 text-teal-700">
            Dominant traditional diagnosis: <span className="font-bold text-teal-900">{dominant}</span> ({dominantCount}/{active.length} active methods).
          </div>
        </div>
      )}
    </div>
  );
}

function StandardEvidence({ row }: { row: DgaRow }) {
  const items = [
    ["Table 1 · concentration screening", asArray(row.ieee_table1_exceeding_gases)],
    ["Table 2 · 95th percentile concentration", asArray(row.ieee_table2_exceeding_gases)],
    ["Table 3 · 95th percentile delta", asArray(row.ieee_table3_exceeding_gases)],
    ["Table 4 · 95th percentile rate", asArray(row.ieee_table4_exceeding_gases)],
  ] as const;

  const status = asNumber(row.ieee_dga_status);
  const continuousRatio = asNumber(row.ieee_continuous_evidence_ratio);
  const continuousBasis = String(row.ieee_continuous_evidence_basis ?? "");

  return (
    <div className="rounded-xl border border-cream-300 bg-white p-4">
      <div className="mb-3">
        <div className="text-[11px] font-semibold uppercase tracking-wide text-teal-400">IEEE C57.104-2019 evidence</div>
        <div className="mt-1 flex flex-wrap items-center gap-2">
          <span className="rounded-full bg-teal-50 px-2 py-0.5 text-xs font-semibold text-teal-800">{ieeeStatusLabel(status)}</span>
          {continuousRatio != null && (
            <span className="rounded-full bg-orange-50 px-2 py-0.5 text-xs font-semibold text-orange-800">
              latest evidence {formatNumber(continuousRatio, 2)}×
            </span>
          )}
        </div>
        {continuousRatio != null && (
          <div className="mt-2 text-[11px] text-teal-500">
            {continuousBasis || "IEEE threshold ratio"}. This ratio belongs to the current sample only.
          </div>
        )}
      </div>

      <div className="space-y-2">
        {items.map(([label, gases]) => {
          const active = gases.length > 0;
          return (
            <div key={label} className={`flex items-start justify-between gap-3 rounded-lg border px-3 py-2 ${active ? "border-red-200 bg-red-50/60" : "border-cream-200 bg-cream-50"}`}>
              <div className="flex items-start gap-2">
                {active ? <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-red-700" /> : <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-teal-500" />}
                <div>
                  <div className="text-xs font-medium text-teal-800">{label}</div>
                  {active && <div className="mt-0.5 text-[11px] text-teal-500">Exceeded gases: {gases.join(", ")}</div>}
                </div>
              </div>
              <span className={`whitespace-nowrap font-mono text-xs font-semibold ${active ? "text-red-800" : "text-teal-600"}`}>
                {active ? `${gases.length} gas${gases.length === 1 ? "" : "es"}` : "Not exceeded"}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function GasEvidence({ row }: { row: DgaRow }) {
  const gases = [
    ["H₂", row.h2],
    ["CH₄", row.ch4],
    ["C₂H₆", row.c2h6],
    ["C₂H₄", row.c2h4],
    ["C₂H₂", row.c2h2],
    ["CO", row.co],
    ["CO₂", row.co2],
  ] as const;
  const tdcg = asNumber(row.tdcg) ?? gases.reduce((sum, [, value]) => sum + (asNumber(value) ?? 0), 0);

  return (
    <div className="rounded-xl border border-cream-300 bg-white p-4">
      <div className="mb-3">
        <div className="text-[11px] font-semibold uppercase tracking-wide text-teal-400">Current DGA observations</div>
        <div className="mt-1 text-xs text-teal-500">Raw gas concentrations. No frontend severity transformation is applied.</div>
      </div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {gases.map(([label, value]) => (
          <div key={label} className="rounded-lg border border-cream-200 bg-cream-50 px-3 py-2">
            <div className="text-[10px] font-semibold uppercase tracking-wide text-teal-400">{label}</div>
            <div className="mt-1 font-mono text-sm font-bold text-teal-900">{asNumber(value) == null ? "—" : formatNumber(asNumber(value), 1)}</div>
            <div className="text-[10px] text-teal-400">ppm</div>
          </div>
        ))}
        <div className="rounded-lg border border-teal-200 bg-teal-50 px-3 py-2">
          <div className="text-[10px] font-semibold uppercase tracking-wide text-teal-500">TDCG</div>
          <div className="mt-1 font-mono text-sm font-bold text-teal-900">{formatNumber(tdcg, 1)}</div>
          <div className="text-[10px] text-teal-500">ppm</div>
        </div>
      </div>
    </div>
  );
}

export function WhyFlagged({ row }: { row: DgaRow }) {
  const status = asNumber(row.ieee_dga_status);
  const fault = asString(row.final_fault ?? row.consensus_fault ?? "ABSTAIN").toUpperCase();
  const ratio = asNumber(row.ieee_continuous_evidence_ratio);
  const ratioBasis = String(row.ieee_continuous_evidence_basis ?? "IEEE threshold ratio");
  const confidence = asNumber(row.diagnostic_confidence);

  const title =
    status === 3
      ? "Why this transformer is IEEE Status 3"
      : status === 2
        ? "Why this transformer is IEEE Status 2"
        : status === 1
          ? "Why this transformer is IEEE Status 1"
          : "Why the DGA result is incomplete";

  const paragraph =
    status === 3
      ? ratio != null
        ? `The latest sample is IEEE ${ieeeStatusLabel(status)}. Its strongest continuous DGA evidence is ${formatNumber(ratio, 2)}× of the applicable reference threshold (${ratioBasis}). This value is sample-specific; it is not the severity of the entire transformer history.`
        : "The latest sample is IEEE Status 3 based on the backend rule-engine evidence."
      : status === 2
        ? "The latest sample is IEEE Status 2 based on intermediate concentration and/or change evidence."
        : status === 1
          ? "The latest sample is IEEE Status 1 under the implemented IEEE screening rules."
          : "The available gas inputs are insufficient for a complete IEEE screening result.";

  return (
    <div className="space-y-4">
      <div className={`rounded-xl border p-4 ${status === 3 ? "border-red-200 bg-red-50/60" : "border-cream-300 bg-white"}`}>
        <div className="flex items-start gap-3">
          {status === 3 ? <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-red-700" /> : <Info className="mt-0.5 h-5 w-5 shrink-0 text-teal-600" />}
          <div>
            <div className={`text-sm font-bold ${status === 3 ? "text-red-900" : "text-teal-900"}`}>{title}</div>
            <p className={`mt-2 text-sm leading-6 ${status === 3 ? "text-red-900/80" : "text-teal-700"}`}>{paragraph}</p>
            {fault !== "ABSTAIN" && (
              <p className="mt-2 text-sm leading-6 text-teal-700">
                Diagnosed fault: <span className="font-semibold">{fault}</span>. {faultExplanation(fault)}
              </p>
            )}
          </div>
        </div>
      </div>

      <StandardEvidence row={row} />
      <MethodAgreement row={row} />
      <GasEvidence row={row} />

      <div className="grid gap-3 sm:grid-cols-2">
        <div className="rounded-xl border border-cream-300 bg-white px-4 py-3">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-teal-500">Diagnostic confidence</div>
          <div className="mt-1 text-lg font-extrabold text-teal-900">
            {confidence == null ? "Not reported" : `${formatNumber(confidence, 0)}%`}
          </div>
          <div className="mt-1 text-[10px] leading-4 text-teal-400">Fault-diagnosis confidence only; not IEEE severity.</div>
        </div>

        <div className="rounded-xl border border-cream-300 bg-white px-4 py-3">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-teal-500">Continuous evidence</div>
          <div className="mt-1 text-lg font-extrabold text-teal-900">
            {ratio == null ? "Not reported" : `${formatNumber(ratio, 2)}×`}
          </div>
          <div className="mt-1 text-[10px] leading-4 text-teal-400">Per-sample IEEE threshold evidence; not a weighted score.</div>
        </div>
      </div>
    </div>
  );
}