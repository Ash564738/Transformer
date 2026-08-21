// src/components/detail/why-flagged.tsx
"use client";

import { AlertTriangle, CheckCircle2, Info } from "lucide-react";

import type { DgaRow } from "@/types/dga";
import { formatNumber } from "@/lib/utils";

type EvidenceItem = {
  label: string;
  value: string;
  detail?: string;
  active: boolean;
};

function asNumber(value: unknown): number | null {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function asString(value: unknown): string {
  return value == null ? "" : String(value);
}

function asArray(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value
      .map((item) => String(item))
      .filter(Boolean);
  }

  if (typeof value === "string") {
    const text = value.trim();

    if (!text) {
      return [];
    }

    try {
      const parsed = JSON.parse(text);

      if (Array.isArray(parsed)) {
        return parsed.map(String).filter(Boolean);
      }
    } catch {
      // Ignore JSON parse failure and fall back to comma-separated text.
    }

    return text
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
  }

  return [];
}

function formatRatio(value: number | null): string {
  if (value == null || !Number.isFinite(value)) {
    return "—";
  }

  return `${formatNumber(value, 2)}×`;
}

function ieeeStatusLabel(status?: number): string {
  switch (Number(status)) {
    case 1:
      return "Status 1";
    case 2:
      return "Status 2";
    case 3:
      return "Status 3";
    default:
      return "Insufficient data";
  }
}

function faultExplanation(fault?: string): string {
  const value = String(fault ?? "").trim().toUpperCase();

  switch (value) {
    case "T1":
      return "Low-temperature thermal fault.";
    case "T2":
      return "Medium-temperature thermal fault.";
    case "T3":
      return "High-temperature thermal fault.";
    case "T3_H":
      return "High-temperature thermal fault, oil-dominant interpretation.";
    case "D1":
      return "Low-energy electrical discharge.";
    case "D2":
      return "High-energy electrical discharge / arcing.";
    case "PD":
      return "Partial discharge.";
    case "DT":
      return "Combined discharge and thermal fault.";
    case "C":
      return "Cellulose carbonization.";
    case "S":
      return "Stray gassing.";
    case "NORMAL":
      return "No significant fault indicated.";
    case "ABSTAIN":
      return "No sufficiently supported fine-grained fault diagnosis.";
    default:
      return "Fault diagnosis returned by the backend.";
  }
}

function buildWhyFlagged(row: DgaRow): {
  title: string;
  paragraphs: string[];
} {
  const status = asNumber(row.ieee_dga_status);
  const fault = asString(
    row.final_fault ??
      row.consensus_fault ??
      "ABSTAIN"
  ).toUpperCase();

  const status3Ratio = asNumber(
    row.ieee_max_status3_standardized_exceedance
  );

  const standardTriggerCount = asNumber(
    row.ieee_standard_trigger_count
  );

  const table1 = asArray(
    row.ieee_table1_exceeding_gases
  );

  const table2 = asArray(
    row.ieee_table2_exceeding_gases
  );

  const table3 = asArray(
    row.ieee_table3_exceeding_gases
  );

  const table4 = asArray(
    row.ieee_table4_exceeding_gases
  );

  const h2 = asNumber(row.h2);
  const ch4 = asNumber(row.ch4);
  const c2h6 = asNumber(row.c2h6);
  const c2h4 = asNumber(row.c2h4);
  const c2h2 = asNumber(row.c2h2);
  const co = asNumber(row.co);
  const co2 = asNumber(row.co2);

  const paragraphs: string[] = [];

  if (status === 3) {
    if (status3Ratio != null) {
      paragraphs.push(
        `The transformer is classified as IEEE ${ieeeStatusLabel(
          status
        )}. The strongest available Status-3 evidence is ${formatRatio(
          status3Ratio
        )} of the applicable standardized Status-3 reference limit.`
      );
    } else {
      paragraphs.push(
        "The transformer is classified as IEEE Status 3 because the backend detected Status-3 DGA evidence."
      );
    }
  } else if (status === 2) {
    paragraphs.push(
      "The transformer is classified as IEEE Status 2 because the backend detected intermediate or possible-gassing evidence."
    );
  } else if (status === 1) {
    paragraphs.push(
      "The transformer is currently classified as IEEE Status 1."
    );
  } else {
    paragraphs.push(
      "The available DGA inputs are insufficient for a complete IEEE screening result."
    );
  }

  if (table2.length > 0) {
    paragraphs.push(
      `Table 2 concentration evidence is exceeded for ${table2.join(
        ", "
      )}.`
    );
  }

  if (table4.length > 0) {
    paragraphs.push(
      `Table 4 rate evidence is exceeded for ${table4.join(
        ", "
      )}.`
    );
  }

  if (table1.length > 0 && table2.length === 0) {
    paragraphs.push(
      `Table 1 screening limits are exceeded for ${table1.join(
        ", "
      )}.`
    );
  }

  if (table3.length > 0) {
    paragraphs.push(
      `Table 3 delta evidence is exceeded for ${table3.join(
        ", "
      )}.`
    );
  }

  if (
    standardTriggerCount != null &&
    standardTriggerCount > 0
  ) {
    paragraphs.push(
      `${standardTriggerCount} standard-derived evidence categories are active in the current sample.`
    );
  }

  if (fault !== "ABSTAIN") {
    paragraphs.push(
      `The diagnosed fault is ${fault}. ${faultExplanation(
        fault
      )}`
    );
  }

  const gasObservations: string[] = [];

  if (ch4 != null && ch4 > 0) {
    gasObservations.push(`CH₄ ${formatNumber(ch4, 1)} ppm`);
  }

  if (c2h4 != null && c2h4 > 0) {
    gasObservations.push(`C₂H₄ ${formatNumber(c2h4, 1)} ppm`);
  }

  if (c2h2 != null && c2h2 > 0) {
    gasObservations.push(`C₂H₂ ${formatNumber(c2h2, 1)} ppm`);
  }

  if (co != null && co > 0) {
    gasObservations.push(`CO ${formatNumber(co, 1)} ppm`);
  }

  if (co2 != null && co2 > 0) {
    gasObservations.push(`CO₂ ${formatNumber(co2, 1)} ppm`);
  }

  if (gasObservations.length > 0) {
    paragraphs.push(
      `Current gas concentrations: ${gasObservations.join(
        ", "
      )}. These concentrations are displayed as direct DGA observations; they are not converted into a frontend severity score.`
    );
  }

  return {
    title:
      status === 3
        ? "Why this transformer is High risk"
        : status === 2
          ? "Why this transformer is under Watch"
          : status === 1
            ? "Why this transformer is currently Normal"
            : "Why the DGA result is incomplete",
    paragraphs,
  };
}

function diagnosticMethods(row: DgaRow): EvidenceItem[] {
  const methods: Array<{
    key: keyof DgaRow;
    label: string;
  }> = [
    {
      key: "keygas_fault",
      label: "Key Gas",
    },
    {
      key: "iec_fault",
      label: "IEC 60599",
    },
    {
      key: "rogers_fault",
      label: "Rogers",
    },
    {
      key: "doernenburg_fault",
      label: "Doernenburg",
    },
    {
      key: "duval_triangle_fault",
      label: "Duval Triangle",
    },
    {
      key: "duval_pentagon_p1_fault",
      label: "Duval Pentagon 1",
    },
    {
      key: "duval_pentagon_p2_fault",
      label: "Duval Pentagon 2",
    },
  ];

  return methods.map((method) => {
    const value = asString(row[method.key]);

    const active =
      value.length > 0 &&
      value.toUpperCase() !== "ABSTAIN" &&
      value !== "-1";

    return {
      label: method.label,
      value: active ? value : "ABSTAIN",
      active,
    };
  });
}

function MethodAgreement({
  row,
}: {
  row: DgaRow;
}) {
  const methods = diagnosticMethods(row);

  const active = methods.filter(
    (method) => method.active
  );

  const counts = new Map<string, number>();

  for (const method of active) {
    const normalized = method.value
      .trim()
      .toUpperCase();

    counts.set(
      normalized,
      (counts.get(normalized) ?? 0) + 1
    );
  }

  const sorted = Array.from(counts.entries()).sort(
    (a, b) => b[1] - a[1]
  );

  const dominant =
    sorted.length > 0 ? sorted[0][0] : "ABSTAIN";

  const dominantCount =
    sorted.length > 0 ? sorted[0][1] : 0;

  return (
    <div className="rounded-xl border border-cream-300 bg-white p-4">
      <div className="mb-3">
        <div className="text-[11px] font-semibold uppercase tracking-wide text-teal-400">
          Traditional diagnostic evidence
        </div>

        <div className="mt-1 text-sm text-teal-700">
          {active.length} of {methods.length} diagnostic
          methods returned an active diagnosis.
        </div>
      </div>

      <div className="grid gap-2 sm:grid-cols-2">
        {methods.map((method) => (
          <div
            key={method.label}
            className="flex items-center justify-between rounded-lg border border-cream-200 bg-cream-50 px-3 py-2"
          >
            <span className="text-xs text-teal-600">
              {method.label}
            </span>

            <span
              className={`font-mono text-xs font-semibold ${
                method.active
                  ? "text-teal-900"
                  : "text-teal-300"
              }`}
            >
              {method.value}
            </span>
          </div>
        ))}
      </div>

      {dominant !== "ABSTAIN" && (
        <div className="mt-3 flex items-start gap-2 rounded-lg border border-teal-200 bg-teal-50 px-3 py-2">
          <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-teal-700" />

          <div className="text-xs leading-5 text-teal-700">
            The dominant traditional diagnosis is{" "}
            <span className="font-bold text-teal-900">
              {dominant}
            </span>{" "}
            with{" "}
            <span className="font-bold text-teal-900">
              {dominantCount}
            </span>{" "}
            active method result
            {dominantCount === 1 ? "" : "s"}.
          </div>
        </div>
      )}
    </div>
  );
}

function StandardEvidence({
  row,
}: {
  row: DgaRow;
}) {
  const table1 = asArray(
    row.ieee_table1_exceeding_gases
  );

  const table2 = asArray(
    row.ieee_table2_exceeding_gases
  );

  const table3 = asArray(
    row.ieee_table3_exceeding_gases
  );

  const table4 = asArray(
    row.ieee_table4_exceeding_gases
  );

  const rows: EvidenceItem[] = [
    {
      label: "Table 1 · concentration screening",
      value:
        table1.length > 0
          ? `${table1.length} gas${table1.length === 1 ? "" : "es"}`
          : "Not exceeded",
      detail:
        table1.length > 0
          ? table1.join(", ")
          : undefined,
      active: table1.length > 0,
    },
    {
      label: "Table 2 · 95th percentile concentration",
      value:
        table2.length > 0
          ? `${table2.length} gas${table2.length === 1 ? "" : "es"}`
          : "Not exceeded",
      detail:
        table2.length > 0
          ? table2.join(", ")
          : undefined,
      active: table2.length > 0,
    },
    {
      label: "Table 3 · 95th percentile delta",
      value:
        table3.length > 0
          ? `${table3.length} gas${table3.length === 1 ? "" : "es"}`
          : "Not exceeded",
      detail:
        table3.length > 0
          ? table3.join(", ")
          : undefined,
      active: table3.length > 0,
    },
    {
      label: "Table 4 · 95th percentile rate",
      value:
        table4.length > 0
          ? `${table4.length} gas${table4.length === 1 ? "" : "es"}`
          : "Not exceeded",
      detail:
        table4.length > 0
          ? table4.join(", ")
          : undefined,
      active: table4.length > 0,
    },
  ];

  const status3Ratio = asNumber(
    row.ieee_max_status3_standardized_exceedance
  );

  return (
    <div className="rounded-xl border border-cream-300 bg-white p-4">
      <div className="mb-3">
        <div className="text-[11px] font-semibold uppercase tracking-wide text-teal-400">
          IEEE C57.104-2019 evidence
        </div>

        <div className="mt-1 flex flex-wrap items-center gap-2">
          <span className="rounded-full bg-teal-50 px-2 py-0.5 text-xs font-semibold text-teal-800">
            {ieeeStatusLabel(
              asNumber(row.ieee_dga_status) ?? 0
            )}
          </span>

          {status3Ratio != null && (
            <span className="rounded-full bg-red-50 px-2 py-0.5 text-xs font-semibold text-red-800">
              strongest Status-3 evidence:{" "}
              {formatRatio(status3Ratio)}
            </span>
          )}
        </div>
      </div>

      <div className="space-y-2">
        {rows.map((item) => (
          <div
            key={item.label}
            className={`flex items-start justify-between gap-3 rounded-lg border px-3 py-2 ${
              item.active
                ? "border-red-200 bg-red-50/60"
                : "border-cream-200 bg-cream-50"
            }`}
          >
            <div className="flex items-start gap-2">
              {item.active ? (
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-red-700" />
              ) : (
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-teal-500" />
              )}

              <div>
                <div className="text-xs font-medium text-teal-800">
                  {item.label}
                </div>

                {item.detail && (
                  <div className="mt-0.5 text-[11px] text-teal-500">
                    Exceeded gases: {item.detail}
                  </div>
                )}
              </div>
            </div>

            <span
              className={`whitespace-nowrap font-mono text-xs font-semibold ${
                item.active
                  ? "text-red-800"
                  : "text-teal-600"
              }`}
            >
              {item.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function GasEvidence({
  row,
}: {
  row: DgaRow;
}) {
  const gases = [
    {
      key: "h2",
      label: "H₂",
      value: asNumber(row.h2),
    },
    {
      key: "ch4",
      label: "CH₄",
      value: asNumber(row.ch4),
    },
    {
      key: "c2h6",
      label: "C₂H₆",
      value: asNumber(row.c2h6),
    },
    {
      key: "c2h4",
      label: "C₂H₄",
      value: asNumber(row.c2h4),
    },
    {
      key: "c2h2",
      label: "C₂H₂",
      value: asNumber(row.c2h2),
    },
    {
      key: "co",
      label: "CO",
      value: asNumber(row.co),
    },
    {
      key: "co2",
      label: "CO₂",
      value: asNumber(row.co2),
    },
  ];

  const tdcg =
    asNumber(row.tdcg) ??
    gases.reduce(
      (sum, gas) =>
        sum + (gas.value ?? 0),
      0
    );

  return (
    <div className="rounded-xl border border-cream-300 bg-white p-4">
      <div className="mb-3">
        <div className="text-[11px] font-semibold uppercase tracking-wide text-teal-400">
          Current DGA observations
        </div>

        <div className="mt-1 text-xs text-teal-500">
          Raw gas concentrations returned by the backend.
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {gases.map((gas) => (
          <div
            key={gas.key}
            className="rounded-lg border border-cream-200 bg-cream-50 px-3 py-2"
          >
            <div className="text-[10px] font-semibold uppercase tracking-wide text-teal-400">
              {gas.label}
            </div>

            <div className="mt-1 font-mono text-sm font-bold text-teal-900">
              {gas.value != null
                ? formatNumber(gas.value, 1)
                : "—"}
            </div>

            <div className="text-[10px] text-teal-400">
              ppm
            </div>
          </div>
        ))}

        <div className="rounded-lg border border-teal-200 bg-teal-50 px-3 py-2">
          <div className="text-[10px] font-semibold uppercase tracking-wide text-teal-500">
            TDCG
          </div>

          <div className="mt-1 font-mono text-sm font-bold text-teal-900">
            {formatNumber(tdcg, 1)}
          </div>

          <div className="text-[10px] text-teal-500">
            ppm
          </div>
        </div>
      </div>
    </div>
  );
}

function FaultSummary({
  row,
}: {
  row: DgaRow;
}) {
  const fault = asString(
    row.final_fault ??
      row.consensus_fault ??
      "ABSTAIN"
  ).toUpperCase();

  const group = asString(
    row.final_fault_group ??
      row.consensus_fault_group ??
      "ABSTAIN"
  ).toUpperCase();

  const criticality = asString(
    row.fault_criticality_class ??
      "UNKNOWN"
  ).toUpperCase();

  const source = asString(
    row.final_fault_source
  );

  return (
    <div className="rounded-xl border border-cream-300 bg-white p-4">
      <div className="mb-3">
        <div className="text-[11px] font-semibold uppercase tracking-wide text-teal-400">
          Fault interpretation
        </div>
      </div>

      <div className="grid gap-2 sm:grid-cols-3">
        <ValueBox
          label="Fault type"
          value={fault}
        />

        <ValueBox
          label="Fault group"
          value={group}
        />

        <ValueBox
          label="Fault criticality"
          value={criticality}
        />
      </div>

      <div className="mt-3 rounded-lg bg-cream-50 px-3 py-2 text-xs leading-5 text-teal-700">
        <span className="font-semibold text-teal-900">
          Interpretation:
        </span>{" "}
        {faultExplanation(fault)}
      </div>

      {source && (
        <div className="mt-2 flex items-start gap-2 text-[11px] text-teal-400">
          <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />

          <span>
            Backend fault source:{" "}
            <span className="font-mono font-semibold text-teal-600">
              {source}
            </span>
          </span>
        </div>
      )}
    </div>
  );
}

function ValueBox({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-lg border border-cream-200 bg-cream-50 px-3 py-2">
      <div className="text-[10px] font-semibold uppercase tracking-wide text-teal-400">
        {label}
      </div>

      <div className="mt-1 font-mono text-sm font-bold text-teal-900">
        {value || "—"}
      </div>
    </div>
  );
}

export function WhyFlagged({
  row,
}: {
  row: DgaRow;
}) {
  const explanation = buildWhyFlagged(row);

  const confidence = asNumber(
    row.diagnostic_confidence
  );

  const confidenceLabel =
    confidence == null
      ? "Not reported"
      : confidence >= 75
        ? "High"
        : confidence >= 50
          ? "Moderate"
          : "Low";

  const c2h2Rate =
    row.c2h2_rate_per_day != null
      ? asNumber(row.c2h2_rate_per_day)
      : null;

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-red-200 bg-red-50/60 p-4">
        <div className="flex items-start gap-3">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-red-700" />

          <div>
            <div className="text-sm font-bold text-red-900">
              {explanation.title}
            </div>

            <div className="mt-2 space-y-2">
              {explanation.paragraphs.map(
                (paragraph, index) => (
                  <p
                    key={index}
                    className="text-sm leading-6 text-red-900/80"
                  >
                    {paragraph}
                  </p>
                )
              )}
            </div>
          </div>
        </div>
      </div>

      <StandardEvidence row={row} />

      <FaultSummary row={row} />

      <MethodAgreement row={row} />

      <GasEvidence row={row} />

      <div className="grid gap-3 sm:grid-cols-2">
        <div className="rounded-xl border border-cream-300 bg-white px-4 py-3">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-teal-500">
            Data confidence
          </div>

          <div className="mt-1 text-lg font-extrabold text-teal-900">
            {confidenceLabel}

            {confidence != null && (
              <span className="ml-1 text-xs font-semibold text-teal-500">
                ({formatNumber(confidence, 0)}%)
              </span>
            )}
          </div>

          <div className="mt-1 text-[10px] leading-4 text-teal-400">
            This is diagnostic confidence reported by the
            backend. It is not an IEEE severity score.
          </div>
        </div>

        <div className="rounded-xl border border-cream-300 bg-white px-4 py-3">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-teal-500">
            C₂H₂ generation rate
          </div>

          <div className="mt-1 text-lg font-extrabold text-teal-900">
            {c2h2Rate != null
              ? `${formatNumber(c2h2Rate, 2)} ppm/day`
              : "Not reported"}
          </div>

          <div className="mt-1 text-[10px] leading-4 text-teal-400">
            The current backend payload does not expose a
            frontend-computed C₂H₂ daily rate unless this field
            is explicitly returned.
          </div>
        </div>
      </div>

      <div className="rounded-xl border border-cream-300 bg-cream-50 px-4 py-3 text-xs leading-5 text-teal-600">
        No weighted severity score is used here. IEEE DGA
        status, fault diagnosis, standardized exceedance
        evidence, and maintenance priority are separate
        evidence layers.
      </div>
    </div>
  );
}