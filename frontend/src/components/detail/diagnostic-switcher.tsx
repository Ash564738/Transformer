// src/components/detail/diagnostic-switcher.tsx
"use client";

import { useMemo, useState } from "react";

import type { DgaRow } from "@/types/dga";

import { FAULT_EXPLANATIONS } from "@/lib/dga-methods";
import { cn, formatDate, formatNumber } from "@/lib/utils";

import { DuvalTriangleSvg } from "../charts/duval-triangle";
import { DuvalPentagon1Svg } from "../charts/duval-pentagon1";
import { DuvalPentagon2Svg } from "../charts/duval-pentagon2";
import { Ratio3DChart } from "../charts/ratio3d-chart";
import { Iec3DChart } from "../charts/Iec3DChart";

type MethodKey =
  | "triangle"
  | "pentagon1"
  | "pentagon2"
  | "doernenburg"
  | "iec"
  | "rogers"
  | "keygas";

const METHODS: { key: MethodKey; label: string }[] = [
  { key: "triangle", label: "Duval Triangle 1" },
  { key: "pentagon1", label: "Duval Pentagon 1" },
  { key: "pentagon2", label: "Duval Pentagon 2" },
  { key: "doernenburg", label: "Doernenburg" },
  { key: "iec", label: "IEC 60599" },
  { key: "rogers", label: "Rogers Ratio" },
  { key: "keygas", label: "Key Gas" },
];

function toOptionalString(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }

  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function toOptionalNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  return undefined;
}

function resultLabel(code: unknown): string {
  const normalized = toOptionalString(code);

  if (!normalized) {
    return "ABSTAIN";
  }

  const explanation = FAULT_EXPLANATIONS[normalized];

  return explanation
    ? `${explanation.toUpperCase()} (${normalized})`
    : normalized;
}

function sampleKey(row: DgaRow, index: number): string {
  return `${row.sample_day}-${index}`;
}

export function DiagnosticSwitcher({ rows }: { rows: DgaRow[] }) {
  const orderedRows = useMemo(
    () =>
      rows
        .map((row, index) => ({ row, index }))
        .sort(
          (a, b) =>
            new Date(a.row.sample_day).getTime() -
              new Date(b.row.sample_day).getTime() ||
            a.index - b.index
        ),
    [rows]
  );

  const [selectedIndex, setSelectedIndex] = useState(
    orderedRows.length > 0 ? orderedRows.length - 1 : -1
  );
  const [active, setActive] = useState<MethodKey>("triangle");

  const selected =
    selectedIndex >= 0 ? orderedRows[selectedIndex]?.row : undefined;

  if (!selected) {
    return (
      <p className="py-8 text-center text-sm text-teal-400">
        No DGA samples available for diagnostic comparison.
      </p>
    );
  }

  const g = {
    h2: Number(selected.h2 ?? 0),
    ch4: Number(selected.ch4 ?? 0),
    c2h6: Number(selected.c2h6 ?? 0),
    c2h4: Number(selected.c2h4 ?? 0),
    c2h2: Number(selected.c2h2 ?? 0),
  };

  const triangleFault = toOptionalString(selected.duval_triangle_fault);
  const pentagon1Fault = toOptionalString(selected.fault_p1);
  const pentagon2Fault = toOptionalString(
    selected.fault_p2 ?? selected.duval_pentagon_fault
  );
  const doernenburgFault = toOptionalString(selected.doernenburg_fault);
  const iecFault = toOptionalString(selected.iec_fault);
  const rogersFault = toOptionalString(selected.rogers_fault);
  const keyGasFault = toOptionalString(selected.keygas_fault);

  return (
    <div>
      <div className="mb-4 flex flex-col gap-2 rounded-xl border border-cream-300 bg-white p-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wide text-teal-400">
            Diagnostic sample
          </div>
          <div className="mt-1 text-sm font-semibold text-teal-900">
            {formatDate(selected.sample_day)}
          </div>
        </div>

        <select
          value={selectedIndex}
          onChange={(event) =>
            setSelectedIndex(Number(event.target.value))
          }
          className="h-9 min-w-[220px] rounded-lg border border-teal-200 bg-white px-3 text-sm text-teal-800"
          aria-label="Select DGA sample"
        >
          {orderedRows.map(({ row, index }, position) => (
            <option key={sampleKey(row, index)} value={position}>
              {formatDate(row.sample_day)}
              {position === orderedRows.length - 1 ? " · Latest" : ""}
            </option>
          ))}
        </select>
      </div>

      <div className="mb-4 flex flex-wrap gap-1.5">
        {METHODS.map((method) => (
          <button
            key={method.key}
            type="button"
            onClick={() => setActive(method.key)}
            className={cn(
              "cursor-pointer rounded-lg px-3 py-1.5 text-xs font-semibold transition-colors",
              active === method.key
                ? "bg-teal-800 text-white"
                : "bg-cream-100 text-teal-700 hover:bg-teal-100"
            )}
          >
            {method.label}
          </button>
        ))}
      </div>

      <div className="rounded-xl bg-cream-50 p-4">
        {active === "triangle" && (
          <div className="space-y-2">
            <DuvalTriangleSvg
              ch4={Number(selected.ch4 ?? 0)}
              c2h4={Number(selected.c2h4 ?? 0)}
              c2h2={Number(selected.c2h2 ?? 0)}
              backendFault={triangleFault}
            />
            <p className="text-center text-sm font-extrabold text-status-critical">
              RESULT: {resultLabel(triangleFault)}
            </p>
          </div>
        )}

        {active === "pentagon1" && (
          <div className="space-y-2">
            <DuvalPentagon1Svg
              h2={g.h2}
              ch4={g.ch4}
              c2h6={g.c2h6}
              c2h4={g.c2h4}
              c2h2={g.c2h2}
              backendFault={pentagon1Fault}
            />
            <p className="text-center text-sm font-extrabold text-status-critical">
              RESULT: {resultLabel(pentagon1Fault)}
            </p>
          </div>
        )}

        {active === "pentagon2" && (
          <div className="space-y-2">
            <DuvalPentagon2Svg
              h2={g.h2}
              ch4={g.ch4}
              c2h6={g.c2h6}
              c2h4={g.c2h4}
              c2h2={g.c2h2}
              backendFault={pentagon2Fault}
            />
            <p className="text-center text-sm font-extrabold text-status-critical">
              RESULT: {resultLabel(pentagon2Fault)}
            </p>
          </div>
        )}

        {active === "doernenburg" && (
          <RatioTable
            rows={[
              ["CH₄/H₂", toOptionalNumber(selected.dr_r1_ch4_h2)],
              ["C₂H₂/C₂H₄", toOptionalNumber(selected.dr_r2_c2h2_c2h4)],
              ["C₂H₂/CH₄", toOptionalNumber(selected.dr_r3_c2h2_ch4)],
              ["C₂H₆/C₂H₂", toOptionalNumber(selected.dr_r4_c2h6_c2h2)],
            ]}
            fault={resultLabel(doernenburgFault)}
          />
        )}

        {active === "iec" && (
          <Iec3DChart
            r1={Number(selected.iec_r1_c2h2_c2h4 ?? 0)}
            r2={Number(selected.iec_r2_ch4_h2 ?? 0)}
            r3={Number(selected.iec_r3_c2h4_c2h6 ?? 0)}
            fault={resultLabel(iecFault)}
          />
        )}

        {active === "rogers" && (
          <Ratio3DChart
            r1={Number(selected.r1_ch4_h2 ?? 0)}
            r2={Number(selected.r2_c2h2_c2h4 ?? 0)}
            r3={Number(selected.r3_c2h4_c2h6 ?? 0)}
            fault={resultLabel(rogersFault)}
          />
        )}

        {active === "keygas" && (
          <KeyGasView
            g={g}
            co={Number(selected.co ?? 0)}
            co2={Number(selected.co2 ?? 0)}
            tdcg={toOptionalNumber(selected.tdcg)}
            fault={resultLabel(keyGasFault)}
          />
        )}
      </div>
    </div>
  );
}

function RatioTable({
  rows,
  fault,
}: {
  rows: [string, number | undefined][];
  fault: string;
}) {
  return (
    <div className="space-y-4">
      <table className="w-full text-sm">
        <tbody>
          {rows.map(([label, value]) => (
            <tr
              key={label}
              className="border-b border-cream-200 last:border-0"
            >
              <td className="py-2 text-teal-600">{label}</td>
              <td className="py-2 text-right font-mono font-semibold text-teal-900">
                {value === undefined || Number.isNaN(value)
                  ? "—"
                  : formatNumber(value, 3)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <p className="text-center text-sm font-extrabold text-status-critical">
        RESULT: {fault}
      </p>
    </div>
  );
}

function KeyGasView({
  g,
  co,
  co2,
  tdcg,
  fault,
}: {
  g: Record<string, number>;
  co?: number;
  co2?: number;
  tdcg?: number;
  fault?: string;
}) {
  const h2 = Number(g.h2 ?? 0);
  const ch4 = Number(g.ch4 ?? 0);
  const c2h6 = Number(g.c2h6 ?? 0);
  const c2h4 = Number(g.c2h4 ?? 0);
  const c2h2 = Number(g.c2h2 ?? 0);
  const coValue = Number(co ?? 0);
  const co2Value = Number(co2 ?? 0);

  const combustibleTotal =
    tdcg != null && Number.isFinite(tdcg)
      ? tdcg
      : h2 + ch4 + c2h6 + c2h4 + c2h2 + coValue;

  const entries: [string, number, string][] = [
    ["H₂", h2, "#ff6b6b"],
    ["CH₄", ch4, "#4ecdc4"],
    ["C₂H₆", c2h6, "#45b7d1"],
    ["C₂H₄", c2h4, "#f9ca24"],
    ["C₂H₂", c2h2, "#6c5ce7"],
    ["CO", coValue, "#a78bfa"],
  ];

  const dominant = entries.reduce(
    (max, entry) => (entry[1] > max[1] ? entry : max),
    entries[0]
  );

  return (
    <div className="space-y-3">
      <div className="mb-2 text-[11px] leading-5 text-teal-500">
        Key Gas uses the combustible gases shown below. CO₂ is displayed as a
        contextual DGA observation but is not included in the Key Gas fault
        proportion calculation or in TDCG.
      </div>

      {entries.map(([label, value, color]) => {
        const percentage =
          combustibleTotal > 0 ? (value / combustibleTotal) * 100 : 0;
        const isDominant = label === dominant[0];

        return (
          <div key={label} className="flex items-center gap-3">
            <span
              className={cn(
                "w-12 text-xs font-semibold",
                isDominant
                  ? "font-extrabold text-teal-900"
                  : "text-teal-600"
              )}
            >
              {label}
            </span>

            <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-cream-200">
              <div
                className={cn(
                  "h-full rounded-full",
                  isDominant ? "ring-2 ring-teal-800" : ""
                )}
                style={{
                  width: `${Math.max(0, Math.min(100, percentage))}%`,
                  background: color,
                }}
              />
            </div>

            <span className="w-20 text-right font-mono text-xs text-teal-500">
              {value.toFixed(1)} ppm
            </span>

            <span className="w-12 text-right font-mono text-xs text-teal-400">
              {percentage.toFixed(1)}%
            </span>
          </div>
        );
      })}

      <div className="flex items-center justify-between border-t border-cream-200 pt-2">
        <span className="text-xs font-semibold text-teal-700">
          TDCG (Total Dissolved Combustible Gas)
        </span>
        <span className="font-mono text-xs font-bold text-teal-900">
          {combustibleTotal.toFixed(1)} ppm
        </span>
      </div>

      <div className="flex items-center justify-between rounded-lg border border-cream-200 bg-white px-3 py-2">
        <span className="text-xs font-semibold text-teal-700">CO₂ observation</span>
        <span className="font-mono text-xs font-bold text-teal-900">
          {co2Value.toFixed(1)} ppm
        </span>
      </div>

      <div className="space-y-1 text-center">
        <p className="text-xs text-teal-500">
          Dominant gas: <span className="font-bold text-teal-800">{dominant[0]}</span>
        </p>
        <p className="text-sm font-extrabold text-status-critical">
          RESULT: {fault ?? "ABSTAIN"}
        </p>
      </div>
    </div>
  );
}