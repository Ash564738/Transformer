"use client";

import { useState } from "react";
import type { DgaRow } from "@/types/dga";
import { FAULT_EXPLANATIONS } from "@/lib/dga-methods";
import { duvalTriangleImageUrl, duvalPentagonImageUrl } from "@/lib/api";
import { cn, formatNumber } from "@/lib/utils";
import { BackendChartImage } from "@/components/charts/backend-chart-image";
import { DuvalTriangleSvg } from "../charts/duval-triangle";
import { DuvalPentagon1Svg } from "../charts/duval-pentagon1";
import { DuvalPentagon2Svg } from "../charts/duval-pentagon2";

type MethodKey = "triangle" | "pentagon1" | "pentagon2" | "doernenburg" | "iec" | "rogers" | "keygas";

const METHODS: { key: MethodKey; label: string }[] = [
  { key: "triangle", label: "Duval Triangle 1" },
  { key: "pentagon1", label: "Duval Pentagon 1" },
  { key: "pentagon2", label: "Duval Pentagon 2" },
  { key: "doernenburg", label: "Doernenburg" },
  { key: "iec", label: "IEC 60599" },
  { key: "rogers", label: "Rogers Ratio" },
  { key: "keygas", label: "Key Gas" },
];

function resultLabel(code?: string) {
  if (!code) return "UNCERTAIN";
  const explanation = FAULT_EXPLANATIONS[code];
  return explanation ? `${explanation.toUpperCase()} (${code})` : code;
}

export function DiagnosticSwitcher({ row }: { row: DgaRow }) {
  const [active, setActive] = useState<MethodKey>("triangle");
  const g = {
    h2: Number(row.h2 ?? 0),
    ch4: Number(row.ch4 ?? 0),
    c2h6: Number(row.c2h6 ?? 0),
    c2h4: Number(row.c2h4 ?? 0),
    c2h2: Number(row.c2h2 ?? 0),
  };
  const faultCode = row.duval_triangle_fault;
  const faultDisplay = faultCode 
    ? (FAULT_EXPLANATIONS[faultCode] || faultCode) 
    : "UNCERTAIN";
  return (
    <div>
      <div className="flex flex-wrap gap-1.5">
        {METHODS.map((m) => (
          <button
            key={m.key}
            onClick={() => setActive(m.key)}
            className={cn(
              "rounded-lg px-3 py-1.5 text-xs font-semibold transition-colors cursor-pointer",
              active === m.key ? "bg-teal-800 text-white" : "bg-cream-100 text-teal-700 hover:bg-teal-100"
            )}
          >
            {m.label}
          </button>
        ))}
      </div>

      <div className="mt-4 rounded-xl bg-cream-50 p-4">
        {active === "triangle" && (
          <div className="space-y-2">
            <DuvalTriangleSvg
              ch4={Number(row.ch4 ?? 0)}
              c2h4={Number(row.c2h4 ?? 0)}
              c2h2={Number(row.c2h2 ?? 0)}
              backendFault={row.duval_triangle_fault}
            />
            <p className="text-center text-sm font-extrabold text-status-critical">
              RESULT: {row.duval_triangle_fault 
                ? (FAULT_EXPLANATIONS[row.duval_triangle_fault] || row.duval_triangle_fault) 
                : "UNCERTAIN"}
            </p>
          </div>
        )}
        {active === "pentagon1" && (
          <div className="space-y-2">
            <DuvalPentagon1Svg
              h2={g.h2} ch4={g.ch4} c2h6={g.c2h6} c2h4={g.c2h4} c2h2={g.c2h2}
              backendFault={row.fault_p1}
            />
            <p className="text-center text-sm font-extrabold text-status-critical">
              RESULT: {resultLabel(row.fault_p1)}
            </p>
          </div>
        )}

        {active === "pentagon2" && (
          <div className="space-y-2">
            <DuvalPentagon2Svg
              h2={g.h2} ch4={g.ch4} c2h6={g.c2h6} c2h4={g.c2h4} c2h2={g.c2h2}
              backendFault={row.duval_pentagon_fault}
            />
            <p className="text-center text-sm font-extrabold text-status-critical">
              RESULT: {resultLabel(row.duval_pentagon_fault)}
            </p>
          </div>
        )}
        {active === "doernenburg" && (
          <RatioTable
            rows={[
              ["CH₄/H₂", row.dr_r1_ch4_h2],
              ["C₂H₂/C₂H₄", row.dr_r2_c2h2_c2h4],
              ["C₂H₂/CH₄", row.dr_r3_c2h2_ch4],
              ["C₂H₆/C₂H₂", row.dr_r4_c2h6_c2h2],
            ]}
            fault={resultLabel(row.doernenburg_fault)}
          />
        )}
        {active === "iec" && (
          <RatioTable
            rows={[
              ["C₂H₂/C₂H₄", row.iec_r1_c2h2_c2h4],
              ["CH₄/H₂", row.iec_r2_ch4_h2],
              ["C₂H₄/C₂H₆", row.iec_r3_c2h4_c2h6],
            ]}
            fault={resultLabel(row.iec_fault)}
          />
        )}
        {active === "rogers" && (
          <RatioTable
            rows={[
              ["CH₄/H₂", row.r1_ch4_h2],
              ["C₂H₂/C₂H₄", row.r2_c2h2_c2h4],
              ["C₂H₄/C₂H₆", row.r3_c2h4_c2h6],
            ]}
            fault={resultLabel(row.rogers_fault)}
          />
        )}
        {active === "keygas" && <KeyGasView g={g} fault={resultLabel(row.keygas_fault)} />}
      </div>
    </div>
  );
}

function RatioTable({ rows, fault }: { rows: [string, number | undefined][]; fault: string }) {
  return (
    <div className="space-y-4">
      <table className="w-full text-sm">
        <tbody>
          {rows.map(([label, val]) => (
            <tr key={label} className="border-b border-cream-200 last:border-0">
              <td className="py-2 text-teal-600">{label}</td>
              <td className="py-2 text-right font-mono font-semibold text-teal-900">
                {val === undefined || Number.isNaN(val) ? "—" : formatNumber(val, 3)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-center text-sm font-extrabold text-status-critical">RESULT: {fault}</p>
    </div>
  );
}

function KeyGasView({ g, fault }: { g: Record<string, number>; fault: string }) {
  const entries = [
    ["H₂", g.h2, "#ff6b6b"],
    ["CH₄", g.ch4, "#4ecdc4"],
    ["C₂H₆", g.c2h6, "#45b7d1"],
    ["C₂H₄", g.c2h4, "#f9ca24"],
    ["C₂H₂", g.c2h2, "#6c5ce7"],
  ] as const;
  const total = entries.reduce((s, [, v]) => s + v, 0) || 1;
  return (
    <div className="space-y-3">
      {entries.map(([label, val, color]) => {
        const pct = (val / total) * 100;
        return (
          <div key={label} className="flex items-center gap-3">
            <span className="w-12 text-xs font-semibold text-teal-600">{label}</span>
            <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-cream-200">
              <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
            </div>
            <span className="w-12 text-right text-xs font-mono text-teal-500">{pct.toFixed(1)}%</span>
          </div>
        );
      })}
      <p className="pt-1 text-center text-sm font-extrabold text-status-critical">RESULT: {fault}</p>
    </div>
  );
}
