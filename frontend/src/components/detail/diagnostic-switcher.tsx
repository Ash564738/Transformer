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
import { RatioZoneChart } from "../charts/ratio-zone-chart";
import { Ratio3DChart } from "../charts/ratio3d-chart";
import { Iec3DChart } from "../charts/Iec3DChart";

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
              RESULT: {resultLabel(row.duval_triangle_fault)}
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
          <Iec3DChart
            r1={Number(row.iec_r1_c2h2_c2h4 ?? 0)}
            r2={Number(row.r1_ch4_h2 ?? 0)}   // tạm dùng R2 của Rogers, vì backend chưa có iec_r2
            r3={Number(row.iec_r3_c2h4_c2h6 ?? 0)}
            fault={resultLabel(row.iec_fault)}
          />
        )}
        {active === "rogers" && (
          <Ratio3DChart
            r1={Number(row.r1_ch4_h2 ?? 0)}
            r2={Number(row.r2_c2h2_c2h4 ?? 0)}
            r3={Number(row.r3_c2h4_c2h6 ?? 0)}
            fault={resultLabel(row.rogers_fault)}
          />
        )}
        {active === "keygas" && (
          <KeyGasView
            g={g}
            co={Number(row.co ?? 0)}
            tdcg={row.tdcg != null ? Number(row.tdcg) : undefined}
            fault={resultLabel(row.keygas_fault)}
          />
        )}      
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

function KeyGasView({ g, co, tdcg, fault }: {
  g: Record<string, number>;
  co?: number;
  tdcg?: number;
  fault?: string;
}) {
  // Ép kiểu tất cả về number (phòng trường hợp row trả về string)
  const h2 = Number(g.h2 ?? 0);
  const ch4 = Number(g.ch4 ?? 0);
  const c2h6 = Number(g.c2h6 ?? 0);
  const c2h4 = Number(g.c2h4 ?? 0);
  const c2h2 = Number(g.c2h2 ?? 0);
  const coVal = Number(co ?? 0);
  const total = Number(tdcg ?? (h2 + ch4 + c2h6 + c2h4 + c2h2 + coVal));

  const entries: [string, number, string][] = [
    ["H₂", h2, "#ff6b6b"],
    ["CH₄", ch4, "#4ecdc4"],
    ["C₂H₆", c2h6, "#45b7d1"],
    ["C₂H₄", c2h4, "#f9ca24"],
    ["C₂H₂", c2h2, "#6c5ce7"],
    ["CO", coVal, "#a78bfa"],
  ];

  const dominant = entries.reduce((max, entry) => entry[1] > max[1] ? entry : max, entries[0]);
  const dominantGas = dominant[0];

  return (
    <div className="space-y-3">
      {entries.map(([label, val, color]) => {
        const pct = total > 0 ? (val / total) * 100 : 0;
        const isDominant = label === dominantGas;
        return (
          <div key={label} className="flex items-center gap-3">
            <span className={`w-12 text-xs font-semibold ${isDominant ? 'text-teal-900 font-extrabold' : 'text-teal-600'}`}>
              {label}
            </span>
            <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-cream-200">
              <div
                className={`h-full rounded-full ${isDominant ? 'ring-2 ring-teal-800' : ''}`}
                style={{ width: `${pct}%`, background: color }}
              />
            </div>
            <span className="w-16 text-right text-xs font-mono text-teal-500">
              {val.toFixed(1)} ppm
            </span>
            <span className="w-10 text-right text-xs font-mono text-teal-400">
              {pct.toFixed(1)}%
            </span>
          </div>
        );
      })}

      <div className="flex items-center justify-between pt-2 border-t border-cream-200">
        <span className="text-xs font-semibold text-teal-700">TDCG (Total Dissolved Combustible Gas)</span>
        <span className="text-xs font-mono font-bold text-teal-900">{total.toFixed(1)} ppm</span>
      </div>

      <div className="text-center space-y-1">
        <p className="text-xs text-teal-500">
          Dominant gas: <span className="font-bold text-teal-800">{dominantGas}</span>
        </p>
        <p className="text-sm font-extrabold text-status-critical">
          RESULT: {fault ?? "UNCERTAIN"}
        </p>
      </div>
    </div>
  );
}