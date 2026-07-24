"use client";

import dynamic from "next/dynamic";
import React, { useMemo } from "react";
import { IEC_3D_ZONES } from "./iec3d-zones";
import { buildIecTraces } from "./build-iec-traces";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface Props {
  r1: number; // C₂H₂ / C₂H₄
  r2: number; // CH₄ / H₂
  r3: number; // C₂H₄ / C₂H₆
  fault?: string;
}

export function Iec3DChart({ r1, r2, r3, fault }: Props) {
  const pointValid =
    Number.isFinite(r1) && r1 > 0 &&
    Number.isFinite(r2) && r2 > 0 &&
    Number.isFinite(r3) && r3 > 0;

  const faultCode = fault ? fault.replace(/^.*\(([^)]+)\)$/, "$1").trim() : undefined;

  const traces = useMemo(() => {
    const zoneTraces = buildIecTraces(IEC_3D_ZONES, faultCode);
    const pointTraces = pointValid
      ? [
          {
            type: "scatter3d",
            mode: "markers",
            x: [r1],
            y: [r2],
            z: [r3],
            marker: {
              color: "#dc2626",
              size: 8,
              symbol: "circle",
              line: { color: "white", width: 2 },
            },
            showlegend: false,
            hoverinfo: "text",
            hovertext: `R1 (C₂H₂/C₂H₄): ${r1.toFixed(3)}<br>R2 (CH₄/H₂): ${r2.toFixed(3)}<br>R3 (C₂H₄/C₂H₆): ${r3.toFixed(3)}`,
          },
        ]
      : [];
    return [...zoneTraces, ...pointTraces];
  }, [r1, r2, r3, faultCode, pointValid]);

  const tickVals = [0.001, 0.01, 0.1, 1, 10];
  const tickTexts = ["0.001", "0.01", "0.1", "1", "10"];

  const layout = {
    margin: { l: 0, r: 0, b: 0, t: 30 },
    paper_bgcolor: "white",
    plot_bgcolor: "#f8fafc",
    scene: {
      xaxis: {
        type: "log",
        tickmode: "array",
        tickvals: tickVals,
        ticktext: tickTexts,
        title: {
          text: "R1 = C₂H₂ / C₂H₄",
          font: { size: 12, color: "#1e293b", family: "Arial, sans-serif" },
        },
        range: [-3, 1.2],
        tickfont: { size: 9, color: "#475569" },
        gridcolor: "#94a3b8",
        gridwidth: 1.2,
        zeroline: false,
        showgrid: true,
        automargin: true,
      },
      yaxis: {
        type: "log",
        tickmode: "array",
        tickvals: tickVals,
        ticktext: tickTexts,
        title: {
          text: "R2 = CH₄ / H₂",
          font: { size: 12, color: "#1e293b", family: "Arial, sans-serif" },
        },
        range: [-3, 1.2],
        tickfont: { size: 9, color: "#475569" },
        gridcolor: "#94a3b8",
        gridwidth: 1.2,
        zeroline: false,
        showgrid: true,
        automargin: true,
      },
      zaxis: {
        type: "log",
        tickmode: "array",
        tickvals: tickVals,
        ticktext: tickTexts,
        title: {
          text: "R3 = C₂H₄ / C₂H₆",
          font: { size: 12, color: "#1e293b", family: "Arial, sans-serif" },
        },
        range: [-3, 1.2],
        tickfont: { size: 9, color: "#475569" },
        gridcolor: "#94a3b8",
        gridwidth: 1.2,
        zeroline: false,
        showgrid: true,
        automargin: true,
      },
      aspectmode: "cube",
      bgcolor: "#f8fafc",
    },
  };

  const legendItems = IEC_3D_ZONES.map((zone) => ({
    name: zone.name,
    desc: zone.desc,
    color: zone.color,
  }));

  return (
    <div className="space-y-3">
      <div className="w-full h-[400px] bg-white rounded-lg shadow-sm overflow-hidden relative">
        <Plot
          data={traces}
          layout={layout}
          useResizeHandler
          style={{ width: "100%", height: "100%" }}
          config={{ displayModeBar: false }}
        />
      </div>

      <div className="flex flex-wrap gap-2 px-1">
        {legendItems.map((item) => (
          <div
            key={item.name}
            className="flex items-center gap-1.5 text-xs text-teal-700 font-medium"
          >
            <div
              className="w-3 h-3 rounded-sm border border-gray-300"
              style={{ backgroundColor: item.color }}
            />
            <span>
              {item.name}: {item.desc}
            </span>
          </div>
        ))}
      </div>

      {fault && (
        <p className="text-center text-sm font-extrabold text-status-critical pt-1">
          RESULT: {fault}
        </p>
      )}

      {!pointValid && (
        <p className="text-center text-xs text-teal-500 italic">
          Sample ratios undefined or invalid.
        </p>
      )}
    </div>
  );
}