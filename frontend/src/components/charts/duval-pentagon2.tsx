"use client";

import React, { useMemo, useState } from "react";

const V = {
  H2: [0, 30],
  C2H2: [28.5, 9.3],
  C2H4: [17.625, -24.3],
  CH4: [-17.625, -24.3],
  C2H6: [-28.5, 9.3],
};

const ZONES_P2: Record<string, [number, number][]> = {
  PD: [[0, 24.75], [-0.75, 24.75], [-0.75, 18.375], [0, 18.375]],
  D1: [[0, 30], [28.5, 9.3], [24, -4.575], [3, 12], [0, 1.125]],
  D2: [[3, 12], [24, -4.575], [18.225, -22.5], [0, -2.25], [0, 1.125]],
  S: [[0, 1.125], [-26.25, 2.325], [-28.5, 9.3], [0, 30], [0, 24.75], [-0.75, 24.75], [-0.75, 18.375], [0, 18.375]],
  T3_H: [[0, -2.25], [18.225, -22.5], [17.625, -24.3], [1.875, -24.3], [-2.625, -2.25]],
  C: [[-2.625, -2.25], [1.875, -24.3], [-16.125, -24.3], [-8.25, -6]],
  O: [[-2.625, -2.25], [-8.25, -6], [-16.125, -24.3], [-17.625, -24.3], [-26.25, 2.325], [0, 1.125], [0, -2.25]],
};

const ZONE_COLORS: Record<string, string> = {
  PD: "#cfff7c",
  D1: "#ffffb3",
  D2: "#cec9ec",
  T3_H: "#90ccc2",
  C: "#fb8072",
  O: "#ffd3ea",
  S: "#ffcfab",
};

const FAULT_EXPLANATIONS: Record<string, string> = {
  PD: "Partial Discharge",
  D1: "Low energy electrical discharge",
  D2: "High energy electrical discharge (arc)",
  T3_H: "Thermal fault > 700°C (oil only)",
  C: "Carbonization of paper insulation",
  O: "Overheating < 250°C",
  S: "Stray Gassing",
  UNCERTAIN: "Uncertain diagnosis",
  NORMAL: "Normal",
};

function centroid(h2: number, ch4: number, c2h6: number, c2h4: number, c2h2: number) {
  const total = h2 + ch4 + c2h6 + c2h4 + c2h2;
  if (total <= 0) return null;
  const p = [h2 / total, c2h2 / total, c2h4 / total, ch4 / total, c2h6 / total];
  const x = p[0] * V.H2[0] + p[1] * V.C2H2[0] + p[2] * V.C2H4[0] + p[3] * V.CH4[0] + p[4] * V.C2H6[0];
  const y = p[0] * V.H2[1] + p[1] * V.C2H2[1] + p[2] * V.C2H4[1] + p[3] * V.CH4[1] + p[4] * V.C2H6[1];
  return { x, y };
}

function pointInPolygon(p: { x: number; y: number }, poly: [number, number][]) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1];
    const xj = poly[j][0], yj = poly[j][1];
    const intersect = yi > p.y !== yj > p.y && p.x < ((xj - xi) * (p.y - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function diagnoseP2(cent: { x: number; y: number }) {
  for (const [zone, poly] of Object.entries(ZONES_P2)) {
    if (pointInPolygon(cent, poly)) return zone;
  }
  return "UNCERTAIN";
}

function clamp(val: number, min: number, max: number) {
  return Math.min(max, Math.max(min, val));
}

export function DuvalPentagon2Svg({
  h2, ch4, c2h6, c2h4, c2h2,
  backendFault,
}: {
  h2: number; ch4: number; c2h6: number; c2h4: number; c2h2: number;
  backendFault?: string;
}) {
  const cent = useMemo(() => centroid(h2, ch4, c2h6, c2h4, c2h2), [h2, ch4, c2h6, c2h4, c2h2]);
  const fault = useMemo(() => (cent ? diagnoseP2(cent) : "UNCERTAIN"), [cent]);

  const total5 = h2 + ch4 + c2h6 + c2h4 + c2h2;
  const pH2 = total5 > 0 ? (h2 / total5) * 100 : 0;
  const pCH4 = total5 > 0 ? (ch4 / total5) * 100 : 0;
  const pC2H6 = total5 > 0 ? (c2h6 / total5) * 100 : 0;
  const pC2H4 = total5 > 0 ? (c2h4 / total5) * 100 : 0;
  const pC2H2 = total5 > 0 ? (c2h2 / total5) * 100 : 0;

  const [hoveredZone, setHoveredZone] = useState<string | null>(null);
  const [showTooltip, setShowTooltip] = useState(false);

  const zoneEls = useMemo(() => {
    return Object.entries(ZONES_P2).map(([zone, poly]) => {
      const pts = poly.map(p => `${p[0]},${-p[1]}`).join(" ");
      const c = poly.reduce((a, p) => ({ x: a.x + p[0] / poly.length, y: a.y + p[1] / poly.length }), { x: 0, y: 0 });
      return { zone, pts, centroid: c };
    });
  }, []);

  const vertexLabels = useMemo(() => {
    return Object.entries(V).map(([gas, [x, y]]) => (
      <text key={gas} x={x * 1.12} y={-y * 1.12} fontSize="2" fontWeight="bold" textAnchor="middle" dominantBaseline="central" fill="#333">
        {gas}
      </text>
    ));
  }, []);

  const spokes = useMemo(() => {
    const lines: React.ReactElement[] = [];
    Object.values(V).forEach(([x, y]) => {
      lines.push(<line key={`spoke-${x}`} x1={0} y1={0} x2={x} y2={-y} stroke="gray" strokeWidth={0.3} strokeDasharray="2 2" opacity={0.4} />);
      [0.2, 0.4, 0.6, 0.8].forEach(k => {
        const tx = x * k, ty = y * k;
        lines.push(
          <text key={`tick-${x}-${k}`} x={tx} y={-ty} fontSize="1.5" fill="#444" textAnchor="middle" dominantBaseline="central" opacity={0.7}>
            {Math.round(k * 100)}%
          </text>
        );
      });
    });
    return lines;
  }, []);

  const tooltipW = 14;   // trước là 20
  const tooltipH = 10;   // trước là 15
  const rawX = cent ? cent.x + 4 : 0;        // lệch sang phải một chút
  const rawY = cent ? -cent.y - tooltipH - 2 : 0; // lên trên (giảm khoảng cách)
  const tx = clamp(rawX, -37, 37 - tooltipW);   // điều chỉnh biên cho vừa viewBox 38
  const ty = clamp(rawY, -37, 37 - tooltipH);

  const hoverZoneY = 32;      // nâng lên (cũ: 30)
  const hoverZoneW = 26;      // thu hẹp một chút (cũ 28)
  const hoverZoneH = 4;

  const hoverText = hoveredZone ? FAULT_EXPLANATIONS[hoveredZone] || hoveredZone : "";

  return (
    <div className="relative w-full max-w-md mx-auto">
      <svg viewBox="-38 -38 76 76" className="w-full">
        {/* Vùng màu và nhãn zone */}
        {zoneEls.map(({ zone, pts, centroid }) => (
          <g key={zone} onMouseEnter={() => setHoveredZone(zone)} onMouseLeave={() => setHoveredZone(null)} className="cursor-pointer">
            <polygon points={pts} fill={ZONE_COLORS[zone] || "#ccc"} stroke={hoveredZone === zone ? "#000" : "none"} strokeWidth={0.3} opacity={0.85} />
            <text x={centroid.x} y={-centroid.y} textAnchor="middle" dominantBaseline="central" fontSize="2" fontWeight="bold" fill={getContrast(ZONE_COLORS[zone])} pointerEvents="none">
              {zone}
            </text>
          </g>
        ))}
        {spokes}
        <polygon points={[V.H2, V.C2H2, V.C2H4, V.CH4, V.C2H6].map(p => `${p[0]},${-p[1]}`).join(" ")} fill="none" />
        {vertexLabels}

        {cent && (
          <g onMouseEnter={() => setShowTooltip(true)} onMouseLeave={() => setShowTooltip(false)}>
            <circle cx={cent.x} cy={-cent.y} r="0.8" fill="none" stroke="#c62828" strokeWidth={0.2} opacity={0.6}>
              <animate attributeName="r" values="0.8;1.6;0.8" dur="2.4s" repeatCount="indefinite" />
              <animate attributeName="opacity" values="0.6;0;0.6" dur="2.4s" repeatCount="indefinite" />
            </circle>
            <circle cx={cent.x} cy={-cent.y} r="0.8" fill="red" stroke="#c62828" strokeWidth={0.2} className="cursor-pointer" />
            {showTooltip && (
              <g transform={`translate(${tx}, ${ty})`}>
                <rect x="0" y="0" width={tooltipW} height={tooltipH} rx="1" fill="white" stroke="darkred" strokeWidth={0.2} opacity={0.95} />
                <text x={tooltipW / 2} y="1.8" textAnchor="middle" fontSize="1" fill="darkred" fontWeight="bold">H₂: {pH2.toFixed(1)}%</text>
                <text x={tooltipW / 2} y="3.2" textAnchor="middle" fontSize="1" fill="darkred" fontWeight="bold">CH₄: {pCH4.toFixed(1)}%</text>
                <text x={tooltipW / 2} y="4.6" textAnchor="middle" fontSize="1" fill="darkred" fontWeight="bold">C₂H₆: {pC2H6.toFixed(1)}%</text>
                <text x={tooltipW / 2} y="6.0" textAnchor="middle" fontSize="1" fill="darkred" fontWeight="bold">C₂H₄: {pC2H4.toFixed(1)}%</text>
                <text x={tooltipW / 2} y="7.4" textAnchor="middle" fontSize="1" fill="darkred" fontWeight="bold">C₂H₂: {pC2H2.toFixed(1)}%</text>
                <text x={tooltipW / 2} y="9.0" textAnchor="middle" fontSize="1" fill="red" fontWeight="bold">Fault: {backendFault || fault}</text>
              </g>
            )}
          </g>
        )}

        {/* Hover zone tooltip */}
        {hoveredZone && (
          <>
            <rect x={-hoverZoneW / 2} y={hoverZoneY} width={hoverZoneW} height={hoverZoneH} rx={1} fill="rgba(0,0,0,0.8)" />
            <text x={0} y={hoverZoneY + hoverZoneH / 2} textAnchor="middle" fontSize="1.4" fill="white" fontWeight="bold" dominantBaseline="middle">
              {hoverText}
            </text>
          </>
        )}
      </svg>
    </div>
  );
}

function getContrast(hex: string) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.5 ? "black" : "white";
}