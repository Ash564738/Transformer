"use client";

import React, { useMemo, useState } from "react";

// ---------- SVG constants ----------
const SVG_W = 600;
const SVG_H = 600;
const MARGIN = 45;
const TRI_W = SVG_W - 2 * MARGIN;
const TRI_H = TRI_W * Math.sqrt(3) / 2;
const TOP_X = SVG_W / 2;
const TOP_Y = MARGIN;
const BOT_LEFT_X = MARGIN;
const BOT_LEFT_Y = TOP_Y + TRI_H;
const BOT_RIGHT_X = SVG_W - MARGIN;
const BOT_RIGHT_Y = BOT_LEFT_Y;

// ---------- Duval Triangle 1 region coordinates (CH4, C2H2, C2H4 in %) ----------
const REGION_COORDS: Record<string, { a: number[]; b: number[]; c: number[] }> = {
  PD: { a: [98, 100, 98], b: [0, 0, 2], c: [2, 0, 0] },
  D1: { a: [0, 0, 64, 87], b: [100, 77, 13, 13], c: [0, 23, 23, 0] },
  D2: { a: [0, 0, 31, 47, 64], b: [77, 29, 29, 13, 13], c: [23, 71, 40, 40, 23] },
  DT: {
    a: [0, 0, 35, 46, 96, 87, 47, 31],
    b: [29, 15, 15, 4, 4, 13, 13, 29],
    c: [71, 85, 50, 50, 0, 0, 40, 40],
  },
  T1: { a: [76, 80, 98, 98, 96], b: [4, 0, 0, 2, 4], c: [20, 20, 2, 0, 0] },
  T2: { a: [46, 50, 80, 76], b: [4, 0, 0, 4], c: [50, 50, 20, 20] },
  T3: { a: [0, 0, 50, 35], b: [15, 0, 0, 15], c: [85, 100, 50, 50] },
};

const ZONE_COLORS: Record<string, string> = {
  PD: "#b3de69",
  T1: "#80b1d3",
  T2: "#fdb462",
  T3: "#8dd3c7",
  D1: "#ffffb3",
  D2: "#bebada",
  DT: "#fb8072",
};

const ZONE_LABELS: Record<string, string> = {
  PD: "PD",
  T1: "T1",
  T2: "T2",
  T3: "T3",
  D1: "D1",
  D2: "D2",
  DT: "DT",
};

const FAULT_EXPLANATIONS: Record<string, string> = {
  PD: "Partial Discharge",
  T1: "Thermal fault < 300°C",
  T2: "Thermal fault 300–700°C",
  T3: "Thermal fault > 700°C",
  D1: "Low energy electrical discharge",
  D2: "High energy electrical discharge (arc)",
  DT: "Mixed discharge + thermal fault",
  UNCERTAIN: "Uncertain diagnosis",
  NORMAL: "Normal",
};

// ---------- Barycentric coordinate conversion ----------
function ternaryToSvg(ch4Pct: number, c2h2Pct: number, c2h4Pct: number) {
  const fCH4 = ch4Pct / 100;
  const fC2H2 = c2h2Pct / 100;
  const fC2H4 = c2h4Pct / 100;
  const x = BOT_LEFT_X * fC2H2 + BOT_RIGHT_X * fC2H4 + TOP_X * fCH4;
  const y = BOT_LEFT_Y * fC2H2 + BOT_RIGHT_Y * fC2H4 + TOP_Y * fCH4;
  return { x, y };
}

// ---------- Diagnosis (point-in-polygon) ----------
function diagnose(ch4Pct: number, c2h2Pct: number, c2h4Pct: number): string {
  const point = ternaryToSvg(ch4Pct, c2h2Pct, c2h4Pct);
  for (const [zone, coords] of Object.entries(REGION_COORDS)) {
    const vertices = coords.a.map((_, i) => ternaryToSvg(coords.a[i], coords.b[i], coords.c[i]));
    if (pointInPolygon(point, vertices)) return zone;
  }
  return "UNCERTAIN";
}

function pointInPolygon(p: { x: number; y: number }, polygon: { x: number; y: number }[]) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x, yi = polygon[i].y;
    const xj = polygon[j].x, yj = polygon[j].y;
    const intersect =
      yi > p.y !== yj > p.y &&
      p.x < ((xj - xi) * (p.y - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

// ---------- Edge label with outward arrow parallel to edge ----------
function EdgeLabel({
  p1,
  p2,
  label,
}: {
  p1: { x: number; y: number };
  p2: { x: number; y: number };
  label: string;
}) {
  const midX = (p1.x + p2.x) / 2;
  const midY = (p1.y + p2.y) / 2;
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const len = Math.sqrt(dx * dx + dy * dy);
  const ux = dx / len;
  const uy = dy / len;

  // Pháp tuyến (chưa xác định hướng)
  let nx = -dy / len;
  let ny = dx / len;

  // Xác định hướng ra ngoài bằng trọng tâm tam giác
  const centroidX = (BOT_LEFT_X + BOT_RIGHT_X + TOP_X) / 3;
  const centroidY = (BOT_LEFT_Y + BOT_RIGHT_Y + TOP_Y) / 3;
  const toCentroidX = centroidX - midX;
  const toCentroidY = centroidY - midY;
  const dot = nx * toCentroidX + ny * toCentroidY;
  if (dot > 0) {
    // pháp tuyến hướng vào trong, đảo ngược
    nx = -nx;
    ny = -ny;
  }

  const arrowOffset = 22; // khoảng cách từ cạnh đến thân mũi tên
  const arrowLen = 30;    // chiều dài mũi tên
  const direction = 1;    // hướng dọc theo cạnh (luôn là chiều p1->p2)
  const startX = midX + nx * arrowOffset - ux * (arrowLen / 2) * direction;
  const startY = midY + ny * arrowOffset - uy * (arrowLen / 2) * direction;
  const endX = startX + ux * arrowLen * direction;
  const endY = startY + uy * arrowLen * direction;

  const textOffset = 36;
  const textX = midX + nx * textOffset;
  const textY = midY + ny * textOffset;

  return (
    <g>
      <line
        x1={startX}
        y1={startY}
        x2={endX}
        y2={endY}
        stroke="black"
        strokeWidth="1.5"
        markerEnd="url(#arrowhead)"
      />
      <text
        x={textX}
        y={textY}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="12"
        fontWeight="bold"
        fill="black"
      >
        {label}
      </text>
    </g>
  );
}

// ---------- Main component ----------
export function DuvalTriangleSvg({
  ch4,
  c2h4,
  c2h2,
  backendFault,   // thêm dòng này
}: {
  ch4: number;
  c2h4: number;
  c2h2: number;
  backendFault?: string;   // thêm dòng này
}) {
  const total = ch4 + c2h4 + c2h2;
  const pCH4 = (ch4 / total) * 100;
  const pC2H4 = (c2h4 / total) * 100;
  const pC2H2 = (c2h2 / total) * 100;

  const samplePos = useMemo(() => ternaryToSvg(pCH4, pC2H2, pC2H4), [pCH4, pC2H2, pC2H4]);
  const fault = useMemo(() => diagnose(pCH4, pC2H2, pC2H4), [pCH4, pC2H2, pC2H4]);

  const [hoveredZone, setHoveredZone] = useState<string | null>(null);
  const [showSampleTooltip, setShowSampleTooltip] = useState(false);
  // Hàm tính centroid thực của đa giác (có thể lồi/lõm)
  function polygonCentroid(polygon: { x: number; y: number }[]) {
    let area = 0;
    let cx = 0;
    let cy = 0;
    const n = polygon.length;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      const cross = polygon[i].x * polygon[j].y - polygon[j].x * polygon[i].y;
      area += cross;
      cx += (polygon[i].x + polygon[j].x) * cross;
      cy += (polygon[i].y + polygon[j].y) * cross;
    }
    area *= 0.5;
    if (Math.abs(area) < 1e-10) {
      // fallback (diện tích gần 0)
      return polygon.reduce((acc, p) => ({ x: acc.x + p.x / n, y: acc.y + p.y / n }), { x: 0, y: 0 });
    }
    cx /= 6 * area;
    cy /= 6 * area;
    return { x: cx, y: cy };
  }

  const regionPolygons = useMemo(() => {
    return Object.entries(REGION_COORDS).map(([zone, coords]) => {
      const pts = coords.a.map((_, i) => ternaryToSvg(coords.a[i], coords.b[i], coords.c[i]));
      const points = pts.map(p => `${p.x},${p.y}`).join(" ");
      const centroid = polygonCentroid(pts);
      return { zone, points, centroid };
    });
  }, []);

  const gridLines = useMemo(() => {
    const lines: React.ReactElement[] = [];
    [20, 40, 60, 80].forEach(tv => {
      const k = tv;
      const left = ternaryToSvg(k, 100 - k, 0);
      const right = ternaryToSvg(k, 0, 100 - k);
      lines.push(<line key={`h-${tv}`} x1={left.x} y1={left.y} x2={right.x} y2={right.y} stroke="#999" strokeWidth={0.5} strokeDasharray="3 3" />);
      const bLeft = ternaryToSvg(0, 100 - k, k);
      const top = ternaryToSvg(100 - k, 0, k);
      lines.push(<line key={`l-${tv}`} x1={bLeft.x} y1={bLeft.y} x2={top.x} y2={top.y} stroke="#999" strokeWidth={0.5} strokeDasharray="3 3" />);
      const bRight = ternaryToSvg(0, k, 100 - k);
      const top2 = ternaryToSvg(100 - k, k, 0);
      lines.push(<line key={`r-${tv}`} x1={bRight.x} y1={bRight.y} x2={top2.x} y2={top2.y} stroke="#999" strokeWidth={0.5} strokeDasharray="3 3" />);
    });
    return lines;
  }, []);

  const tickLabels = useMemo(() => {
    const labels: React.ReactElement[] = [];
    [20, 40, 60, 80].forEach(tv => {
      const pLeft = ternaryToSvg(tv, 100 - tv, 0);
      labels.push(<text key={`tl-${tv}`} x={pLeft.x - 8} y={pLeft.y} textAnchor="end" fontSize="10" fill="#555">{tv}%</text>);
      const pRight = ternaryToSvg(100 - tv, 0, tv);
      labels.push(<text key={`tr-${tv}`} x={pRight.x + 8} y={pRight.y} textAnchor="start" fontSize="10" fill="#555">{tv}%</text>);
      const pBottom = ternaryToSvg(0, tv, 100 - tv);
      labels.push(<text key={`tb-${tv}`} x={pBottom.x} y={pBottom.y + 14} textAnchor="middle" fontSize="10" fill="#555">{tv}%</text>);
    });
    return labels;
  }, []);

  // Fault explanation text width estimation for foreignObject
  const faultText = hoveredZone ? FAULT_EXPLANATIONS[hoveredZone] || hoveredZone : "";

  return (
    <div className="relative w-full max-w-md mx-auto">
      <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} className="w-full">
        <defs>
          <marker id="arrowhead" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
            <polygon points="0 0, 6 2, 0 4" fill="black" />
          </marker>
        </defs>

        {/* Fault zones */}
        {regionPolygons.map(({ zone, points, centroid }) => (
          <g
            key={zone}
            onMouseEnter={() => setHoveredZone(zone)}
            onMouseLeave={() => setHoveredZone(null)}
            className="cursor-pointer"
          >
            <polygon
              points={points}
              fill={ZONE_COLORS[zone] || "gray"}
              stroke={hoveredZone === zone ? "#000" : "none"}
              strokeWidth={hoveredZone === zone ? 2 : 0}
              opacity={0.85}
            />
            <text
              x={centroid.x}
              y={centroid.y}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize="12"
              fontWeight="bold"
              fill={getContrastText(ZONE_COLORS[zone] || "#ccc")}
              pointerEvents="none"
            >
              {ZONE_LABELS[zone]}
            </text>
          </g>
        ))}

        {/* Grid and ticks */}
        {gridLines}
        {tickLabels}

        {/* Edge arrows */}
        <EdgeLabel p1={{ x: BOT_LEFT_X, y: BOT_LEFT_Y }} p2={{ x: TOP_X, y: TOP_Y }} label="CH₄" />
        <EdgeLabel p1={{ x: TOP_X, y: TOP_Y }} p2={{ x: BOT_RIGHT_X, y: BOT_RIGHT_Y }} label="C₂H₄" />
        <EdgeLabel p1={{ x: BOT_LEFT_X, y: BOT_LEFT_Y }} p2={{ x: BOT_RIGHT_X, y: BOT_RIGHT_Y }} label="C₂H₂" />

        {/* Sample point */}
        <g onMouseEnter={() => setShowSampleTooltip(true)} onMouseLeave={() => setShowSampleTooltip(false)}>
          {/* Pulse ring (lớp ngoài cùng, mờ dần và lan rộng) */}
          <circle cx={samplePos.x} cy={samplePos.y} r="8" fill="none" stroke="#c62828" strokeWidth="1" opacity="0.6">
            <animate attributeName="r" values="6;14;6" dur="2.4s" repeatCount="indefinite" />
            <animate attributeName="opacity" values="0.6;0;0.6" dur="2.4s" repeatCount="indefinite" />
          </circle>
          {/* Điểm đỏ chính */}
          <circle cx={samplePos.x} cy={samplePos.y} r="6" fill="red" stroke="#c62828" strokeWidth="2" className="cursor-pointer" />
          
          {showSampleTooltip && (
            <g transform={`translate(${samplePos.x + 18}, ${samplePos.y - 35})`}>
              <rect x="-62" y="-28" width="124" height="52" rx="6" fill="white" stroke="darkred" strokeWidth="1.5" opacity="0.95" />
              <text x="0" y="-16" textAnchor="middle" fontSize="9" fill="darkred" fontWeight="bold">CH₄: {pCH4.toFixed(1)}%</text>
              <text x="0" y="-4" textAnchor="middle" fontSize="9" fill="darkred" fontWeight="bold">C₂H₄: {pC2H4.toFixed(1)}%</text>
              <text x="0" y="8" textAnchor="middle" fontSize="9" fill="darkred" fontWeight="bold">C₂H₂: {pC2H2.toFixed(1)}%</text>
              <text x="0" y="20" textAnchor="middle" fontSize="9" fill="red" fontWeight="bold">
                Fault: {backendFault || fault || "UNCERTAIN"}
              </text>
            </g>
          )}
        </g>

        {/* Hover zone tooltip (auto-size with foreignObject) */}
        {hoveredZone && (
          <foreignObject x={SVG_W / 2 - 140} y={SVG_H - 55} width="280" height="50">
            <div
              style={{
                background: "rgba(0,0,0,0.8)",
                color: "white",
                borderRadius: "8px",
                fontSize: "12px",
                fontWeight: "bold",
                textAlign: "center",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                height: "80%",
                lineHeight: 1.3,
                wordBreak: "break-word",
                overflow: "hidden",
              }}
            >
              {faultText}
            </div>
          </foreignObject>
        )}
      </svg>
    </div>
  );
}

function getContrastText(hex: string) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return luminance > 0.5 ? "black" : "white";
}