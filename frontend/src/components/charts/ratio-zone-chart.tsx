"use client";

import React, { useRef, useEffect, useState, useMemo } from "react";
import { Chart, registerables, TooltipItem } from "chart.js";
import annotationPlugin from "chartjs-plugin-annotation";

Chart.register(...registerables, annotationPlugin);

const ZONE_COLORS: Record<string, string> = {
  PD: "rgba(239, 68, 68, 0.4)",
  D1: "rgba(245, 158, 11, 0.4)",
  D2: "rgba(220, 38, 38, 0.4)",
  T1: "rgba(234, 179, 8, 0.4)",
  T2: "rgba(249, 115, 22, 0.4)",
  T3: "rgba(153, 27, 27, 0.4)",
  NORMAL: "rgba(74, 222, 128, 0.4)",
  THERMAL: "rgba(249, 115, 22, 0.4)",
};

const METHOD_CONFIGS = {
  iec: {
    title: "IEC 60599 (R1 vs R3)",
    xLabel: "R3 (C₂H₄ / C₂H₆)",
    yLabel: "R1 (C₂H₂ / C₂H₄)",
    zones: [
      { key: "pd_zone", label: "PD", xMin: 0, xMax: 1, yMin: 0, yMax: 0.01, color: ZONE_COLORS.PD, desc: "Partial Discharge" },
      { key: "t1_zone", label: "T1", xMin: 0, xMax: 1, yMin: 0.01, yMax: 0.1, color: ZONE_COLORS.T1, desc: "Thermal fault < 300°C" },
      { key: "t2_zone", label: "T2", xMin: 1, xMax: 5, yMin: 0.01, yMax: 1, color: ZONE_COLORS.T2, desc: "Thermal fault 300–700°C" },
      { key: "t3_zone", label: "T3", xMin: 5, xMax: 15, yMin: 0.01, yMax: 0.2, color: ZONE_COLORS.T3, desc: "Thermal fault > 700°C" },
      { key: "d1_zone", label: "D1", xMin: 1, xMax: 15, yMin: 1, yMax: 15, color: ZONE_COLORS.D1, desc: "Low-energy discharge" },
      { key: "d2_zone", label: "D2", xMin: 5, xMax: 15, yMin: 0.2, yMax: 1, color: ZONE_COLORS.D2, desc: "High-energy discharge" },
    ],
    legends: [
      "PD: Partial Discharge",
      "T1: Thermal fault < 300°C",
      "T2: Thermal fault 300–700°C",
      "T3: Thermal fault > 700°C",
      "D1: Low-energy discharge",
      "D2: High-energy discharge",
    ],
  },
  rogers: {
    title: "Rogers Ratio (R1 vs R2)",
    xLabel: "R1 (CH₄ / H₂)",
    yLabel: "R2 (C₂H₂ / C₂H₄)",
    zones: [
      { key: "normal_zone", label: "N", xMin: 0.1, xMax: 1.0, yMin: 0.001, yMax: 0.1, color: ZONE_COLORS.NORMAL, desc: "Normal" },
      { key: "pd_zone", label: "PD", xMin: 0.001, xMax: 0.1, yMin: 0.001, yMax: 0.1, color: ZONE_COLORS.PD, desc: "Partial Discharge" },
      { key: "d2_zone", label: "D2", xMin: 0.1, xMax: 1.0, yMin: 0.1, yMax: 1.0, color: ZONE_COLORS.D2, desc: "Low-energy arcing (D2)" },
      { key: "d1_zone", label: "D1", xMin: 0.1, xMax: 1.0, yMin: 1.0, yMax: 10, color: ZONE_COLORS.D1, desc: "High-energy arcing (D1)" },
      { key: "t2_zone", label: "T2", xMin: 1.0, xMax: 10, yMin: 0.001, yMax: 0.1, color: ZONE_COLORS.T2, desc: "Thermal 300–700°C" },
      { key: "t3_zone", label: "T3", xMin: 1.0, xMax: 10, yMin: 0.1, yMax: 10, color: ZONE_COLORS.T3, desc: "Thermal > 700°C" },
    ],
    legends: [
      "Normal",
      "PD: Partial Discharge",
      "D2: Low-energy arcing",
      "D1: High-energy arcing",
      "T2: Thermal 300–700°C",
      "T3: Thermal > 700°C",
    ],
  },
  doernenburg: {
    title: "Doernenburg (Ratio 2 vs Ratio 1)",
    xLabel: "Ratio 2 (C₂H₂ / C₂H₄)",
    yLabel: "Ratio 1 (CH₄ / H₂)",
    zones: [
      { key: "pd_zone", label: "PD", xMin: 0, xMax: 15, yMin: 0, yMax: 0.1, color: ZONE_COLORS.PD, desc: "Partial Discharge" },
      { key: "t3_zone", label: "T3", xMin: 0, xMax: 0.75, yMin: 1.0, yMax: 15, color: ZONE_COLORS.T3, desc: "Thermal fault > 700°C" },
      { key: "d2_zone", label: "D2", xMin: 0.75, xMax: 15, yMin: 0.1, yMax: 1.0, color: ZONE_COLORS.D2, desc: "High-energy discharge" },
    ],
    legends: [
      "PD: Partial Discharge",
      "T3: Thermal fault > 700°C",
      "D2: High-energy discharge",
    ],
  },
};

interface HoverPoint {
  x: number;
  y: number;
  label: string;
  desc: string;
}

export type RatioMethod = keyof typeof METHOD_CONFIGS;

export function RatioZoneChart({
  method,
  xValue,
  yValue,
  fault,
}: {
  method: RatioMethod;
  xValue: number;
  yValue: number;
  fault?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<Chart | null>(null);
  const [error, setError] = useState(false);

  const config = METHOD_CONFIGS[method];
  const pointValid = Number.isFinite(xValue) && xValue > 0 && Number.isFinite(yValue) && yValue > 0;

  // Tạo dataset ẩn chứa điểm tại tâm mỗi vùng để kích hoạt tooltip
  const hoverPoints: HoverPoint[] = useMemo(() => {
    return config.zones.map(zone => ({
      x: (zone.xMin + zone.xMax) / 2,
      y: (zone.yMin + zone.yMax) / 2,
      label: zone.label,
      desc: zone.desc || zone.label,
    }));
  }, [config]);

  useEffect(() => {
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;

    if (chartRef.current) chartRef.current.destroy();

    // Xác định vùng chứa lỗi (nếu có) để tô đậm viền
    const faultCode = fault ? fault.replace(/^.*\(([^)]+)\)$/, "$1").trim() : null;
    const matchedZone = faultCode ? config.zones.find(z => z.label === faultCode) : null;

    const annotations: any = {};
    config.zones.forEach((zone) => {
      const isActive = matchedZone && zone.key === matchedZone.key;
      annotations[zone.key] = {
        type: "box",
        xMin: zone.xMin,
        xMax: zone.xMax,
        yMin: zone.yMin,
        yMax: zone.yMax,
        backgroundColor: zone.color,
        borderColor: isActive ? "#1e293b" : "transparent",
        borderWidth: isActive ? 2 : 0,
      };

      // Nhãn tĩnh trong vùng (chỉ khi đủ rộng)
      const w = zone.xMax - zone.xMin;
      const h = zone.yMax - zone.yMin;
      if (w > 0.02 && h > 0.02) {
        const cx = zone.xMin + w / 2;
        const cy = zone.yMin + h / 2;
        annotations[zone.key + "_label"] = {
          type: "label",
          xValue: cx,
          yValue: cy,
          backgroundColor: "transparent",
          content: zone.label,
          font: { size: 10, weight: "bold" },
          color: "#1e293b",
        };
      }
    });

    try {
      chartRef.current = new Chart(ctx, {
        type: "scatter",
        data: {
          datasets: [
            {
              label: "Current sample",
              data: pointValid ? [{ x: xValue, y: yValue }] : [],
              backgroundColor: "#0ea5e9",
              borderColor: "#fff",
              borderWidth: 2,
              pointRadius: 7,
              order: 1, // đè lên trên
            },
            // Dataset ẩn cho tooltip vùng (điểm trong suốt)
            {
              label: "Zone hover",
              data: hoverPoints as any,
              backgroundColor: "transparent",
              borderColor: "transparent",
              pointRadius: 0.1, // rất nhỏ để không thấy
              pointHitRadius: 20, // vùng bắt hover
              showLine: false,
              order: 0,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              type: "logarithmic",
              title: { display: true, text: config.xLabel, color: "#1e293b", font: { size: 12, weight: "bold" } },
              grid: { color: "#e2e8f0" },
              ticks: { color: "#475569" },
              min: 0.001,
              max: 15,
            },
            y: {
              type: "logarithmic",
              title: { display: true, text: config.yLabel, color: "#1e293b", font: { size: 12, weight: "bold" } },
              grid: { color: "#e2e8f0" },
              ticks: { color: "#475569" },
              min: 0.001,
              max: 15,
            },
          },
          plugins: {
            legend: { display: false },
            annotation: { annotations },
            tooltip: {
              callbacks: {
                title: (items: TooltipItem<"scatter">[]) => {
                  if (items[0].datasetIndex === 0) return "Current sample";
                  // Ép kiểu raw để truy xuất label
                  const raw = items[0].raw as HoverPoint;
                  return raw?.label ?? "";
                },
                label: (item: TooltipItem<"scatter">) => {
                  if (item.datasetIndex === 0) {
                    const raw = item.raw as { x: number; y: number };
                    return `(${raw.x.toFixed(3)}, ${raw.y.toFixed(3)})`;
                  }
                  const raw = item.raw as HoverPoint;
                  return raw?.desc ?? "";
                },
              },
            },
          },
        },
      });
      setError(false);
    } catch {
      setError(true);
    }

    return () => {
      chartRef.current?.destroy();
      chartRef.current = null;
    };
  }, [method, xValue, yValue, pointValid, config, hoverPoints, fault]);

  return (
    <div className="space-y-3">
      {/* Legends phía trên */}
      <div className="flex flex-wrap gap-2">
        {config.legends.map((text, idx) => {
          const color = config.zones[idx]?.color || "#ccc";
          return (
            <div key={text} className="flex items-center gap-1.5 text-xs text-teal-700 font-medium">
              <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: color }} />
              <span>{text}</span>
            </div>
          );
        })}
      </div>

      {/* Biểu đồ */}
      <div className="relative w-full h-[400px] bg-white border border-cream-200 rounded-lg p-4 shadow-sm">
        {error ? (
          <div className="flex items-center justify-center h-full text-red-500 text-sm">Chart rendering error.</div>
        ) : (
          <canvas ref={canvasRef} />
        )}
      </div>

      {/* Thông báo nếu không có điểm */}
      {!pointValid && !error && (
        <p className="text-center text-xs text-teal-500 italic">
          {fault === "NORMAL"
            ? "Sample ratios undefined — diagnosis: NORMAL"
            : "Sample point not available (ratio undefined)"}
        </p>
      )}

      {/* Kết quả chẩn đoán – luôn hiển thị nếu có fault */}
      {fault && (
        <p className="text-center text-sm font-extrabold text-status-critical">
          RESULT: {fault}
        </p>
      )}
    </div>
  );
}