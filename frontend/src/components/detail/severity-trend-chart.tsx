// src/components/detail/severity-trend-chart.tsx
"use client";

import { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DgaRow, RiskStatus } from "@/types/dga";
import { ieeeStatusToRiskStatus, STATUS_HEX, STATUS_STYLES } from "@/lib/severity";
import { formatDate, formatNumber } from "@/lib/utils";

interface TrendPoint {
  day: string;
  risk: number | null;
  evidence: number | null;
  evidenceBasis: string;
  status: RiskStatus;
  fault: string;
}

function StatusDot(props: {
  cx?: number;
  cy?: number;
  payload?: TrendPoint;
  singlePoint?: boolean;
}) {
  const { cx, cy, payload, singlePoint } = props;
  if (cx == null || cy == null || !payload) return null;

  return (
    <circle
      cx={cx}
      cy={cy}
      r={singlePoint ? 7 : 4}
      fill={STATUS_HEX[payload.status]}
      stroke="white"
      strokeWidth={singlePoint ? 2.5 : 1.5}
    />
  );
}

const STATUS_TICK_FORMATTER = (value: number) => {
  switch (value) {
    case 1:
      return "S1 · Normal";
    case 2:
      return "S2 · Watch";
    case 3:
      return "S3 · High";
    default:
      return "—";
  }
};

function evidenceBasisLabel(value: string): string {
  switch (value) {
    case "TABLE_2_95TH_CONCENTRATION":
      return "Table 2 concentration";
    case "TABLE_3_95TH_DELTA":
      return "Table 3 positive delta";
    case "TABLE_4_95TH_RATE":
      return "Table 4 positive rate";
    case "TABLE_1_90TH_CONCENTRATION":
      return "Table 1 concentration";
    default:
      return value || "No continuous evidence";
  }
}

export function SeverityTrendChart({ rows }: { rows: DgaRow[] }) {
  const data = useMemo<TrendPoint[]>(() => {
    return rows
      .map((row) => {
        const numericStatus = Number(row.ieee_dga_status);
        const status = ieeeStatusToRiskStatus(numericStatus);
        const evidence = Number(row.ieee_continuous_evidence_ratio);

        return {
          day: row.sample_day,
          risk: [1, 2, 3].includes(numericStatus) ? numericStatus : null,
          evidence: Number.isFinite(evidence) ? evidence : null,
          evidenceBasis: String(
            row.ieee_continuous_evidence_basis ?? "NO_CONTINUOUS_EVIDENCE"
          ),
          status,
          fault: String(
            row.final_fault ?? row.consensus_fault ?? "ABSTAIN"
          ),
        };
      })
      .filter((point) => point.risk != null || point.evidence != null)
      .sort(
        (a, b) =>
          new Date(a.day).getTime() - new Date(b.day).getTime()
      );
  }, [rows]);

  if (data.length === 0) {
    return (
      <p className="py-14 text-center text-sm text-teal-400">
        No DGA condition history available.
      </p>
    );
  }

  const singlePoint = data.length === 1;

  return (
    <div>
      <div className="mb-3 rounded-lg border border-teal-100 bg-teal-50 px-3 py-2 text-[11px] leading-5 text-teal-700">
        IEEE Status is the discrete condition class (S1/S2/S3). The orange
        trajectory is the latest-sample continuous DGA evidence ratio derived
        directly from the applicable IEEE threshold tables. It is not a
        weighted severity score, failure probability, or health index.
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart
          data={data}
          margin={{ top: 8, right: 20, left: 0, bottom: 0 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#e8e5d9"
            vertical={false}
          />

          <XAxis
            dataKey="day"
            tickFormatter={(value) => formatDate(String(value))}
            tick={{ fontSize: 11, fill: "#4f8f83" }}
            axisLine={{ stroke: "#d9d5c4" }}
            tickLine={false}
            minTickGap={18}
          />

          <YAxis
            yAxisId="status"
            type="number"
            domain={[1, 3]}
            ticks={[1, 2, 3]}
            tickFormatter={STATUS_TICK_FORMATTER}
            tick={{ fontSize: 11, fill: "#4f8f83" }}
            axisLine={false}
            tickLine={false}
            width={88}
            allowDecimals={false}
          />

          <YAxis
            yAxisId="evidence"
            orientation="right"
            type="number"
            domain={[0, "auto"]}
            tick={{ fontSize: 11, fill: "#c96f28" }}
            axisLine={false}
            tickLine={false}
            width={62}
            tickFormatter={(value) => `${formatNumber(Number(value), 1)}×`}
          />

          <Tooltip
            labelFormatter={(value) => formatDate(String(value))}
            contentStyle={{
              borderRadius: 10,
              borderColor: "#d9d5c4",
              fontSize: 12,
            }}
            formatter={(value, name, item) => {
              if (name === "IEEE Status") {
                return [`S${Number(value)}`, "IEEE Status"];
              }

              if (name === "DGA evidence") {
                const payload = item?.payload as TrendPoint | undefined;
                const basis = evidenceBasisLabel(payload?.evidenceBasis ?? "");
                return [
                  `${formatNumber(Number(value), 2)}×`,
                  `DGA evidence · ${basis}`,
                ];
              }

              return [value, name];
            }}
          />

          <Legend wrapperStyle={{ fontSize: 11 }} />

          <Line
            yAxisId="status"
            type="monotone"
            dataKey="risk"
            stroke="#184843"
            strokeWidth={2.5}
            dot={(props: {
              cx?: number;
              cy?: number;
              payload?: TrendPoint;
              index?: number;
            }) => (
              <StatusDot
                key={`status-${props.index}`}
                cx={props.cx}
                cy={props.cy}
                payload={props.payload}
                singlePoint={singlePoint}
              />
            )}
            activeDot={{ r: singlePoint ? 9 : 6, stroke: "white", strokeWidth: 2 }}
            name="IEEE Status"
            connectNulls
            isAnimationActive={!singlePoint}
          />

          <Line
            yAxisId="evidence"
            type="monotone"
            dataKey="evidence"
            stroke="#c96f28"
            strokeWidth={2.5}
            dot={{ r: singlePoint ? 6 : 3 }}
            activeDot={{ r: singlePoint ? 8 : 5 }}
            name="DGA evidence"
            connectNulls
            isAnimationActive={!singlePoint}
          />
        </LineChart>
      </ResponsiveContainer>

      {singlePoint && (
        <p className="mt-1 text-center text-[11px] text-teal-400">
          Only one record is available; no temporal trend can be inferred.
        </p>
      )}

      <div className="mt-2 flex flex-wrap justify-center gap-3 text-[11px] text-teal-500">
        {( ["Normal", "Watch", "High", "Insufficient data"] as const).map(
          (status) => (
            <span key={status} className="flex items-center gap-1">
              <span
                className={`h-2 w-2 rounded-full ${STATUS_STYLES[status].dot}`}
              />
              {status}
            </span>
          )
        )}
        <span className="flex items-center gap-1">
          <span className="h-2 w-5 rounded-full bg-[#c96f28]" />
          Continuous DGA evidence
        </span>
      </div>
    </div>
  );
}