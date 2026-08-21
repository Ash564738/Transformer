// src/lib/severity.ts
import type {
  MaintenancePriority,
  NativeSeverityLabel,
  RiskStatus,
  TransformerSummary,
} from "@/types/dga";

export type StatusStyle = {
  text: string;
  bg: string;
  border: string;
  bar: string;
  dot: string;
  topBorder: string;
  leftBorder: string;
};

export type MaintenancePriorityStyle = {
  text: string;
  bg: string;
  border: string;
};

export const STATUS_ORDER: RiskStatus[] = [
  "Normal",
  "Watch",
  "High",
  "Insufficient data",
];

export const MAINTENANCE_PRIORITY_ORDER: MaintenancePriority[] = [
  "CRITICAL",
  "HIGH_RISK",
  "WATCH",
  "NORMAL",
  "DATA_REVIEW",
];

export const STATUS_HEX: Record<RiskStatus, string> = {
  Normal: "#1f7a4d",
  Watch: "#a16a07",
  High: "#9a4a1f",
  "Insufficient data": "#718096",
};

export const MAINTENANCE_PRIORITY_HEX: Record<MaintenancePriority, string> = {
  CRITICAL: "#b42318",
  HIGH_RISK: "#9a4a1f",
  WATCH: "#a16a07",
  NORMAL: "#1f7a4d",
  DATA_REVIEW: "#718096",
};

export const STATUS_STYLES: Record<RiskStatus, StatusStyle> = {
  Normal: {
    text: "text-status-normal",
    bg: "bg-status-normal-soft",
    border: "border-status-normal-border",
    bar: "bg-status-normal",
    dot: "bg-status-normal",
    topBorder: "border-t-status-normal",
    leftBorder: "border-l-status-normal",
  },
  Watch: {
    text: "text-status-watch",
    bg: "bg-status-watch-soft",
    border: "border-status-watch-border",
    bar: "bg-status-watch",
    dot: "bg-status-watch",
    topBorder: "border-t-status-watch",
    leftBorder: "border-l-status-watch",
  },
  High: {
    text: "text-status-high",
    bg: "bg-status-high-soft",
    border: "border-status-high-border",
    bar: "bg-status-high",
    dot: "bg-status-high",
    topBorder: "border-t-status-high",
    leftBorder: "border-l-status-high",
  },
  "Insufficient data": {
    text: "text-slate-500",
    bg: "bg-slate-50",
    border: "border-slate-300",
    bar: "bg-slate-400",
    dot: "bg-slate-400",
    topBorder: "border-t-slate-400",
    leftBorder: "border-l-slate-400",
  },
};

export const MAINTENANCE_PRIORITY_STYLES: Record<
  MaintenancePriority,
  MaintenancePriorityStyle
> = {
  CRITICAL: {
    text: "text-red-800",
    bg: "bg-red-50",
    border: "border-red-300",
  },
  HIGH_RISK: {
    text: "text-orange-800",
    bg: "bg-orange-50",
    border: "border-orange-300",
  },
  WATCH: {
    text: "text-amber-800",
    bg: "bg-amber-50",
    border: "border-amber-300",
  },
  NORMAL: {
    text: "text-emerald-800",
    bg: "bg-emerald-50",
    border: "border-emerald-300",
  },
  DATA_REVIEW: {
    text: "text-slate-600",
    bg: "bg-slate-50",
    border: "border-slate-300",
  },
};

export const STATUS_ICON_LABEL: Record<RiskStatus, string> = {
  Normal: "✓",
  Watch: "◉",
  High: "▲",
  "Insufficient data": "—",
};

export function ieeeStatusToRiskStatus(
  ieeeStatus: number | null | undefined
): RiskStatus {
  if (ieeeStatus == null || !Number.isFinite(Number(ieeeStatus))) {
    return "Insufficient data";
  }

  switch (Number(ieeeStatus)) {
    case 1:
      return "Normal";
    case 2:
      return "Watch";
    case 3:
      return "High";
    default:
      return "Insufficient data";
  }
}

export function nativeToStatus(label: NativeSeverityLabel): RiskStatus {
  switch (label) {
    case "STATUS_1":
      return "Normal";
    case "STATUS_2":
      return "Watch";
    case "STATUS_3":
      return "High";
    default:
      return "Insufficient data";
  }
}

/** Backend has two independent axes. Status always comes from IEEE status. */
export function statusFromSummary(summary: TransformerSummary): RiskStatus {
  return ieeeStatusToRiskStatus(summary.ieee_status);
}

/**
 * Backwards-compatible fallback for legacy UI code. Do not use this for
 * severity distribution: maintenance priority and IEEE status are separate.
 */
export function maintenancePriorityToRiskStatus(
  priority: MaintenancePriority | string | null | undefined,
  ieeeStatus?: number | null
): RiskStatus {
  switch (String(priority ?? "").toUpperCase()) {
    case "CRITICAL":
    case "HIGH_RISK":
      return "High";
    case "WATCH":
      return "Watch";
    case "NORMAL":
      return "Normal";
    case "DATA_REVIEW":
      return "Insufficient data";
    default:
      return ieeeStatusToRiskStatus(ieeeStatus);
  }
}

export function statusFromValues(
  ieeeStatus?: number | null,
  label?: NativeSeverityLabel | string
): RiskStatus {
  if (ieeeStatus != null) return ieeeStatusToRiskStatus(ieeeStatus);
  if (label === "STATUS_1" || label === "STATUS_2" || label === "STATUS_3") {
    return nativeToStatus(label);
  }
  return "Insufficient data";
}

export function normalizeMaintenancePriority(
  priority: MaintenancePriority | string | null | undefined
): MaintenancePriority {
  switch (String(priority ?? "").toUpperCase()) {
    case "CRITICAL":
      return "CRITICAL";
    case "HIGH_RISK":
      return "HIGH_RISK";
    case "WATCH":
      return "WATCH";
    case "NORMAL":
      return "NORMAL";
    default:
      return "DATA_REVIEW";
  }
}

export function maintenancePriorityFromSummary(
  summary: TransformerSummary
): MaintenancePriority {
  return normalizeMaintenancePriority(summary.maintenance_priority);
}

export function isCriticalSummary(summary: TransformerSummary): boolean {
  return maintenancePriorityFromSummary(summary) === "CRITICAL" || summary.critical_front === true;
}

export function maintenancePriorityLabel(
  priority: MaintenancePriority | string | null | undefined
): string {
  switch (normalizeMaintenancePriority(priority)) {
    case "CRITICAL":
      return "Critical";
    case "HIGH_RISK":
      return "High Risk";
    case "WATCH":
      return "Watch";
    case "NORMAL":
      return "Normal";
    default:
      return "Data Review";
  }
}

export function formatTrend(
  trend?: string
): { label: string; icon: string } {
  switch (trend) {
    case "worsening":
      return { label: "Worsening", icon: "↑" };
    case "improving":
      return { label: "Improving", icon: "↓" };
    default:
      return { label: "Stable", icon: "→" };
  }
}