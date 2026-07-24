import type { NativeSeverityLabel, RiskStatus, TransformerSummary } from "@/types/dga";

/**
 * Mirrors config.py `SEVERITY_CLASS_BOUNDARIES = [4, 8, 13]` used by
 * severity.severity_class_from_score() on the backend. The backend collapses
 * these 4 native buckets down to 3 UI strings (Severe/Moderate/Low) for the
 * legacy Streamlit view; this dashboard instead reproduces the full 4-tier
 * Normal/Watch/High/Critical status from UI-SPEC-DGA-DASHBOARD-01 directly
 * from the same raw `severity_score`, so no backend change is required.
 */
const SEVERITY_BOUNDARIES = [4, 8, 13] as const;

export function classifyScore(score: number): NativeSeverityLabel {
  if (score < SEVERITY_BOUNDARIES[0]) return "NORMAL";
  if (score < SEVERITY_BOUNDARIES[1]) return "WATCHLIST";
  if (score < SEVERITY_BOUNDARIES[2]) return "WARNING";
  return "CRITICAL";
}

export function nativeToStatus(label: NativeSeverityLabel): RiskStatus {
  switch (label) {
    case "NORMAL":
      return "Normal";
    case "WATCHLIST":
      return "Watch";
    case "WARNING":
      return "High";
    case "CRITICAL":
      return "Critical";
  }
}

export function scoreToStatus(score: number): RiskStatus {
  return nativeToStatus(classifyScore(score));
}

/**
 * Normalizes the raw severity_score onto the 0-100 risk scale used in
 * UI-SPEC-DGA-DASHBOARD-01 (0-30 Normal, 31-60 Watch, 61-89 High, 90-100
 * Critical), anchoring on the same boundaries the backend classifies with
 * so a transformer's risk-bar percentage always agrees with its status pill.
 */
export function scoreToRisk(score: number): number {
  const anchors: [number, number][] = [
    [0, 0],
    [SEVERITY_BOUNDARIES[0], 30],
    [SEVERITY_BOUNDARIES[1], 60],
    [SEVERITY_BOUNDARIES[2], 90],
    [SEVERITY_BOUNDARIES[2] * 2, 100],
  ];
  const clamped = Math.max(0, score);
  for (let i = 1; i < anchors.length; i++) {
    const [x0, y0] = anchors[i - 1];
    const [x1, y1] = anchors[i];
    if (clamped <= x1) {
      const t = x1 === x0 ? 1 : (clamped - x0) / (x1 - x0);
      return Math.round(y0 + t * (y1 - y0));
    }
  }
  return 100;
}

export const STATUS_ORDER: RiskStatus[] = ["Normal", "Watch", "High", "Critical"];

// Raw hex twins of the --color-status-* CSS variables (globals.css) — SVG
// chart libraries (recharts) need real color values for fill/stroke, not
// Tailwind class names, so this can't just reuse STATUS_STYLES below.
export const STATUS_HEX: Record<RiskStatus, string> = {
  Normal: "#1f7a4d",
  Watch: "#a16a07",
  High: "#9a4a1f",
  Critical: "#c62828",
};

export const STATUS_STYLES: Record<
  RiskStatus,
  {
    text: string;
    bg: string;
    border: string;
    bar: string;
    dot: string;
    topBorder: string;
    leftBorder: string;
  }
> = {
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
  Critical: {
    text: "text-status-critical",
    bg: "bg-status-critical-soft",
    border: "border-status-critical-border",
    bar: "bg-status-critical",
    dot: "bg-status-critical",
    topBorder: "border-t-status-critical",
    leftBorder: "border-l-status-critical",
  },
};

export const STATUS_ICON_LABEL: Record<RiskStatus, string> = {
  Normal: "✓",
  Watch: "◉",
  High: "▲",
  Critical: "⊘",
};

export function statusFromSummary(summary: TransformerSummary): RiskStatus {
  return scoreToStatus(summary.latest_score);
}

export function formatTrend(trend: string): { label: string; icon: string } {
  switch (trend) {
    case "worsening":
      return { label: "Worsening", icon: "↑" };
    case "improving":
      return { label: "Improving", icon: "↓" };
    default:
      return { label: "Stable", icon: "→" };
  }
}
