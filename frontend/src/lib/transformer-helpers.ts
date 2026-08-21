// src/lib/transformer-helpers.ts
import type {
  DgaPayload,
  DgaRow,
  MaintenancePriority,
  RiskStatus,
  TransformerSummary,
} from "@/types/dga";

export const GAS_LABELS: Record<string, string> = {
  h2: "Hydrogen (H₂)",
  ch4: "Methane (CH₄)",
  c2h6: "Ethane (C₂H₆)",
  c2h4: "Ethylene (C₂H₄)",
  c2h2: "Acetylene (C₂H₂)",
  co: "Carbon monoxide (CO)",
  co2: "Carbon dioxide (CO₂)",
};

const TOP_GAS_KEYS = ["h2", "ch4", "c2h6", "c2h4", "c2h2"] as const;

export function latestRowFor(
  payload: DgaPayload | null,
  transformerId: string
): DgaRow | undefined {
  if (!payload) return undefined;
  let best: DgaRow | undefined;
  for (const row of payload.rows) {
    if (row.transformer_id !== transformerId) continue;
    if (!best || new Date(row.sample_day).getTime() > new Date(best.sample_day).getTime()) {
      best = row;
    }
  }
  return best;
}

export function rowsForTransformer(
  payload: DgaPayload | null,
  transformerId: string
): DgaRow[] {
  if (!payload) return [];
  return payload.rows
    .filter((r) => r.transformer_id === transformerId)
    .sort(
      (a, b) =>
        new Date(a.sample_day).getTime() - new Date(b.sample_day).getTime()
    );
}

export function topGasLabel(row: DgaRow | undefined, status?: RiskStatus): string {
  if (!row || status === "Normal" || status === "Insufficient data") return "N/A";
  let best: (typeof TOP_GAS_KEYS)[number] | null = null;
  let bestVal = 0;
  for (const key of TOP_GAS_KEYS) {
    const value = Number(row[key] ?? 0);
    if (Number.isFinite(value) && value > bestVal) {
      bestVal = value;
      best = key;
    }
  }
  if (!best) return "N/A";
  return GAS_LABELS[best];
}

export function stationOf(summary: TransformerSummary): string {
  return summary.loc || summary.name || "Unassigned";
}

export function getStations(payload: DgaPayload | null): string[] {
  if (!payload) return [];
  return Array.from(new Set(payload.transformer_summary.map(stationOf))).sort();
}

export function isMaintenanceCritical(
  summary?: TransformerSummary | null
): boolean {
  return !!summary &&
    (summary.maintenance_priority === "CRITICAL" || summary.critical_front === true);
}

export function maintenancePriorityLabel(
  priority?: MaintenancePriority | string
): string {
  switch (priority) {
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