import type { DgaPayload, DgaRow, RiskStatus, TransformerSummary } from "@/types/dga";

export const GAS_LABELS: Record<string, string> = {
  h2: "Hydrogen (H2)",
  ch4: "Methane (CH4)",
  c2h6: "Ethane (C2H6)",
  c2h4: "Ethylene (C2H4)",
  c2h2: "Acetylene (C2H2)",
};

// CO/CO2 are excluded: they track cellulose ageing, not the fault-indicating
// hydrocarbon gas the Overview table's "Top Gas" column is meant to surface.
const TOP_GAS_KEYS = ["h2", "ch4", "c2h6", "c2h4", "c2h2"] as const;

export function latestRowFor(payload: DgaPayload | null, transformerId: string): DgaRow | undefined {
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

export function rowsForTransformer(payload: DgaPayload | null, transformerId: string): DgaRow[] {
  if (!payload) return [];
  return payload.rows
    .filter((r) => r.transformer_id === transformerId)
    .sort((a, b) => new Date(a.sample_day).getTime() - new Date(b.sample_day).getTime());
}

export function topGasLabel(row: DgaRow | undefined, status?: RiskStatus): string {
  if (!row || status === "Normal") return "N/A";
  let best: (typeof TOP_GAS_KEYS)[number] | null = null;
  let bestVal = 0;
  for (const key of TOP_GAS_KEYS) {
    const v = Number(row[key] ?? 0);
    if (v > bestVal) {
      bestVal = v;
      best = key;
    }
  }
  if (!best || bestVal <= 0) return "N/A";
  return GAS_LABELS[best];
}

export function stationOf(summary: TransformerSummary): string {
  return summary.loc || summary.name || "Unassigned";
}

export function getStations(payload: DgaPayload | null): string[] {
  if (!payload) return [];
  const set = new Set<string>();
  for (const s of payload.transformer_summary) set.add(stationOf(s));
  return Array.from(set).sort();
}
