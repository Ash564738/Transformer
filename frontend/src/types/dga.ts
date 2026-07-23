// Types mirror the payload produced by inference_service.create_payload()
// in the Flask backend (app.py -> /predict). Field names match the raw
// pandas columns the pipeline emits (dga/*.py, consensus.py, severity.py,
// ranking.py) so the frontend can render diagnostic detail without
// re-deriving anything the backend already computed.

export type UiSeverity = "Severe" | "Moderate" | "Low";

/** Native 4-tier severity bucket computed by severity.py (SEVERITY_CLASS_BOUNDARIES). */
export type NativeSeverityLabel = "NORMAL" | "WATCHLIST" | "WARNING" | "CRITICAL";

/** Fixed 4-tier status shown in the UI, matching UI-SPEC-DGA-DASHBOARD-01. */
export type RiskStatus = "Normal" | "Watch" | "High" | "Critical";

export type TrendDirection = "worsening" | "stable" | "improving";

export interface TopFeature {
  feature: string;
  importance: number;
}

export interface Prediction {
  row_index: number;
  transformer_id: string;
  pred_ensemble: number;
  severity: UiSeverity;
  fault_type: string;
  reason: string;
  top_features: TopFeature[];
}

export interface DiagnosticVotes {
  keygas_fault?: string;
  iec_fault?: string;
  rogers_fault?: string;
  doernenburg_fault?: string;
  duval_triangle_fault?: string;
  duval_pentagon_p1_fault?: string;
  duval_pentagon_p2_fault?: string;
  [method: string]: string | undefined;
}

/** Raw feature-engineered row — one DGA sample. Superset of known fields. */
export interface DgaRow {
  transformer_id: string;
  sample_day: string;
  loc?: string;
  name?: string;
  ser?: string;
  codetx?: string;
  mfg?: string;

  h2?: number;
  ch4?: number;
  c2h6?: number;
  c2h4?: number;
  c2h2?: number;
  co?: number;
  co2?: number;
  tdcg?: number;
  o2?: number;
  n2?: number;
  water?: number;
  temp?: number;

  severity_score?: number;
  severity_label?: NativeSeverityLabel;
  consensus_fault?: string;
  mixed_components?: string[];
  diagnostic_confidence?: number;
  diagnostic_votes?: DiagnosticVotes;

  keygas_fault?: string;
  iec_fault?: string;
  rogers_fault?: string;
  doernenburg_fault?: string;
  duval_triangle_fault?: string;
  fault_p1?: string;
  duval_pentagon_fault?: string;

  iec_r1_c2h2_c2h4?: number;
  iec_r2_ch4_h2?: number;
  iec_r3_c2h4_c2h6?: number;
  r1_ch4_h2?: number;
  r2_c2h2_c2h4?: number;
  r3_c2h4_c2h6?: number;
  dr_r1_ch4_h2?: number;
  dr_r2_c2h2_c2h4?: number;
  dr_r3_c2h2_ch4?: number;
  dr_r4_c2h6_c2h2?: number;
  ratio_co2_co?: number;

  h2_rate_per_day?: number;
  c2h2_rate_per_day?: number;
  tdcg_rate_per_day?: number;

  t_x?: number;
  t_y?: number;
  p_x?: number;
  p_y?: number;

  [key: string]: unknown;
}

/** Composition of the fleet-wide ranking score (backend/ranking.py final_score),
 * distinct from the per-record severity_score breakdown: this explains why a
 * transformer ranks where it does relative to the rest of the fleet — current
 * severity weighted most heavily, but historical (EWM) severity, trend,
 * critical-event history, and diagnostic confidence all contribute. */
export interface RankingBreakdown {
  weights: { current: number; history: number; trend: number; critical_history: number; confidence: number };
  current_severity: number;
  current_contribution: number;
  historical_severity: number;
  historical_contribution: number;
  trend_bonus: number;
  trend_contribution: number;
  critical_history_count: number;
  critical_history_contribution: number;
  diagnostic_confidence: number;
  confidence_contribution: number;
  persistence_bonus_factor: number;
  fault_persistence: number;
  days_since_last_critical: number | null;
}

export interface TransformerSummary {
  rank: number;
  transformer_id: string;
  latest_sample_day: string;
  latest_score: number;
  severity: UiSeverity;
  fault_type: string;
  trend: TrendDirection;
  priority_score: number;
  priority_label: UiSeverity;
  recommended_action: string;
  reason: string;
  features: Record<string, unknown>;
  loc?: string;
  name?: string;
  ranking_breakdown?: RankingBreakdown;
}

export interface TimeseriesPoint {
  "Sample Day": string;
  H2: number;
  C2H2: number;
  TCG: number;
  pred_ensemble: number;
  fault_type: string;
  severity: NativeSeverityLabel | string;
}

export interface DatasetSummary {
  total_transformers: number;
  total_rows: number;
}

export interface ChatContextPayload {
  transformer_summary: TransformerSummary[];
  dataset_summary: DatasetSummary;
}

export interface DgaPayload {
  predictions: Prediction[];
  rows: DgaRow[];
  preview_rows: DgaRow[];
  transformer_summary: TransformerSummary[];
  transformer_timeseries: Record<string, TimeseriesPoint[]>;
  dataset_summary: DatasetSummary;
  chat_context_payload: ChatContextPayload;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sourceChip?: string;
  createdAt: number;
}

export interface FieldInspectionConfirmation {
  id: string;
  transformer_id: string;
  confirmed_by: string;
  confirmed_fault_type: string;
  notes: string;
  confirmed_at: string;
}
