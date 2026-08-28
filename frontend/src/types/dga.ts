// src/types/dga.ts
export type NativeSeverityLabel =
  | "STATUS_1"
  | "STATUS_2"
  | "STATUS_3"
  | "INSUFFICIENT_DATA";

export type RiskStatus =
  | "Normal"
  | "Watch"
  | "High"
  | "Insufficient data";

export type MaintenancePriority =
  | "HIGH_RISK"
  | "WATCH"
  | "NORMAL"
  | "DATA_REVIEW"
  | "STATUS_3"
  | "STATUS_2"
  | "STATUS_1"
  | string;

export type FaultCriticalityClass =
  | "NO_FAULT"
  | "LOWER_URGENCY"
  | "ELEVATED_URGENCY"
  | "HIGH_CONCERN"
  | "CONTEXT_DEPENDENT"
  | "UNKNOWN"
  | string;

export type TrendDirection =
  | "worsening"
  | "stable"
  | "improving";

export interface TopFeature {
  feature: string;
  importance: number;
}

export interface Prediction {
  row_index: number;
  transformer_id: string;

  pred_ensemble: number;

  ieee_status?: number;
  ieee_status_label?: NativeSeverityLabel | string;

  status?: RiskStatus;

  severity: RiskStatus | string;
  severity_label?: NativeSeverityLabel | string;

  maintenance_priority?: MaintenancePriority | string;
  maintenance_rank?: number;

  critical_front?: boolean;
  maintenance_priority_reason?: string;

  critical_evidence_ratio?: number | null;
  continuous_evidence_ratio?: number | null;
  continuous_evidence_basis?: string;

  fault_type: string;
  fault_group?: string;

  fault_criticality_class?: FaultCriticalityClass;
  fault_source?: string;

  fault_confidence?: number | null;
  fault_entropy?: number | null;
  fault_evidence_level?: string;

  reason: string;

  confirmation_required?: boolean;

  anomaly_percentile?: number | null;

  top_features: TopFeature[];
}

export interface DiagnosticVotes {
  keygas_fault?: string;
  iec_fault?: string;
  rogers_fault?: string;
  doernenburg_fault?: string;
  duval_triangle_fault?: string;
  duval_pentagon_fault?: string;
  duval_pentagon_p1_fault?: string;
  duval_pentagon_p2_fault?: string;

  [method: string]: string | undefined;
}

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

  severity_label_text?: string;
  severity_source?: string;
  severity_score_type?: string;
  severity_is_failure_probability?: boolean;
  severity_composite_weighted?: number | null;
  severity_uses_manual_weights?: boolean;
  severity_anomaly_used?: boolean;
  severity_nei_used?: boolean;
  severity_is_not_a_health_score?: boolean;

  ieee_dga_status?: number;
  ieee_dga_status_label?: NativeSeverityLabel | string;
  ieee_dga_status_reason?: string;

  ieee_recommended_action?: string;
  ieee_confirmation_required?: boolean;

  ieee_max_standardized_exceedance?: number;
  ieee_max_status3_standardized_exceedance?: number;

  ieee_standard_trigger_count?: number;

  ieee_table1_max_exceedance_ratio?: number;
  ieee_table2_max_exceedance_ratio?: number;
  ieee_table3_max_exceedance_ratio?: number;
  ieee_table4_max_exceedance_ratio?: number;

  ieee_table1_concentration_ratio_all?: number;
  ieee_table2_concentration_ratio_all?: number;
  ieee_table3_delta_ratio_all?: number;
  ieee_table4_rate_ratio_all?: number;

  ieee_continuous_evidence_ratio?: number;
  ieee_continuous_evidence_basis?: string;
  ieee_continuous_evidence_label?: string;
  ieee_continuous_evidence_is_score?: boolean;
  ieee_continuous_evidence_is_probability?: boolean;
  ieee_continuous_evidence_reference?: string;

  ieee_maintenance_priority_hint?: string;
  ieee_maintenance_priority_hint_reason?: string;

  ieee_table1_exceeding_gases?: string[];
  ieee_table2_exceeding_gases?: string[];
  ieee_table3_exceeding_gases?: string[];
  ieee_table4_exceeding_gases?: string[];

  ieee_delta_available?: boolean;
  ieee_rate_available?: boolean;
  ieee_rate_span_months?: number;

  ieee_o2_n2_ratio?: number;
  ieee_o2_n2_section?: string;

  ieee_age_bucket?: string;
  ieee_transformer_age_years?: number;

  ieee_tdcg_ppm?: number;

  iec_60599_standard?: string;
  iec_60599_ratio_available?: boolean;
  iec_60599_ratio_count?: number;
  iec_60599_ratios?: Record<string, number | null>;
  iec_60599_interpretation_flags?: string[];

  ieee_delta?: Record<string, number>;
  ieee_gas_rate_ppm_per_year?: Record<string, number>;

  consensus_fault?: string;
  consensus_fault_traditional?: string;
  consensus_fault_group?: string;

  final_fault?: string;
  final_fault_group?: string;
  final_fault_source?: string;

  final_fault_conflict?: boolean;
  final_fault_same_coarse_different_fine?: boolean;
  final_fault_conflict_level?: string;

  fault_criticality_class?: FaultCriticalityClass;
  fault_criticality_source?: string;

  diagnostic_confidence?: number;
  diagnostic_coverage?: number;
  diagnostic_agreement_ratio?: number;

  weak_fine_fault?: string;
  weak_fine_fault_group?: string;
  weak_fine_posterior_max?: number;
  weak_fine_entropy?: number;
  weak_fine_lf_active_count?: number;
  weak_fine_lf_coverage?: number;
  weak_fine_is_ABSTAIN?: boolean;

  weak_coarse_fault?: string;
  weak_coarse_fault_group?: string;
  weak_coarse_posterior_max?: number;
  weak_coarse_entropy?: number;
  weak_coarse_lf_active_count?: number;
  weak_coarse_lf_coverage?: number;
  weak_coarse_is_ABSTAIN?: boolean;

  anomaly_percentile?: number;
  anomaly_is_severity_input?: boolean;
  anomaly_interpretation?: string;

  keygas_fault?: string;
  iec_fault?: string;
  rogers_fault?: string;
  doernenburg_fault?: string;
  duval_triangle_fault?: string;
  duval_pentagon_fault?: string;
  duval_pentagon_p1_fault?: string;
  duval_pentagon_p2_fault?: string;

  fault_p1?: string;
  fault_p2?: string;

  student_fault_label?: string;
  student_fault_group?: string;
  student_fault_confidence?: number;
  student_model_name?: string;
  student_training_type?: string;
  student_feature_set?: string;
  student_used_as_fallback?: boolean;

  [key: string]: unknown;
}

export interface RankingBreakdown {
  current_status: number;

  fleet_priority_percent?: number | null;

  maintenance_priority?: MaintenancePriority | string;
  maintenance_rank?: number;

  current_standardized_exceedance?: number | null;
  current_status3_standardized_exceedance?: number | null;

  current_concentration_exceedance_ratio?: number | null;
  current_delta_exceedance_ratio?: number | null;
  current_rate_exceedance_ratio?: number | null;

  current_delta_exceedance?: number;
  current_standard_trigger_count?: number;

  current_continuous_evidence_ratio?: number | null;
  current_continuous_evidence_basis?: string;

  historical_max_continuous_evidence_ratio?: number | null;
  historical_max_standardized_exceedance?: number | null;

  pareto_dominance_count?: number;
  pareto_front?: boolean;

  history_max_status_before_current?: number;
  history_abnormal_record_ratio?: number | null;
  history_critical_record_ratio?: number | null;
  history_fault_recurrence_rate?: number | null;

  history_worsening_transition_ratio?: number | null;
  history_improving_transition_ratio?: number | null;

  trend_slope?: number | null;
  evidence_trend_slope?: number | null;
}

export interface TransformerSummary {
  rank: number;

  maintenance_rank?: number;
  rank_tie?: boolean;
  rank_group_size?: number;

  transformer_id: string;
  latest_sample_day: string;

  loc?: string;
  name?: string;
  ser?: string;
  codetx?: string;
  mfg?: string;

  ieee_status?: number;
  ieee_status_label?: NativeSeverityLabel | string;

  status?: RiskStatus;

  severity: RiskStatus | string;
  severity_label?: NativeSeverityLabel | string;

  maintenance_priority: MaintenancePriority | string;
  maintenance_priority_ordinal?: number;
  maintenance_priority_reason?: string;

  maintenance_priority_score_type?: string;
  maintenance_priority_is_weighted?: boolean;

  critical_front?: boolean;
  critical_front_flag?: boolean;

  critical_rule?: string;
  critical_reference?: string;
  critical_evidence_table?: string | null;
  critical_evidence_gas?: string | null;
  critical_evidence_ratio?: number | null;
  critical_evidence_scope?: string;

  fault_type: string;
  fault_group?: string;

  fault_criticality_class?: FaultCriticalityClass;
  fault_criticality_source?: string;

  current_fault_posterior_max?: number | null;
  current_fault_entropy?: number | null;
  diagnostic_agreement_ratio?: number | null;

  trend?: TrendDirection;

  priority_score?: number | null;
  priority_label?: MaintenancePriority | string;

  recommended_action: string;
  reason: string;

  current_standardized_exceedance?: number | null;
  current_concentration_exceedance_ratio?: number | null;
  current_delta_exceedance_ratio?: number | null;
  current_rate_exceedance_ratio?: number | null;

  current_status3_standardized_exceedance?: number | null;

  current_continuous_evidence_ratio?: number | null;
  current_continuous_evidence_basis?: string;

  historical_max_continuous_evidence_ratio?: number | null;
  historical_max_standardized_exceedance?: number | null;

  current_delta_exceedance?: number;
  current_standard_trigger_count?: number;

  history_mean_status_before_current?: number | null;
  history_status_slope_per_year?: number | null;
  history_status_slope_basis?: string;

  history_max_status_before_current?: number;
  history_record_count?: number;
  history_span_days?: number;

  history_nonzero_status_record_count?: number;
  history_abnormal_record_count?: number;
  history_critical_record_count?: number;

  history_abnormal_record_ratio?: number | null;
  history_critical_record_ratio?: number | null;

  history_fault_occurrence_count?: number;
  history_dominant_fault?: string;
  history_fault_entropy?: number | null;
  history_dominant_fault_count?: number;

  history_recurrent_fault_fraction?: number | null;
  history_current_fault_recurrence_fraction?: number | null;
  history_fault_persistence?: number;

  history_worsening_transition_count?: number;
  history_improving_transition_count?: number;
  history_stable_transition_count?: number;
  history_observed_transition_count?: number;

  history_worsening_transition_ratio?: number | null;
  history_improving_transition_ratio?: number | null;

  history_has_observed_trend?: boolean;

  single_record_transformer?: boolean;

  history_data_sufficiency_level?: number;

  pareto_dominance_count?: number;
  pareto_front?: boolean;

  severity_evidence_vector?: string;

  maintenance_priority_rank_percentile?: number | null;

  priority_score_type?: string;
  ranking_policy?: string;

  ranking_is_weighted?: boolean;
  ranking_is_health_score?: boolean;

  transformer_overall_severity_level?: number;
  transformer_overall_severity_label?: NativeSeverityLabel | string;
  transformer_overall_severity_score?: number;
  overall_score_type?: string;
  overall_score_formula?: string;
  overall_score_is_weighted?: boolean;
  overall_score_is_failure_probability?: boolean;
  overall_score_history_is_bounded?: boolean;

  features: Record<string, unknown>;

  ranking_breakdown?: RankingBreakdown;
}

export interface TimeseriesPoint {
  "Sample Day": string;

  H2: number;
  C2H2: number;
  TDCG: number;

  pred_ensemble: number;

  ieee_status?: number;
  ieee_status_label?: NativeSeverityLabel | string;

  status?: RiskStatus;

  fault_type: string;
  fault_group?: string;

  fault_criticality_class?: FaultCriticalityClass;

  severity: NativeSeverityLabel | string;

  maintenance_priority?: MaintenancePriority | string;

  recommended_action?: string;

  critical_front?: boolean;
  critical_evidence_ratio?: number | null;

  continuous_evidence_ratio?: number | null;
  continuous_evidence_basis?: string;

  continuous_evidence_is_score?: boolean;
  continuous_evidence_reference?: string;

  table2_concentration_ratio?: number | null;
  table3_delta_ratio?: number | null;
  table4_rate_ratio?: number | null;

  confirmation_required?: boolean;
}

export interface DatasetSummary {
  total_transformers: number;
  total_rows: number;

  severity_status_1?: number;
  severity_status_2?: number;
  severity_status_3?: number;
  severity_insufficient_data?: number;

  high_risk_transformer_count?: number;
  watch_transformer_count?: number;
  normal_transformer_count?: number;

  maintenance_priority_counts?: Partial<
    Record<MaintenancePriority, number>
  > &
    Record<string, number>;

  critical_transformer_count?: number;

  critical_queue_top20?: Array<Record<string, unknown>>;
  maintenance_queue_top20?: Array<Record<string, unknown>>;

  first_priority_transformer_id?: string | null;
  first_priority_rank?: number | null;

  critical_rule?: string;
  critical_reference?: string;

  fault_criticality_context_counts?: Record<string, number>;
  fault_criticality_source?: string;

  traditional_abstain_rows?: number;
  student_fallback_rows?: number;
  student_traditional_physical_conflicts?: number;
}

export interface StudentTraditionalComparisonRow {
  transformer_id: string;

  n_samples: number;
  n_joint_active: number;

  coarse_agreement_rate?: number;
  fine_agreement_rate?: number;

  traditional_abstain_count: number;
  student_abstain_count: number;

  student_used_as_fallback_count: number;
  physical_conflict_count: number;
}

export interface ChatContextPayload {
  transformer_summary: TransformerSummary[];
  dataset_summary?: DatasetSummary;
}

export interface DgaPayload {
  predictions: Prediction[];

  rows: DgaRow[];
  preview_rows: DgaRow[];

  transformer_summary: TransformerSummary[];

  transformer_timeseries: Record<
    string,
    TimeseriesPoint[]
  >;

  dataset_summary: DatasetSummary;

  student_traditional_comparison?: StudentTraditionalComparisonRow[];

  chat_context_payload: ChatContextPayload;

  pipeline?: {
    elapsed_seconds?: number;
    timings?: Record<string, number>;
    timings_by_cost?: Record<string, number>;
    slowest_stage?: string | null;
    slowest_stage_seconds?: number | null;
    [key: string]: unknown;
  };
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