// src/lib/dga-methods.ts
export const FAULT_EXPLANATIONS: Record<string, string> = {
  NORMAL: "Normal",
  PD: "Partial discharge",
  D1: "Low-energy electrical discharge",
  D2: "High-energy electrical discharge (arcing)",
  DT: "Mixed discharge + thermal fault",
  T1: "Thermal fault < 300°C",
  T2: "Thermal fault 300°C–700°C",
  T3: "Thermal fault > 700°C",
  T1_T2: "Thermal fault spanning T1/T2 classification",
  T3_H: "Thermal fault > 700°C (oil only)",
  C: "Carbonization of paper insulation",
  O: "Overheating < 250°C",
  S: "Stray gassing",
  THERMAL_OIL: "Thermal fault (oil)",
  THERMAL_CELLULOSE: "Thermal fault (cellulose / CO dominant)",
  MIXED: "Mixed fault (methods disagree)",
  ABSTAIN: "ABSTAIN diagnosis",
};

const FAULT_GROUPS: Record<string, string> = {
  NORMAL: "NORMAL",
  PD: "DISCHARGE",
  D1: "DISCHARGE",
  D2: "DISCHARGE",
  DT: "MIXED",
  T1: "THERMAL",
  T2: "THERMAL",
  T3: "THERMAL",
  T1_T2: "THERMAL",
  T3_H: "THERMAL",
  THERMAL_OIL: "THERMAL",
  THERMAL_CELLULOSE: "CELLULOSE",
  C: "CELLULOSE",
  O: "THERMAL",
  S: "STRAY_GASSING",
  ABSTAIN: "ABSTAIN",
  MIXED: "MIXED",
};

export function unifyFault(label?: string): string {
  if (!label) return "ABSTAIN";
  const key = label.trim().toUpperCase().replace("-", "_");
  return FAULT_GROUPS[key] ?? "ABSTAIN";
}

export function faultExplanation(label?: string): string {
  if (!label) return "Unknown fault";
  const key = label.trim().toUpperCase();
  return FAULT_EXPLANATIONS[key] ?? label;
}