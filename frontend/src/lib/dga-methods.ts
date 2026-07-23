// Small shared glossary/taxonomy for DGA fault codes. The actual diagnostic
// LOGIC (Duval Triangle/Pentagon geometry, Rogers/IEC/Doernenburg/Key Gas
// thresholds) lives only in the Python backend (backend/dga/*.py) — this
// frontend does not reimplement it. Duval Triangle/Pentagon diagrams are
// rendered server-side (backend/app.py `/chart/*` routes reuse the existing
// matplotlib plotting functions) and embedded here as <img>; ratio numbers
// and fault codes shown elsewhere come directly from the row data the
// backend already computed.

export const FAULT_EXPLANATIONS: Record<string, string> = {
  NORMAL: "Normal",
  PD: "Partial discharge",
  D1: "Low-energy electrical discharge",
  D2: "High-energy electrical discharge (arcing)",
  DT: "Mixed discharge + thermal fault",
  T1: "Thermal fault < 300°C",
  T2: "Thermal fault 300°C–700°C",
  T3: "Thermal fault > 700°C",
  T3_H: "Thermal fault > 700°C (oil only)",
  C: "Carbonization of paper insulation",
  O: "Overheating < 250°C",
  S: "Stray gassing",
  THERMAL_OIL: "Thermal fault (oil)",
  THERMAL_CELLULOSE: "Thermal fault (cellulose / CO dominant)",
  MIXED: "Mixed fault (methods disagree)",
  UNCERTAIN: "Uncertain diagnosis",
};

// Mirrors config.py's FAULT_GROUPS — a static label taxonomy (not a
// diagnostic algorithm) used only to decide whether a method's vote agrees
// with the transformer's consensus fault group.
const FAULT_GROUPS: Record<string, string> = {
  NORMAL: "NORMAL",
  PD: "DISCHARGE",
  D1: "DISCHARGE",
  D2: "DISCHARGE",
  DT: "MIXED",
  T1: "THERMAL",
  T2: "THERMAL",
  T3: "THERMAL",
  T3_H: "THERMAL",
  THERMAL_OIL: "THERMAL",
  THERMAL_CELLULOSE: "CELLULOSE",
  C: "CELLULOSE",
  O: "THERMAL",
  S: "STRAY_GASSING",
  UNCERTAIN: "UNCERTAIN",
  MIXED: "MIXED",
};

/** Mirrors consensus.py's unify_fault(): maps a specific fault code to its broad group. */
export function unifyFault(label?: string): string {
  if (!label) return "UNCERTAIN";
  const key = label.trim().toUpperCase();
  return FAULT_GROUPS[key === "T3-H" ? "T3_H" : key] ?? "UNCERTAIN";
}
