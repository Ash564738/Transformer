// src/components/charts/iec3d-zones.ts
//
// These boxes must match backend/dga/iec60599.py's iec_ratio_method() exactly
// — that function is the actual classifier; this file only visualizes its
// decision boundaries. r1 = C2H2/C2H4 (x), r2 = CH4/H2 (y), r3 = C2H4/C2H6 (z).
// FLOOR/CEIL stand in for the open ends of one-sided backend conditions
// (e.g. D1's "r1 >= 0.1" has no upper bound) — they match the chart's own
// axis range (10^-3 to ~10^1.2), not an arbitrary guess.
//
// NOTE on NORMAL: unlike Rogers (which has a genuine ratio-space "quiet"
// zone), IEC 60599's NORMAL is decided by an absolute-ppm gate *before* any
// ratio is computed (see iec_ratio_method's first `if not any([...])`
// check) — there is no ratio-space box that means "Normal" for this method.
// Drawing one anyway would misrepresent the real algorithm, so this array
// intentionally has no NORMAL entry; Iec3DChart.tsx shows a text note
// instead of a fabricated zone.
import { Iec3DZone } from "./iec3d-config";

const FLOOR = 0.001;
const CEIL = 15;

export const IEC_3D_ZONES: Iec3DZone[] = [
  {
    // r1 < 0.1 AND r2 < 0.1 AND r3 < 0.2
    name: "PD",
    desc: "Partial Discharge",
    color: "#EF4444",
    xMin: FLOOR, xMax: 0.1,
    yMin: FLOOR, yMax: 0.1,
    zMin: FLOOR, zMax: 0.2,
  },
  {
    // r2 > 1.0 AND r3 < 1.0 (r1 unconstrained)
    name: "T1",
    desc: "Thermal fault < 300°C",
    color: "#3B82F6",
    xMin: FLOOR, xMax: CEIL,
    yMin: 1.0, yMax: CEIL,
    zMin: FLOOR, zMax: 1.0,
  },
  {
    // r1 < 0.1 AND r2 > 1.0 AND 1.0 <= r3 <= 4.0
    name: "T2",
    desc: "Thermal fault 300–700°C",
    color: "#F59E0B",
    xMin: FLOOR, xMax: 0.1,
    yMin: 1.0, yMax: CEIL,
    zMin: 1.0, zMax: 4.0,
  },
  {
    // r1 < 0.2 AND r2 > 1.0 AND r3 > 4.0
    name: "T3",
    desc: "Thermal fault > 700°C",
    color: "#8B5CF6",
    xMin: FLOOR, xMax: 0.2,
    yMin: 1.0, yMax: CEIL,
    zMin: 4.0, zMax: CEIL,
  },
  {
    // r1 >= 0.1 AND r3 > 3.0 (r2 unconstrained)
    name: "D1",
    desc: "Low-energy discharge",
    color: "#EC4899",
    xMin: 0.1, xMax: CEIL,
    yMin: FLOOR, yMax: CEIL,
    zMin: 3.0, zMax: CEIL,
  },
  {
    // 0.6 <= r1 <= 2.5 AND 0.1 <= r2 <= 1.0 AND r3 >= 1.0
    name: "D2",
    desc: "High-energy discharge",
    color: "#FCD34D",
    xMin: 0.6, xMax: 2.5,
    yMin: 0.1, yMax: 1.0,
    zMin: 1.0, zMax: CEIL,
  },
];
