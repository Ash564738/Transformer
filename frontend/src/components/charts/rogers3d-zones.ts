// rogers3d-zones.ts
//
// These boxes must match backend/dga/rogers.py's rogers_ratio_method()
// exactly. r1 = CH4/H2 (x), r2 = C2H2/C2H4 (y), r3 = C2H4/C2H6 (z). Verified
// against every branch of that function — unlike IEC 60599, Rogers genuinely
// does have a ratio-space "quiet" reading (0.1<=r1<=1.0, r2<0.1, r3<1.0) that
// the backend itself maps to NORMAL, which is why this array (unlike
// iec3d-zones.ts) legitimately includes a NORMAL box.
import { Ratio3DZone } from "./ratio3d-config";

export const ROGERS_3D_ZONES: Ratio3DZone[] = [
  {
    name: "PD",
    desc: "Partial Discharge",
    color: "#FF3B30",   // đỏ
    xMin: 0.001, xMax: 0.1,
    yMin: 0.001, yMax: 0.1,
    zMin: 0.001, zMax: 1,
  },
  {
    name: "NORMAL",
    desc: "Normal operating condition",
    color: "#34C759",   // xanh lá
    xMin: 0.1, xMax: 1,
    yMin: 0.001, yMax: 0.1,
    zMin: 0.001, zMax: 1,
  },
  {
    name: "T1",
    desc: "Thermal fault < 300°C",
    color: "#0A84FF",   // xanh dương
    xMin: 0.1, xMax: 1,
    yMin: 0.001, yMax: 0.1,
    zMin: 1, zMax: 3,
  },
  {
    name: "T2",
    desc: "Thermal fault 300–700°C",
    color: "#FF9500",   // cam
    xMin: 1, xMax: 15,
    yMin: 0.001, yMax: 0.1,
    zMin: 1, zMax: 3,
  },
  {
    name: "T3",
    desc: "Thermal fault > 700°C",
    color: "#AF52DE",   // tím
    xMin: 1, xMax: 15,
    yMin: 0.001, yMax: 0.1,
    zMin: 3, zMax: 15,
  },
  {
    name: "D2",
    desc: "High-energy discharge (arcing)",
    color: "#FFD60A",   // vàng
    xMin: 0.1, xMax: 1,
    yMin: 1, yMax: 3,
    zMin: 0.1, zMax: 3,
  },
  {
    name: "D1",
    desc: "Low-energy discharge",
    color: "#EC4899",   // hồng đậm (mới)
    xMin: 0.1, xMax: 1,
    yMin: 1, yMax: 15,
    zMin: 3, zMax: 15,
  },
];