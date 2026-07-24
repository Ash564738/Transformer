// iec3d-zones.ts
import { Iec3DZone } from "./iec3d-config";

export const IEC_3D_ZONES: Iec3DZone[] = [
  {
    name: "PD",
    desc: "Partial Discharge",
    color: "#EF4444",   // đỏ
    xMin: 0.001, xMax: 0.1,   // R1 rất thấp
    yMin: 0.001, yMax: 0.1,   // R2 thấp
    zMin: 0.001, zMax: 0.1,   // R3 thấp (thực tế IEC cho PD: R1<0.1, R2<0.1, R3<1)
  },
  {
    name: "T1",
    desc: "Thermal fault < 300°C",
    color: "#3B82F6",   // xanh dương
    xMin: 0.001, xMax: 0.1,
    yMin: 0.1, yMax: 1.0,
    zMin: 0.001, zMax: 1.0,
  },
  {
    name: "T2",
    desc: "Thermal fault 300–700°C",
    color: "#F59E0B",   // cam
    xMin: 0.001, xMax: 0.1,
    yMin: 1.0, yMax: 15,
    zMin: 1.0, zMax: 5.0,
  },
  {
    name: "T3",
    desc: "Thermal fault > 700°C",
    color: "#8B5CF6",   // tím
    xMin: 0.001, xMax: 0.1,
    yMin: 1.0, yMax: 15,
    zMin: 5.0, zMax: 15,
  },
  {
    name: "D1",
    desc: "Low-energy discharge",
    color: "#EC4899",   // hồng
    xMin: 0.1, xMax: 15,
    yMin: 0.001, yMax: 0.1,
    zMin: 1.0, zMax: 15,
  },
  {
    name: "D2",
    desc: "High-energy discharge",
    color: "#FCD34D",   // vàng
    xMin: 0.1, xMax: 15,
    yMin: 0.1, yMax: 15,
    zMin: 0.001, zMax: 15,  // D2 bao trùm rộng
  },
];