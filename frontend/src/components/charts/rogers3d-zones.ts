// rogers3d-zones.ts
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