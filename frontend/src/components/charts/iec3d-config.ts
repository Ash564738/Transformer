// iec3d-config.ts
export interface Iec3DZone {
  name: string;
  desc: string;
  color: string;
  xMin: number; // R1 (C2H2/C2H4)
  xMax: number;
  yMin: number; // R2 (CH4/H2)
  yMax: number;
  zMin: number; // R3 (C2H4/C2H6)
  zMax: number;
}