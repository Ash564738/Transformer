// ratio3d-config.ts
export interface Ratio3DZone {
  name: string;
  desc: string;       // mô tả lỗi
  color: string;
  xMin: number; // R1 (CH4/H2)
  xMax: number;
  yMin: number; // R2 (C2H2/C2H4)
  yMax: number;
  zMin: number; // R3 (C2H4/C2H6)
  zMax: number;
}