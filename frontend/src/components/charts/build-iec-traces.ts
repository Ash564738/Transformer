// build-iec-traces.ts
import { Iec3DZone } from "./iec3d-config";

export function buildIecTraces(
  zones: Iec3DZone[],
  activeFault?: string
): any[] {
  const traces: any[] = [];

  for (const zone of zones) {
    const isActive = zone.name === activeFault;
    const { xMin, xMax, yMin, yMax, zMin, zMax, color, name, desc } = zone;

    const x = [xMin, xMax, xMax, xMin, xMin, xMax, xMax, xMin];
    const y = [yMin, yMin, yMax, yMax, yMin, yMin, yMax, yMax];
    const z = [zMin, zMin, zMin, zMin, zMax, zMax, zMax, zMax];

    const i = [0,0, 4,4, 0,0, 2,2, 0,0, 1,1];
    const j = [1,2, 5,6, 1,5, 3,7, 3,7, 2,6];
    const k = [2,3, 6,7, 5,4, 7,6, 7,4, 6,5];

    traces.push({
      type: "mesh3d",
      x, y, z,
      i, j, k,
      color: color,
      opacity: isActive ? 1 : 0.15,
      flatshading: true,
      hoverinfo: "text",
      text: `${name}: ${desc}`,
      showscale: false,
      lighting: {
        ambient: 0.6,
        diffuse: 0.8,
        specular: 0.6,
        roughness: 0.3,
        fresnel: 0.2,
      },
      lightposition: {
        x: 100,
        y: 150,
        z: 200,
      },
    });

    if (isActive) {
      const edges = [
        [0,1], [1,2], [2,3], [3,0],
        [4,5], [5,6], [6,7], [7,4],
        [0,4], [1,5], [2,6], [3,7]
      ];
      let wireX: number[] = [], wireY: number[] = [], wireZ: number[] = [];
      for (const [a,b] of edges) {
        wireX.push(x[a], x[b], null as any);
        wireY.push(y[a], y[b], null as any);
        wireZ.push(z[a], z[b], null as any);
      }
      traces.push({
        type: "scatter3d",
        mode: "lines",
        x: wireX, y: wireY, z: wireZ,
        line: { color: "#ffffff", width: 4 },
        showlegend: false,
        hoverinfo: "none",
      });
    }

    const cx = (xMin + xMax) / 2;
    const cy = (yMin + yMax) / 2;
    const cz = (zMin + zMax) / 2;
    traces.push({
      type: "scatter3d",
      mode: "text",
      x: [cx], y: [cy], z: [cz],
      text: [name],
      textfont: {
        color: isActive ? "#000000" : "#1e293b",
        size: isActive ? 18 : 13,
        weight: "bold" as const,
      },
      showlegend: false,
      hoverinfo: "none",
    });
  }

  return traces;
}