import type { DgaRow } from "@/types/dga";
import { formatNumber } from "@/lib/utils";

export function GasIndicatorCards({ row }: { row: DgaRow }) {
  const tdcg = Number(row.tdcg ?? 0) || (Number(row.h2 ?? 0) + Number(row.ch4 ?? 0) + Number(row.c2h6 ?? 0) + Number(row.c2h4 ?? 0) + Number(row.c2h2 ?? 0) + Number(row.co ?? 0));
  const items = [
    { label: "H₂", sub: "Hydrogen", value: row.h2 },
    { label: "CH₄", sub: "Methane", value: row.ch4 },
    { label: "C₂H₂", sub: "Acetylene", value: row.c2h2 },
    { label: "C₂H₄", sub: "Ethylene", value: row.c2h4 },
    { label: "CO", sub: "Carbon Monoxide", value: row.co },
    { label: "TDCG", sub: "Total Combustible Gas", value: tdcg },
  ];

  return (
    <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-6">
      {items.map((item) => (
        <div key={item.label} className="rounded-xl border border-cream-300 bg-white px-3 py-3 text-center">
          <div className="text-xs font-semibold uppercase tracking-wide text-teal-400">{item.label}</div>
          <div className="mt-1 text-lg font-extrabold text-teal-900">{formatNumber(Number(item.value ?? 0), 1)}</div>
          <div className="text-[10px] text-teal-400">ppm · {item.sub}</div>
        </div>
      ))}
    </div>
  );
}
