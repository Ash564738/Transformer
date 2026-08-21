// src/components/detail/gas-indicator-cards.tsx
"use client";

import type { DgaRow } from "@/types/dga";
import { formatNumber } from "@/lib/utils";

function finiteNumber(value: unknown): number | null {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function fallbackTdcg(row: DgaRow): number | null {
  const direct = finiteNumber(row.tdcg);

  if (direct != null && direct >= 0) {
    return direct;
  }

  const combustible: Array<number | null> = [
    finiteNumber(row.h2),
    finiteNumber(row.ch4),
    finiteNumber(row.c2h6),
    finiteNumber(row.c2h4),
    finiteNumber(row.c2h2),
    finiteNumber(row.co),
  ];

  const numericCombustible: number[] = combustible.filter(
    (value): value is number => value != null
  );

  if (numericCombustible.length !== combustible.length) {
    return null;
  }

  return numericCombustible.reduce(
    (sum: number, value: number) => sum + value,
    0
  );
}

export function GasIndicatorCards({ row }: { row: DgaRow }) {
  const tdcg = fallbackTdcg(row);

  const items: Array<{
    label: string;
    sub: string;
    value: number | null;
  }> = [
    {
      label: "H₂",
      sub: "Hydrogen",
      value: finiteNumber(row.h2),
    },
    {
      label: "CH₄",
      sub: "Methane",
      value: finiteNumber(row.ch4),
    },
    {
      label: "C₂H₂",
      sub: "Acetylene",
      value: finiteNumber(row.c2h2),
    },
    {
      label: "C₂H₄",
      sub: "Ethylene",
      value: finiteNumber(row.c2h4),
    },
    {
      label: "CO",
      sub: "Carbon Monoxide",
      value: finiteNumber(row.co),
    },
    {
      label: "CO₂",
      sub: "Carbon Dioxide",
      value: finiteNumber(row.co2),
    },
    {
      label: "TDCG",
      sub: "Total Combustible Gas",
      value: tdcg,
    },
  ];

  return (
    <div className="grid gap-2.5 sm:grid-cols-3 lg:grid-cols-7">
      {items.map((item) => (
        <div
          key={item.label}
          className="rounded-xl border border-cream-300 bg-white px-3 py-3 text-center"
        >
          <div className="text-xs font-semibold uppercase tracking-wide text-teal-400">
            {item.label}
          </div>

          <div className="mt-1 text-lg font-extrabold text-teal-900">
            {item.value == null
              ? "—"
              : formatNumber(item.value, 1)}
          </div>

          <div className="text-[10px] text-teal-400">
            ppm · {item.sub}
          </div>
        </div>
      ))}
    </div>
  );
}