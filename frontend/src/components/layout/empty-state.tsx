"use client";

import { UploadCloud } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useDashboardStore } from "@/store/use-dashboard-store";

export function EmptyState({ title, subtitle }: { title: string; subtitle: string }) {
  const setDataPanelOpen = useDashboardStore((s) => s.setDataPanelOpen);

  return (
    <div className="flex flex-col items-center justify-center gap-4 rounded-2xl border border-dashed border-teal-200 bg-white/60 px-6 py-20 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-teal-50 text-teal-500">
        <UploadCloud className="h-7 w-7" />
      </div>
      <div>
        <h2 className="text-lg font-bold text-teal-900">{title}</h2>
        <p className="mx-auto mt-1 max-w-sm text-sm text-teal-500">{subtitle}</p>
      </div>
      <Button variant="outline" onClick={() => setDataPanelOpen(true)}>
        <UploadCloud className="h-4 w-4" /> Upload DGA dataset
      </Button>
    </div>
  );
}
