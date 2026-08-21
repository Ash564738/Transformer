// src/components/layout/alert-banner.tsx
"use client";

import { AlertTriangle, X } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { useDashboardStore } from "@/store/use-dashboard-store";
import { statusFromSummary } from "@/lib/severity";

export function AlertBanner() {
  const payload = useDashboardStore((s) => s.payload);
  const dismissed = useDashboardStore((s) => s.bannerDismissed);
  const dismiss = useDashboardStore((s) => s.dismissBanner);

  const highRiskCount = payload
    ? payload.transformer_summary.filter(
        (s) => statusFromSummary(s) === "High"
      ).length
    : 0;

  return (
    <AnimatePresence initial={false}>
      {highRiskCount > 0 && !dismissed && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: "auto", opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          className="no-print overflow-hidden border-b border-status-high-border bg-status-high-soft"
        >
          <div className="mx-auto flex max-w-[1600px] items-center gap-3 px-4 py-3 sm:px-6">
            <div className="flex flex-1 items-center gap-2 text-sm font-medium text-status-high">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              <span>
                <strong className="font-bold">{highRiskCount}</strong> transformer
                {highRiskCount === 1 ? " is" : "s are"} currently at High risk —
                field inspection recommended.
              </span>
            </div>
            <button
              onClick={dismiss}
              className="shrink-0 rounded-full p-1 text-status-high/70 hover:bg-status-high/10 hover:text-status-high cursor-pointer"
              aria-label="Dismiss alert"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}