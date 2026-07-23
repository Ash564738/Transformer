"use client";

import { AlertTriangle, X } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { useDashboardStore } from "@/store/use-dashboard-store";

export function AlertBanner() {
  const payload = useDashboardStore((s) => s.payload);
  const dismissed = useDashboardStore((s) => s.bannerDismissed);
  const dismiss = useDashboardStore((s) => s.dismissBanner);

  const criticalCount = payload
    ? payload.transformer_summary.filter((s) => s.latest_score >= 13).length
    : 0;

  return (
    <AnimatePresence initial={false}>
      {criticalCount > 0 && !dismissed && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: "auto", opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          className="no-print overflow-hidden border-b border-status-critical-border bg-status-critical-soft"
        >
          <div className="mx-auto flex max-w-[1600px] items-center gap-3 px-4 py-3 sm:px-6">
            <div className="flex flex-1 items-center gap-2 text-sm font-medium text-status-critical">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              <span>
                <strong className="font-bold">{criticalCount}</strong> transformer
                {criticalCount === 1 ? " is" : "s are"} currently at Critical status — field inspection
                required within 48 hours.
              </span>
            </div>
            <button
              onClick={dismiss}
              className="shrink-0 rounded-full p-1 text-status-critical/70 hover:bg-status-critical/10 hover:text-status-critical cursor-pointer"
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
