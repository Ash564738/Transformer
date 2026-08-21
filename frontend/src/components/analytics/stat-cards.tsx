// src/components/analytics/stat-cards.tsx
"use client";

import { motion } from "framer-motion";
import type { RiskStatus, TransformerSummary } from "@/types/dga";
import { STATUS_ORDER, STATUS_STYLES, statusFromSummary } from "@/lib/severity";

export function StatCards({ summaries }: { summaries: TransformerSummary[] }) {
  const counts: Record<RiskStatus, number> = {
    Normal: 0,
    Watch: 0,
    High: 0,
    "Insufficient data": 0,
  };

  for (const summary of summaries) {
    counts[statusFromSummary(summary)] += 1;
  }

  return (
    <div className="grid grid-cols-2 gap-3 sm:gap-4 lg:grid-cols-4">
      {STATUS_ORDER.map((status, index) => {
        const style = STATUS_STYLES[status];

        return (
          <motion.div
            key={status}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
            className={`card-surface border-t-4 ${style.topBorder} px-5 py-4`}
          >
            <div className="text-sm font-semibold text-teal-500">
              {status === "High" ? "Status 3" : status}
            </div>

            <div className={`mt-1 text-3xl font-extrabold ${style.text}`}>
              {counts[status]}
            </div>

            <div className="mt-1 text-[10px] leading-4 text-slate-500">
              {status === "High"
                ? "IEEE Status 3; fleet maintenance order is separate."
                : status === "Watch"
                  ? "IEEE Status 2."
                  : status === "Normal"
                    ? "IEEE Status 1."
                    : "IEEE screening result unavailable or insufficient."}
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}