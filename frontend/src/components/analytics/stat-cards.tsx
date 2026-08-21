// src/components/analytics/stat-cards.tsx
"use client";

import { motion } from "framer-motion";
import type { RiskStatus, TransformerSummary } from "@/types/dga";
import {
  STATUS_ORDER,
  STATUS_STYLES,
  statusFromSummary,
  isCriticalSummary,
} from "@/lib/severity";

export function StatCards({
  summaries,
}: {
  summaries: TransformerSummary[];
}) {
  const counts: Record<RiskStatus, number> = {
    Normal: 0,
    Watch: 0,
    High: 0,
    "Insufficient data": 0,
  };

  let criticalCount = 0;

  for (const summary of summaries) {
    counts[statusFromSummary(summary)] += 1;
    if (isCriticalSummary(summary)) criticalCount += 1;
  }

  return (
    <div className="grid grid-cols-2 gap-3 sm:gap-4 lg:grid-cols-5">
      {STATUS_ORDER.map((status, index) => {
        const count = counts[status];
        // Chỉ bỏ qua nếu là "Insufficient data" và count === 0
        if (status === "Insufficient data" && count === 0) {
          return null;
        }

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
              {status === "High" ? "High Risk" : status}
            </div>
            <div className={`mt-1 text-3xl font-extrabold ${style.text}`}>
              {counts[status]}
            </div>
          </motion.div>
        );
      })}

      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="card-surface border-t-4 border-t-red-700 px-5 py-4"
      >
        <div className="text-sm font-semibold text-red-700">Critical Queue</div>
        <div className="mt-1 text-3xl font-extrabold text-red-800">{criticalCount}</div>
        <div className="mt-1 text-[10px] leading-4 text-slate-500">
          Operational priority only; not IEEE Status 4.
        </div>
      </motion.div>
    </div>
  );
}