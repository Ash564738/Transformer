"use client";

import { motion } from "framer-motion";
import type { TransformerSummary } from "@/types/dga";
import { STATUS_ORDER, STATUS_STYLES, scoreToStatus } from "@/lib/severity";

export function StatCards({ summaries }: { summaries: TransformerSummary[] }) {
  const counts = { Normal: 0, Watch: 0, High: 0, Critical: 0 };
  for (const s of summaries) counts[scoreToStatus(s.latest_score)]++;

  return (
    <div className="grid grid-cols-2 gap-3 sm:gap-4 md:grid-cols-4">
      {STATUS_ORDER.map((status, i) => {
        const style = STATUS_STYLES[status];
        return (
          <motion.div
            key={status}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05 }}
            className={`card-surface border-t-4 ${style.topBorder} px-5 py-4`}
          >
            <div className="text-sm font-semibold text-teal-500">
              {status === "High" ? "High Risk" : status}
            </div>
            <div className={`mt-1 text-3xl font-extrabold ${style.text}`}>{counts[status]}</div>
          </motion.div>
        );
      })}
    </div>
  );
}
