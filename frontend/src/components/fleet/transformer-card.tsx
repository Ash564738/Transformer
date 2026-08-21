// src/components/fleet/transformer-card.tsx
"use client";

import Link from "next/link";
import { motion } from "framer-motion";

import type { TransformerSummary } from "@/types/dga";

import { StatusBadge } from "@/components/ui/badge";

import {
  statusFromSummary,
  STATUS_STYLES,
  formatTrend,
} from "@/lib/severity";

import { formatDate } from "@/lib/utils";

export function TransformerCard({
  summary,
}: {
  summary: TransformerSummary;
}) {
  const status = statusFromSummary(summary);
  const style = STATUS_STYLES[status];
  const trend = formatTrend(summary.trend);

  return (
    <Link href={`/transformer/${encodeURIComponent(summary.transformer_id)}`}>
      <motion.div
        whileHover={{ y: -2 }}
        className={`card-surface h-full border-l-4 ${style.leftBorder} px-4 py-3.5 transition-shadow hover:shadow-lg`}
      >
        <div className="flex items-start justify-between gap-2">
          <div>
            <div className="font-bold text-teal-900">
              {summary.transformer_id}
            </div>

            <div className="text-xs text-teal-400">
              {summary.fault_type}
            </div>
          </div>

          <StatusBadge status={status} />
        </div>

        <div className="mt-3 flex items-center justify-between text-xs text-teal-400">
          <span>
            {formatDate(summary.latest_sample_day)}
          </span>

          <span className="flex items-center gap-1">
            {trend.icon} {trend.label}
          </span>
        </div>
      </motion.div>
    </Link>
  );
}