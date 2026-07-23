import Link from "next/link";
import { ArrowLeft, MapPin, CalendarClock } from "lucide-react";
import type { TransformerSummary } from "@/types/dga";
import { StatusBadge } from "@/components/ui/badge";
import { scoreToStatus } from "@/lib/severity";
import { stationOf } from "@/lib/transformer-helpers";
import { formatDate } from "@/lib/utils";
import { ExportButton } from "@/components/detail/export-button";

export function DetailHeader({ summary }: { summary: TransformerSummary }) {
  const status = scoreToStatus(summary.latest_score);
  return (
    <div className="rounded-2xl bg-gradient-to-b from-teal-900 to-teal-800 px-5 py-5 text-white sm:px-7 sm:py-6 print:bg-white print:text-teal-950">
      <Link
        href="/"
        className="no-print inline-flex items-center gap-1.5 text-xs font-bold uppercase tracking-wide text-teal-300 hover:text-white"
      >
        <ArrowLeft className="h-3.5 w-3.5" /> Back to Dashboard
      </Link>
      <div className="mt-3 flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
        <div>
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-3xl font-extrabold tracking-tight">{summary.transformer_id}</h1>
            <StatusBadge status={status} />
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-teal-200 print:text-teal-600">
            <span className="flex items-center gap-1.5">
              <MapPin className="h-3.5 w-3.5" /> Station: {stationOf(summary)}
            </span>
            <span className="flex items-center gap-1.5">
              <CalendarClock className="h-3.5 w-3.5" /> Last Test: {formatDate(summary.latest_sample_day)}
            </span>
          </div>
        </div>
        <ExportButton />
      </div>
    </div>
  );
}
