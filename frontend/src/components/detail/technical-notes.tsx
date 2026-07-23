import type { DgaRow } from "@/types/dga";
import { formatDate } from "@/lib/utils";
import { scoreToStatus } from "@/lib/severity";

export function TechnicalNotes({ rows }: { rows: DgaRow[] }) {
  const latest = rows[rows.length - 1];
  const prior = rows.length > 1 ? rows[rows.length - 2] : undefined;
  if (!latest) return <p className="text-sm text-teal-400">No technical notes available.</p>;

  const status = scoreToStatus(Number(latest.severity_score ?? 0));
  const entries: { title: string; date: string; tone: "flag" | "routine" }[] = [];

  if (status === "Critical" || status === "High") {
    entries.push({ title: "Auto-Flagged", date: latest.sample_day, tone: "flag" });
  }
  entries.push({ title: "Routine Sample", date: latest.sample_day, tone: "routine" });
  if (prior) entries.push({ title: "Routine Sample", date: prior.sample_day, tone: "routine" });

  return (
    <ul className="space-y-3">
      {entries.map((e, i) => (
        <li key={i} className="flex items-start justify-between gap-3 border-b border-cream-200 pb-3 last:border-0 last:pb-0">
          <span className={`text-sm font-bold ${e.tone === "flag" ? "text-copper-600" : "text-teal-800"}`}>
            {e.title}
          </span>
          <span className={`text-xs ${e.tone === "flag" ? "text-copper-500" : "text-teal-400"}`}>
            {formatDate(e.date)}
          </span>
        </li>
      ))}
    </ul>
  );
}
