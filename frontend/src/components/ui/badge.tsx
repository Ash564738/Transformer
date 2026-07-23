import { cn } from "@/lib/utils";
import { STATUS_STYLES, STATUS_ICON_LABEL } from "@/lib/severity";
import type { RiskStatus } from "@/types/dga";

export function StatusBadge({ status, className }: { status: RiskStatus; className?: string }) {
  const style = STATUS_STYLES[status];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold",
        style.text,
        style.bg,
        style.border,
        className
      )}
    >
      <span aria-hidden>{STATUS_ICON_LABEL[status]}</span>
      {status === "High" ? "High" : status}
    </span>
  );
}

export function Dot({ status, className }: { status: RiskStatus; className?: string }) {
  const style = STATUS_STYLES[status];
  return <span className={cn("inline-block h-2.5 w-2.5 rounded-full", style.dot, className)} />;
}
