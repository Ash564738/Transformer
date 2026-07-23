"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Command } from "cmdk";
import { Search, LayoutDashboard, BarChart3, ListTree, MessageCircle } from "lucide-react";
import { useDashboardStore } from "@/store/use-dashboard-store";
import { StatusBadge } from "@/components/ui/badge";
import { scoreToStatus } from "@/lib/severity";

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const router = useRouter();
  const payload = useDashboardStore((s) => s.payload);
  const toggleChat = useDashboardStore((s) => s.toggleChat);

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((v) => !v);
      }
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, []);

  function go(path: string) {
    router.push(path);
    setOpen(false);
  }

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-[60] flex items-start justify-center bg-teal-950/50 pt-[12vh] backdrop-blur-sm"
      onClick={() => setOpen(false)}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-lg overflow-hidden rounded-2xl border border-teal-200 bg-white shadow-2xl"
      >
        <Command label="Command palette" className="flex flex-col">
          <div className="flex items-center gap-2 border-b border-cream-200 px-4 py-3">
            <Search className="h-4 w-4 text-teal-400" />
            <Command.Input
              autoFocus
              placeholder="Search transformers or jump to a screen…"
              className="flex-1 bg-transparent text-sm outline-none placeholder:text-teal-400"
            />
            <kbd className="rounded border border-cream-300 bg-cream-100 px-1.5 py-0.5 text-[10px] text-teal-500">
              Esc
            </kbd>
          </div>
          <Command.List className="max-h-96 overflow-y-auto scrollbar-thin p-2">
            <Command.Empty className="px-3 py-6 text-center text-sm text-teal-400">
              No results found.
            </Command.Empty>

            <Command.Group heading="Screens" className="px-1 py-1 text-[11px] font-semibold uppercase tracking-wide text-teal-400">
              <Command.Item
                onSelect={() => go("/")}
                className="flex cursor-pointer items-center gap-2.5 rounded-lg px-3 py-2 text-sm text-teal-800 data-[selected=true]:bg-teal-50"
              >
                <LayoutDashboard className="h-4 w-4 text-teal-500" /> Fleet Overview
              </Command.Item>
              <Command.Item
                onSelect={() => go("/analytics")}
                className="flex cursor-pointer items-center gap-2.5 rounded-lg px-3 py-2 text-sm text-teal-800 data-[selected=true]:bg-teal-50"
              >
                <BarChart3 className="h-4 w-4 text-teal-500" /> Analytics
              </Command.Item>
              <Command.Item
                onSelect={() => go("/fleet")}
                className="flex cursor-pointer items-center gap-2.5 rounded-lg px-3 py-2 text-sm text-teal-800 data-[selected=true]:bg-teal-50"
              >
                <ListTree className="h-4 w-4 text-teal-500" /> Fleet Directory
              </Command.Item>
              <Command.Item
                onSelect={() => {
                  toggleChat(true);
                  setOpen(false);
                }}
                className="flex cursor-pointer items-center gap-2.5 rounded-lg px-3 py-2 text-sm text-teal-800 data-[selected=true]:bg-teal-50"
              >
                <MessageCircle className="h-4 w-4 text-teal-500" /> Ask DGA Assistant
              </Command.Item>
            </Command.Group>

            {payload && payload.transformer_summary.length > 0 && (
              <Command.Group heading="Transformers" className="mt-2 px-1 py-1 text-[11px] font-semibold uppercase tracking-wide text-teal-400">
                {payload.transformer_summary.map((t) => (
                  <Command.Item
                    key={t.transformer_id}
                    value={`${t.transformer_id} ${t.loc ?? ""} ${t.fault_type}`}
                    onSelect={() => go(`/transformer/${encodeURIComponent(t.transformer_id)}`)}
                    className="flex cursor-pointer items-center justify-between gap-2.5 rounded-lg px-3 py-2 text-sm text-teal-800 data-[selected=true]:bg-teal-50"
                  >
                    <span className="flex items-center gap-2">
                      <span className="font-semibold">{t.transformer_id}</span>
                      <span className="text-xs text-teal-400">{t.loc}</span>
                    </span>
                    <StatusBadge status={scoreToStatus(t.latest_score)} />
                  </Command.Item>
                ))}
              </Command.Group>
            )}
          </Command.List>
        </Command>
      </div>
    </div>
  );
}
