"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Bell, Settings, Zap, LogOut } from "lucide-react";
import { cn } from "@/lib/utils";
import { useDashboardStore } from "@/store/use-dashboard-store";
import { useAuthStore } from "@/store/use-auth-store";

const NAV_LINKS = [
  { href: "/", label: "Overview" },
  { href: "/analytics", label: "Analytics" },
  { href: "/fleet", label: "Fleet" },
];

export function TopNav({ onOpenDataPanel }: { onOpenDataPanel: () => void }) {
  const pathname = usePathname();
  const router = useRouter();
  const payload = useDashboardStore((s) => s.payload);
  const user = useAuthStore((s) => s.user);
  const logout = useAuthStore((s) => s.logout);
  const [menuOpen, setMenuOpen] = useState(false);
  const criticalCount = payload
    ? payload.transformer_summary.filter((s) => s.latest_score >= 13).length
    : 0;

  const initials = (user?.name ?? "?")
    .split(" ")
    .map((p) => p[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();

  return (
    <div className="no-print sticky top-0 z-40">
      <header className="bg-gradient-to-b from-teal-900 to-teal-950 text-cream-50">
        <div className="mx-auto flex h-16 max-w-[1600px] items-center justify-between px-4 sm:px-6">
          <div className="flex items-center gap-8">
            <Link href="/" className="flex items-center gap-2 font-extrabold text-lg tracking-tight">
              <span className="flex h-7 w-7 items-center justify-center rounded-md bg-copper-500/90">
                <Zap className="h-4 w-4 text-teal-950" fill="currentColor" />
              </span>
              DGA Monitor
            </Link>
            <nav className="hidden md:flex items-center gap-6 text-sm font-medium">
              {NAV_LINKS.map((link) => {
                const active = link.href === "/" ? pathname === "/" : pathname.startsWith(link.href);
                return (
                  <Link
                    key={link.href}
                    href={link.href}
                    className={cn(
                      "relative py-1 text-teal-200 transition-colors hover:text-white",
                      active && "text-white"
                    )}
                  >
                    {link.label}
                    {active && (
                      <span className="absolute -bottom-[19px] left-0 right-0 h-0.5 bg-copper-400" />
                    )}
                  </Link>
                );
              })}
            </nav>
          </div>

          <div className="flex items-center gap-3">
            <button
              className="relative rounded-full p-2 text-teal-200 hover:bg-white/10 hover:text-white transition-colors cursor-pointer"
              aria-label="Notifications"
            >
              <Bell className="h-5 w-5" />
              {criticalCount > 0 && (
                <span className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-status-critical ring-2 ring-teal-950" />
              )}
            </button>
            <button
              onClick={onOpenDataPanel}
              className="rounded-full p-2 text-teal-200 hover:bg-white/10 hover:text-white transition-colors cursor-pointer"
              aria-label="Data source settings"
            >
              <Settings className="h-5 w-5" />
            </button>
            <div className="relative">
              <button
                onClick={() => setMenuOpen((v) => !v)}
                className="flex h-8 w-8 items-center justify-center rounded-full bg-copper-500 text-xs font-bold text-teal-950 cursor-pointer hover:bg-copper-400 transition-colors"
                aria-label="Account menu"
              >
                {initials}
              </button>
              {menuOpen && (
                <>
                  <div className="fixed inset-0 z-10" onClick={() => setMenuOpen(false)} />
                  <div className="absolute right-0 top-11 z-20 w-56 rounded-xl border border-teal-800 bg-teal-950 p-1.5 shadow-2xl">
                    <div className="px-3 py-2 text-xs">
                      <div className="font-semibold text-white">{user?.name}</div>
                      <div className="text-teal-400">{user?.email}</div>
                    </div>
                    <div className="my-1 h-px bg-teal-800" />
                    <button
                      onClick={async () => {
                        setMenuOpen(false);
                        await logout();
                        router.replace("/login");
                      }}
                      className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm text-teal-200 hover:bg-white/5 hover:text-white cursor-pointer"
                    >
                      <LogOut className="h-4 w-4" /> Sign out
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </header>
      <div className="h-[3px] bg-gradient-to-r from-copper-400 via-copper-500 to-copper-400" />
    </div>
  );
}
