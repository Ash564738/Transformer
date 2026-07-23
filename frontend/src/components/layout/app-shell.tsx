"use client";

import { usePathname } from "next/navigation";
import { TopNav } from "@/components/layout/top-nav";
import { AlertBanner } from "@/components/layout/alert-banner";
import { DataPanel } from "@/components/layout/data-panel";
import { CommandPalette } from "@/components/layout/command-palette";
import { FloatingChat } from "@/components/chat/floating-chat";
import { AuthGuard } from "@/components/auth/auth-guard";
import { useDashboardStore } from "@/store/use-dashboard-store";

const PUBLIC_ROUTES = ["/login"];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const dataPanelOpen = useDashboardStore((s) => s.dataPanelOpen);
  const setDataPanelOpen = useDashboardStore((s) => s.setDataPanelOpen);

  if (PUBLIC_ROUTES.includes(pathname)) {
    // Login is unauthenticated by definition — no top nav, no auth-gated
    // chrome, just the form itself. There is no self-service registration
    // (single-user login, seeded via backend/seed_user.py).
    return <>{children}</>;
  }

  return (
    <AuthGuard>
      <TopNav onOpenDataPanel={() => setDataPanelOpen(true)} />
      <AlertBanner />
      <main className="mx-auto w-full max-w-[1600px] flex-1 px-4 py-6 sm:px-6 sm:py-8">{children}</main>
      <DataPanel open={dataPanelOpen} onClose={() => setDataPanelOpen(false)} />
      <CommandPalette />
      <FloatingChat />
    </AuthGuard>
  );
}
