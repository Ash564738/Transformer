"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { Loader2, Zap } from "lucide-react";
import { useAuthStore } from "@/store/use-auth-store";

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const status = useAuthStore((s) => s.status);
  const init = useAuthStore((s) => s.init);
  const router = useRouter();

  useEffect(() => {
    init();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (status === "unauthenticated") {
      router.replace("/login");
    }
  }, [status, router]);

  if (status === "loading" || status === "unauthenticated") {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-3 bg-cream-50">
        <span className="flex h-12 w-12 items-center justify-center rounded-xl bg-teal-800">
          <Zap className="h-6 w-6 text-copper-400" fill="currentColor" />
        </span>
        <Loader2 className="h-5 w-5 animate-spin text-teal-500" />
      </div>
    );
  }

  return <>{children}</>;
}
