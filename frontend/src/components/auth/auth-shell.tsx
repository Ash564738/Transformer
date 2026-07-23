"use client";

import Link from "next/link";
import { Zap } from "lucide-react";

export function AuthShell({
  title,
  subtitle,
  children,
  footer,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
  footer: React.ReactNode;
}) {
  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-gradient-to-b from-teal-900 to-teal-950 px-4 py-12">
      <div
        className="pointer-events-none absolute inset-0 opacity-40"
        style={{
          backgroundImage:
            "radial-gradient(circle at 15% 20%, rgba(217,119,6,0.18), transparent 40%), radial-gradient(circle at 85% 80%, rgba(217,119,6,0.12), transparent 45%)",
        }}
      />
      <div className="relative w-full max-w-md">
        <div className="mb-8 flex flex-col items-center gap-3 text-center">
          <span className="flex h-12 w-12 items-center justify-center rounded-xl bg-copper-500/90 shadow-lg">
            <Zap className="h-6 w-6 text-teal-950" fill="currentColor" />
          </span>
          <div>
            <h1 className="text-xl font-extrabold tracking-tight text-cream-50">DGA Monitor</h1>
            <p className="text-sm text-teal-300">Transformer Degradation Dashboard</p>
          </div>
        </div>

        <div className="rounded-2xl border border-teal-800 bg-cream-50 p-7 shadow-2xl">
          <h2 className="text-lg font-bold text-teal-900">{title}</h2>
          <p className="mt-1 text-sm text-teal-500">{subtitle}</p>
          <div className="mt-6">{children}</div>
        </div>

        <p className="mt-5 text-center text-sm text-teal-300">{footer}</p>
      </div>
    </div>
  );
}
