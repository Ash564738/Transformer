import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { AppShell } from "@/components/layout/app-shell";
import { Toaster } from "@/components/ui/toaster";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "DGA Monitor — Transformer Degradation Dashboard",
  description:
    "DGA-based transformer health scoring, ranking, and diagnostic explanation dashboard.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    // suppressHydrationWarning on <html>: browser auto-translate (Chrome/Edge
    // offering to translate an English page when the browser's locale is
    // Vietnamese) rewrites `lang` and adds a `translated-ltr` class right
    // after React hydrates, which otherwise logs a harmless but noisy
    // hydration-mismatch warning. This is the fix React/Next.js documents
    // for exactly this class of DOM interference (translate extensions,
    // etc.) — it does not suppress real hydration bugs in children.
    <html lang="en" className={`${inter.variable} h-full antialiased`} suppressHydrationWarning>
      <body className="min-h-full flex flex-col bg-cream-50 text-teal-950">
        <AppShell>{children}</AppShell>
        <Toaster />
      </body>
    </html>
  );
}
