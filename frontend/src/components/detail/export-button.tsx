"use client";

import { Download } from "lucide-react";
import { Button } from "@/components/ui/button";

export function ExportButton() {
  return (
    <Button variant="outline" size="sm" onClick={() => window.print()}>
      <Download className="h-3.5 w-3.5" /> Export Summary (PDF)
    </Button>
  );
}
