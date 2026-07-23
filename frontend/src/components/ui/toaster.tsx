"use client";

import { Toaster as SonnerToaster } from "sonner";

export function Toaster() {
  return (
    <SonnerToaster
      position="bottom-right"
      toastOptions={{
        style: {
          background: "#ffffff",
          border: "1px solid #d9d5c4",
          color: "#0f2f2c",
          borderRadius: "0.75rem",
          fontSize: "0.875rem",
        },
      }}
    />
  );
}
