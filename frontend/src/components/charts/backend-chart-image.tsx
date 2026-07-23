"use client";

import { useState } from "react";
import { ImageOff } from "lucide-react";

/** Embeds a chart image rendered server-side by the Flask backend (reuses the
 * existing matplotlib plotting code in backend/dga/*.py) — nothing is redrawn
 * on the frontend. Falls back to a message when the backend isn't reachable
 * (e.g. the Flask server isn't running). */
export function BackendChartImage({ src, alt }: { src: string; alt: string }) {
  const [failed, setFailed] = useState(false);

  if (failed) {
    return (
      <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-teal-200 bg-white px-6 py-10 text-center">
        <ImageOff className="h-6 w-6 text-teal-300" />
        <p className="text-sm text-teal-500">
          This diagram is rendered by the Python backend and requires it to be running.
        </p>
      </div>
    );
  }

  // eslint-disable-next-line @next/next/no-img-element
  return <img src={src} alt={alt} className="mx-auto max-w-full rounded-lg" onError={() => setFailed(true)} />;
}
