"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { X, UploadCloud, FileJson, Trash2, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { useDashboardStore } from "@/store/use-dashboard-store";

export function DataPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [mode, setMode] = useState<"file" | "json">("file");
  const [jsonText, setJsonText] = useState('[{"transformer_id":"T01","h2":120,"co":340}]');
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [pendingFile, setPendingFile] = useState<File | null>(null);

  const loading = useDashboardStore((s) => s.loading);
  const payload = useDashboardStore((s) => s.payload);
  const uploadFile = useDashboardStore((s) => s.uploadFile);
  const uploadJson = useDashboardStore((s) => s.uploadJson);
  const clearData = useDashboardStore((s) => s.clearData);

  const [elapsedSec, setElapsedSec] = useState(0);
  const startRef = useRef<number | null>(null);
  useEffect(() => {
    if (!loading) {
      startRef.current = null;
      return;
    }
    startRef.current = Date.now();
    const id = setInterval(() => {
      setElapsedSec(startRef.current ? Math.floor((Date.now() - startRef.current) / 1000) : 0);
    }, 1000);
    return () => clearInterval(id);
  }, [loading]);

  // File mode has no separate "Run" step — picking or dropping a file
  // immediately sends it to the backend and closes the panel on success.
  async function runFile(file: File) {
    setElapsedSec(0);
    setPendingFile(file);
    try {
      await uploadFile(file);
      toast.success(`Prediction complete for ${file.name}.`);
      onClose();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Could not run prediction.");
    }
  }

  async function runJson() {
    setElapsedSec(0);
    try {
      const parsed = JSON.parse(jsonText);
      const rows = Array.isArray(parsed) ? parsed : parsed.data ?? [parsed];
      await uploadJson(rows);
      toast.success(`Prediction complete for ${rows.length} row(s).`);
      onClose();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Could not run prediction.");
    }
  }

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-teal-950/40 backdrop-blur-[1px]"
            onClick={loading ? undefined : onClose}
          />
          <motion.aside
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 28, stiffness: 260 }}
            className="fixed right-0 top-0 z-50 h-full w-full max-w-md overflow-y-auto scrollbar-thin bg-cream-50 shadow-2xl"
          >
            <div className="flex items-center justify-between border-b border-cream-300 bg-white px-5 py-4">
              <div>
                <h2 className="text-lg font-bold text-teal-900">Data Source</h2>
                <p className="text-xs text-teal-600">Upload a DGA dataset to run prediction.</p>
              </div>
              <button
                onClick={onClose}
                disabled={loading}
                className="rounded-full p-1.5 text-teal-500 hover:bg-teal-50 cursor-pointer disabled:cursor-not-allowed disabled:opacity-30"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="space-y-5 p-5">
              {payload && (
                <div className="flex items-center justify-between rounded-xl border border-teal-200 bg-teal-50 px-4 py-3 text-sm">
                  <div>
                    <span className="font-semibold text-teal-900">
                      {payload.dataset_summary.total_transformers} transformers
                    </span>
                    <span className="text-teal-600"> · {payload.dataset_summary.total_rows} rows loaded</span>
                  </div>
                  <button
                    onClick={() => {
                      clearData();
                      toast("Cleared loaded data.");
                    }}
                    className="text-teal-500 hover:text-status-critical cursor-pointer"
                    aria-label="Clear data"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              )}

              <div className="flex rounded-lg border border-teal-200 bg-white p-1 text-sm font-medium">
                <button
                  onClick={() => setMode("file")}
                  className={`flex-1 rounded-md py-1.5 transition-colors cursor-pointer ${mode === "file" ? "bg-teal-800 text-white" : "text-teal-700 hover:bg-teal-50"}`}
                >
                  File upload
                </button>
                <button
                  onClick={() => setMode("json")}
                  className={`flex-1 rounded-md py-1.5 transition-colors cursor-pointer ${mode === "json" ? "bg-teal-800 text-white" : "text-teal-700 hover:bg-teal-50"}`}
                >
                  JSON text
                </button>
              </div>

              {mode === "file" ? (
                <div
                  onDragOver={(e) => {
                    e.preventDefault();
                    if (!loading) setDragOver(true);
                  }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setDragOver(false);
                    if (loading) return;
                    const f = e.dataTransfer.files?.[0];
                    if (f) runFile(f);
                  }}
                  onClick={() => !loading && fileInputRef.current?.click()}
                  className={`flex flex-col items-center gap-2 rounded-xl border-2 border-dashed px-4 py-8 text-center transition-colors ${
                    loading ? "cursor-not-allowed opacity-60" : "cursor-pointer"
                  } ${dragOver ? "border-teal-500 bg-teal-50" : "border-teal-200 bg-white hover:bg-teal-50/50"}`}
                >
                  {loading ? (
                    <Loader2 className="h-7 w-7 animate-spin text-teal-400" />
                  ) : (
                    <UploadCloud className="h-7 w-7 text-teal-400" />
                  )}
                  <div className="text-sm font-medium text-teal-800">
                    {loading
                      ? `Running prediction on ${pendingFile?.name}… ${elapsedSec}s`
                      : "Drop a CSV or Excel file, or click to browse"}
                  </div>
                  <div className="text-xs text-teal-500">Supported formats: .csv, .xlsx, .xls</div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv,.xlsx,.xls"
                    className="hidden"
                    disabled={loading}
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) runFile(f);
                      e.target.value = "";
                    }}
                  />
                </div>
              ) : (
                <div className="space-y-2">
                  <div className="flex items-center gap-1.5 text-xs font-medium text-teal-600">
                    <FileJson className="h-3.5 w-3.5" /> Paste JSON rows
                  </div>
                  <textarea
                    value={jsonText}
                    onChange={(e) => setJsonText(e.target.value)}
                    rows={10}
                    className="w-full rounded-lg border border-teal-200 bg-white p-3 font-mono text-xs text-teal-900 outline-none focus:border-teal-500"
                  />
                  <Button onClick={runJson} disabled={loading} className="w-full" size="lg">
                    {loading ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" /> Running prediction… {elapsedSec}s
                      </>
                    ) : (
                      "Run prediction"
                    )}
                  </Button>
                </div>
              )}

              {loading ? (
                <p className="text-center text-xs text-copper-600">
                  Cleaning data, running 6 diagnostic methods, and scoring every transformer — large files
                  (thousands of rows) can take up to a minute. This will not disappear if you wait.
                </p>
              ) : (
                <p className="text-center text-xs text-teal-400">
                  Sends data directly to the Flask inference service (/predict).
                </p>
              )}
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
