"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle2, X } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import type { FieldInspectionConfirmation } from "@/types/dga";

export function ConfirmInspectionDialog({
  transformerId,
  suggestedFaultType,
  open,
  onClose,
}: {
  transformerId: string;
  suggestedFaultType: string;
  open: boolean;
  onClose: () => void;
}) {
  const [confirmedBy, setConfirmedBy] = useState("");
  const [faultType, setFaultType] = useState(suggestedFaultType);
  const [notes, setNotes] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!confirmedBy.trim()) {
      toast.error("Enter the confirming engineer's name.");
      return;
    }
    setSubmitting(true);
    try {
      const res = await fetch("/api/confirmations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transformer_id: transformerId,
          confirmed_by: confirmedBy,
          confirmed_fault_type: faultType,
          notes,
        } satisfies Partial<FieldInspectionConfirmation>),
      });
      if (!res.ok) throw new Error("Could not save the confirmation.");
      toast.success(`Field inspection result recorded for ${transformerId}.`);
      onClose();
      setConfirmedBy("");
      setNotes("");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setSubmitting(false);
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
            className="fixed inset-0 z-50 bg-teal-950/45 backdrop-blur-[1px]"
            onClick={onClose}
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: 12 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 12 }}
            className="fixed left-1/2 top-1/2 z-50 w-full max-w-md -translate-x-1/2 -translate-y-1/2 rounded-2xl bg-white p-6 shadow-2xl"
          >
            <div className="mb-4 flex items-start justify-between">
              <div className="flex items-center gap-2.5">
                <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-teal-50 text-teal-700">
                  <CheckCircle2 className="h-5 w-5" />
                </span>
                <div>
                  <h2 className="text-base font-bold text-teal-900">Confirm Field Inspection Result</h2>
                  <p className="text-xs text-teal-500">{transformerId} · ground-truth for model retraining</p>
                </div>
              </div>
              <button onClick={onClose} className="rounded-full p-1 text-teal-400 hover:bg-teal-50 cursor-pointer">
                <X className="h-4 w-4" />
              </button>
            </div>

            <form onSubmit={submit} className="space-y-3.5">
              <Field label="Confirming engineer">
                <input
                  value={confirmedBy}
                  onChange={(e) => setConfirmedBy(e.target.value)}
                  placeholder="Full name"
                  className="h-10 w-full rounded-lg border border-teal-200 px-3 text-sm outline-none focus:border-teal-500"
                  required
                />
              </Field>
              <Field label="Confirmed fault type">
                <input
                  value={faultType}
                  onChange={(e) => setFaultType(e.target.value)}
                  placeholder="e.g. D2 — high-energy arcing"
                  className="h-10 w-full rounded-lg border border-teal-200 px-3 text-sm outline-none focus:border-teal-500"
                />
              </Field>
              <Field label="Field notes">
                <textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  rows={4}
                  placeholder="Findings from the physical inspection…"
                  className="w-full rounded-lg border border-teal-200 px-3 py-2 text-sm outline-none focus:border-teal-500"
                />
              </Field>
              <p className="text-[11px] text-teal-400">
                This record is stored as ground-truth and timestamped automatically for future model retraining.
              </p>
              <div className="flex justify-end gap-2 pt-1">
                <Button type="button" variant="ghost" size="sm" onClick={onClose}>
                  Cancel
                </Button>
                <Button type="submit" size="sm" disabled={submitting}>
                  {submitting ? "Saving…" : "Confirm Field Inspection Result"}
                </Button>
              </div>
            </form>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-semibold text-teal-600">{label}</span>
      {children}
    </label>
  );
}
