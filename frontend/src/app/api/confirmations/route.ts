import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";
import type { FieldInspectionConfirmation } from "@/types/dga";

// SCR-03 (Field Inspection Confirmation) needs somewhere to persist the
// ground-truth record the maintenance engineer submits. The Flask backend
// (app.py) has no endpoint for this and per the project's scope this
// frontend rewrite must not touch backend Python files, so confirmations
// are stored locally within the Next.js app as a small JSON store.
const DATA_FILE = path.join(process.cwd(), "data", "confirmations.json");

async function readAll(): Promise<FieldInspectionConfirmation[]> {
  try {
    const raw = await fs.readFile(DATA_FILE, "utf-8");
    return JSON.parse(raw);
  } catch {
    return [];
  }
}

async function writeAll(records: FieldInspectionConfirmation[]) {
  await fs.mkdir(path.dirname(DATA_FILE), { recursive: true });
  await fs.writeFile(DATA_FILE, JSON.stringify(records, null, 2), "utf-8");
}

export async function GET(request: NextRequest) {
  const transformerId = request.nextUrl.searchParams.get("transformer_id");
  const all = await readAll();
  const filtered = transformerId ? all.filter((r) => r.transformer_id === transformerId) : all;
  return NextResponse.json(filtered);
}

export async function POST(request: NextRequest) {
  const body = await request.json();
  if (!body.transformer_id || !body.confirmed_by) {
    return NextResponse.json({ error: "transformer_id and confirmed_by are required." }, { status: 400 });
  }
  const record: FieldInspectionConfirmation = {
    id: `${body.transformer_id}-${Date.now()}`,
    transformer_id: body.transformer_id,
    confirmed_by: body.confirmed_by,
    confirmed_fault_type: body.confirmed_fault_type ?? "",
    notes: body.notes ?? "",
    confirmed_at: new Date().toISOString(),
  };
  const all = await readAll();
  all.unshift(record);
  await writeAll(all);
  return NextResponse.json(record, { status: 201 });
}
