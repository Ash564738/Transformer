import type { DgaPayload, DgaRow } from "@/types/dga";

// Called directly from the browser (not through Next.js's rewrite proxy):
// large-dataset /predict calls can run close to a minute, and the dev
// server's proxy was resetting the connection on requests that long. Flask
// has CORS enabled (backend/app.py) specifically so this direct call works.
const BACKEND_PREFIX = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:5000";

const AUTH_TOKEN_KEY = "dga-auth-token";

export class ApiError extends Error {}

export function getAuthToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(AUTH_TOKEN_KEY);
}

function authHeaders(): Record<string, string> {
  const token = getAuthToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export interface AuthUser {
  id: number;
  email: string;
  name: string;
}

async function handleAuthResponse(res: Response): Promise<{ user: AuthUser; token: string }> {
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new ApiError(body.error ?? "Authentication request failed.");
  }
  return body;
}

export async function loginAccount(email: string, password: string) {
  const res = await fetch(`${BACKEND_PREFIX}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  return handleAuthResponse(res);
}

export async function fetchCurrentUser(): Promise<AuthUser | null> {
  const token = getAuthToken();
  if (!token) return null;
  try {
    const res = await fetch(`${BACKEND_PREFIX}/auth/me`, { headers: authHeaders(), cache: "no-store" });
    if (!res.ok) return null;
    const body = await res.json();
    return body.user as AuthUser;
  } catch {
    return null;
  }
}

export async function logoutAccount(): Promise<void> {
  try {
    await fetch(`${BACKEND_PREFIX}/auth/logout`, { method: "POST", headers: authHeaders() });
  } catch {
    // best-effort — the client clears its own token regardless
  }
}

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BACKEND_PREFIX}/health`, { cache: "no-store" });
    return res.ok;
  } catch {
    return false;
  }
}

export async function runPredictionFromFile(file: File): Promise<DgaPayload> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BACKEND_PREFIX}/predict`, { method: "POST", headers: authHeaders(), body: form });
  return handlePredictResponse(res);
}

export async function runPredictionFromJson(rows: unknown[]): Promise<DgaPayload> {
  const res = await fetch(`${BACKEND_PREFIX}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ data: rows }),
  });
  return handlePredictResponse(res);
}

async function handlePredictResponse(res: Response): Promise<DgaPayload> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }));
    throw new ApiError(body.error ?? "Prediction request failed.");
  }
  return res.json();
}

// These build URLs for the backend's matplotlib-rendered chart images
// (backend/app.py `/chart/*` routes, which call backend/dga/duval_triangle.py
// and backend/dga/duval_pentagon.py directly) — the frontend never redraws
// the diagram itself, only embeds the image the backend already produced.
export function duvalTriangleImageUrl(row: DgaRow): string {
  const params = new URLSearchParams({
    ch4: String(row.ch4 ?? 0),
    c2h4: String(row.c2h4 ?? 0),
    c2h2: String(row.c2h2 ?? 0),
    fault: String(row.duval_triangle_fault ?? ""),
  });
  return `${BACKEND_PREFIX}/chart/duval-triangle?${params.toString()}`;
}

export function duvalPentagonImageUrl(row: DgaRow): string {
  const params = new URLSearchParams({
    h2: String(row.h2 ?? 0),
    ch4: String(row.ch4 ?? 0),
    c2h6: String(row.c2h6 ?? 0),
    c2h4: String(row.c2h4 ?? 0),
    c2h2: String(row.c2h2 ?? 0),
    fault_p1: String(row.fault_p1 ?? ""),
    fault_p2: String(row.duval_pentagon_fault ?? ""),
  });
  return `${BACKEND_PREFIX}/chart/duval-pentagon?${params.toString()}`;
}

export async function askChatBackend(
  question: string,
  context: unknown
): Promise<string> {
  const res = await fetch(`${BACKEND_PREFIX}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ question, context }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }));
    throw new ApiError(body.error ?? "Chat request failed.");
  }
  const data = await res.json();
  return data.answer as string;
}
