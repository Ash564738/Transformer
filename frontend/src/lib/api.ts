// src/lib/api.ts
import type { DgaPayload } from "@/types/dga";

const DEFAULT_PRODUCTION_BACKEND =
  "https://transformer-kgen.onrender.com";

const configuredBackend =
  process.env.NEXT_PUBLIC_BACKEND_URL?.trim();

const BACKEND_PREFIX = (
  configuredBackend ||
  (process.env.NODE_ENV === "production"
    ? DEFAULT_PRODUCTION_BACKEND
    : "http://127.0.0.1:5000")
).replace(/\/+$/, "");

const AUTH_TOKEN_KEY = "dga-auth-token";

const PREDICTION_START_TIME_TIMEOUT_MS = 20 * 60 * 1000;
const DEFAULT_REQUEST_TIMEOUT_MS = 60_000;

export class ApiError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ApiError";
  }
}

export function getAuthToken(): string | null {
  if (typeof window === "undefined") {
    return null;
  }

  return window.localStorage.getItem(AUTH_TOKEN_KEY);
}

function authHeaders(): Record<string, string> {
  const token = getAuthToken();

  return token
    ? {
        Authorization: `Bearer ${token}`,
      }
    : {};
}

export interface AuthUser {
  id: number;
  email: string;
  name: string;
}

async function parseJsonResponse(
  res: Response
): Promise<Record<string, unknown>> {
  const body = await res.json().catch(() => ({}));

  if (
    body &&
    typeof body === "object" &&
    !Array.isArray(body)
  ) {
    return body as Record<string, unknown>;
  }

  return {};
}

async function fetchWithTimeout(
  input: RequestInfo | URL,
  init: RequestInit = {},
  timeoutMs = DEFAULT_REQUEST_TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();

  const timer = window.setTimeout(() => {
    controller.abort();
  }, timeoutMs);

  try {
    return await fetch(input, {
      ...init,
      signal: controller.signal,
    });
  } catch (error) {
    if (
      error instanceof DOMException &&
      error.name === "AbortError"
    ) {
      throw new ApiError(
        `Backend request timed out after ${Math.round(
          timeoutMs / 1000
        )} seconds. URL: ${String(input)}`
      );
    }

    const message =
      error instanceof Error
        ? error.message
        : String(error);

    throw new ApiError(
      `Backend request failed: ${message}. URL: ${String(input)}`
    );
  } finally {
    window.clearTimeout(timer);
  }
}

async function fetchBackend(
  input: RequestInfo | URL,
  init?: RequestInit
): Promise<Response> {
  return fetchWithTimeout(
    input,
    init,
    DEFAULT_REQUEST_TIMEOUT_MS
  );
}

async function handleAuthResponse(
  res: Response
): Promise<{
  user: AuthUser;
  token: string;
}> {
  const body = await parseJsonResponse(res);

  if (!res.ok) {
    const message =
      typeof body.error === "string"
        ? body.error
        : `Authentication request failed (${res.status}).`;

    throw new ApiError(message);
  }

  if (
    !body.user ||
    typeof body.user !== "object" ||
    typeof body.token !== "string"
  ) {
    throw new ApiError(
      "Backend returned an invalid authentication response."
    );
  }

  return {
    user: body.user as AuthUser,
    token: body.token,
  };
}

export async function loginAccount(
  email: string,
  password: string
): Promise<{
  user: AuthUser;
  token: string;
}> {
  const res = await fetchBackend(
    `${BACKEND_PREFIX}/auth/login`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        email,
        password,
      }),
      cache: "no-store",
    }
  );

  return handleAuthResponse(res);
}

export async function fetchCurrentUser(): Promise<AuthUser | null> {
  const token = getAuthToken();

  if (!token) {
    return null;
  }

  try {
    const res = await fetch(
      `${BACKEND_PREFIX}/auth/me`,
      {
        method: "GET",
        headers: authHeaders(),
        cache: "no-store",
      }
    );

    if (!res.ok) {
      return null;
    }

    const body = await parseJsonResponse(res);

    if (
      !body.user ||
      typeof body.user !== "object"
    ) {
      return null;
    }

    return body.user as AuthUser;
  } catch {
    return null;
  }
}

export async function logoutAccount(): Promise<void> {
  try {
    await fetch(
      `${BACKEND_PREFIX}/auth/logout`,
      {
        method: "POST",
        headers: authHeaders(),
        cache: "no-store",
      }
    );
  } catch {
    // Best effort.
  }
}

export async function resetDataset(): Promise<void> {
  try {
    await fetch(
      `${BACKEND_PREFIX}/dataset/reset`,
      {
        method: "POST",
        headers: authHeaders(),
        cache: "no-store",
      }
    );
  } catch {
    // Best effort.
  }
}

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const res = await fetch(
      `${BACKEND_PREFIX}/health`,
      {
        method: "GET",
        cache: "no-store",
      }
    );

    return res.ok;
  } catch {
    return false;
  }
}

async function handlePredictResponse(
  res: Response
): Promise<DgaPayload> {
  const body = await res.json().catch(() => null);

  if (!res.ok) {
    const message =
      body &&
      typeof body === "object" &&
      typeof (body as { error?: unknown }).error === "string"
        ? (body as { error: string }).error
        : `Prediction request failed (${res.status}).`;

    throw new ApiError(message);
  }

  if (!body || typeof body !== "object") {
    throw new ApiError(
      "Backend returned an invalid prediction response."
    );
  }

  return body as DgaPayload;
}

async function startPrediction(init: RequestInit): Promise<DgaPayload> {
  const response = await fetchWithTimeout(
    `${BACKEND_PREFIX}/predict`,
    init,
    PREDICTION_START_TIME_TIMEOUT_MS
  );

  return handlePredictResponse(response);
}

export async function runPredictionFromFile(file: File): Promise<DgaPayload> {
  const form = new FormData();
  form.append("file", file);

  return startPrediction({
    method: "POST",
    headers: authHeaders(),
    body: form,
    cache: "no-store",
    mode: "cors",
  });
}

export async function runPredictionFromJson(rows: unknown[]): Promise<DgaPayload> {
  return startPrediction({
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
    },
    body: JSON.stringify({ data: rows }),
    cache: "no-store",
    mode: "cors",
  });
}

export interface ChatHistoryTurn {
  role: "user" | "assistant";
  content: string;
}

export async function askChatBackend(
  question: string,
  context: unknown,
  history?: ChatHistoryTurn[]
): Promise<string> {
  const res = await fetchBackend(
    `${BACKEND_PREFIX}/chat`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...authHeaders(),
      },
      body: JSON.stringify({
        question,
        context,
        history,
      }),
      cache: "no-store",
      mode: "cors",
    }
  );

  if (!res.ok) {
    const body = await parseJsonResponse(res);

    const message =
      typeof body.error === "string"
        ? body.error
        : `Chat request failed (${res.status}).`;

    throw new ApiError(message);
  }

  const body = await parseJsonResponse(res);

  if (typeof body.answer !== "string") {
    throw new ApiError(
      "Backend returned an invalid chat response."
    );
  }

  return body.answer;
}

export function getBackendUrl(): string {
  return BACKEND_PREFIX;
}