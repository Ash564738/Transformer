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

const PREDICTION_REQUEST_TIMEOUT_MS = 120_000;
const DEFAULT_REQUEST_TIMEOUT_MS = 30_000;
const HEALTH_REQUEST_TIMEOUT_MS = 15_000;

const RETRYABLE_STATUS = new Set([502, 503, 504]);
const MAX_NETWORK_RETRIES = 2;

export class ApiError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ApiError";
  }
}

export interface AuthUser {
  id: number;
  email: string;
  name: string;
}

export interface ChatHistoryTurn {
  role: "user" | "assistant";
  content: string;
}

export function getAuthToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(AUTH_TOKEN_KEY);
}

function authHeaders(): Record<string, string> {
  const token = getAuthToken();
  return token
    ? { Authorization: `Bearer ${token}` }
    : {};
}

async function parseJsonResponse(
  res: Response,
): Promise<Record<string, unknown>> {
  const body = await res.json().catch(() => ({}));
  return body &&
    typeof body === "object" &&
    !Array.isArray(body)
    ? (body as Record<string, unknown>)
    : {};
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchWithTimeout(
  input: RequestInfo | URL,
  init: RequestInit = {},
  timeoutMs = DEFAULT_REQUEST_TIMEOUT_MS,
): Promise<Response> {
  const controller = new AbortController();
  const timer = window.setTimeout(
    () => controller.abort(),
    timeoutMs,
  );

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
          timeoutMs / 1000,
        )} seconds.`,
      );
    }

    const message =
      error instanceof Error
        ? error.message
        : String(error);

    throw new ApiError(
      `Backend request failed: ${message}.`,
    );
  } finally {
    window.clearTimeout(timer);
  }
}

async function fetchBackend(
  input: RequestInfo | URL,
  init: RequestInit = {},
  timeoutMs = DEFAULT_REQUEST_TIMEOUT_MS,
): Promise<Response> {
  let lastError: unknown = null;

  for (
    let attempt = 0;
    attempt <= MAX_NETWORK_RETRIES;
    attempt += 1
  ) {
    try {
      const response = await fetchWithTimeout(
        input,
        init,
        timeoutMs,
      );

      if (
        RETRYABLE_STATUS.has(response.status) &&
        attempt < MAX_NETWORK_RETRIES
      ) {
        await sleep(750 * 2 ** attempt);
        continue;
      }

      return response;
    } catch (error) {
      lastError = error;

      if (attempt >= MAX_NETWORK_RETRIES) {
        throw error;
      }

      await sleep(750 * 2 ** attempt);
    }
  }

  throw lastError instanceof Error
    ? lastError
    : new ApiError("Backend request failed.");
}

async function handleAuthResponse(
  res: Response,
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
      "Backend returned an invalid authentication response.",
    );
  }

  return {
    user: body.user as AuthUser,
    token: body.token,
  };
}

export async function loginAccount(
  email: string,
  password: string,
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
      mode: "cors",
    },
    30_000,
  );

  return handleAuthResponse(res);
}

export async function fetchCurrentUser(): Promise<AuthUser | null> {
  const token = getAuthToken();

  if (!token) return null;

  try {
    const res = await fetchBackend(
      `${BACKEND_PREFIX}/auth/me`,
      {
        method: "GET",
        headers: authHeaders(),
        cache: "no-store",
        mode: "cors",
      },
      20_000,
    );

    if (!res.ok) return null;

    const body = await parseJsonResponse(res);

    return body.user &&
      typeof body.user === "object"
      ? (body.user as AuthUser)
      : null;
  } catch {
    return null;
  }
}

export async function logoutAccount(): Promise<void> {
  try {
    await fetchBackend(
      `${BACKEND_PREFIX}/auth/logout`,
      {
        method: "POST",
        headers: authHeaders(),
        cache: "no-store",
        mode: "cors",
      },
      15_000,
    );
  } catch {
    // Best effort.
  }
}

export async function resetDataset(): Promise<void> {
  const res = await fetchBackend(
    `${BACKEND_PREFIX}/dataset/reset`,
    {
      method: "POST",
      headers: authHeaders(),
      cache: "no-store",
      mode: "cors",
    },
    15_000,
  );

  if (!res.ok) {
    const body = await parseJsonResponse(res);

    throw new ApiError(
      typeof body.error === "string"
        ? body.error
        : `Dataset reset failed (${res.status}).`,
    );
  }
}

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const res = await fetchWithTimeout(
      `${BACKEND_PREFIX}/health`,
      {
        method: "GET",
        cache: "no-store",
        mode: "cors",
      },
      HEALTH_REQUEST_TIMEOUT_MS,
    );

    return res.ok;
  } catch {
    return false;
  }
}

async function handlePredictResponse(
  res: Response,
): Promise<DgaPayload> {
  const body = await res.json().catch(() => null);

  if (!res.ok) {
    const message =
      body &&
      typeof body === "object" &&
      typeof (body as { error?: unknown }).error === "string"
        ? (body as { error: string }).error
        : `Prediction request failed (${res.status}).`;

    if (res.status === 409) {
      throw new ApiError(
        `${message} Please wait for the current prediction to finish.`,
      );
    }

    throw new ApiError(message);
  }

  if (
    !body ||
    typeof body !== "object" ||
    Array.isArray(body)
  ) {
    throw new ApiError(
      "Backend returned an invalid prediction response.",
    );
  }

  return body as DgaPayload;
}

async function startPrediction(
  init: RequestInit,
): Promise<DgaPayload> {
  const response = await fetchBackend(
    `${BACKEND_PREFIX}/predict`,
    init,
    PREDICTION_REQUEST_TIMEOUT_MS,
  );

  return handlePredictResponse(response);
}

export async function runPredictionFromFile(
  file: File,
): Promise<DgaPayload> {
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

export async function runPredictionFromJson(
  rows: unknown[],
): Promise<DgaPayload> {
  return startPrediction({
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
    },
    body: JSON.stringify({
      data: rows,
    }),
    cache: "no-store",
    mode: "cors",
  });
}

export async function askChatBackend(
  question: string,
  context: unknown,
  history?: ChatHistoryTurn[],
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
    },
    45_000,
  );

  if (!res.ok) {
    const body = await parseJsonResponse(res);

    throw new ApiError(
      typeof body.error === "string"
        ? body.error
        : `Chat request failed (${res.status}).`,
    );
  }

  const body = await parseJsonResponse(res);

  if (typeof body.answer !== "string") {
    throw new ApiError(
      "Backend returned an invalid chat response.",
    );
  }

  return body.answer;
}

export function getBackendUrl(): string {
  return BACKEND_PREFIX;
}