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

const PREDICTION_START_TIMEOUT_MS = 60_000;
const DEFAULT_REQUEST_TIMEOUT_MS = 60_000;
const PREDICTION_POLL_INTERVAL_MS = 2_000;
const PREDICTION_MAX_WAIT_MS = 30 * 60 * 1000;
const PREDICTION_NETWORK_RETRY_LIMIT = 8;

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

interface PredictionStartResponse {
  job_id: string;
  status: "queued";
  poll_url?: string;
}

type PredictionJobStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "not_found";

interface PredictionStatusResponse {
  job_id: string;
  status: PredictionJobStatus;
  result?: DgaPayload;
  error?: string;
  elapsed_seconds?: number;
  running_seconds?: number;
}

function isRecord(
  value: unknown
): value is Record<string, unknown> {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value)
  );
}

function isPredictionStatusResponse(
  value: unknown
): value is PredictionStatusResponse {
  if (!isRecord(value)) {
    return false;
  }

  const status = value.status;

  return (
    typeof value.job_id === "string" &&
    (
      status === "queued" ||
      status === "running" ||
      status === "completed" ||
      status === "failed" ||
      status === "not_found"
    )
  );
}

async function pollPredictionStatus(
  jobId: string
): Promise<DgaPayload> {
  const deadline =
    Date.now() + PREDICTION_MAX_WAIT_MS;

  let consecutiveNetworkFailures = 0;
  let consecutiveNotFound = 0;

  while (Date.now() < deadline) {
    await new Promise<void>((resolve) => {
      window.setTimeout(
        resolve,
        PREDICTION_POLL_INTERVAL_MS
      );
    });

    try {
      const response = await fetchWithTimeout(
        `${BACKEND_PREFIX}/predict/status/${encodeURIComponent(jobId)}`,
        {
          method: "GET",
          headers: authHeaders(),
          cache: "no-store",
          mode: "cors",
        },
        DEFAULT_REQUEST_TIMEOUT_MS
      );

      const raw =
        await response.json().catch(() => null);

      if (
        !raw ||
        !isPredictionStatusResponse(raw)
      ) {
        consecutiveNetworkFailures += 1;

        if (
          consecutiveNetworkFailures >=
          PREDICTION_NETWORK_RETRY_LIMIT
        ) {
          throw new ApiError(
            `Backend returned an invalid prediction status response. URL: ${BACKEND_PREFIX}/predict/status/${jobId}`
          );
        }

        continue;
      }

      consecutiveNetworkFailures = 0;

      if (raw.status === "completed") {
        if (!raw.result) {
          throw new ApiError(
            "Prediction completed but returned no result."
          );
        }

        return raw.result;
      }

      if (raw.status === "failed") {
        throw new ApiError(
          raw.error ||
            "Prediction worker failed."
        );
      }

      if (raw.status === "not_found") {
        consecutiveNotFound += 1;

        // A transient restart/proxy race should not immediately kill a
        // prediction that has already been accepted by the backend.
        if (consecutiveNotFound < 10) {
          continue;
        }

        throw new ApiError(
          raw.error ||
            "Prediction job not found or expired."
        );
      }

      consecutiveNotFound = 0;
    } catch (error) {
      if (error instanceof ApiError) {
        const message = error.message.toLowerCase();

        if (
          message.includes("not found or expired")
        ) {
          throw error;
        }

        consecutiveNetworkFailures += 1;

        if (
          consecutiveNetworkFailures >=
          PREDICTION_NETWORK_RETRY_LIMIT
        ) {
          throw error;
        }

        continue;
      }

      consecutiveNetworkFailures += 1;

      if (
        consecutiveNetworkFailures >=
        PREDICTION_NETWORK_RETRY_LIMIT
      ) {
        throw new ApiError(
          `Unable to reach prediction status endpoint after ${consecutiveNetworkFailures} attempts. URL: ${BACKEND_PREFIX}/predict/status/${jobId}`
        );
      }
    }
  }

  throw new ApiError(
    "Prediction exceeded the 30-minute client wait limit."
  );
}

async function startPrediction(
  init: RequestInit
): Promise<DgaPayload> {
  const startResponse =
    await fetchWithTimeout(
      `${BACKEND_PREFIX}/predict`,
      init,
      PREDICTION_START_TIMEOUT_MS
    );

  if (startResponse.status !== 202) {
    return handlePredictResponse(startResponse);
  }

  const raw =
    await startResponse.json().catch(() => null);

  if (
    !raw ||
    typeof raw !== "object" ||
    Array.isArray(raw) ||
    typeof (raw as PredictionStartResponse).job_id !== "string"
  ) {
    throw new ApiError(
      "Backend returned an invalid prediction job response."
    );
  }

  return pollPredictionStatus(
    (raw as PredictionStartResponse).job_id
  );
}

export async function runPredictionFromFile(
  file: File
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
  rows: unknown[]
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