// src/lib/api.ts
import type { DgaPayload } from "@/types/dga";

const DEFAULT_PRODUCTION_BACKEND =
  "https://transformer-kgen.onrender.com";

const configuredBackend = process.env.NEXT_PUBLIC_BACKEND_URL?.trim();

const BACKEND_PREFIX = (
  configuredBackend ||
  (process.env.NODE_ENV === "production"
    ? DEFAULT_PRODUCTION_BACKEND
    : "http://127.0.0.1:5000")
).replace(/\/+$/, "");

const AUTH_TOKEN_KEY = "dga-auth-token";

const DEFAULT_REQUEST_TIMEOUT_MS = 30_000;
const HEALTH_REQUEST_TIMEOUT_MS = 15_000;

/*
 * This is only the timeout for uploading the source file and receiving the
 * initial job acknowledgement. It is NOT the prediction timeout.
 *
 * The actual inference runs in the backend background executor and is polled
 * via /predict/status/<job_id>.
 */
const PREDICTION_UPLOAD_TIMEOUT_MS = 15 * 60_000;

const PREDICTION_POLL_INTERVAL_MS = 1_000;
const PREDICTION_POLL_SLOW_INTERVAL_MS = 2_000;

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

interface PredictionJobResponse {
  job_id?: string;
  prediction_job_id?: string;
  jobId?: string;

  status?: "queued" | "running" | "completed" | "failed";
  message?: string;
  error?: string;

  result?: DgaPayload;

  /*
   * Compatibility with the old synchronous backend.
   * If an older Render deployment returns the prediction payload directly,
   * the frontend must accept it instead of reporting "missing job id".
   */
  predictions?: unknown;
  rows?: unknown;
  transformer_summary?: unknown;
  dataset_summary?: unknown;
}

export function getAuthToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(AUTH_TOKEN_KEY);
}

function authHeaders(): Record<string, string> {
  const token = getAuthToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
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

  if (!token) {
    return null;
  }

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

    if (!res.ok) {
      return null;
    }

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

/**
 * The new backend returns a job identifier.
 *
 * We also accept:
 *   - prediction_job_id
 *   - jobId
 *   - a complete legacy DgaPayload
 *
 * The last case is important during a rolling Render deployment or when the
 * frontend is newer than the currently running backend.
 */
function extractPredictionResult(
  body: PredictionJobResponse,
): DgaPayload | null {
  if (
    body.result &&
    typeof body.result === "object"
  ) {
    return body.result;
  }

  if (
    Array.isArray(body.predictions) &&
    Array.isArray(body.rows) &&
    Array.isArray(body.transformer_summary) &&
    body.dataset_summary &&
    typeof body.dataset_summary === "object"
  ) {
    return body as unknown as DgaPayload;
  }

  return null;
}

function extractPredictionJobId(
  body: PredictionJobResponse,
): string | null {
  const candidates = [
    body.job_id,
    body.prediction_job_id,
    body.jobId,
  ];

  for (const candidate of candidates) {
    if (
      typeof candidate === "string" &&
      candidate.trim()
    ) {
      return candidate.trim();
    }
  }

  return null;
}

async function parsePredictionJobResponse(
  res: Response,
): Promise<PredictionJobResponse> {
  const body = await parseJsonResponse(res);

  /*
   * Some valid acknowledgement responses are 202; normal status endpoints
   * are 200. Any other non-2xx response is an API failure.
   */
  if (!res.ok) {
    throw new ApiError(
      typeof body.error === "string"
        ? body.error
        : `Prediction request failed (${res.status}).`,
    );
  }

  return body as PredictionJobResponse;
}

async function waitForPredictionJob(
  jobId: string,
): Promise<DgaPayload> {
  let pollCount = 0;

  for (;;) {
    const res = await fetchBackend(
      `${BACKEND_PREFIX}/predict/status/${encodeURIComponent(jobId)}`,
      {
        method: "GET",
        headers: authHeaders(),
        cache: "no-store",
        mode: "cors",
      },
      DEFAULT_REQUEST_TIMEOUT_MS,
    );

    const body = await parsePredictionJobResponse(res);

    const status = body.status;

    if (status === "completed") {
      const result = extractPredictionResult(body);

      if (!result) {
        throw new ApiError(
          "Prediction job completed but backend returned no prediction result.",
        );
      }

      return result;
    }

    if (status === "failed") {
      throw new ApiError(
        body.error || "Prediction job failed.",
      );
    }

    /*
     * 404 is checked after parsing because it is a terminal API response and
     * must not be treated as a transient polling state.
     */
    if (res.status === 404) {
      throw new ApiError(
        "Prediction job was not found or has expired.",
      );
    }

    if (
      status !== "queued" &&
      status !== "running"
    ) {
      throw new ApiError(
        `Backend returned an invalid prediction job status: ${String(
          status ?? "undefined",
        )}.`,
      );
    }

    pollCount += 1;

    await sleep(
      pollCount > 30
        ? PREDICTION_POLL_SLOW_INTERVAL_MS
        : PREDICTION_POLL_INTERVAL_MS,
    );
  }
}

async function startPredictionJob(
  init: RequestInit,
): Promise<DgaPayload> {
  const response = await fetchBackend(
    `${BACKEND_PREFIX}/predict`,
    init,
    PREDICTION_UPLOAD_TIMEOUT_MS,
  );

  const body = await parsePredictionJobResponse(response);

  /*
   * Case 1:
   * New async backend already completed the request.
   */
  const immediateResult = extractPredictionResult(body);

  if (
    body.status === "completed" &&
    immediateResult
  ) {
    return immediateResult;
  }

  /*
   * Case 2:
   * New async backend returned queued/running with a job id.
   */
  const jobId = extractPredictionJobId(body);

  if (jobId) {
    return waitForPredictionJob(jobId);
  }

  /*
   * Case 3:
   * Very old backend returned the DgaPayload directly without any async job
   * wrapper. Accept it instead of producing the misleading:
   * "Backend did not return a prediction job id."
   */
  const legacyResult = extractPredictionResult(body);

  if (legacyResult) {
    return legacyResult;
  }

  /*
   * At this point the backend genuinely returned a malformed prediction
   * response. Include the status and visible response shape to make diagnosis
   * deterministic rather than guessing.
   */
  const keys = Object.keys(body);

  throw new ApiError(
    [
      "Backend returned neither a prediction job id nor a prediction payload.",
      `HTTP ${response.status}.`,
      keys.length
        ? `Response fields: ${keys.join(", ")}.`
        : "Response body was empty.",
    ].join(" "),
  );
}

let predictionInFlight: Promise<DgaPayload> | null = null;

export async function runPredictionFromFile(
  file: File,
): Promise<DgaPayload> {
  if (predictionInFlight) {
    return predictionInFlight;
  }

  const form = new FormData();
  form.append("file", file);

  predictionInFlight = startPredictionJob({
    method: "POST",
    headers: authHeaders(),
    body: form,
    cache: "no-store",
    mode: "cors",
  }).finally(() => {
    predictionInFlight = null;
  });

  return predictionInFlight;
}

export async function runPredictionFromJson(
  rows: unknown[],
): Promise<DgaPayload> {
  if (predictionInFlight) {
    return predictionInFlight;
  }

  predictionInFlight = startPredictionJob({
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
  }).finally(() => {
    predictionInFlight = null;
  });

  return predictionInFlight;
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