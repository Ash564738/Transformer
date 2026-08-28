// src/lib/api.ts
import type { DgaPayload } from "@/types/dga";

const BACKEND_PREFIX =
  "https://transformer-kgen.onrender.com";

const AUTH_TOKEN_KEY = "dga-auth-token";

const DEFAULT_REQUEST_TIMEOUT_MS = 30_000;
const HEALTH_REQUEST_TIMEOUT_MS = 15_000;

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

  status?:
    | "queued"
    | "running"
    | "completed"
    | "failed";

  message?: string;
  error?: string;

  result?: DgaPayload;

  predictions?: unknown;
  rows?: unknown;
  preview_rows?: unknown;
  transformer_summary?: unknown;
  transformer_timeseries?: unknown;
  dataset_summary?: unknown;
  student_traditional_comparison?: unknown;
  chat_context_payload?: unknown;

  [key: string]: unknown;
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

async function parseJsonResponse(
  res: Response
): Promise<Record<string, unknown>> {
  const body = await res.json().catch(() => ({}));

  return body &&
    typeof body === "object" &&
    !Array.isArray(body)
    ? (body as Record<string, unknown>)
    : {};
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function requestErrorMessage(
  error: unknown,
  url: string
): string {
  const message =
    error instanceof Error
      ? error.message
      : String(error);

  if (
    /failed to fetch|networkerror|load failed|connection refused/i.test(
      message
    )
  ) {
    return [
      `Production backend request failed: ${message}.`,
      `Attempted URL: ${url}.`,
      `Backend: ${BACKEND_PREFIX}.`,
      "The frontend is locked to the production Render backend.",
      "Verify that the Render service is live and that CORS permits the Vercel frontend origin.",
    ].join(" ");
  }

  return [
    `Backend request failed: ${message}.`,
    `URL: ${url}.`,
  ].join(" ");
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

  const attemptedUrl =
    input instanceof URL
      ? input.toString()
      : String(input);

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
        `Production backend request timed out after ${Math.round(
          timeoutMs / 1000
        )} seconds. URL: ${attemptedUrl}.`
      );
    }

    throw new ApiError(
      requestErrorMessage(error, attemptedUrl)
    );
  } finally {
    window.clearTimeout(timer);
  }
}

async function fetchBackend(
  input: RequestInfo | URL,
  init: RequestInit = {},
  timeoutMs = DEFAULT_REQUEST_TIMEOUT_MS
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
        timeoutMs
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
    : new ApiError(
        "Production backend request failed."
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
    throw new ApiError(
      typeof body.error === "string"
        ? body.error
        : `Authentication request failed (${res.status}).`
    );
  }

  if (
    !body.user ||
    typeof body.user !== "object" ||
    Array.isArray(body.user) ||
    typeof body.token !== "string"
  ) {
    throw new ApiError(
      "Production backend returned an invalid authentication response."
    );
  }

  const user = body.user as Partial<AuthUser>;

  if (
    typeof user.id !== "number" ||
    typeof user.email !== "string" ||
    typeof user.name !== "string"
  ) {
    throw new ApiError(
      "Production backend returned an invalid authenticated user."
    );
  }

  return {
    user: user as AuthUser,
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
        Accept: "application/json",
      },
      body: JSON.stringify({
        email,
        password,
      }),
      cache: "no-store",
      mode: "cors",
    },
    DEFAULT_REQUEST_TIMEOUT_MS
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
        headers: {
          ...authHeaders(),
          Accept: "application/json",
        },
        cache: "no-store",
        mode: "cors",
      },
      20_000
    );

    if (res.status === 401) {
      window.localStorage.removeItem(
        AUTH_TOKEN_KEY
      );
      return null;
    }

    if (!res.ok) {
      return null;
    }

    const body = await parseJsonResponse(res);
    const user = body.user;

    if (
      !user ||
      typeof user !== "object" ||
      Array.isArray(user)
    ) {
      return null;
    }

    return user as AuthUser;
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
        headers: {
          ...authHeaders(),
          Accept: "application/json",
        },
        cache: "no-store",
        mode: "cors",
      },
      15_000
    );
  } catch {
    // Best effort.
  } finally {
    if (typeof window !== "undefined") {
      window.localStorage.removeItem(
        AUTH_TOKEN_KEY
      );
    }
  }
}

export async function resetDataset(): Promise<void> {
  const res = await fetchBackend(
    `${BACKEND_PREFIX}/dataset/reset`,
    {
      method: "POST",
      headers: {
        ...authHeaders(),
        Accept: "application/json",
      },
      cache: "no-store",
      mode: "cors",
    },
    15_000
  );

  if (!res.ok) {
    const body = await parseJsonResponse(res);

    throw new ApiError(
      typeof body.error === "string"
        ? body.error
        : `Dataset reset failed (${res.status}).`
    );
  }
}

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const res = await fetchWithTimeout(
      `${BACKEND_PREFIX}/health`,
      {
        method: "GET",
        headers: {
          Accept: "application/json",
        },
        cache: "no-store",
        mode: "cors",
      },
      HEALTH_REQUEST_TIMEOUT_MS
    );

    return res.ok;
  } catch {
    return false;
  }
}

function extractPredictionResult(
  body: PredictionJobResponse
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
    Array.isArray(body.preview_rows) &&
    body.dataset_summary &&
    typeof body.dataset_summary === "object"
  ) {
    return body as unknown as DgaPayload;
  }

  return null;
}

function extractPredictionJobId(
  body: PredictionJobResponse
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
  res: Response
): Promise<PredictionJobResponse> {
  const body = await parseJsonResponse(res);

  if (!res.ok) {
    throw new ApiError(
      typeof body.error === "string"
        ? body.error
        : `Production prediction request failed (${res.status}).`
    );
  }

  return body as PredictionJobResponse;
}

async function waitForPredictionJob(
  jobId: string
): Promise<DgaPayload> {
  let pollCount = 0;

  for (;;) {
    const res = await fetchBackend(
      `${BACKEND_PREFIX}/predict/status/${encodeURIComponent(
        jobId
      )}`,
      {
        method: "GET",
        headers: {
          ...authHeaders(),
          Accept: "application/json",
        },
        cache: "no-store",
        mode: "cors",
      },
      DEFAULT_REQUEST_TIMEOUT_MS
    );

    if (res.status === 404) {
      throw new ApiError(
        "Production prediction job was not found or has expired."
      );
    }

    const body =
      await parsePredictionJobResponse(res);

    const status = body.status;

    if (status === "completed") {
      const result =
        extractPredictionResult(body);

      if (!result) {
        throw new ApiError(
          "Prediction job completed but production backend returned no prediction result."
        );
      }

      return result;
    }

    if (status === "failed") {
      throw new ApiError(
        body.error ||
          "Production prediction job failed."
      );
    }

    if (
      status !== "queued" &&
      status !== "running"
    ) {
      throw new ApiError(
        `Production backend returned an invalid prediction job status: ${String(
          status ?? "undefined"
        )}.`
      );
    }

    pollCount += 1;

    await sleep(
      pollCount > 30
        ? PREDICTION_POLL_SLOW_INTERVAL_MS
        : PREDICTION_POLL_INTERVAL_MS
    );
  }
}

async function startPredictionJob(
  init: RequestInit
): Promise<DgaPayload> {
  const response = await fetchBackend(
    `${BACKEND_PREFIX}/predict`,
    init,
    PREDICTION_UPLOAD_TIMEOUT_MS
  );

  const body =
    await parsePredictionJobResponse(response);

  const immediateResult =
    extractPredictionResult(body);

  if (
    body.status === "completed" &&
    immediateResult
  ) {
    return immediateResult;
  }

  const jobId =
    extractPredictionJobId(body);

  if (jobId) {
    return waitForPredictionJob(jobId);
  }

  const legacyResult =
    extractPredictionResult(body);

  if (legacyResult) {
    return legacyResult;
  }

  const keys = Object.keys(body);

  throw new ApiError(
    [
      "Production backend returned neither a prediction job id nor a prediction payload.",
      `HTTP ${response.status}.`,
      keys.length
        ? `Response fields: ${keys.join(", ")}.`
        : "Response body was empty.",
    ].join(" ")
  );
}

let predictionInFlight:
  | Promise<DgaPayload>
  | null = null;

export async function runPredictionFromFile(
  file: File
): Promise<DgaPayload> {
  if (predictionInFlight) {
    return predictionInFlight;
  }

  const form = new FormData();
  form.append("file", file);

  predictionInFlight =
    startPredictionJob({
      method: "POST",
      headers: {
        ...authHeaders(),
      },
      body: form,
      cache: "no-store",
      mode: "cors",
    }).finally(() => {
      predictionInFlight = null;
    });

  return predictionInFlight;
}

export async function runPredictionFromJson(
  rows: unknown[]
): Promise<DgaPayload> {
  if (predictionInFlight) {
    return predictionInFlight;
  }

  predictionInFlight =
    startPredictionJob({
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
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
  history?: ChatHistoryTurn[]
): Promise<string> {
  const res = await fetchBackend(
    `${BACKEND_PREFIX}/chat`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
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
    45_000
  );

  if (!res.ok) {
    const body = await parseJsonResponse(res);

    throw new ApiError(
      typeof body.error === "string"
        ? body.error
        : `Chat request failed (${res.status}).`
    );
  }

  const body = await parseJsonResponse(res);

  if (typeof body.answer !== "string") {
    throw new ApiError(
      "Production backend returned an invalid chat response."
    );
  }

  return body.answer;
}

export function getBackendUrl(): string {
  return BACKEND_PREFIX;
}

export function isUsingProductionBackend(): boolean {
  return true;
}

export function isUsingLocalBackend(): boolean {
  return false;
}