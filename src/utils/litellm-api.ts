import type { LiteLLMModel, LiteLLMModelInfo, LiteLLMModelInfoResponse, LiteLLMModelsResponse } from '../types'
import { CAPABILITY_FLAGS } from './model-capabilities'
import { resolveReasoningEfforts } from './reasoning-efforts'

export const DEFAULT_LITELLM_URL = 'http://localhost:4000'
const MODELS_ENDPOINT = '/v1/models'
const MODEL_INFO_ENDPOINT = '/v1/model/info'
// Health checks originally failed fast (3 s) so auto-detection stayed
// snappy, but a proxy that boots lazily — behind a VM, container or
// remote host start-up — is then reported as offline and discovery is
// skipped until OpenCode restarts. The budget is configurable now and
// generous by default; the discovery fetches keep their own budget
// because `/v1/model/info` payloads from remote proxies with many
// database-defined models can be large and slow to generate.
const DEFAULT_HEALTH_TIMEOUT_MS = 15000
const DEFAULT_FETCH_TIMEOUT_MS = 15000
// A single probe is not enough at start-up: a proxy that is still opening
// its listener refuses the first connection and answers a second later,
// and a DNS name that is not resolvable yet (the guest behind a VPN, a
// container that is still being created) recovers the same way. Three
// attempts cover that without turning a genuinely offline proxy into a
// long wait.
const DEFAULT_HEALTH_ATTEMPTS = 3
// Fixed gaps between attempts, not exponential: the window being covered
// is a proxy that is seconds away from being ready, so there is nothing
// to gain from backing off further. Later attempts reuse the last value.
const HEALTH_RETRY_BACKOFF_MS: readonly number[] = [400, 1200]

/**
 * Historical name of the health-check budget, kept exported for callers
 * that referenced it. The effective value is `getHealthTimeoutMs()`.
 */
export const HEALTH_TIMEOUT_MS = DEFAULT_HEALTH_TIMEOUT_MS

/**
 * Per-request timeout for discovery fetches (`/v1/models`,
 * `/v1/model/info`), overridable via `LITELLM_REQUEST_TIMEOUT_MS` for
 * proxies that legitimately need longer than the 15 s default (issue
 * #20). Invalid values fall back to the default.
 */
export function getRequestTimeoutMs(): number {
  const raw = process.env.LITELLM_REQUEST_TIMEOUT_MS
  if (!raw) return DEFAULT_FETCH_TIMEOUT_MS
  // Number() (not parseInt) so '12.5' or '60abc' fall back to the
  // default instead of silently truncating to a bogus 12 ms timeout.
  const parsed = Number(raw)
  if (!Number.isInteger(parsed) || parsed <= 0) return DEFAULT_FETCH_TIMEOUT_MS
  return parsed
}

/**
 * Timeout for the health probe (`/v1/models`), overridable via
 * `OPENCODE_LITELLM_HEALTH_TIMEOUT_MS`. Unset or invalid values fall
 * back to the default, which is long enough for a proxy that is still
 * starting inside a VM or container.
 */
export function getHealthTimeoutMs(): number {
  const raw = process.env.OPENCODE_LITELLM_HEALTH_TIMEOUT_MS
  if (!raw) return DEFAULT_HEALTH_TIMEOUT_MS
  // Number() (not parseInt) so '7.5' or '15s' fall back to the default
  // instead of silently truncating to a bogus 7 ms timeout.
  const parsed = Number(raw)
  if (!Number.isInteger(parsed) || parsed <= 0) return DEFAULT_HEALTH_TIMEOUT_MS
  return parsed
}

/**
 * Number of health-probe attempts per start, overridable via
 * `OPENCODE_LITELLM_HEALTH_ATTEMPTS`. Unset, non-integer or `< 1` values
 * fall back to 3. An operator debugging a genuinely offline proxy can set
 * `1` to get the old fail-fast behaviour.
 */
export function getHealthAttempts(): number {
  const raw = process.env.OPENCODE_LITELLM_HEALTH_ATTEMPTS
  if (!raw) return DEFAULT_HEALTH_ATTEMPTS
  // Number() (not parseInt) so '2.5' or '3x' fall back to the default
  // instead of silently truncating to a surprising attempt count.
  const parsed = Number(raw)
  if (!Number.isInteger(parsed) || parsed < 1) return DEFAULT_HEALTH_ATTEMPTS
  return parsed
}

/**
 * How long to wait before attempt number `completed + 1`, given that
 * `completed` attempts have already failed. Past the last configured gap
 * the final value repeats, so raising the attempt count with
 * `OPENCODE_LITELLM_HEALTH_ATTEMPTS` never yields a zero-delay hammering
 * loop.
 */
function healthBackoffMs(completed: number): number {
  const index = Math.min(Math.max(completed, 1), HEALTH_RETRY_BACKOFF_MS.length) - 1
  return HEALTH_RETRY_BACKOFF_MS[index]
}

/**
 * Worst-case wall time of the whole retry sequence: every attempt using
 * its full budget plus every gap between them. Callers that cap an
 * operation containing a health probe use it so the cap cannot silently
 * cut the retries short.
 */
export function getHealthRetryBudgetMs(): number {
  const attempts = getHealthAttempts()
  let budget = getHealthTimeoutMs() * attempts
  for (let completed = 1; completed < attempts; completed++) {
    budget += healthBackoffMs(completed)
  }
  return budget
}

/** Pause the retry loop without holding anything else. */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

/**
 * Normalise a base URL so the rest of the plugin can rely on a
 * predictable shape (no trailing slash, no `/v1` suffix).
 */
export function normalizeBaseURL(baseURL: string = DEFAULT_LITELLM_URL): string {
  let normalized = baseURL.replace(/\/+$/, '')
  if (normalized.endsWith('/v1')) {
    normalized = normalized.slice(0, -3)
  }
  return normalized
}

/** Build a full URL for a given API endpoint. */
export function buildAPIURL(baseURL: string, endpoint: string = MODELS_ENDPOINT): string {
  return `${normalizeBaseURL(baseURL)}${endpoint}`
}

function buildHeaders(apiKey?: string, customHeaders?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  }
  const key = apiKey ?? process.env.LITELLM_API_KEY ?? process.env.LITELLM_MASTER_KEY
  if (key) {
    headers['Authorization'] = `Bearer ${key}`
  }
  if (customHeaders) {
    Object.assign(headers, customHeaders)
  }
  return headers
}

/**
 * The transport-level cause behind an `unreachable` result, so callers can
 * tell "this name does not resolve yet" from "something is listening but
 * refused us" from "the connection broke mid-flight". The raw error text
 * is still reported verbatim in `message`; this only labels it.
 */
export type LiteLLMTransportCause = 'dns' | 'refused' | 'timeout' | 'socket' | 'other'

/**
 * Why a health probe failed.
 *
 * `timeout` means the request was still pending when the budget expired:
 * the proxy may simply be starting up. `http` carries the non-2xx status
 * (401/403 included) and `unreachable` a transport error (DNS, refused
 * connection). The distinction matters because the fixes differ, so the
 * caller must be able to log them separately.
 *
 * `attempts` records how many probes the verdict took (`1` for a single
 * probe) so the caller can say so in its log line.
 */
export type LiteLLMHealthFailure =
  | { healthy: false; kind: 'timeout'; timeoutMs: number; attempts?: number }
  | { healthy: false; kind: 'http'; status: number; statusText: string; attempts?: number }
  | {
      healthy: false
      kind: 'unreachable'
      message: string
      cause?: LiteLLMTransportCause
      attempts?: number
    }

/** Outcome of a health probe: either alive, or an identified failure. */
export type LiteLLMHealthResult =
  | { healthy: true; status: number; attempts?: number }
  | LiteLLMHealthFailure

/**
 * Identify an abort that came from the timeout budget rather than from a
 * caller-cancelled request. `AbortSignal.timeout` rejects with a
 * `TimeoutError` DOMException on current Node and Bun.
 */
function isTimeoutError(error: unknown): boolean {
  if (!error || typeof error !== 'object') return false
  const name = (error as { name?: unknown }).name
  return name === 'TimeoutError' || name === 'AbortError'
}

/**
 * Fold a fetch failure's message chain and its socket code into one search
 * string. Bun reports `Unable to connect. Is the computer able to access the
 * url?` and keeps `ECONNREFUSED`-style detail in `error.code`, so message
 * alone is not enough; Node nests the real error in `cause`, and an
 * `AggregateError` in `errors`.
 */
function transportErrorText(error: unknown): { text: string; code?: string } {
  const parts: string[] = []
  let code: string | undefined
  let current: unknown = error
  // Bounded walk: a cyclic `cause` chain must not hang the probe.
  for (let depth = 0; depth < 5 && current; depth++) {
    if (!current || typeof current !== 'object') {
      parts.push(String(current))
      break
    }
    const candidate = current as {
      message?: unknown
      code?: unknown
      cause?: unknown
      errors?: unknown
    }
    if (typeof candidate.message === 'string') parts.push(candidate.message)
    if (!code && typeof candidate.code === 'string' && candidate.code) code = candidate.code
    if (Array.isArray(candidate.errors)) {
      for (const inner of candidate.errors) {
        if (!inner || typeof inner !== 'object') continue
        const nested = inner as { message?: unknown; code?: unknown }
        if (typeof nested.message === 'string') parts.push(nested.message)
        if (!code && typeof nested.code === 'string' && nested.code) code = nested.code
      }
    }
    current = candidate.cause
  }
  return { text: parts.join(' '), code }
}

/**
 * Label a transport error with the layer that failed, leaving the raw
 * message untouched so `getaddrinfo ENOTFOUND …`, `connect ECONNREFUSED …`
 * and socket timeouts stay greppable and verbatim.
 */
function classifyTransportError(haystack: string): {
  cause: LiteLLMTransportCause
  label?: string
} {
  if (/ENOTFOUND|EAI_AGAIN|EAI_FAIL|getaddrinfo|FailedToResolve|NameResolution/i.test(haystack)) {
    return { cause: 'dns', label: 'DNS lookup failed' }
  }
  if (/ECONNREFUSED|ConnectionRefused/i.test(haystack)) {
    return { cause: 'refused', label: 'connection refused' }
  }
  if (/ETIMEDOUT|ESOCKETTIMEDOUT|TimedOut|timed out/i.test(haystack)) {
    return { cause: 'timeout', label: 'connection timed out' }
  }
  if (
    /ECONNRESET|EPIPE|EHOSTUNREACH|ENETUNREACH|socket hang up|ConnectionClosed|ConnectionReset|FailedToOpenSocket/i.test(
      haystack,
    )
  ) {
    return { cause: 'socket', label: 'connection failed' }
  }
  return { cause: 'other' }
}

/**
 * The text reported in parentheses: the failure's own message, prefixed by
 * the failing layer and suffixed with its socket code when the message does
 * not already carry it (Bun's generic `Unable to connect` does not).
 */
function describeTransportError(error: unknown): {
  message: string
  cause: LiteLLMTransportCause
} {
  const raw = error instanceof Error ? error.message : String(error)
  const { text, code } = transportErrorText(error)
  const { cause, label } = classifyTransportError(`${text} ${code ?? ''}`)
  const codeSuffix = code && !raw.includes(code) ? ` [${code}]` : ''
  return {
    message: `${label ? `${label}: ` : ''}${raw}${codeSuffix}`,
    cause,
  }
}

/**
 * Lightweight ping that reports *why* a LiteLLM server is unusable, so a
 * slow start-up is not mistaken for a missing or misconfigured proxy.
 *
 * A single attempt only: use {@link probeLiteLLMHealthWithRetry} for the
 * start-up path, which is the one that must survive a proxy that is a
 * second away from accepting connections.
 */
export async function probeLiteLLMHealth(
  baseURL: string = DEFAULT_LITELLM_URL,
  apiKey?: string,
  customHeaders?: Record<string, string>,
): Promise<LiteLLMHealthResult> {
  const timeoutMs = getHealthTimeoutMs()
  try {
    const response = await fetch(buildAPIURL(baseURL), {
      method: 'GET',
      headers: buildHeaders(apiKey, customHeaders),
      signal: AbortSignal.timeout(timeoutMs),
    })
    if (response.ok) return { healthy: true, status: response.status }
    // 401 still means a server is alive — we just don't have the right
    // credentials. Surface that as "unhealthy" so the user is prompted
    // to set LITELLM_API_KEY.
    return { healthy: false, kind: 'http', status: response.status, statusText: response.statusText }
  } catch (error) {
    if (isTimeoutError(error)) return { healthy: false, kind: 'timeout', timeoutMs }
    const { message, cause } = describeTransportError(error)
    return { healthy: false, kind: 'unreachable', message, cause }
  }
}

/**
 * Start-up health probe: up to `attempts` (default 3) single probes, each
 * with its own `getHealthTimeoutMs()` budget, separated by a fixed 400 ms
 * then 1200 ms wait. Returns the first healthy verdict; when every attempt
 * fails it returns the *last* failure, stamped with the attempt count, so
 * the caller's warning still names the real cause instead of a generic
 * "offline".
 */
export async function probeLiteLLMHealthWithRetry(
  baseURL: string = DEFAULT_LITELLM_URL,
  apiKey?: string,
  customHeaders?: Record<string, string>,
  attempts: number = getHealthAttempts(),
): Promise<LiteLLMHealthResult> {
  const total = Number.isInteger(attempts) && attempts >= 1 ? attempts : DEFAULT_HEALTH_ATTEMPTS
  let lastFailure: LiteLLMHealthFailure | undefined
  for (let attempt = 1; attempt <= total; attempt++) {
    const result = await probeLiteLLMHealth(baseURL, apiKey, customHeaders)
    if (result.healthy) return { ...result, attempts: attempt }
    lastFailure = result
    if (attempt < total) await sleep(healthBackoffMs(attempt))
  }
  // `total >= 1`, so the loop always set a failure here.
  return { ...(lastFailure as LiteLLMHealthFailure), attempts: total }
}

/** Boolean form of `probeLiteLLMHealth`, for callers that only need a verdict. */
export async function checkLiteLLMHealth(
  baseURL: string = DEFAULT_LITELLM_URL,
  apiKey?: string,
  customHeaders?: Record<string, string>,
): Promise<boolean> {
  return (await probeLiteLLMHealth(baseURL, apiKey, customHeaders)).healthy
}

/** Discover all models exposed by a LiteLLM proxy. */
export async function discoverLiteLLMModels(
  baseURL: string = DEFAULT_LITELLM_URL,
  apiKey?: string,
  customHeaders?: Record<string, string>,
): Promise<LiteLLMModel[]> {
  const url = buildAPIURL(baseURL)
  const response = await fetch(url, {
    method: 'GET',
    headers: buildHeaders(apiKey, customHeaders),
    signal: AbortSignal.timeout(getRequestTimeoutMs()),
  })

  if (!response.ok) {
    throw new Error(`LiteLLM responded with HTTP ${response.status} ${response.statusText}`)
  }

  const data = (await response.json()) as LiteLLMModelsResponse
  return data.data ?? []
}

/**
 * Fetch per-model metadata (`mode`, token limits, capability flags and the
 * optional `display_name`) from `/v1/model/info`, keyed by model name.
 * `/v1/models` omits these fields for database-defined models, so
 * classification (e.g. filtering out embedding models) relies on this
 * endpoint.
 */
export async function discoverLiteLLMModelInfo(
  baseURL: string = DEFAULT_LITELLM_URL,
  apiKey?: string,
  customHeaders?: Record<string, string>,
): Promise<Map<string, LiteLLMModelInfo>> {
  const url = buildAPIURL(baseURL, MODEL_INFO_ENDPOINT)
  const response = await fetch(url, {
    method: 'GET',
    headers: buildHeaders(apiKey, customHeaders),
    signal: AbortSignal.timeout(getRequestTimeoutMs()),
  })

  if (!response.ok) {
    throw new Error(`LiteLLM responded with HTTP ${response.status} ${response.statusText}`)
  }

  const data = (await response.json()) as LiteLLMModelInfoResponse
  const infoByName = new Map<string, LiteLLMModelInfo>()
  for (const entry of data.data ?? []) {
    if (!entry.model_info) continue
    // Some deployments set capability flags on the params block rather
    // than inside model_info. Fill those gaps so enrichment sees them.
    const info: LiteLLMModelInfo = { ...entry.model_info }
    for (const flag of CAPABILITY_FLAGS) {
      const paramsValue = entry.litellm_params?.[flag]
      if (info[flag] == null && typeof paramsValue === 'boolean') {
        info[flag] = paramsValue
      }
    }

    info.supports_reasoning_efforts = resolveReasoningEfforts(info, entry.litellm_params)

    // Index under every alias LiteLLM may use for this model — the
    // `/v1/models` id can match any of them depending on how the
    // deployment names its entries (alias vs upstream model string).
    const keys = [
      entry.model_name,
      entry.model_info.key,
      typeof entry.litellm_params?.model === 'string' ? entry.litellm_params.model : undefined,
    ]
    for (const key of keys) {
      if (key && !infoByName.has(key)) {
        infoByName.set(key, info)
      }
    }
  }
  return infoByName
}

/**
 * Use an explicit LITELLM_BASE_URL when set; otherwise try `ports` on localhost,
 * by default the most common ones. The default `litellm --port` is 4000, with
 * 8000 and 8080 also common.
 */
export async function autoDetectLiteLLM(
  apiKey?: string,
  customHeaders?: Record<string, string>,
  ports: readonly number[] = [4000, 8000, 8080],
): Promise<string | null> {
  const configuredBaseURL = process.env.LITELLM_BASE_URL?.trim()
  if (configuredBaseURL) return normalizeBaseURL(configuredBaseURL)

  for (const port of ports) {
    const baseURL = `http://localhost:${port}`
    if (await checkLiteLLMHealth(baseURL, apiKey, customHeaders)) {
      return baseURL
    }
  }
  return null
}
