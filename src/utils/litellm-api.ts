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
 * Why a health probe failed.
 *
 * `timeout` means the request was still pending when the budget expired:
 * the proxy may simply be starting up. `http` carries the non-2xx status
 * (401/403 included) and `unreachable` a transport error (DNS, refused
 * connection). The distinction matters because the fixes differ, so the
 * caller must be able to log them separately.
 */
export type LiteLLMHealthFailure =
  | { healthy: false; kind: 'timeout'; timeoutMs: number }
  | { healthy: false; kind: 'http'; status: number; statusText: string }
  | { healthy: false; kind: 'unreachable'; message: string }

/** Outcome of a health probe: either alive, or an identified failure. */
export type LiteLLMHealthResult = { healthy: true; status: number } | LiteLLMHealthFailure

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
 * Lightweight ping that reports *why* a LiteLLM server is unusable, so a
 * slow start-up is not mistaken for a missing or misconfigured proxy.
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
    return {
      healthy: false,
      kind: 'unreachable',
      message: error instanceof Error ? error.message : String(error),
    }
  }
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
