/**
 * Per-model token-limit overrides keyed by exact model id
 * (`providers.<id>.settings.modelLimits`).
 *
 * Discovery copies whatever the proxy reports into each model's `limit`
 * block, and every launch re-reads that cache, so a limit the proxy gets
 * wrong (or reports as `0`) reaches OpenCode unchanged. This option lets
 * the user correct it from the config without hand-maintaining a model
 * list — the same escape hatch `modelCapabilities` gives for capability
 * flags.
 *
 * `output` is the one that matters most in practice: OpenCode sends it as
 * the request's `max_tokens`, so a proxy that advertises `0` (or a value
 * smaller than the model's real ceiling) caps — or silently truncates —
 * long answers, which reads as a model that "thinks forever and never
 * answers".
 */
export interface ModelLimitOverride {
  context?: number
  output?: number
}

/** Per-model limit overrides, keyed by exact model id. */
export type ModelLimits = Record<string, ModelLimitOverride>

/** A positive, finite token count; anything else is not a usable limit. */
function isTokenCount(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0
}

/**
 * Defensively parse the `modelLimits` provider option. Non-object input,
 * non-object per-model values, and non-positive/non-finite numbers are
 * dropped; a per-model entry that keeps neither field is dropped too.
 */
export function parseModelLimits(raw: unknown): ModelLimits {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return {}
  const out: ModelLimits = {}
  for (const [modelId, limits] of Object.entries(raw as Record<string, unknown>)) {
    if (!limits || typeof limits !== 'object' || Array.isArray(limits)) continue
    const { context, output } = limits as Record<string, unknown>
    const parsed: ModelLimitOverride = {}
    if (isTokenCount(context)) parsed.context = context
    if (isTokenCount(output)) parsed.output = output
    if (Object.keys(parsed).length > 0) out[modelId] = parsed
  }
  return out
}

/**
 * Overlay a user-configured limit override onto the `limit` block discovery
 * built for a model. Returns the entry unchanged when there is no override,
 * and only the fields the override names are replaced.
 */
export function applyLimitOverrides<T extends Record<string, unknown>>(
  entry: T,
  override?: ModelLimitOverride,
): T {
  if (!override) return entry
  const current = entry.limit
  const limit: Record<string, unknown> =
    current && typeof current === 'object' && !Array.isArray(current)
      ? { ...(current as Record<string, unknown>) }
      : {}
  if (override.context !== undefined) limit.context = override.context
  if (override.output !== undefined) limit.output = override.output
  return { ...entry, limit }
}
