import { describe, expect, it } from 'vitest'
import { applyLimitOverrides, parseModelLimits } from '../src/utils/model-limits'
import { buildCacheKey } from '../src/utils/model-cache'
import { toConfigModel } from '../src/plugin/index'
import type { LiteLLMModel } from '../src/types'

describe('parseModelLimits', () => {
  it('keeps positive finite context/output numbers', () => {
    expect(
      parseModelLimits({
        'mistral/zai-glm-5-3-flash': { context: 1048576, output: 131072 },
      }),
    ).toEqual({ 'mistral/zai-glm-5-3-flash': { context: 1048576, output: 131072 } })
  })

  it('drops non-positive, non-finite and non-numeric values', () => {
    expect(
      parseModelLimits({
        a: { context: 0, output: -1 },
        b: { output: Number.POSITIVE_INFINITY },
        c: { context: '32768' },
        d: {},
      }),
    ).toEqual({})
  })

  it('ignores malformed input', () => {
    expect(parseModelLimits(undefined)).toEqual({})
    expect(parseModelLimits([])).toEqual({})
    expect(parseModelLimits({ a: 42 })).toEqual({})
    expect(parseModelLimits({ a: ['32768'] })).toEqual({})
  })
})

describe('applyLimitOverrides', () => {
  const entry = { name: 'GLM', limit: { context: 1048576, output: 0 } }

  it('replaces only the fields the override names', () => {
    expect(applyLimitOverrides(entry, { output: 32768 })).toEqual({
      name: 'GLM',
      limit: { context: 1048576, output: 32768 },
    })
  })

  it('creates the limit block when discovery emitted none', () => {
    expect(applyLimitOverrides({ name: 'GLM' }, { output: 32768 })).toEqual({
      name: 'GLM',
      limit: { output: 32768 },
    })
  })

  it('returns the entry untouched without an override', () => {
    expect(applyLimitOverrides(entry, undefined)).toBe(entry)
  })

  it('does not mutate the input entry', () => {
    applyLimitOverrides(entry, { output: 32768 })
    expect(entry.limit.output).toBe(0)
  })
})

describe('buildCacheKey with modelLimits', () => {
  const base = 'http://127.0.0.1:4000'

  it('keeps the plain key when no override exists', () => {
    expect(buildCacheKey('litellm', base, {}, {}, {}, undefined, {})).toBe(
      buildCacheKey('litellm', base, {}, {}, {}),
    )
  })

  it('starts a fresh cache when an override is added or changed', () => {
    const none = buildCacheKey('litellm', base, {}, {}, {}, undefined, {})
    const one = buildCacheKey('litellm', base, {}, {}, {}, undefined, {
      'mistral/zai-glm-5-3-flash': { output: 32768 },
    })
    const other = buildCacheKey('litellm', base, {}, {}, {}, undefined, {
      'mistral/zai-glm-5-3-flash': { output: 131072 },
    })
    expect(new Set([none, one, other]).size).toBe(3)
  })

  it('is order-insensitive across model ids', () => {
    const a = buildCacheKey('litellm', base, {}, {}, {}, undefined, {
      a: { output: 1 },
      b: { output: 2 },
    })
    const b = buildCacheKey('litellm', base, {}, {}, {}, undefined, {
      b: { output: 2 },
      a: { output: 1 },
    })
    expect(a).toBe(b)
  })
})

describe('discovered entry with an output limit of 0', () => {
  it('carries the corrected output limit OpenCode sends as max_tokens', () => {
    const model: LiteLLMModel = {
      id: 'mistral/zai-glm-5-3-flash',
      object: 'model',
      max_input_tokens: 1048576,
      max_output_tokens: 0,
    }
    const entry = toConfigModel(model)
    expect(entry?.limit).toEqual({ context: 1048576, output: 0 })
    expect(applyLimitOverrides(entry!, { output: 32768 }).limit).toEqual({
      context: 1048576,
      output: 32768,
    })
  })
})
