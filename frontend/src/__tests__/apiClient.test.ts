import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, buildQuery, request } from '@/api/client'

describe('buildQuery', () => {
  it('omits empty values and expands arrays', () => {
    expect(buildQuery({ a: 1, b: undefined, c: null, d: '', e: ['x', 'y'] })).toBe('?a=1&e=x&e=y')
  })

  it('returns an empty string when there is nothing to send', () => {
    expect(buildQuery({})).toBe('')
    expect(buildQuery()).toBe('')
  })
})

describe('request error handling', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('unwraps the API error envelope', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ error: { code: 'not_found', message: 'Product 9 was not found.', request_id: 'abc' } }),
      { status: 404, headers: { 'Content-Type': 'application/json' } },
    )))

    await expect(request('/products/9')).rejects.toMatchObject({
      status: 404, code: 'not_found', message: 'Product 9 was not found.', requestId: 'abc',
    })
  })

  it('flags an unavailable model so the UI can degrade instead of erroring', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ error: { code: 'model_unavailable', message: 'not trained', request_id: 'x' } }),
      { status: 503, headers: { 'Content-Type': 'application/json' } },
    )))

    await expect(request('/forecast/1')).rejects.toSatisfy(
      (error: unknown) => error instanceof ApiError && error.isModelUnavailable,
    )
  })

  it('converts a network failure into a readable error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    await expect(request('/health')).rejects.toMatchObject({ code: 'network_error' })
  })

  it('parses a successful JSON response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ status: 'ok' }), { status: 200, headers: { 'Content-Type': 'application/json' } },
    )))
    await expect(request<{ status: string }>('/health')).resolves.toEqual({ status: 'ok' })
  })
})
