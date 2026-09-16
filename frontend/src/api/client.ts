/** Typed fetch wrapper around the E-Commerce Intelligence API. */

const RAW_BASE = import.meta.env.VITE_API_BASE_URL ?? ''
export const API_BASE = `${RAW_BASE.replace(/\/$/, '')}/api/v1`

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public requestId?: string,
    public details?: unknown,
  ) {
    super(message)
    this.name = 'ApiError'
  }

  /** True when the failure is a missing model rather than a real fault. */
  get isModelUnavailable(): boolean {
    return this.code === 'model_unavailable'
  }
}

const TOKEN_KEY = 'eci.token'
const SESSION_KEY = 'eci.session'

/** In-memory auth state. Persisted to sessionStorage when available. */
let authToken: string | null = null

function safeStorage(): Storage | null {
  try {
    return typeof window !== 'undefined' ? window.sessionStorage : null
  } catch {
    return null
  }
}

export function setToken(token: string | null): void {
  authToken = token
  const store = safeStorage()
  if (!store) return
  if (token) store.setItem(TOKEN_KEY, token)
  else store.removeItem(TOKEN_KEY)
}

export function getToken(): string | null {
  if (authToken) return authToken
  authToken = safeStorage()?.getItem(TOKEN_KEY) ?? null
  return authToken
}

/** Stable per-tab session id so anonymous behaviour still personalises. */
export function getSessionId(): string {
  const store = safeStorage()
  const existing = store?.getItem(SESSION_KEY)
  if (existing) return existing
  const generated = `web-${Math.random().toString(36).slice(2, 10)}-${Date.now().toString(36)}`
  store?.setItem(SESSION_KEY, generated)
  return generated
}

export type QueryValue = string | number | boolean | undefined | null | (string | number)[]

export function buildQuery(params: Record<string, QueryValue> = {}): string {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '') continue
    if (Array.isArray(value)) value.forEach((v) => search.append(key, String(v)))
    else search.append(key, String(value))
  }
  const qs = search.toString()
  return qs ? `?${qs}` : ''
}

interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE'
  body?: unknown
  params?: Record<string, QueryValue>
  signal?: AbortSignal
  auth?: boolean
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = 'GET', body, params, signal, auth = true } = options
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'X-Session-Id': getSessionId(),
  }
  const token = auth ? getToken() : null
  if (token) headers.Authorization = `Bearer ${token}`

  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}${buildQuery(params)}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    })
  } catch (cause) {
    if ((cause as Error).name === 'AbortError') throw cause
    throw new ApiError(0, 'network_error', 'Could not reach the API. Is the backend running?')
  }

  if (response.status === 204) return undefined as T

  const text = await response.text()
  let payload: unknown = null
  if (text) {
    try {
      payload = JSON.parse(text)
    } catch {
      payload = text
    }
  }

  if (!response.ok) {
    const envelope = (payload as { error?: { code?: string; message?: string; request_id?: string; details?: unknown } })?.error
    throw new ApiError(
      response.status,
      envelope?.code ?? 'http_error',
      envelope?.message ?? `Request failed with status ${response.status}`,
      envelope?.request_id,
      envelope?.details,
    )
  }
  return payload as T
}

export const api = {
  get: <T>(path: string, params?: Record<string, QueryValue>, signal?: AbortSignal) =>
    request<T>(path, { method: 'GET', params, signal }),
  post: <T>(path: string, body?: unknown, params?: Record<string, QueryValue>) =>
    request<T>(path, { method: 'POST', body, params }),
}
