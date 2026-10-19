// Minimal GitHub REST client. The only place a token is read into a request, and only as an Authorization header
// to https://api.github.com. Error messages are built from status codes, never from response bodies or the token.

const BASE = "https://api.github.com"

export class GitHubError extends Error {
  constructor(readonly status: number, message: string, readonly retryAfterMs: number | null) {
    super(message)
  }
}
export class NetworkError extends Error {}

export interface ApiResult { status: number; body: unknown }
export interface Api {
  /** Resolves for 2xx and for statuses listed in `allow`; throws GitHubError otherwise, NetworkError if fetch throws. */
  call(method: string, path: string, body?: unknown, allow?: number[]): Promise<ApiResult>
}

/** Delay implied by a rate-limit response: Retry-After (seconds) or X-RateLimit-Reset (epoch seconds). */
export function retryDelayMs(h: Headers, nowMs: number): number | null {
  const ra = h.get("retry-after")
  if (ra !== null && ra.trim() !== "" && Number.isFinite(Number(ra))) return Math.max(0, Number(ra)) * 1000
  const reset = h.get("x-ratelimit-reset")
  if (reset !== null && reset.trim() !== "" && Number.isFinite(Number(reset))) return Math.max(0, Number(reset) * 1000 - nowMs)
  return null
}

export function createApi(token: string, fetchFn: typeof fetch, now: () => number = Date.now): Api {
  return {
    async call(method, path, body, allow = []) {
      let res: Response
      try {
        res = await fetchFn(BASE + path, {
          method,
          headers: {
            Authorization: `Bearer ${token}`,
            "X-GitHub-Api-Version": "2022-11-28",
            Accept: "application/vnd.github+json",
            ...(body === undefined ? {} : { "Content-Type": "application/json" }),
          },
          ...(body === undefined ? {} : { body: JSON.stringify(body) }),
        })
      } catch {
        throw new NetworkError("network request failed")
      }
      let parsed: unknown = null
      try { parsed = await res.json() } catch { /* empty or non-JSON body */ }
      if (res.ok || allow.includes(res.status)) return { status: res.status, body: parsed }
      const limited = res.status === 403 || res.status === 429
      throw new GitHubError(res.status, `GitHub answered ${res.status}`, limited ? retryDelayMs(res.headers, now()) : null)
    },
  }
}

/** UTF-8 safe base64 that works in browsers and Node. */
export function toBase64(s: string): string {
  const bytes = new TextEncoder().encode(s)
  let bin = ""
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  return btoa(bin)
}
