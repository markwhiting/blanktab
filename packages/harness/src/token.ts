import { GitHubError, NetworkError, createApi } from "./github"

export const REPO = { owner: "markwhiting", name: "blanktab" } as const

// The token lives here and nowhere else (decision 0010). Never IndexedDB, logs, or request bodies.
const KEY = "blanktab:githubToken"

export function getToken(ls: Storage): string | null {
  const t = ls.getItem(KEY)?.trim()
  return t ? t : null
}
export function setToken(ls: Storage, t: string): void {
  const v = t.trim()
  if (v) ls.setItem(KEY, v)
  else ls.removeItem(KEY)
}
export function clearToken(ls: Storage): void {
  ls.removeItem(KEY)
}

export const REJECTED_MESSAGE = "Your GitHub token was rejected — paste a new one."

export interface Identity { login: string; canPush: boolean }

/** Who the token belongs to and whether it can push to upstream. Throws GitHubError / NetworkError. */
export async function whoAmI(token: string, fetchFn: typeof fetch): Promise<Identity> {
  const api = createApi(token, fetchFn)
  const user = await api.call("GET", "/user")
  const repo = await api.call("GET", `/repos/${REPO.owner}/${REPO.name}`)
  const login = (user.body as { login?: unknown } | null)?.login
  if (typeof login !== "string") throw new GitHubError(502, "unexpected /user response", null)
  const push = (repo.body as { permissions?: { push?: unknown } } | null)?.permissions?.push
  return { login, canPush: push === true }
}

export async function checkToken(token: string, fetchFn: typeof fetch): Promise<{ ok: true; login: string; canPush: boolean } | { ok: false; reason: string }> {
  try {
    return { ok: true, ...(await whoAmI(token, fetchFn)) }
  } catch (e) {
    if (e instanceof GitHubError && e.status === 401) return { ok: false, reason: REJECTED_MESSAGE }
    if (e instanceof GitHubError) return { ok: false, reason: `GitHub answered ${e.status}; try again shortly.` }
    if (e instanceof NetworkError) return { ok: false, reason: "Could not reach GitHub — check your connection." }
    throw e
  }
}
