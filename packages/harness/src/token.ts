import { GitHubError, NetworkError, createApi } from "./github"

export const REPO = { owner: "markwhiting", name: "learner" } as const

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

export const MISSING_SCOPE_MESSAGE = "This token lacks the public_repo scope, so it can't open pull requests — create one with the link above."

interface Who extends Identity { /** Classic tokens list their scopes in X-OAuth-Scopes; null when GitHub sends no such header. */ scopes: string[] | null }

async function identify(token: string, fetchFn: typeof fetch): Promise<Who> {
  const api = createApi(token, fetchFn)
  const user = await api.call("GET", "/user")
  const repo = await api.call("GET", `/repos/${REPO.owner}/${REPO.name}`)
  const login = (user.body as { login?: unknown } | null)?.login
  if (typeof login !== "string") throw new GitHubError(502, "unexpected /user response", null)
  const push = (repo.body as { permissions?: { push?: unknown } } | null)?.permissions?.push
  const header = user.headers.get("x-oauth-scopes")
  const scopes = header === null ? null : header.split(",").map(x => x.trim()).filter(x => x !== "")
  return { login, canPush: push === true, scopes }
}

/** Who the token belongs to and whether it can push to upstream. Throws GitHubError / NetworkError. */
export async function whoAmI(token: string, fetchFn: typeof fetch): Promise<Identity> {
  const { login, canPush } = await identify(token, fetchFn)
  return { login, canPush }
}

export async function checkToken(token: string, fetchFn: typeof fetch): Promise<{ ok: true; login: string; canPush: boolean } | { ok: false; reason: string }> {
  try {
    const { login, canPush, scopes } = await identify(token, fetchFn)
    if (scopes !== null && !scopes.includes("public_repo") && !scopes.includes("repo")) return { ok: false, reason: MISSING_SCOPE_MESSAGE }
    return { ok: true, login, canPush }
  } catch (e) {
    if (e instanceof GitHubError && e.status === 401) return { ok: false, reason: REJECTED_MESSAGE }
    if (e instanceof GitHubError) return { ok: false, reason: `GitHub answered ${e.status}; try again shortly.` }
    if (e instanceof NetworkError) return { ok: false, reason: "Could not reach GitHub — check your connection." }
    throw e
  }
}
