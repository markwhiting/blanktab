// A fake api.github.com: records every request and simulates the endpoints the contributor uses.
export interface Recorded { url: string; method: string; headers: Record<string, string>; body: string | undefined }
type Repo = { fork: boolean; parent?: string; branches: Map<string, string>; files: Map<string, { content: string; sha: string }> }
export interface Pull { head: string; base: string; title: string; draft: boolean; body: string }

const json = (status: number, body: unknown, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } })

export function fakeGithub(opts: { token: string; login: string; canPush: boolean; existingForkAs?: string }) {
  const UP = "markwhiting/blanktab"
  const repos = new Map<string, Repo>()
  const mk = (fork: boolean, parent?: string): Repo => ({ fork, ...(parent ? { parent } : {}), branches: new Map(), files: new Map() })
  const upstream = mk(false)
  upstream.branches.set("main", "a".repeat(40))
  repos.set(UP, upstream)
  if (opts.existingForkAs) repos.set(opts.existingForkAs, mk(true, UP))
  const pulls: Pull[] = []
  const requests: Recorded[] = []
  /** Interceptors run first; return a Response (or throw) to override, undefined to fall through. */
  const interceptors: Array<(r: Recorded) => Response | undefined> = []
  let forkPollsBeforeReady = 0
  let counter = 0

  const handle = (r: Recorded): Response => {
    const u = new URL(r.url)
    const p = u.pathname
    let m: RegExpMatchArray | null
    if (r.headers.Authorization !== `Bearer ${opts.token}`) return json(401, { message: "Bad credentials" })
    if (p === "/user") return json(200, { login: opts.login })
    if (r.method === "GET" && p === `/repos/${UP}`) return json(200, { full_name: UP, permissions: { push: opts.canPush } })
    if (r.method === "POST" && p === `/repos/${UP}/forks`) {
      const full = opts.existingForkAs ?? `${opts.login}/blanktab`
      if (!repos.has(full)) { repos.set(full, mk(true, UP)); forkPollsBeforeReady = 2 }
      return json(202, { full_name: full })
    }
    if ((m = p.match(/^\/repos\/([^/]+\/[^/]+)\/pulls$/)) && m[1] === UP) {
      if (r.method === "GET") {
        const head = u.searchParams.get("head")
        return json(200, pulls.filter(x => x.head === head).map(x => ({ ...x, number: 1 })))
      }
      const b = JSON.parse(r.body ?? "{}") as Pull
      pulls.push(b)
      return json(201, { number: pulls.length })
    }
    if ((m = p.match(/^\/repos\/([^/]+\/[^/]+)\/git\/ref\/heads\/(.+)$/))) {
      const repo = repos.get(m[1]!)
      const sha = repo?.branches.get(decodeURIComponent(m[2]!))
      return sha ? json(200, { object: { sha } }) : json(404, { message: "Not Found" })
    }
    if (r.method === "POST" && (m = p.match(/^\/repos\/([^/]+\/[^/]+)\/git\/refs$/))) {
      const repo = repos.get(m[1]!)
      if (!repo) return json(404, { message: "Not Found" })
      const b = JSON.parse(r.body ?? "{}") as { ref: string; sha: string }
      const name = b.ref.replace(/^refs\/heads\//, "")
      if (repo.branches.has(name)) return json(422, { message: "Reference already exists" })
      repo.branches.set(name, b.sha)
      return json(201, { ref: b.ref })
    }
    if ((m = p.match(/^\/repos\/([^/]+\/[^/]+)\/contents\/(.+)$/))) {
      const repo = repos.get(m[1]!)
      if (!repo) return json(404, { message: "Not Found" })
      const path = decodeURIComponent(m[2]!)
      if (r.method === "GET") {
        const f = repo.files.get(`${u.searchParams.get("ref")}:${path}`)
        return f ? json(200, { sha: f.sha, content: f.content, encoding: "base64" }) : json(404, { message: "Not Found" })
      }
      const b = JSON.parse(r.body ?? "{}") as { content: string; branch: string; sha?: string; message: string }
      if (!repo.branches.has(b.branch)) return json(404, { message: "Branch not found" })
      const key = `${b.branch}:${path}`
      const old = repo.files.get(key)
      if (old && old.sha !== b.sha) return json(422, { message: "Invalid request. \"sha\" wasn't supplied." })
      repo.files.set(key, { content: b.content, sha: `blob${++counter}` })
      return json(old ? 200 : 201, { content: { path } })
    }
    if (r.method === "GET" && (m = p.match(/^\/repos\/([^/]+\/[^/]+)$/))) {
      const repo = repos.get(m[1]!)
      if (!repo) return json(404, { message: "Not Found" })
      if (forkPollsBeforeReady > 0) { forkPollsBeforeReady--; return json(404, { message: "Not Found" }) }
      return json(200, { full_name: m[1], fork: repo.fork, ...(repo.parent ? { parent: { full_name: repo.parent } } : {}) })
    }
    return json(404, { message: `unhandled ${r.method} ${p}` })
  }

  const fetchFn = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const headers: Record<string, string> = {}
    new Headers(init?.headers).forEach((v, k) => { headers[k === "authorization" ? "Authorization" : k] = v })
    const rec: Recorded = { url: String(input), method: init?.method ?? "GET", headers, body: typeof init?.body === "string" ? init.body : undefined }
    requests.push(rec)
    for (const i of interceptors) {
      const res = i(rec)
      if (res) return res
    }
    return handle(rec)
  }) as typeof fetch

  const decode = (b64: string) => new TextDecoder().decode(Uint8Array.from(atob(b64), c => c.charCodeAt(0)))
  return {
    fetchFn, requests, interceptors, pulls, repos, decode,
    /** Parsed content of a file on a repo's branch. */
    file: (repo: string, branch: string, path: string): unknown => {
      const f = repos.get(repo)?.files.get(`${branch}:${path}`)
      return f ? JSON.parse(decode(f.content)) : undefined
    },
  }
}
