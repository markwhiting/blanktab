// A fake api.github.com: records every request and simulates the endpoints the contributor uses.
// Git objects (blobs, trees, commits) live in one store shared by upstream and its forks, as on GitHub.
export interface Recorded { url: string; method: string; headers: Record<string, string>; body: string | undefined }
type Repo = { fork: boolean; parent?: string; branches: Map<string, string> }
export interface Pull { number?: number; head: string; base: string; title: string; draft: boolean; body: string; state?: "open" | "closed"; merged?: boolean; headRepo?: string }
interface Commit { tree: string; parents: string[] }

const json = (status: number, body: unknown, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } })

export function fakeGithub(opts: { token: string; login: string; canPush: boolean; existingForkAs?: string; scopes?: string }) {
  const UP = "markwhiting/learner"
  const repos = new Map<string, Repo>()
  const mk = (fork: boolean, parent?: string): Repo => ({ fork, ...(parent ? { parent } : {}), branches: new Map() })

  // Content-addressed blobs and trees (same content -> same sha); commits are always new.
  const interned = new Map<string, string>()
  let counter = 0
  const newSha = () => (++counter).toString(16).padStart(40, "0")
  const intern = (key: string) => { let s = interned.get(key); if (!s) { s = newSha(); interned.set(key, s) } return s }
  const blobs = new Map<string, string>() // sha -> base64 content
  const trees = new Map<string, Map<string, string>>() // sha -> path -> blob sha
  const commits = new Map<string, Commit>()
  const putTree = (t: Map<string, string>): string => {
    const sha = intern("tree:" + JSON.stringify([...t].sort(([a], [b]) => (a < b ? -1 : 1))))
    trees.set(sha, new Map(t))
    return sha
  }
  const MAIN0 = "a".repeat(40)
  commits.set(MAIN0, { tree: putTree(new Map()), parents: [] })

  const upstream = mk(false)
  upstream.branches.set("main", MAIN0)
  repos.set(UP, upstream)
  if (opts.existingForkAs) repos.set(opts.existingForkAs, mk(true, UP))
  const pulls: Pull[] = []
  const requests: Recorded[] = []
  /** Interceptors run first; return a Response (or throw) to override, undefined to fall through. */
  const interceptors: Array<(r: Recorded) => Response | undefined> = []
  let forkPollsBeforeReady = 0

  const ancestors = (sha: string): Set<string> => {
    const seen = new Set<string>()
    const todo = [sha]
    while (todo.length) { const s = todo.pop()!; if (seen.has(s)) continue; seen.add(s); todo.push(...(commits.get(s)?.parents ?? [])) }
    return seen
  }
  const treeOf = (commit: string) => trees.get(commits.get(commit)!.tree)!
  /** The repo a PR head "owner:branch" lives in: upstream, or the owner's fork of it. */
  const repoOfOwner = (owner: string): string | undefined =>
    [...repos.keys()].find(full => full.split("/")[0] === owner && (full === UP || repos.get(full)!.parent === UP))
  /** Files a PR proposes, as GitHub shows them: head tree vs. the merge base with main. */
  const diff = (p: Pull): Map<string, string> => {
    const [owner, branch] = p.head.split(":") as [string, string]
    const head = repos.get(p.headRepo ?? repoOfOwner(owner)!)?.branches.get(branch)
    if (!head) return new Map()
    const onMain = ancestors(upstream.branches.get("main")!)
    let base = head
    while (!onMain.has(base)) base = commits.get(base)!.parents[0]!
    const from = treeOf(base)
    return new Map([...treeOf(head)].filter(([path, sha]) => from.get(path) !== sha))
  }

  const handle = (r: Recorded): Response => {
    const u = new URL(r.url)
    const p = u.pathname
    let m: RegExpMatchArray | null
    if (r.headers.Authorization !== `Bearer ${opts.token}`) return json(401, { message: "Bad credentials" })
    const body = () => JSON.parse(r.body ?? "{}") as Record<string, unknown>
    if (p === "/user") return json(200, { login: opts.login }, opts.scopes === undefined ? {} : { "x-oauth-scopes": opts.scopes })
    if (r.method === "GET" && p === `/repos/${UP}`) return json(200, { full_name: UP, permissions: { push: opts.canPush } })
    if (r.method === "POST" && p === `/repos/${UP}/forks`) {
      const full = opts.existingForkAs ?? `${opts.login}/learner`
      if (!repos.has(full)) { repos.set(full, mk(true, UP)); forkPollsBeforeReady = 2 }
      return json(202, { full_name: full })
    }
    if ((m = p.match(/^\/repos\/([^/]+\/[^/]+)\/pulls$/)) && m[1] === UP) {
      if (r.method === "GET") {
        const head = u.searchParams.get("head")
        const want = u.searchParams.get("state") ?? "open"
        return json(200, pulls.filter(x => x.head === head && (want === "all" || (x.state ?? "open") === want)))
      }
      const b = body() as unknown as Pull
      const headRepo = repoOfOwner(b.head.split(":")[0]!)
      const pr: Pull = { ...b, state: "open", number: pulls.length + 1, ...(headRepo ? { headRepo } : {}) }
      if (diff(pr).size === 0) return json(422, { message: "Validation Failed", errors: [{ message: "No commits between main and head" }] })
      pulls.push(pr)
      return json(201, { number: pr.number })
    }
    if (r.method === "GET" && (m = p.match(/^\/repos\/([^/]+\/[^/]+)\/compare\/main\.\.\.(.+)$/)) && m[1] === UP) {
      const head = decodeURIComponent(m[2]!)
      const [owner, branch] = head.split(":") as [string, string]
      if (!repos.get(repoOfOwner(owner) ?? "")?.branches.has(branch)) return json(404, { message: "Not Found" })
      return json(200, { files: [...diff({ head, base: "main", title: "", draft: true, body: "" }).keys()].map(filename => ({ filename })) })
    }
    if (!(m = p.match(/^\/repos\/([^/]+\/[^/]+)(\/.*)?$/))) return json(404, { message: `unhandled ${r.method} ${p}` })
    const full = m[1]!
    const rest = m[2] ?? ""
    const repo = repos.get(full)
    if (!repo) return json(404, { message: "Not Found" })
    if (r.method === "GET" && rest === "") {
      if (forkPollsBeforeReady > 0) { forkPollsBeforeReady--; return json(404, { message: "Not Found" }) }
      return json(200, { full_name: full, fork: repo.fork, ...(repo.parent ? { parent: { full_name: repo.parent } } : {}) })
    }
    if (r.method === "GET" && (m = rest.match(/^\/git\/ref\/heads\/(.+)$/))) {
      const sha = repo.branches.get(decodeURIComponent(m[1]!))
      return sha ? json(200, { object: { sha } }) : json(404, { message: "Not Found" })
    }
    if (r.method === "POST" && rest === "/git/refs") {
      const b = body() as { ref: string; sha: string }
      const name = b.ref.replace(/^refs\/heads\//, "")
      if (repo.branches.has(name)) return json(422, { message: "Reference already exists" })
      if (!commits.has(b.sha)) return json(422, { message: "Object does not exist" })
      repo.branches.set(name, b.sha)
      return json(201, { ref: b.ref, object: { sha: b.sha } })
    }
    if (r.method === "PATCH" && (m = rest.match(/^\/git\/refs\/heads\/(.+)$/))) {
      const name = decodeURIComponent(m[1]!)
      const b = body() as { sha: string; force?: boolean }
      const old = repo.branches.get(name)
      if (!old) return json(422, { message: "Reference does not exist" })
      if (!commits.has(b.sha)) return json(422, { message: "Object does not exist" })
      if (b.force !== true && !ancestors(b.sha).has(old)) return json(422, { message: "Update is not a fast forward" })
      repo.branches.set(name, b.sha)
      return json(200, { object: { sha: b.sha } })
    }
    if (r.method === "GET" && (m = rest.match(/^\/git\/commits\/([0-9a-f]{40})$/))) {
      const c = commits.get(m[1]!)
      return c ? json(200, { sha: m[1], tree: { sha: c.tree }, parents: c.parents.map(sha => ({ sha })) }) : json(404, { message: "Not Found" })
    }
    if (r.method === "POST" && rest === "/git/blobs") {
      const b = body() as { content: string; encoding: string }
      if (b.encoding !== "base64") return json(422, { message: "encoding must be base64 in this fake" })
      const sha = intern("blob:" + b.content)
      blobs.set(sha, b.content)
      return json(201, { sha })
    }
    if (r.method === "POST" && rest === "/git/trees") {
      const b = body() as { base_tree?: string; tree: { path: string; mode: string; type: string; sha: string }[] }
      const base = b.base_tree === undefined ? new Map<string, string>() : trees.get(b.base_tree)
      if (!base) return json(422, { message: "base_tree not found" })
      const t = new Map(base)
      for (const e of b.tree) {
        if (e.mode !== "100644" || e.type !== "blob" || !blobs.has(e.sha)) return json(422, { message: "bad tree entry" })
        t.set(e.path, e.sha)
      }
      return json(201, { sha: putTree(t) })
    }
    if (r.method === "POST" && rest === "/git/commits") {
      const b = body() as { message: string; tree: string; parents: string[] }
      if (!trees.has(b.tree) || !b.parents.every(s => commits.has(s))) return json(422, { message: "bad commit" })
      const sha = newSha()
      commits.set(sha, { tree: b.tree, parents: b.parents })
      return json(201, { sha })
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
  /** path -> blob sha on a repo's branch, or undefined when the branch does not exist. */
  const branchTree = (repo: string, branch: string): Map<string, string> | undefined => {
    const head = repos.get(repo)?.branches.get(branch)
    return head ? treeOf(head) : undefined
  }
  return {
    fetchFn, requests, interceptors, pulls, repos, blobs, decode,
    /** Parsed content of a file on a repo's branch. */
    file: (repo: string, branch: string, path: string): unknown => {
      const sha = branchTree(repo, branch)?.get(path)
      return sha ? JSON.parse(decode(blobs.get(sha)!)) : undefined
    },
    /** Sorted paths on a repo's branch. */
    paths: (repo: string, branch: string): string[] => [...(branchTree(repo, branch)?.keys() ?? [])].sort(),
    /** Sorted paths a PR proposes (head vs. merge base with main). */
    prFiles: (pr: Pull): string[] => [...diff(pr).keys()].sort(),
    /** Squash-merges a PR into upstream main, like the gate does; optionally deletes the head branch. */
    merge(pr: Pull, o: { deleteBranch: boolean }): void {
      const main = upstream.branches.get("main")!
      const t = new Map(treeOf(main))
      for (const [path, sha] of diff(pr)) t.set(path, sha)
      const sha = newSha()
      commits.set(sha, { tree: putTree(t), parents: [main] })
      upstream.branches.set("main", sha)
      pr.state = "closed"
      pr.merged = true
      if (o.deleteBranch) repos.get(pr.headRepo!)!.branches.delete(pr.head.split(":")[1]!)
    },
  }
}
