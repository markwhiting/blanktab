import type { Queue } from "./queue"
import type { LogFile } from "./log"
import { GitHubError, NetworkError, createApi, toBase64, type Api } from "./github"
import { REPO, REJECTED_MESSAGE, getToken, whoAmI, type Identity } from "./token"

export interface ContributorStatus { state: "no-token" | "idle" | "sending" | "error"; message: string; pending: number }

export interface ContributorDeps {
  queue: Queue
  ls: Storage
  fetchFn: typeof fetch
  now: () => Date
  onStatus: (s: ContributorStatus) => void
  /** Injectable for tests. Default: real timers. */
  sleep?: (ms: number) => Promise<void>
  schedule?: (fn: () => void, ms: number) => void
}

const UPSTREAM = `${REPO.owner}/${REPO.name}`
const BACKOFF_START_MS = 5_000
const BACKOFF_MAX_MS = 600_000
const FORK_POLLS = 10
const FORK_POLL_MS = 2_000
const DEFAULT_RATE_LIMIT_MS = 60_000
const PERMISSION_MESSAGE = "Your GitHub token can't write to the repo — check its permissions."

class Wait extends Error { constructor(readonly ms: number, readonly why: string) { super(why) } }

const runPath = (id: string) => `data/runs/${id}`
const branchOf = (id: string) => `run/${id}`

export function createContributor(deps: ContributorDeps): { flush(): Promise<void> } {
  const { queue, ls, fetchFn, now, onStatus } = deps
  const sleep = deps.sleep ?? (ms => new Promise<void>(r => setTimeout(r, ms)))
  const schedule = deps.schedule ?? ((fn, ms) => void setTimeout(fn, ms))
  let inFlight: Promise<void> | null = null
  let retryArmed = false
  let backoff = BACKOFF_START_MS

  const status = (state: ContributorStatus["state"], message: string, pending: number) => onStatus({ state, message, pending })

  const armRetry = (ms: number) => {
    if (retryArmed) return
    retryArmed = true
    schedule(() => { retryArmed = false; void contributor.flush() }, ms)
  }

  async function destination(api: Api, me: Identity): Promise<string> {
    if (me.canPush) return UPSTREAM
    const mine = await api.call("GET", `/repos/${me.login}/${REPO.name}`, undefined, [404])
    const b = mine.body as { full_name?: string; fork?: boolean; parent?: { full_name?: string } } | null
    if (mine.status === 200 && b?.fork === true && b.parent?.full_name === UPSTREAM && typeof b.full_name === "string") return b.full_name
    const created = await api.call("POST", `/repos/${UPSTREAM}/forks`, {})
    const full = (created.body as { full_name?: unknown } | null)?.full_name
    if (typeof full !== "string") throw new GitHubError(502, "unexpected fork response", null)
    for (let i = 0; i < FORK_POLLS; i++) {
      const r = await api.call("GET", `/repos/${full}`, undefined, [404])
      if (r.status === 200) return full
      await sleep(FORK_POLL_MS)
    }
    throw new Wait(BACKOFF_START_MS, "Your fork is still being created")
  }

  async function ensureBranch(api: Api, dest: string, learnerId: string): Promise<void> {
    const have = await api.call("GET", `/repos/${dest}/git/ref/heads/${branchOf(learnerId)}`, undefined, [404])
    if (have.status === 200) return
    const main = await api.call("GET", `/repos/${UPSTREAM}/git/ref/heads/main`)
    const sha = (main.body as { object?: { sha?: unknown } } | null)?.object?.sha
    if (typeof sha !== "string") throw new GitHubError(502, "unexpected ref response", null)
    try {
      await api.call("POST", `/repos/${dest}/git/refs`, { ref: `refs/heads/${branchOf(learnerId)}`, sha })
    } catch (e) {
      if (!(e instanceof GitHubError && e.status === 422)) throw e // 422: someone else created it first
    }
  }

  async function putFile(api: Api, dest: string, learnerId: string, path: string, content: string, message: string, update: boolean): Promise<void> {
    const ref = encodeURIComponent(branchOf(learnerId))
    const existing = update ? await api.call("GET", `/repos/${dest}/contents/${path}?ref=${ref}`, undefined, [404]) : null
    const sha = existing?.status === 200 ? (existing.body as { sha?: unknown } | null)?.sha : undefined
    const body = { message, content: toBase64(content), branch: branchOf(learnerId), ...(typeof sha === "string" ? { sha } : {}) }
    try {
      await api.call("PUT", `/repos/${dest}/contents/${path}`, body)
    } catch (e) {
      // A log file is write-once. If it is already there (sent earlier, never marked), that is success.
      if (!update && e instanceof GitHubError && e.status === 422) {
        const there = await api.call("GET", `/repos/${dest}/contents/${path}?ref=${ref}`, undefined, [404])
        if (there.status === 200) return
      }
      throw e
    }
  }

  async function ensurePr(api: Api, dest: string, owner: string, learnerId: string): Promise<void> {
    const head = `${owner}:${branchOf(learnerId)}`
    // Only an OPEN PR counts: once the gate merges or closes one, later logs need a fresh PR.
    const open = async () => {
      const found = await api.call("GET", `/repos/${UPSTREAM}/pulls?head=${encodeURIComponent(head)}&state=open`)
      return Array.isArray(found.body) && found.body.length > 0
    }
    if (await open()) return
    try {
      await createPr(api, dest, owner, learnerId, head)
    } catch (e) {
      // Two tabs share one outbox: the other tab may have opened it first.
      if (e instanceof GitHubError && e.status === 422 && (await open())) return
      throw e
    }
  }

  async function createPr(api: Api, dest: string, owner: string, learnerId: string, head: string): Promise<void> {
    await api.call("POST", `/repos/${UPSTREAM}/pulls`, {
      title: `run: ${learnerId}`,
      head,
      base: "main",
      draft: true,
      body: `Session logs from learner ${learnerId}, submitted by the blanktab page on behalf of @${owner}. Branch: ${dest}@${branchOf(learnerId)}.`,
    })
  }

  async function sendAll(token: string): Promise<void> {
    const pending = await queue.pending()
    if (pending.length === 0) return
    const api = createApi(token, fetchFn, () => now().getTime())
    const me = await whoAmI(token, fetchFn)
    const byLearner = new Map<string, LogFile[]>()
    for (const f of pending) byLearner.set(f.learnerId, [...(byLearner.get(f.learnerId) ?? []), f])
    const dest = await destination(api, me)
    const owner = dest.split("/")[0]!
    for (const [id, files] of byLearner) {
      files.sort((a, b) => a.seq - b.seq)
      await ensureBranch(api, dest, id)
      // Each log is marked sent right after its PUT. The last stays queued until the heartbeat and PR
      // are done, so a failure there is retried instead of leaving a branch with no PR.
      const last = files[files.length - 1]!
      for (const f of files) {
        status("sending", `Sending session log ${f.seq} of learner ${id}…`, (await queue.pending()).length)
        await putFile(api, dest, id, `${runPath(id)}/${f.seq}.json`, JSON.stringify(f), `log: ${id} #${f.seq}`, false)
        if (f !== last) await queue.markSent(f.learnerId, f.seq)
      }
      const hb = { learnerId: id, writtenAt: now().toISOString() }
      await putFile(api, dest, id, `${runPath(id)}/heartbeat.json`, JSON.stringify(hb), `heartbeat: ${id}`, true)
      await ensurePr(api, dest, owner, id)
      await queue.markSent(last.learnerId, last.seq)
    }
  }

  async function run(): Promise<void> {
    const count = async () => (await queue.pending()).length
    const token = getToken(ls)
    if (token === null) return status("no-token", "Add a GitHub token to share your session logs.", await count())
    const before = await count()
    if (before === 0) return status("idle", "All session logs are shared.", 0)
    status("sending", `Sending ${before} session log${before === 1 ? "" : "s"} to GitHub…`, before)
    try {
      await sendAll(token)
      backoff = BACKOFF_START_MS
      status("idle", "All session logs are shared.", await count())
    } catch (e) {
      const pending = await count()
      if (e instanceof GitHubError && e.status === 401) return status("error", REJECTED_MESSAGE, pending)
      let ms: number
      let msg: string
      if (e instanceof GitHubError && e.status === 403 && !e.rateLimited) return status("error", PERMISSION_MESSAGE, pending)
      if (e instanceof GitHubError && e.rateLimited) {
        ms = Math.max(e.retryAfterMs ?? DEFAULT_RATE_LIMIT_MS, 1000)
        msg = `GitHub is limiting requests; retrying in ${Math.ceil(ms / 1000)} s.`
      } else if (e instanceof Wait) {
        ms = e.ms
        msg = `${e.why}; retrying in ${Math.ceil(ms / 1000)} s.`
      } else {
        ms = backoff
        backoff = Math.min(backoff * 2, BACKOFF_MAX_MS)
        msg = e instanceof NetworkError
          ? `Can't reach GitHub; retrying in ${Math.ceil(ms / 1000)} s.`
          : `GitHub had a problem (${e instanceof GitHubError ? e.status : "unknown"}); retrying in ${Math.ceil(ms / 1000)} s.`
      }
      status("error", msg, pending)
      armRetry(ms)
    }
  }

  const contributor = {
    flush(): Promise<void> {
      inFlight ??= run().finally(() => { inFlight = null })
      return inFlight
    },
  }
  return contributor
}
