import { describe, it, expect } from "vitest"
import { IDBFactory } from "fake-indexeddb"
import { openQueue, type Queue } from "../src/queue"
import { createContributor, type ContributorStatus } from "../src/contributor"
import { setToken } from "../src/token"
import { parseHeartbeat, parseLogFile } from "../src/log"
import { stubStorage, mkFile } from "./helpers"
import { fakeGithub } from "./fakeGithub"

const TOKEN = "ghp_SECRET_token_1234567890"
const ID = "lrn_0123456789abcdef"
const NOW = new Date(1_700_000_000_000)

async function setup(opts: { canPush?: boolean; withToken?: boolean; files?: number; existingForkAs?: string } = {}) {
  const idb = new IDBFactory()
  const queue: Queue = await openQueue(idb)
  const n = opts.files ?? 6
  for (let s = 0; s < n; s++) await queue.put(mkFile(ID, s, { final: s === n - 1 }))
  const ls = stubStorage()
  if (opts.withToken !== false) setToken(ls, TOKEN)
  const gh = fakeGithub({ token: TOKEN, login: "ann", canPush: opts.canPush ?? true, ...(opts.existingForkAs ? { existingForkAs: opts.existingForkAs } : {}) })
  const statuses: ContributorStatus[] = []
  const scheduled: number[] = []
  const timers: Array<() => void> = []
  const c = createContributor({
    queue, ls, fetchFn: gh.fetchFn, now: () => NOW, onStatus: s => statuses.push(s),
    sleep: async () => {},
    schedule: (fn, ms) => { scheduled.push(ms); timers.push(fn) },
  })
  return { idb, queue, ls, gh, c, statuses, scheduled, timers, last: () => statuses.at(-1)! }
}

describe("contributor: push access", () => {
  it("creates the branch, puts 6 files + heartbeat, opens 1 draft PR, empties the queue", async () => {
    const t = await setup()
    await t.c.flush()
    const reqs = t.gh.requests
    expect(reqs.filter(r => r.method === "POST" && r.url.endsWith("/git/refs"))).toHaveLength(1)
    const puts = reqs.filter(r => r.method === "PUT")
    expect(puts.map(r => new URL(r.url).pathname)).toEqual([
      ...[0, 1, 2, 3, 4, 5].map(s => `/repos/markwhiting/blanktab/contents/data/runs/${ID}/${s}.json`),
      `/repos/markwhiting/blanktab/contents/data/runs/${ID}/heartbeat.json`,
    ])
    for (const p of puts) expect(JSON.parse(p.body!).branch).toBe(`run/${ID}`)
    for (let s = 0; s < 6; s++) {
      const f = t.gh.file("markwhiting/blanktab", `run/${ID}`, `data/runs/${ID}/${s}.json`)
      expect(parseLogFile(f).ok).toBe(true)
      expect(f).toEqual(mkFile(ID, s, { final: s === 5 }))
    }
    const hb = t.gh.file("markwhiting/blanktab", `run/${ID}`, `data/runs/${ID}/heartbeat.json`)
    expect(parseHeartbeat(hb).ok).toBe(true)
    expect(hb).toEqual({ learnerId: ID, writtenAt: NOW.toISOString() })
    expect(t.gh.pulls).toEqual([expect.objectContaining({ title: `run: ${ID}`, head: `markwhiting:run/${ID}`, base: "main", draft: true })])
    expect(await t.queue.pending()).toEqual([])
    expect(t.last()).toMatchObject({ state: "idle", pending: 0 })
    expect(reqs.some(r => r.url.includes("/forks"))).toBe(false)
  })

  it("a second flush with nothing pending makes no requests", async () => {
    const t = await setup()
    await t.c.flush()
    const n = t.gh.requests.length
    await t.c.flush()
    expect(t.gh.requests.length).toBe(n)
  })

  it("updates an existing heartbeat using its blob sha and reuses an existing branch and PR", async () => {
    const t = await setup({ files: 2 })
    await t.c.flush()
    await t.queue.put(mkFile(ID, 2))
    t.gh.requests.length = 0
    await t.c.flush()
    const hbPut = t.gh.requests.filter(r => r.method === "PUT" && r.url.endsWith("heartbeat.json"))
    expect(JSON.parse(hbPut[0]!.body!).sha).toMatch(/^blob/)
    expect(t.gh.requests.some(r => r.method === "POST")).toBe(false)
    expect(t.gh.pulls).toHaveLength(1)
  })

  it("never overwrites a file already on the branch (crash between PUT and markSent)", async () => {
    const t = await setup({ files: 2 })
    // File 0 landed and was marked sent; file 1 fails with a 503 and is retried.
    t.gh.interceptors.push(r => (r.method === "PUT" && r.url.endsWith("/1.json") ? new Response("{}", { status: 503 }) : undefined))
    await t.c.flush()
    expect((await t.queue.pending()).map(f => f.seq)).toEqual([1])
    t.gh.interceptors.length = 0
    await t.c.flush()
    expect(await t.queue.pending()).toEqual([])
    expect(t.gh.file("markwhiting/blanktab", `run/${ID}`, `data/runs/${ID}/0.json`)).toEqual(mkFile(ID, 0))
  })

  it("groups several learners, one branch and PR each", async () => {
    const t = await setup({ files: 2 })
    const other = "lrn_fedcba9876543210"
    await t.queue.put(mkFile(other, 0))
    await t.c.flush()
    expect(t.gh.pulls.map(p => p.head).sort()).toEqual([`markwhiting:run/${other}`, `markwhiting:run/${ID}`].sort())
    expect(await t.queue.pending()).toEqual([])
  })

  it("encodes the body as base64 of UTF-8 JSON", async () => {
    const t = await setup({ files: 1 })
    await t.queue.put(mkFile("lrn_aaaaaaaaaaaaaaaa", 0, { settings: { note: "héllo ✓" } }))
    await t.c.flush()
    expect(t.gh.file("markwhiting/blanktab", "run/lrn_aaaaaaaaaaaaaaaa", "data/runs/lrn_aaaaaaaaaaaaaaaa/0.json")).toMatchObject({ settings: { note: "héllo ✓" } })
  })
})

describe("contributor: no push access", () => {
  it("creates a fork first, polls until ready, writes to the fork, PR goes to upstream", async () => {
    const t = await setup({ canPush: false, files: 2 })
    await t.c.flush()
    const reqs = t.gh.requests.map(r => `${r.method} ${new URL(r.url).pathname}`)
    const forkAt = reqs.indexOf("POST /repos/markwhiting/blanktab/forks")
    expect(forkAt).toBeGreaterThan(-1)
    expect(reqs.indexOf("PUT /repos/ann/blanktab/contents/data/runs/" + ID + "/0.json")).toBeGreaterThan(forkAt)
    expect(reqs.filter(r => r === "GET /repos/ann/blanktab").length).toBeGreaterThan(2)
    expect(t.gh.file("ann/blanktab", `run/${ID}`, `data/runs/${ID}/1.json`)).toBeTruthy()
    expect(t.gh.pulls[0]).toMatchObject({ head: `ann:run/${ID}`, base: "main", draft: true })
    expect(reqs.some(r => r.startsWith("PUT /repos/markwhiting"))).toBe(false)
    expect(await t.queue.pending()).toEqual([])
  })

  it("uses the full_name an existing differently named fork, without creating another", async () => {
    const t = await setup({ canPush: false, files: 1, existingForkAs: "ann/blanktab-1" })
    await t.c.flush()
    expect(t.gh.file("ann/blanktab-1", `run/${ID}`, `data/runs/${ID}/0.json`)).toBeTruthy()
    expect(t.gh.pulls[0]!.head).toBe(`ann:run/${ID}`)
  })

  it("does not create a second fork when one already exists", async () => {
    const t = await setup({ canPush: false, files: 1, existingForkAs: "ann/blanktab" })
    await t.c.flush()
    expect(t.gh.requests.some(r => r.url.endsWith("/forks"))).toBe(false)
  })
})

describe("contributor: GitHub failures (Review Focus 4)", () => {
  it("401 leaves the queue intact, shows the rejection message, and does not auto-retry", async () => {
    const t = await setup()
    setToken(t.ls, "ghp_bad")
    await t.c.flush()
    expect((await t.queue.pending()).length).toBe(6)
    expect(t.last()).toEqual({ state: "error", message: "Your GitHub token was rejected — paste a new one.", pending: 6 })
    expect(t.scheduled).toEqual([])
    expect(t.gh.requests.some(r => r.method === "PUT")).toBe(false)
  })

  it("401 midway keeps unsent files pending", async () => {
    const t = await setup()
    let puts = 0
    t.gh.interceptors.push(r => (r.method === "PUT" && ++puts === 3 ? new Response('{"message":"Bad credentials"}', { status: 401 }) : undefined))
    await t.c.flush()
    expect(t.last().message).toMatch(/rejected/)
    expect((await t.queue.pending()).map(f => f.seq)).toEqual([2, 3, 4, 5]) // 0 and 1 were sent and marked
  })

  it("403 with Retry-After: 30 schedules a retry of at least 30 s, queue intact", async () => {
    const t = await setup()
    t.gh.interceptors.push(r => (r.method === "PUT" ? new Response('{"message":"rate"}', { status: 403, headers: { "Retry-After": "30" } }) : undefined))
    await t.c.flush()
    expect(t.scheduled).toHaveLength(1)
    expect(t.scheduled[0]!).toBeGreaterThanOrEqual(30_000)
    expect(t.last().state).toBe("error")
    expect((await t.queue.pending()).length).toBe(6)
    // the scheduled retry works once the limit clears
    t.gh.interceptors.length = 0
    t.timers[0]!()
    await new Promise(r => setTimeout(r, 50))
    expect(await t.queue.pending()).toEqual([])
  })

  it("429 honours X-RateLimit-Reset", async () => {
    const t = await setup()
    const reset = Math.floor(NOW.getTime() / 1000) + 120
    t.gh.interceptors.push(r => (r.method === "PUT" ? new Response("{}", { status: 429, headers: { "X-RateLimit-Reset": String(reset), "X-RateLimit-Remaining": "0" } }) : undefined))
    await t.c.flush()
    expect(t.scheduled[0]!).toBeGreaterThanOrEqual(120_000)
  })

  it("offline (fetch throws) retries with exponential backoff from 5 s, capped at 10 min", async () => {
    const t = await setup()
    t.gh.interceptors.push(() => { throw new TypeError("Failed to fetch") })
    await t.c.flush()
    for (let i = 0; i < 9; i++) { t.timers.shift()!(); await new Promise(r => setTimeout(r, 20)) }
    expect(t.scheduled).toEqual([5000, 10000, 20000, 40000, 80000, 160000, 320000, 600000, 600000, 600000])
    expect((await t.queue.pending()).length).toBe(6)
    expect(t.last()).toMatchObject({ state: "error", pending: 6 })
    t.gh.interceptors.length = 0
    t.timers.shift()!()
    await new Promise(r => setTimeout(r, 50))
    expect(await t.queue.pending()).toEqual([])
    expect(t.last().state).toBe("idle")
  })

  it("concurrent flushes do not double-send", async () => {
    const t = await setup()
    await Promise.all([t.c.flush(), t.c.flush()])
    expect(t.gh.requests.filter(r => r.method === "PUT" && r.url.endsWith("/0.json"))).toHaveLength(1)
  })
})

describe("contributor: no token (Review Focus 5)", () => {
  it("queues with state no-token and no requests; after setToken all 6 send in seq order", async () => {
    const t = await setup({ withToken: false })
    await t.c.flush()
    expect(t.last()).toMatchObject({ state: "no-token", pending: 6 })
    expect(t.gh.requests).toEqual([])
    setToken(t.ls, TOKEN)
    await t.c.flush()
    const order = t.gh.requests.filter(r => r.method === "PUT" && !r.url.endsWith("heartbeat.json")).map(r => r.url.match(/(\d+)\.json$/)![1])
    expect(order).toEqual(["0", "1", "2", "3", "4", "5"])
    expect(await t.queue.pending()).toEqual([])
  })
})

describe("contributor: token never leaks (Review Focus 1)", () => {
  it("only api.github.com; token only in Authorization header; never in a body or in IndexedDB", async () => {
    for (const canPush of [true, false]) {
      const t = await setup({ canPush })
      await t.c.flush()
      expect(t.gh.requests.length).toBeGreaterThan(10)
      for (const r of t.gh.requests) {
        expect(new URL(r.url).host).toBe("api.github.com")
        expect(new URL(r.url).protocol).toBe("https:")
        expect(r.body ?? "").not.toContain(TOKEN)
        expect(r.url).not.toContain(TOKEN)
        expect(r.headers.Authorization).toBe(`Bearer ${TOKEN}`)
        expect(r.headers["x-github-api-version"]).toBe("2022-11-28")
        const others = Object.entries(r.headers).filter(([k]) => k !== "Authorization")
        for (const [, v] of others) expect(v).not.toContain(TOKEN)
      }
      for (const s of t.statuses) expect(JSON.stringify(s)).not.toContain(TOKEN)
      for (const p of t.gh.repos.values()) for (const f of p.files.values()) expect(atob(f.content)).not.toContain(TOKEN)
      // dump every IndexedDB record, raw
      const db = await new Promise<IDBDatabase>((res, rej) => { const q = t.idb.open("blanktab"); q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error) })
      const rows = await new Promise<unknown[]>((res, rej) => { const q = db.transaction("outbox").objectStore("outbox").getAll(); q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error) })
      expect(rows).toHaveLength(6)
      expect(JSON.stringify(rows)).not.toContain(TOKEN)
      expect(Array.from(db.objectStoreNames)).toEqual(["outbox"])
    }
  })

  it("error statuses never contain the token", async () => {
    const t = await setup()
    t.gh.interceptors.push(() => new Response(JSON.stringify({ message: `bad ${TOKEN}` }), { status: 500 }))
    await t.c.flush()
    expect(t.statuses.length).toBeGreaterThan(0)
    for (const s of t.statuses) expect(JSON.stringify(s)).not.toContain(TOKEN)
  })
})

describe("contributor: review fixes", () => {
  it("opens a new PR when the earlier one was merged/closed", async () => {
    const t = await setup({ files: 2 })
    await t.c.flush()
    expect(t.gh.pulls).toHaveLength(1)
    t.gh.pulls[0]!.state = "closed"
    await t.queue.put(mkFile(ID, 2))
    await t.c.flush()
    expect(t.gh.pulls).toHaveLength(2)
    expect(t.gh.pulls[1]).toMatchObject({ state: "open", draft: true })
    expect(await t.queue.pending()).toEqual([])
  })

  it("does not open a second PR while one is open", async () => {
    const t = await setup({ files: 2 })
    await t.c.flush()
    await t.queue.put(mkFile(ID, 2))
    await t.c.flush()
    expect(t.gh.pulls).toHaveLength(1)
  })

  it("403 with x-ratelimit-remaining 0 waits until the reset time", async () => {
    const t = await setup()
    const reset = Math.floor(NOW.getTime() / 1000) + 300
    t.gh.interceptors.push(r => (r.method === "PUT" ? new Response("{}", { status: 403, headers: { "x-ratelimit-remaining": "0", "x-ratelimit-reset": String(reset) } }) : undefined))
    await t.c.flush()
    expect(t.scheduled[0]!).toBeGreaterThanOrEqual(300_000)
  })

  it("429 with no hints defaults to at least 60 s; reset is ignored while quota remains", async () => {
    const t = await setup()
    const reset = Math.floor(NOW.getTime() / 1000) + 1
    t.gh.interceptors.push(r => (r.method === "PUT" ? new Response("{}", { status: 429, headers: { "x-ratelimit-remaining": "40", "x-ratelimit-reset": String(reset) } }) : undefined))
    await t.c.flush()
    expect(t.scheduled[0]!).toBeGreaterThanOrEqual(60_000)
  })

  it("a plain 403 is a permissions error: no auto-retry, queue intact", async () => {
    const t = await setup()
    t.gh.interceptors.push(r => (r.method === "PUT" ? new Response('{"message":"Resource not accessible"}', { status: 403, headers: { "x-ratelimit-remaining": "4000" } }) : undefined))
    await t.c.flush()
    expect(t.last()).toEqual({ state: "error", message: "Your GitHub token can't write to the repo — check its permissions.", pending: 6 })
    expect(t.scheduled).toEqual([])
    expect((await t.queue.pending()).length).toBe(6)
  })

  it("marks each log sent right after its PUT; the last stays pending until heartbeat and PR succeed", async () => {
    const t = await setup({ files: 3 })
    t.gh.interceptors.push(r => (r.method === "PUT" && r.url.endsWith("heartbeat.json") ? new Response("{}", { status: 503 }) : undefined))
    await t.c.flush()
    expect((await t.queue.pending()).map(f => f.seq)).toEqual([2])
    expect(t.gh.pulls).toHaveLength(0)
    t.gh.interceptors.length = 0
    await t.c.flush()
    expect(await t.queue.pending()).toEqual([])
    expect(t.gh.pulls).toHaveLength(1)
  })

  it("treats a 422 'pull request already exists' (another tab) as success", async () => {
    const t = await setup({ files: 1 })
    t.gh.interceptors.push(r => {
      if (r.method !== "POST" || !r.url.endsWith("/pulls")) return undefined
      t.gh.pulls.push({ ...JSON.parse(r.body!), state: "open" }) // the other tab won the race
      return new Response('{"message":"A pull request already exists"}', { status: 422 })
    })
    await t.c.flush()
    expect(t.last()).toMatchObject({ state: "idle", pending: 0 })
    expect(t.gh.pulls).toHaveLength(1)
  })
})
