import { test, expect, type Page, type Request } from "@playwright/test"
import { fakeGithub } from "../../packages/harness/test/fakeGithub"

const TOKEN = "test-token"
const UP = "markwhiting/learner"
/** The real learner on the CPU at tiny widths: a 0.05 CU life is a few dozen steps, with a sleep (a curve point) every 5. */
const KNOBS = "testBackend=cpu&testWidths=tiny&testSleepInterval=5"
const LIFETIME_CU = 0.05
const PAGE = `/learner/?${KNOBS}&testLifetimeCU=${LIFETIME_CU}&testStepsPerSecond=10`
/** A life that outlasts the test. */
const LONG = `/learner/?${KNOBS}&testLifetimeCU=50&testStepsPerSecond=20`
const TINY = { conv1: 1, conv2: 1, enc: 2, hidden: 2 }
const ID_RE = /lrn_[0-9a-f]{16}/

const row = (id: string) => ({
  learnerId: id, settings: { moveProbability: 0.5 }, hostSig: { cores: 4, deviceMemoryGb: null, webgpu: false, tabsOnMachine: 1 },
  stepsTotal: 3000, final: true, codeSha: "a".repeat(40), knowledgeSha: "b".repeat(64), sessionsSeenAtBirth: 0,
})
/** The settings space the e2e learners are born from: fixed values, so a life's length in steps is predictable. */
const SPACE = [
  { name: "lr", min: 1e-3, max: 1e-3 }, { name: "entropyWeight", min: 0.01, max: 0.01 }, { name: "curiosityWeight", min: 1, max: 1 },
  { name: "bufferSize", min: 500, max: 500 }, { name: "replayPerSleep", min: 16, max: 16 }, { name: "headLrScale", min: 0.1, max: 0.1 },
  { name: "maxStepsPerSec", min: 30, max: 30 }, { name: "moveProbability", min: 0.1, max: 1 },
]
const KNOWLEDGE: Record<string, unknown> = {
  "summary.json": { schemaVersion: 1, sessions: [row("lrn_aaaaaaaaaaaaaaaa"), { ...row("lrn_bbbbbbbbbbbbbbbb"), cu: 1024.5, lossRatio: 0.9123 }] },
  "manifest.json": { knowledgeSha: "c".repeat(64), codeSha: "a".repeat(40) },
  "settings-space.json": { schemaVersion: 1, settings: SPACE },
}

const CORS = {
  "access-control-allow-origin": "*",
  "access-control-allow-headers": "authorization, content-type, x-github-api-version, accept",
  "access-control-allow-methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
}

/** Mocks knowledge/*.json and api.github.com, and records every request the page (or its worker) makes. */
async function setup(page: Page, opts: { knowledge: "ok" | "unreachable"; token?: string }) {
  const gh = fakeGithub({ token: TOKEN, login: "tester", canPush: true })
  const all: Request[] = []
  page.context().on("request", r => all.push(r))
  await page.route("**/knowledge/*.json", route => {
    if (opts.knowledge === "unreachable") return route.abort("internetdisconnected")
    const name = new URL(route.request().url()).pathname.split("/").pop()!
    const body = KNOWLEDGE[name]
    return body === undefined ? route.fulfill({ status: 404 }) : route.fulfill({ json: body })
  })
  await page.route("https://api.github.com/**", async route => {
    const req = route.request()
    if (req.method() === "OPTIONS") return route.fulfill({ status: 204, headers: CORS })
    const res = await gh.fetchFn(req.url(), { method: req.method(), headers: await req.allHeaders(), ...(req.postData() ? { body: req.postData()! } : {}) })
    return route.fulfill({ status: res.status, headers: { ...CORS, "content-type": "application/json" }, body: await res.text() })
  })
  if (opts.token !== undefined) {
    const token = opts.token
    await page.addInitScript(t => { if (localStorage.getItem("blanktab:githubToken") === null) localStorage.setItem("blanktab:githubToken", t) }, token)
  }
  const githubCalls = () => gh.requests.filter(r => r.method !== "OPTIONS")
  const commits = () => githubCalls().filter(r => r.method === "POST" && r.url === `https://api.github.com/repos/${UP}/git/commits`)
  return { gh, all, githubCalls, commits }
}

const stepsLived = async (page: Page) => Number((await page.getByTestId("steps").textContent())?.match(/^(\d+)/)?.[1] ?? NaN)
const cuLived = async (page: Page) => Number((await page.getByTestId("cu").textContent())?.match(/^([\d.]+) of/)?.[1] ?? NaN)
const curvePoints = async (page: Page) => Number(await page.getByTestId("curve").getAttribute("data-points"))

/** Writes rows straight into the page's IndexedDB (as an earlier page version or a crashed life would have left them). */
async function seedDb(page: Page, rows: { outbox?: unknown[]; checkpoints?: unknown[] }) {
  await page.evaluate(r => new Promise<void>((res, rej) => {
    const req = indexedDB.open("blanktab")
    req.onerror = () => rej(req.error)
    req.onsuccess = () => {
      const tx = req.result.transaction(["outbox", "checkpoints"], "readwrite")
      for (const row of r.outbox ?? []) tx.objectStore("outbox").put(row)
      for (const row of r.checkpoints ?? []) tx.objectStore("checkpoints").put(row)
      tx.oncomplete = () => { req.result.close(); res() }
      tx.onerror = () => rej(tx.error)
    }
  }), rows)
}

test("the loop: knowledge in, the real learner lives its life in CU, its v2 log goes to GitHub as one PR, a newborn follows", async ({ page, baseURL }) => {
  const { gh, all, githubCalls } = await setup(page, { knowledge: "ok" })
  await page.goto(PAGE)

  // Sessions before this one: the visible proof that the loop is closed.
  await expect(page.getByText("2 sessions before this one")).toBeVisible()
  await expect(page.getByText(/offline/i)).toHaveCount(0)

  const learner = page.getByTestId("learner-id")
  await expect(learner).toHaveText(/lrn_[0-9a-f]{16}/)
  const id = (await learner.textContent())!.match(/lrn_[0-9a-f]{16}/)![0]
  expect(await page.evaluate(() => sessionStorage.getItem("blanktab:learnerId"))).toBe(id)
  // The settings it was born with, as the learner uses them (moveProbability is not one of them).
  await expect(page.getByTestId("setting")).toContainText("replayPerSleep = 16")
  await expect(page.getByTestId("setting")).toContainText("maxStepsPerSec = 10") // the speed knob is the cap it lives at
  await expect(page.getByTestId("setting")).not.toContainText("moveProbability")
  await expect(page.getByTestId("backend")).toHaveText(/^CPU/)

  // Steps, compute and the loss curve all move.
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0)
  const first = await stepsLived(page)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(first)
  await expect(page.getByTestId("cu")).toContainText(`of ${LIFETIME_CU} CU`)
  await expect.poll(() => cuLived(page)).toBeGreaterThan(0)
  await expect.poll(() => curvePoints(page)).toBeGreaterThan(0)
  await expect(page.getByTestId("ratio")).toHaveText(/^\d+(\.\d+)? \(its guesses are/)
  await expect(page.getByTestId("rate")).toHaveText(/^[\d.]+ steps\/s$/)
  await expect(page.getByTestId("curve-legend")).toContainText("how wrong its guesses are")
  await expect(page.getByTestId("curve-legend")).toContainText("the lazy guess: nothing changes")

  // No token: the panel explains how to get one; nothing has gone to GitHub.
  const panel = page.getByTestId("token-panel")
  await expect(panel).toBeVisible()
  await expect(page.getByTestId("change-token")).toBeHidden()
  await page.screenshot({ path: test.info().outputPath("page-notoken.png"), fullPage: true })
  await expect(panel.getByRole("link")).toHaveAttribute("href", "https://github.com/settings/tokens/new?scopes=public_repo&description=learner")
  await expect(page.getByTestId("contribution")).toContainText(/token/i)
  expect(githubCalls()).toHaveLength(0)

  await panel.getByLabel(/GitHub token/i).fill(TOKEN)
  await panel.getByRole("button", { name: "Check & save" }).click()
  await expect(panel).toBeHidden()
  expect(githubCalls()[0]?.url).toBe("https://api.github.com/user") // checkToken ran before the token was used

  // The life goes final below the first crossing (8 CU = 2^3), so its one log file is the final one: data/runs/<id>/0.json, in one PR.
  await expect(page.getByText("This learner's life is complete")).toBeVisible({ timeout: 30_000 })
  const pulls = () => githubCalls().filter(r => r.method === "POST" && r.url === `https://api.github.com/repos/${UP}/pulls` && JSON.parse(r.body ?? "{}").head === `markwhiting:run/${id}`)
  await expect.poll(() => pulls().length, { timeout: 30_000 }).toBe(1)
  expect(gh.prFiles(gh.pulls[0]!)).toEqual([`data/runs/${id}/0.json`])
  expect(githubCalls().some(r => r.method === "PUT" || r.url.includes("/contents/") || (r.body ?? "").includes("heartbeat"))).toBe(false)
  await expect(page.getByTestId("change-token")).toBeVisible()
  const final = gh.file(UP, `run/${id}`, `data/runs/${id}/0.json`) as Record<string, unknown> & { cu: number; curve: { predLoss: number; baselineLoss: number }[]; settings: Record<string, number> }
  expect(final).toMatchObject({
    schemaVersion: 2, seq: 0, final: true, sessionsSeenAtBirth: 2, knowledgeSha: "c".repeat(64), codeSha: "dev", condition: "A",
    widths: TINY, hostSig: { webgpu: false }, restores: 0, lostCU: 0,
  })
  expect(final.cu).toBeGreaterThanOrEqual(LIFETIME_CU)
  expect(final.curve.length).toBeGreaterThan(0)
  for (const p of final.curve) expect(Number.isFinite(p.predLoss) && Number.isFinite(p.baselineLoss)).toBe(true)
  expect(final.settings).toMatchObject({ replayPerSleep: 16, bufferSize: 500, maxStepsPerSec: 10 })
  expect(final.settings).not.toHaveProperty("moveProbability")

  // A newborn with a new learnerId takes its place.
  await expect(learner).not.toHaveText(id)
  await expect(learner).toHaveText(/lrn_[0-9a-f]{16}/)
  expect(await page.evaluate(() => sessionStorage.getItem("blanktab:learnerId"))).not.toBe(id)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0)

  // Review Focus 1: no host but api.github.com and the page's own origin; the token is in no body, no file, no IndexedDB row.
  const origin = new URL(baseURL!).origin
  const strays = all.map(r => r.url()).filter(u => !u.startsWith("data:") && !u.startsWith("blob:") && new URL(u).origin !== origin && new URL(u).origin !== "https://api.github.com")
  expect(strays).toEqual([])
  for (const r of all) expect(r.postData() ?? "").not.toContain(TOKEN)
  for (const r of all) if (new URL(r.url()).origin !== "https://api.github.com") expect(JSON.stringify(await r.allHeaders())).not.toContain(TOKEN)
  for (const r of gh.requests) {
    const content = r.body ? (JSON.parse(r.body) as { content?: string }).content : undefined
    if (content) expect(gh.decode(content)).not.toContain(TOKEN)
  }
  for (const b of gh.blobs.values()) expect(gh.decode(b)).not.toContain(TOKEN)
  const idb = await page.evaluate(() => new Promise<string>((res, rej) => {
    const req = indexedDB.open("blanktab")
    req.onerror = () => rej(req.error)
    req.onsuccess = () => {
      const get = req.result.transaction("outbox").objectStore("outbox").getAll()
      get.onsuccess = () => res(JSON.stringify(get.result))
    }
  }))
  expect(idb).toContain(id)
  expect(idb).not.toContain(TOKEN)
})

test("knowledge unreachable: the page says offline and the learner runs anyway", async ({ page }) => {
  const { githubCalls } = await setup(page, { knowledge: "unreachable" })
  await page.goto(LONG)
  await expect(page.getByTestId("knowledge")).toHaveText(/offline/i)
  await expect(page.getByText("0 sessions before this one")).toBeVisible()
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0)
  const first = await stepsLived(page)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(first)
  expect(githubCalls()).toHaveLength(0)
})

test("cadence: with a token, a whole life goes to GitHub as one commit, sent when the final log is queued", async ({ page }) => {
  const { gh, githubCalls, commits } = await setup(page, { knowledge: "ok", token: TOKEN })
  // At the CPU half widths (not tiny): far more compute per step, so 8 CU takes seconds, not a minute.
  await page.goto("/learner/?testBackend=cpu&testSleepInterval=50&testLifetimeCU=8.5&testStepsPerSecond=300") // a log at 8 CU (not final) and the final one
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  const id = (await page.getByTestId("learner-id").textContent())!.match(ID_RE)![0]
  await expect.poll(() => cuLived(page), { timeout: 60_000 }).toBeGreaterThan(8.05) // the non-final log is queued by now
  expect(githubCalls().filter(r => r.method !== "GET")).toEqual([]) // ...and nothing was written for it
  await expect(page.getByText("This learner's life is complete")).toBeVisible({ timeout: 60_000 })
  await expect.poll(() => gh.pulls.length, { timeout: 30_000 }).toBe(1)
  expect(commits()).toHaveLength(1)
  expect(gh.prFiles(gh.pulls[0]!)).toEqual([0, 1].map(s => `data/runs/${id}/${s}.json`))
  const files = [0, 1].map(s => gh.file(UP, `run/${id}`, `data/runs/${id}/${s}.json`) as { cu: number; final: boolean })
  expect(files.map(f => f.final)).toEqual([false, true])
  expect(files[0]!.cu).toBeGreaterThanOrEqual(8)
  expect(files[1]!.cu).toBeGreaterThanOrEqual(8.5)
})

test("token stored: a small 'change token' control clears it and brings the panel back", async ({ page }) => {
  await setup(page, { knowledge: "ok", token: TOKEN })
  await page.goto(LONG)
  const panel = page.getByTestId("token-panel")
  const change = page.getByTestId("change-token")
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  await expect(change).toBeVisible()
  await expect(panel).toBeHidden()
  await page.screenshot({ path: test.info().outputPath("page-token.png"), fullPage: true })
  await change.click()
  await expect(panel).toBeVisible()
  await expect(change).toBeHidden()
  expect(await page.evaluate(() => localStorage.getItem("blanktab:githubToken"))).toBeNull()
  await expect(page.getByTestId("contribution")).toContainText(/token/i)
})

test("permission error: the token panel comes back (with 'change token') instead of a dead end", async ({ page }) => {
  const { gh } = await setup(page, { knowledge: "ok", token: TOKEN })
  gh.interceptors.push(r => (r.method === "POST" && r.url.endsWith("/git/blobs") ? new Response('{"message":"Resource not accessible by personal access token"}', { status: 403 }) : undefined))
  await page.goto(PAGE)
  await expect(page.getByTestId("token-panel")).toBeHidden()
  await expect(page.getByTestId("contribution")).toContainText("can't write to the repo", { timeout: 30_000 })
  await expect(page.getByTestId("token-panel")).toBeVisible()
  await expect(page.getByTestId("change-token")).toBeVisible()
  await page.screenshot({ path: test.info().outputPath("page-permission.png"), fullPage: true })
})

test("an unexpected failure while checking a token shows an error instead of hanging", async ({ page }) => {
  await setup(page, { knowledge: "ok" })
  await page.addInitScript(() => {
    const orig = Storage.prototype.setItem
    Storage.prototype.setItem = function (k: string, v: string) { if (k === "blanktab:githubToken") throw new Error("storage is full"); return orig.call(this, k, v) }
  })
  await page.goto(LONG)
  const panel = page.getByTestId("token-panel")
  await panel.getByLabel(/GitHub token/i).fill(TOKEN)
  await panel.getByRole("button", { name: "Check & save" }).click()
  await expect(page.locator("#token-error")).toHaveText(/went wrong/i)
  await expect(panel.getByRole("button", { name: "Check & save" })).toBeEnabled()
})

test("the learner always runs: a worker error starts a newborn", async ({ page }) => {
  await setup(page, { knowledge: "ok" })
  await page.goto(LONG)
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  const broken = "lrn_badbadbadbadbad0"
  // A checkpoint whose learner state the session refuses: the life throws at start.
  await seedDb(page, { checkpoints: [{ learnerId: broken, savedAt: new Date().toISOString(), steps: 10, cu: 0.01, seq: 0, curveTail: [], restores: 0, lostCU: 0, lostSkips: 0, learnerState: { garbage: true } }] })
  await page.evaluate(id => sessionStorage.setItem("blanktab:learnerId", id), broken)
  await page.reload()
  await expect(page.getByTestId("life")).toContainText("The learner stopped")
  await expect(page.getByTestId("learner-id")).not.toHaveText(broken, { timeout: 10_000 })
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  const first = await stepsLived(page)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(first)
})

test("a learner with logs but no checkpoint (e.g. the old flailer's) cannot be resumed: a newborn takes its place", async ({ page }) => {
  await setup(page, { knowledge: "ok" })
  await page.goto(LONG)
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  const old = "lrn_01d01d01d01d01d0"
  const file = { schemaVersion: 1, learnerId: old, seq: 0, codeSha: "dev", knowledgeSha: "0".repeat(64), writtenAt: new Date().toISOString(), stepsTotal: 500,
    settings: { moveProbability: 0.5 }, hostSig: { cores: 1, deviceMemoryGb: null, webgpu: false, tabsOnMachine: 1 }, sessionsSeenAtBirth: 0, final: false }
  await seedDb(page, { outbox: [{ learnerId: old, seq: 0, sent: false, file }] })
  await page.evaluate(id => sessionStorage.setItem("blanktab:learnerId", id), old)
  await page.reload()
  await expect(page.getByTestId("life")).toContainText("cannot be resumed")
  await expect(page.getByTestId("life")).not.toContainText("stopped")
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  await expect(page.getByTestId("learner-id")).not.toHaveText(old)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0)
})

test("a reload resumes the learner from its checkpoint; a duplicated tab (same sessionStorage) gets a newborn", async ({ page, context }) => {
  await setup(page, { knowledge: "ok" })
  await page.goto(LONG)
  const learner = page.getByTestId("learner-id")
  await expect(learner).toHaveText(ID_RE)
  const id = (await learner.textContent())!.match(ID_RE)![0]
  await expect.poll(() => curvePoints(page)).toBeGreaterThan(0)

  await page.reload()
  await expect(learner).toHaveText(id)
  await expect(page.getByTestId("status")).toContainText("resumed from its checkpoint")
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0)

  // Chrome's "Duplicate tab" copies sessionStorage, so the copy starts with the same learnerId.
  const dup = await context.newPage()
  await dup.addInitScript(i => { if (sessionStorage.getItem("blanktab:dup") === null) { sessionStorage.setItem("blanktab:learnerId", i); sessionStorage.setItem("blanktab:dup", "1") } }, id)
  await dup.goto(LONG)
  await expect(dup.getByTestId("learner-id")).toHaveText(ID_RE)
  const dupId = (await dup.getByTestId("learner-id").textContent())!.match(ID_RE)![0]
  expect(dupId).not.toBe(id)
  expect(await dup.evaluate(() => sessionStorage.getItem("blanktab:learnerId"))).toBe(dupId)
  await expect(learner).toHaveText(id) // the original keeps its learner
  await expect.poll(() => stepsLived(dup)).toBeGreaterThan(0)
})

test("a closed tab's learner is adopted by the next new tab; surplus and stale orphans are deleted", async ({ page, context }) => {
  await setup(page, { knowledge: "ok" })
  await page.goto(LONG)
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  const id = (await page.getByTestId("learner-id").textContent())!.match(ID_RE)![0]
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0) // its birth checkpoint is written
  // Six more closed tabs' learners: copies of its checkpoint saved 1..5 hours ago, and one 25 hours ago (stale).
  const others = [1, 2, 3, 4, 5, 25].map(h => ({ learnerId: `lrn_${h.toString(16).padStart(16, "0")}`, hoursAgo: h }))
  await page.evaluate(({ id, others }) => new Promise<void>((res, rej) => {
    const req = indexedDB.open("blanktab")
    req.onerror = () => rej(req.error)
    req.onsuccess = () => {
      const db = req.result
      const get = db.transaction("checkpoints").objectStore("checkpoints").get(id)
      get.onsuccess = () => {
        const tx = db.transaction("checkpoints", "readwrite")
        for (const o of others) tx.objectStore("checkpoints").put({ ...get.result, learnerId: o.learnerId, savedAt: new Date(Date.now() - o.hoursAgo * 3_600_000).toISOString() })
        tx.oncomplete = () => { db.close(); res() }
        tx.onerror = () => rej(tx.error)
      }
    }
  }), { id, others })
  await page.close() // the tab closes mid-life: its learner is an orphan now

  const next = await context.newPage() // a new tab: no learnerId yet
  await setup(next, { knowledge: "ok" })
  await next.goto(LONG)
  await expect(next.getByTestId("learner-id")).toHaveText(id)
  await expect(next.getByTestId("life")).toContainText(`Adopted learner ${id}`)
  await expect(next.getByTestId("status")).toContainText("resumed from its checkpoint")
  expect(await next.evaluate(() => sessionStorage.getItem("blanktab:learnerId"))).toBe(id)
  await expect.poll(() => stepsLived(next)).toBeGreaterThan(0)
  await next.screenshot({ path: test.info().outputPath("page-adopted.png"), fullPage: true })
  // The newest three orphans stay (the adopted one among them); the older ones and the stale one are gone.
  const ids = await next.evaluate(() => new Promise<string[]>((res, rej) => {
    const req = indexedDB.open("blanktab")
    req.onerror = () => rej(req.error)
    req.onsuccess = () => {
      const keys = req.result.transaction("checkpoints").objectStore("checkpoints").getAllKeys()
      keys.onsuccess = () => { req.result.close(); res(keys.result as string[]) }
    }
  }))
  expect(ids.sort()).toEqual([id, others[0]!.learnerId, others[1]!.learnerId].sort())
})

test("a hidden page gets no frames; the learner lives on, and frames come back when the page is shown", async ({ page }) => {
  await setup(page, { knowledge: "ok" })
  await page.goto(LONG)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0)
  const setHidden = (h: boolean) => page.evaluate(hidden => {
    Object.defineProperty(document, "hidden", { configurable: true, get: () => hidden })
    document.dispatchEvent(new Event("visibilitychange"))
  }, h)
  await setHidden(true)
  await page.waitForTimeout(300) // frames already in flight land
  const frozen = await stepsLived(page)
  await page.waitForTimeout(1_500) // ~30 steps at 20 steps/s
  expect(await stepsLived(page)).toBe(frozen)
  await setHidden(false)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(frozen + 10)
})

test("an older tab holding the database open: the page says to close other learner tabs", async ({ page, context }) => {
  await setup(page, { knowledge: "ok" })
  // An old page version (database v1, no versionchange handler) open in another tab of this origin.
  const old = await context.newPage()
  await old.route("**/knowledge/*.json", route => route.fulfill({ json: {} }))
  await old.goto("/learner/knowledge/manifest.json")
  await old.evaluate(() => new Promise<void>((res, rej) => {
    const req = indexedDB.open("blanktab", 1)
    req.onupgradeneeded = () => req.result.createObjectStore("outbox", { keyPath: ["learnerId", "seq"] })
    req.onsuccess = () => { (globalThis as { held?: unknown }).held = req.result; res() }
    req.onerror = () => rej(req.error)
  }))
  await page.goto(LONG)
  await expect(page.getByTestId("life")).toContainText("Close other learner tabs to update", { timeout: 20_000 })
})

test("without a speed knob the learner lives at most maxStepsPerSec = 30 steps a second, on the backend this browser offers", async ({ page }) => {
  await setup(page, { knowledge: "ok" })
  await page.goto("/learner/?testWidths=tiny&testLifetimeCU=50") // no testBackend: the real WebGPU-or-CPU choice
  await expect(page.getByTestId("backend")).toHaveText(/^(WebGPU|CPU)/)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0)
  const t0 = Date.now(), s0 = await stepsLived(page)
  await page.waitForTimeout(5_000)
  const rate = (await stepsLived(page) - s0) / ((Date.now() - t0) / 1000)
  expect(rate).toBeGreaterThan(10)
  expect(rate).toBeLessThan(31)
})

const GH = `https://github.com/${UP}`

test("tabs: LEARNER by default; HISTORY lists past sessions newest first; ABOUT explains and links; links on every tab", async ({ page }) => {
  await setup(page, { knowledge: "ok" })
  await page.goto(LONG)
  const tab = (name: string) => page.getByRole("tab", { name })
  await expect(tab("Learner")).toHaveAttribute("aria-selected", "true")
  await expect(tab("History")).toHaveAttribute("aria-selected", "false")
  await expect(page.getByTestId("learner-id")).toBeVisible()
  await expect(page.getByTestId("history")).toBeHidden()
  await expect(page.getByTestId("build")).toHaveText(/^dev built \d{4}-\d\d-\d\d \d\d:\d\d UTC$/)
  await expect(page.getByTestId("build")).toHaveAttribute("href", GH)

  const footer = page.getByTestId("footer-links")
  const footerLinks = async () => footer.getByRole("link").evaluateAll(as => as.map(a => (a as HTMLAnchorElement).href))
  const expected = [GH, `${GH}/blob/main/docs/specs/2026-10-19-blanktab-v1-design.md`, `${GH}/tree/main/docs/decisions`,
    `${GH}/blob/main/docs/research/2026-10-16-research-synthesis-v0.1.md`, `${GH}/blob/main/docs/validate-plan-01.md`,
    `${GH}/pulls`, `${GH}/actions`, `${GH}/tree/main/data/runs`, "https://markwhiting.github.io/life/"]
  expect(await footerLinks()).toEqual(expected)

  await tab("History").click()
  await expect(tab("History")).toHaveAttribute("aria-selected", "true")
  await expect(tab("Learner")).toHaveAttribute("aria-selected", "false")
  await expect(page.getByTestId("learner-id")).toBeHidden()
  const rows = page.getByTestId("history").locator("tbody tr")
  await expect(rows).toHaveCount(2)
  await expect(rows.nth(0)).toContainText("lrn_bbbbbbbbbbbbbbbb")
  await expect(rows.nth(1)).toContainText("lrn_aaaaaaaaaaaaaaaa")
  await expect(rows.nth(0).getByRole("link")).toHaveAttribute("href", `${GH}/tree/main/data/runs/lrn_bbbbbbbbbbbbbbbb`)
  await expect(rows.nth(0)).toContainText("3000")
  await expect(rows.nth(0)).toContainText("moveProbability = 0.5")
  await expect(rows.nth(0)).toContainText("4 cores")
  await expect(rows.nth(0)).toContainText("0.912") // its loss ratio
  await expect(rows.nth(0)).toContainText("1024.5") // its CU
  await expect(page.getByTestId("history").locator("thead")).toContainText(/loss ratio/i)
  await expect(page.getByTestId("history").locator("thead")).toContainText(/CU/)
  await expect(footer).toBeVisible()

  await tab("About").click()
  await expect(tab("About")).toHaveAttribute("aria-selected", "true")
  const about = page.getByTestId("about")
  await expect(about).toBeVisible()
  await expect(about).toContainText(/only pixels/i)
  await expect(about).toContainText(/api\.github\.com/)
  await expect(about).toContainText(/learns to predict what it sees/i)
  await expect(about).toContainText(/doesn't yet move between rooms by itself/i)
  await expect(about).not.toContainText(/flails/i)
  expect(await about.getByRole("link").evaluateAll(as => as.map(a => (a as HTMLAnchorElement).href))).toEqual(expected)
  await expect(footer).toBeVisible()

  // The learner keeps living while another tab is shown.
  const before = await stepsLived(page)
  await tab("Learner").click()
  await expect(page.getByTestId("learner-id")).toBeVisible()
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(before)
})

test("HISTORY says so when there are no past sessions", async ({ page }) => {
  await setup(page, { knowledge: "unreachable" })
  await page.goto(LONG)
  await page.getByRole("tab", { name: "History" }).click()
  await expect(page.getByTestId("history")).toContainText("No past sessions yet")
  await expect(page.getByTestId("history").locator("tbody tr")).toHaveCount(0)
})

test("a malformed summary row is dropped: the learner runs and HISTORY shows only the good rows, newest first", async ({ page }) => {
  await setup(page, { knowledge: "ok" })
  await page.route("**/knowledge/summary.json", route => route.fulfill({ json: { schemaVersion: 1, sessions: [
    { ...row("lrn_aaaaaaaaaaaaaaaa"), lastWrittenAt: "2026-10-19T12:00:00.000Z" },
    { ...row("lrn_dddddddddddddddd"), hostSig: null },
    { learnerId: "garbage" },
    { ...row("lrn_bbbbbbbbbbbbbbbb"), lastWrittenAt: "2026-10-20T12:00:00.000Z" },
  ] } }))
  await page.goto(LONG)
  await expect(page.getByText("2 sessions before this one")).toBeVisible()
  await expect(page.getByTestId("knowledge")).not.toHaveText(/offline/i)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0)
  await page.getByRole("tab", { name: "History" }).click()
  const rows = page.getByTestId("history").locator("tbody tr")
  await expect(rows).toHaveCount(2)
  await expect(rows.nth(0)).toContainText("lrn_bbbbbbbbbbbbbbbb")
  await expect(rows.nth(1)).toContainText("lrn_aaaaaaaaaaaaaaaa")
})
