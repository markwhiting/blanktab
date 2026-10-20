import { test, expect, type Page, type Request } from "@playwright/test"
import { fakeGithub } from "../../packages/harness/test/fakeGithub"

const TOKEN = "test-token"
const UP = "markwhiting/learner"
const LIFETIME = 600
const PAGE = `/learner/?testLifetime=${LIFETIME}&testStepsPerSecond=100`
const ID_RE = /lrn_[0-9a-f]{16}/

const row = (id: string) => ({
  learnerId: id, settings: { moveProbability: 0.5 }, hostSig: { cores: 4, deviceMemoryGb: null, webgpu: false, tabsOnMachine: 1 },
  stepsTotal: 3000, final: true, codeSha: "a".repeat(40), knowledgeSha: "b".repeat(64), sessionsSeenAtBirth: 0,
})
const KNOWLEDGE: Record<string, unknown> = {
  "summary.json": { schemaVersion: 1, sessions: [row("lrn_aaaaaaaaaaaaaaaa"), row("lrn_bbbbbbbbbbbbbbbb")] },
  "manifest.json": { knowledgeSha: "c".repeat(64), codeSha: "a".repeat(40) },
  "settings-space.json": { schemaVersion: 1, settings: [{ name: "moveProbability", min: 0.1, max: 1 }] },
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

const stepsLived = async (page: Page) => Number((await page.getByTestId("steps").textContent())?.match(/(\d+) of/)?.[1] ?? NaN)

test("the loop: knowledge in, a session runs, logs go to GitHub as one PR, a newborn follows", async ({ page, baseURL }) => {
  const { gh, all, githubCalls } = await setup(page, { knowledge: "ok" })
  await page.goto(PAGE)

  // Sessions before this one: the visible proof that the loop is closed.
  await expect(page.getByText("2 sessions before this one")).toBeVisible()
  await expect(page.getByText(/offline/i)).toHaveCount(0)

  const learner = page.getByTestId("learner-id")
  await expect(learner).toHaveText(/lrn_[0-9a-f]{16}/)
  const id = (await learner.textContent())!.match(/lrn_[0-9a-f]{16}/)![0]
  expect(await page.evaluate(() => sessionStorage.getItem("blanktab:learnerId"))).toBe(id)
  await expect(page.getByTestId("setting")).toHaveText(/moveProbability = 0\.\d+|moveProbability = 1/)

  // The steps counter increases.
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0)
  const first = await stepsLived(page)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(first)
  await expect(page.getByTestId("steps")).toContainText(`of ${LIFETIME}`)

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

  // The session goes final; its logs are submitted as data/runs/<id>/0.json and 1.json with one PR.
  await expect(page.getByText("This learner's life is complete")).toBeVisible({ timeout: 30_000 })
  const pulls = () => githubCalls().filter(r => r.method === "POST" && r.url === `https://api.github.com/repos/${UP}/pulls` && JSON.parse(r.body ?? "{}").head === `markwhiting:run/${id}`)
  await expect.poll(() => pulls().length, { timeout: 30_000 }).toBe(1)
  expect(gh.prFiles(gh.pulls[0]!)).toEqual([`data/runs/${id}/0.json`, `data/runs/${id}/1.json`])
  expect(githubCalls().some(r => r.method === "PUT" || r.url.includes("/contents/") || (r.body ?? "").includes("heartbeat"))).toBe(false)
  await expect(page.getByTestId("change-token")).toBeVisible()
  const final = gh.file(UP, `run/${id}`, `data/runs/${id}/1.json`) as { final: boolean; stepsTotal: number; sessionsSeenAtBirth: number; knowledgeSha: string; codeSha: string }
  expect(final).toMatchObject({ final: true, stepsTotal: LIFETIME, sessionsSeenAtBirth: 2, knowledgeSha: "c".repeat(64), codeSha: "dev" })
  expect(gh.file(UP, `run/${id}`, `data/runs/${id}/0.json`)).toMatchObject({ seq: 0, final: false })

  // A newborn with a new learnerId takes its place.
  await expect(learner).not.toHaveText(id)
  await expect(learner).toHaveText(/lrn_[0-9a-f]{16}/)
  expect(await page.evaluate(() => sessionStorage.getItem("blanktab:learnerId"))).not.toBe(id)

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
  await page.goto(PAGE)
  await expect(page.getByTestId("knowledge")).toHaveText(/offline/i)
  await expect(page.getByText("0 sessions before this one")).toBeVisible()
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0)
  const first = await stepsLived(page)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(first)
  expect(githubCalls()).toHaveLength(0)
})

test("cadence: with a token, a whole life goes to GitHub as one commit, sent when the final log is queued", async ({ page }) => {
  const { gh, githubCalls, commits } = await setup(page, { knowledge: "ok", token: TOKEN })
  await page.goto(`/learner/?testLifetime=1500&testStepsPerSecond=150`) // logs at 500, 1000 (not final) and 1500 (final)
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  const id = (await page.getByTestId("learner-id").textContent())!.match(ID_RE)![0]
  await expect.poll(() => stepsLived(page), { timeout: 20_000 }).toBeGreaterThan(1100) // two non-final logs are queued by now
  expect(githubCalls().filter(r => r.method !== "GET")).toEqual([]) // ...and nothing was written for them
  await expect(page.getByText("This learner's life is complete")).toBeVisible({ timeout: 30_000 })
  await expect.poll(() => gh.pulls.length, { timeout: 30_000 }).toBe(1)
  expect(commits()).toHaveLength(1)
  expect(gh.prFiles(gh.pulls[0]!)).toEqual([0, 1, 2].map(s => `data/runs/${id}/${s}.json`))
})

test("token stored: a small 'change token' control clears it and brings the panel back", async ({ page }) => {
  await setup(page, { knowledge: "ok", token: TOKEN })
  await page.goto(PAGE)
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
  await page.goto(PAGE)
  const panel = page.getByTestId("token-panel")
  await panel.getByLabel(/GitHub token/i).fill(TOKEN)
  await panel.getByRole("button", { name: "Check & save" }).click()
  await expect(page.locator("#token-error")).toHaveText(/went wrong/i)
  await expect(panel.getByRole("button", { name: "Check & save" })).toBeEnabled()
})

test("the learner always runs: a worker error starts a newborn", async ({ page }) => {
  await setup(page, { knowledge: "ok" })
  await page.goto(PAGE)
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  const broken = "lrn_badbadbadbadbad0"
  // A resumed learner whose recorded settings the worker refuses: its session throws at start.
  await page.evaluate(id => new Promise<void>((res, rej) => {
    sessionStorage.setItem("blanktab:learnerId", id)
    const req = indexedDB.open("blanktab")
    req.onerror = () => rej(req.error)
    req.onsuccess = () => {
      const file = { schemaVersion: 1, learnerId: id, seq: 0, codeSha: "dev", knowledgeSha: "0".repeat(64), writtenAt: new Date().toISOString(), stepsTotal: 500,
        settings: { moveProbability: "not a number" }, hostSig: { cores: 1, deviceMemoryGb: null, webgpu: false, tabsOnMachine: 1 }, sessionsSeenAtBirth: 0, final: false }
      const tx = req.result.transaction("outbox", "readwrite")
      tx.objectStore("outbox").put({ learnerId: id, seq: 0, sent: false, file })
      tx.oncomplete = () => res()
      tx.onerror = () => rej(tx.error)
    }
  }), broken)
  await page.reload()
  await expect(page.getByTestId("life")).toContainText("The learner stopped")
  await expect(page.getByTestId("learner-id")).not.toHaveText(broken, { timeout: 10_000 })
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  const first = await stepsLived(page)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(first)
})

test("a reload keeps the learner; a duplicated tab (same sessionStorage) gets a newborn", async ({ page, context }) => {
  await setup(page, { knowledge: "ok" })
  await page.goto(PAGE)
  const learner = page.getByTestId("learner-id")
  await expect(learner).toHaveText(ID_RE)
  const id = (await learner.textContent())!.match(ID_RE)![0]
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0)

  await page.reload()
  await expect(learner).toHaveText(id)

  // Chrome's "Duplicate tab" copies sessionStorage, so the copy starts with the same learnerId.
  const dup = await context.newPage()
  await dup.addInitScript(i => { if (sessionStorage.getItem("blanktab:dup") === null) { sessionStorage.setItem("blanktab:learnerId", i); sessionStorage.setItem("blanktab:dup", "1") } }, id)
  await dup.goto(PAGE)
  await expect(dup.getByTestId("learner-id")).toHaveText(ID_RE)
  const dupId = (await dup.getByTestId("learner-id").textContent())!.match(ID_RE)![0]
  expect(dupId).not.toBe(id)
  expect(await dup.evaluate(() => sessionStorage.getItem("blanktab:learnerId"))).toBe(dupId)
  await expect(learner).toHaveText(id) // the original keeps its learner
  await expect.poll(() => stepsLived(dup)).toBeGreaterThan(0)
})

test("without the test knob the learner lives at 2 steps a second", async ({ page }) => {
  await setup(page, { knowledge: "ok" })
  await page.goto(`/learner/?testLifetime=${LIFETIME}`)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0)
  const t0 = Date.now(), s0 = await stepsLived(page)
  await page.waitForTimeout(5_000)
  const rate = (await stepsLived(page) - s0) / ((Date.now() - t0) / 1000)
  expect(rate).toBeGreaterThan(1.4)
  expect(rate).toBeLessThan(2.6)
})

const GH = `https://github.com/${UP}`

test("tabs: LEARNER by default; HISTORY lists past sessions newest first; ABOUT explains and links; links on every tab", async ({ page }) => {
  await setup(page, { knowledge: "ok" })
  await page.goto(PAGE)
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
  await expect(footer).toBeVisible()

  await tab("About").click()
  await expect(tab("About")).toHaveAttribute("aria-selected", "true")
  const about = page.getByTestId("about")
  await expect(about).toBeVisible()
  await expect(about).toContainText(/only pixels/i)
  await expect(about).toContainText(/api\.github\.com/)
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
  await page.goto(PAGE)
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
  await page.goto(PAGE)
  await expect(page.getByText("2 sessions before this one")).toBeVisible()
  await expect(page.getByTestId("knowledge")).not.toHaveText(/offline/i)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0)
  await page.getByRole("tab", { name: "History" }).click()
  const rows = page.getByTestId("history").locator("tbody tr")
  await expect(rows).toHaveCount(2)
  await expect(rows.nth(0)).toContainText("lrn_bbbbbbbbbbbbbbbb")
  await expect(rows.nth(1)).toContainText("lrn_aaaaaaaaaaaaaaaa")
})
