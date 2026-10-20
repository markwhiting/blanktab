import { test, expect, type Page, type Request } from "@playwright/test"
import { fakeGithub } from "../../packages/harness/test/fakeGithub"

const TOKEN = "test-token"
const UP = "markwhiting/blanktab"
const LIFETIME = 600
const PAGE = `/blanktab/?testLifetime=${LIFETIME}&testStepsPerSecond=100`

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
async function setup(page: Page, opts: { knowledge: "ok" | "unreachable" }) {
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
  const githubCalls = () => gh.requests.filter(r => r.method !== "OPTIONS")
  return { gh, all, githubCalls }
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
  await expect(panel.getByRole("link")).toHaveAttribute("href", "https://github.com/settings/tokens/new?scopes=public_repo&description=blanktab")
  await expect(page.getByTestId("contribution")).toContainText(/token/i)
  expect(githubCalls()).toHaveLength(0)

  await panel.getByLabel(/GitHub token/i).fill(TOKEN)
  await panel.getByRole("button", { name: "Check & save" }).click()
  await expect(panel).toBeHidden()
  expect(githubCalls()[0]?.url).toBe("https://api.github.com/user") // checkToken ran before the token was used

  // The session goes final; its logs are submitted as data/runs/<id>/0.json and 1.json with one PR.
  await expect(page.getByText("This learner's life is complete")).toBeVisible({ timeout: 30_000 })
  const put = (seq: number) => githubCalls().filter(r => r.method === "PUT" && r.url === `https://api.github.com/repos/${UP}/contents/data/runs/${id}/${seq}.json`)
  const pulls = () => githubCalls().filter(r => r.method === "POST" && r.url === `https://api.github.com/repos/${UP}/pulls` && JSON.parse(r.body ?? "{}").head === `markwhiting:run/${id}`)
  await expect.poll(() => put(1).length, { timeout: 30_000 }).toBe(1)
  await expect.poll(() => pulls().length, { timeout: 30_000 }).toBe(1)
  expect(put(0)).toHaveLength(1)
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
