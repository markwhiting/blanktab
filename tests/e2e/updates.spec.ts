// Decision 0014: a learner lives as long as its tab, through new knowledge and new code (Review Focus 1 and 2).
import { test, expect, type Page, type Request } from "@playwright/test"

const ID_RE = /lrn_[0-9a-f]{16}/
const SPACE = [
  { name: "lr", min: 1e-3, max: 1e-3 }, { name: "entropyWeight", min: 0.01, max: 0.01 }, { name: "curiosityWeight", min: 1, max: 1 },
  { name: "bufferSize", min: 500, max: 500 }, { name: "replayPerSleep", min: 16, max: 16 }, { name: "headLrScale", min: 0.1, max: 0.1 },
  { name: "maxStepsPerSec", min: 30, max: 30 },
]
const row = (id: string) => ({
  learnerId: id, settings: {}, hostSig: { cores: 4, deviceMemoryGb: null, webgpu: false, tabsOnMachine: 1 },
  stepsTotal: 3000, final: true, codeSha: "a".repeat(40), knowledgeSha: "b".repeat(64), sessionsSeenAtBirth: 0,
})
/** Polls every 500 ms (no jitter); a code reload waits at most 300 ms. */
const FAST = "testPollMs=500&testReloadJitterMs=300"
const NEW_CODE = "b".repeat(40)

/** Serves knowledge/*.json from `files`, which a test changes as a deploy would; records every request. */
async function setup(page: Page) {
  const files: Record<string, unknown> = {
    "summary.json": { schemaVersion: 1, sessions: [row("lrn_aaaaaaaaaaaaaaaa"), row("lrn_bbbbbbbbbbbbbbbb")] },
    // The e2e build is code "dev" with LEARNER_STATE_VERSION 1: at first, nothing new is deployed.
    "manifest.json": { knowledgeSha: "c".repeat(64), codeSha: "dev", learnerStateVersion: 1 },
    "settings-space.json": { schemaVersion: 1, settings: SPACE },
  }
  const all: Request[] = []
  page.context().on("request", r => all.push(r))
  await page.route("**/knowledge/*.json", route => {
    const body = files[new URL(route.request().url()).pathname.split("/").pop()!]
    return body === undefined ? route.fulfill({ status: 404 }) : route.fulfill({ json: body })
  })
  // Nothing here has a token: any GitHub request at all is a failure.
  await page.route("https://api.github.com/**", route => route.abort())
  let loads = 0
  page.on("load", () => { loads++ })
  const deploy = (m: Record<string, unknown>) => { files["manifest.json"] = { ...(files["manifest.json"] as object), ...m } }
  return { files, all, deploy, loads: () => loads }
}

const stepsLived = async (page: Page) => Number((await page.getByTestId("steps").textContent())?.match(/^(\d+)/)?.[1] ?? NaN)
const cuLived = async (page: Page) => Number((await page.getByTestId("cu").textContent())?.match(/^([\d.]+) (?:of |CU)/)?.[1] ?? NaN)
const learnerId = async (page: Page) => (await page.getByTestId("learner-id").textContent())!.match(ID_RE)![0]

/** The outbox's log files for `id`, in seq order. */
const outbox = (page: Page, id: string) => page.evaluate(i => new Promise<{ seq: number; stepsTotal: number; cu: number; final: boolean; knowledgeSha: string; sessionsSeenAtBirth: number }[]>((res, rej) => {
  const req = indexedDB.open("blanktab")
  req.onerror = () => rej(req.error)
  req.onsuccess = () => {
    const get = req.result.transaction("outbox").objectStore("outbox").getAll()
    get.onsuccess = () => { req.result.close(); res((get.result as { learnerId: string; file: never }[]).filter(r => r.learnerId === i).map(r => r.file).sort((a: { seq: number }, b: { seq: number }) => a.seq - b.seq)) }
  }
}), id)

function onlyOwnOrigin(all: Request[], baseURL: string) {
  const origin = new URL(baseURL).origin
  return all.map(r => r.url()).filter(u => !u.startsWith("data:") && !u.startsWith("blob:") && new URL(u).origin !== origin)
}

test("new compatible code: the tab saves its learner, reloads once, and the same learner resumes with steps, CU and seq continuing", async ({ page, baseURL }) => {
  test.setTimeout(180_000)
  const { all, deploy, loads } = await setup(page)
  // At the CPU half widths 8 CU (the first log file) takes seconds; the life ends at 12 CU. The checkpoint throttle is
  // the default 5 min, so after the 8 CU crossing only a forced checkpoint can save the learner before the reload.
  await page.goto(`/learner/?testBackend=cpu&testSleepInterval=50&testLifetimeCU=12&testStepsPerSecond=300&${FAST}`)
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  const id = await learnerId(page)
  await expect.poll(() => cuLived(page), { timeout: 90_000 }).toBeGreaterThan(9) // the 8 CU log (seq 0) and its checkpoint are written
  const [first] = await outbox(page, id)
  expect(first).toMatchObject({ seq: 0, final: false })

  let navigated = false
  page.on("framenavigated", f => { if (f === page.mainFrame()) navigated = true })
  let lastSteps = 0, lastCU = 0
  deploy({ codeSha: NEW_CODE })
  const t0 = Date.now()
  while (!navigated) {
    try {
      const [s, c] = [await stepsLived(page), await cuLived(page)]
      if (!navigated && Number.isFinite(s) && Number.isFinite(c)) { lastSteps = s; lastCU = c }
    } catch { /* the page is going away */ }
    expect(Date.now() - t0).toBeLessThan(20_000)
    await page.waitForTimeout(25)
  }
  await page.waitForLoadState("load")

  // The same learner, resumed from the forced checkpoint (not the 8 CU one): steps and CU go on from where they were.
  await expect(page.getByTestId("learner-id")).toHaveText(id)
  await expect(page.getByTestId("status")).toContainText("resumed from its checkpoint at step")
  const resumedAt = Number((await page.getByTestId("status").textContent())!.match(/at step (\d+)/)![1])
  expect(resumedAt).toBeGreaterThan(first!.stepsTotal)
  expect(resumedAt).toBeGreaterThanOrEqual(lastSteps - 30) // at most the few steps lived between the save and the reload
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(resumedAt)
  expect(await cuLived(page)).toBeGreaterThanOrEqual(lastCU - 0.5)

  // The life goes on to its end on the new page: seq 1 follows seq 0, nothing reused, counters never go back.
  await expect(page.getByText("This learner's life is complete")).toBeVisible({ timeout: 90_000 })
  const files = await outbox(page, id)
  expect(files.map(f => [f.seq, f.final])).toEqual([[0, false], [1, true]])
  expect(files[1]!.stepsTotal).toBeGreaterThan(resumedAt)
  expect(files[1]!.cu).toBeGreaterThanOrEqual(12)
  // One reload only: the new page (still code "dev" here) does not reload again for the same codeSha.
  expect(loads()).toBe(2)
  await page.waitForTimeout(2_000)
  expect(loads()).toBe(2)
  expect(onlyOwnOrigin(all, baseURL!)).toEqual([])
})

test("new incompatible code: the learner keeps running here and the page says so", async ({ page, baseURL }) => {
  const { all, deploy, loads } = await setup(page)
  await page.goto(`/learner/?testBackend=cpu&testWidths=tiny&testSleepInterval=5&testLifetimeCU=50&testStepsPerSecond=20&${FAST}`)
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  const id = await learnerId(page)
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0)
  deploy({ codeSha: NEW_CODE, learnerStateVersion: 2 })
  await expect(page.getByTestId("update")).toHaveText("A newer version is available; this learner continues on the version it was born with.")
  await expect(page.getByTestId("update")).toBeVisible()
  const before = await stepsLived(page)
  await page.waitForTimeout(3_000) // six poll periods
  expect(loads()).toBe(1)
  await expect(page.getByTestId("learner-id")).toHaveText(id)
  expect(await stepsLived(page)).toBeGreaterThan(before)
  await expect(page.getByTestId("life")).not.toContainText("stopped")
  await page.screenshot({ path: test.info().outputPath("page-incompatible.png"), fullPage: true })
  expect(onlyOwnOrigin(all, baseURL!)).toEqual([])
})

test("refreshed knowledge: the running learner's next log file records the new knowledgeSha; the page shows it", async ({ page }) => {
  const { deploy } = await setup(page)
  await page.goto(`/learner/?testBackend=cpu&testWidths=tiny&testSleepInterval=5&testLifetimeCU=0.3&testStepsPerSecond=20&${FAST}`)
  await expect(page.getByTestId("learner-id")).toHaveText(ID_RE)
  const id = await learnerId(page)
  await expect(page.getByTestId("knowledge")).toContainText("cccccccccccc")
  await expect.poll(() => stepsLived(page)).toBeGreaterThan(0)
  deploy({ knowledgeSha: "d".repeat(64) }) // new priors; the code is the same
  await expect(page.getByTestId("knowledge")).toContainText("dddddddddddd")
  await expect(page.getByText("This learner's life is complete")).toBeVisible({ timeout: 60_000 })
  const files = await outbox(page, id)
  expect(files).toHaveLength(1)
  expect(files[0]).toMatchObject({ final: true, knowledgeSha: "d".repeat(64), sessionsSeenAtBirth: 2 })
  expect(await page.getByTestId("life").textContent()).not.toContain("stopped")
})
