// The page's main thread: knowledge, identity, the GitHub contributor and the token live here.
// The learner itself runs in ./worker.ts; this thread only starts it and shows what it reports.
import {
  DB_BLOCKED_MESSAGE, DEFAULT_LIFETIME_CU, PERMISSION_MESSAGE, REJECTED_MESSAGE, checkToken, chooseSettings, clearToken, createContributor,
  getOrCreateLearnerId, getToken, hasCheckpoint, loadKnowledge, newLearnerId, openQueue, setToken,
  type Knowledge, type LogFile, type Queue,
} from "@blanktab/harness"
import { seedsOf, type LifeConfig } from "./life"
import type { FromWorker, ToWorker } from "./protocol"
import { readTestParams, type TestParams } from "./params"
import { flushPolicy } from "./schedule"
import { birthSettings } from "./settings"
import { createUi, type Ui } from "./ui"

const NEWBORN_DELAY_MS = 2_000
const PRESENCE_CHANNEL = "blanktab:presence"
const PRESENCE_WAIT_MS = 250

// Test-only overrides. In the production build MODE is the literal "production" and this is dropped.
const test: TestParams = import.meta.env.MODE !== "production" ? readTestParams(location.search) : {}
const lifetimeCU = test.lifetimeCU ?? DEFAULT_LIFETIME_CU

const fetchFn: typeof fetch = (input, init) => fetch(input, init)

/** Every blanktab tab answers pings, including this one, so the count includes self. */
function answerPresence(): void {
  const ch = new BroadcastChannel(PRESENCE_CHANNEL)
  ch.onmessage = (e: MessageEvent<{ type?: string; from?: string }>) => {
    if (e.data?.type === "ping") ch.postMessage({ type: "pong", to: e.data.from })
  }
}
async function countTabs(): Promise<number> {
  const me = crypto.randomUUID()
  const ch = new BroadcastChannel(PRESENCE_CHANNEL)
  let n = 0
  ch.onmessage = (e: MessageEvent<{ type?: string; to?: string }>) => { if (e.data?.type === "pong" && e.data.to === me) n++ }
  ch.postMessage({ type: "ping", from: me })
  await new Promise(r => setTimeout(r, PRESENCE_WAIT_MS))
  ch.close()
  return Math.max(1, n)
}
/** webgpu is filled in by the worker: whether the learner actually runs on WebGPU. */
async function hostSig(): Promise<LifeConfig["hostSig"]> {
  const nav = navigator as Navigator & { deviceMemory?: number }
  return {
    cores: Number.isFinite(nav.hardwareConcurrency) ? nav.hardwareConcurrency : 1,
    deviceMemoryGb: typeof nav.deviceMemory === "number" && Number.isFinite(nav.deviceMemory) ? nav.deviceMemory : null,
    tabsOnMachine: await countTabs(),
  }
}

/**
 * Birth facts are fixed for a learner's whole life. A resumed learner's seed, widths and learner settings come from
 * its checkpoint (the session ignores these); its knowledgeSha, sessionsSeenAtBirth and speed cap from its latest log.
 */
function birth(latest: LogFile | undefined, knowledge: Knowledge): Pick<LifeConfig, "knowledgeSha" | "sessionsSeenAtBirth" | "learnerSettings" | "maxStepsPerSec"> {
  const fresh = birthSettings(chooseSettings(knowledge, Math.random))
  if (!latest) return { ...fresh, knowledgeSha: knowledge.manifest.knowledgeSha, sessionsSeenAtBirth: knowledge.summary.sessions.length }
  const cap = latest.settings.maxStepsPerSec
  return {
    learnerSettings: fresh.learnerSettings,
    maxStepsPerSec: typeof cap === "number" && cap > 0 ? cap : fresh.maxStepsPerSec,
    knowledgeSha: latest.knowledgeSha, sessionsSeenAtBirth: latest.sessionsSeenAtBirth,
  }
}

/**
 * One tab per learner. Chrome's "Duplicate tab" copies sessionStorage, so two tabs can start with the same
 * learnerId; the Web Lock tells them apart. Resolves to a release function, or null if another tab holds it.
 * Without the Locks API (very old browsers) every tab simply proceeds.
 */
function lockLearner(id: string): Promise<(() => void) | null> {
  if (!("locks" in navigator)) return Promise.resolve(() => {})
  return new Promise(resolve => {
    void navigator.locks.request(`blanktab:learner:${id}`, { ifAvailable: true }, lock => {
      if (lock === null) { resolve(null); return }
      return new Promise<void>(release => resolve(release)) // held until the learner's life ends
    })
  })
}

interface Ctx { ui: Ui; queue: Queue; knowledge: Knowledge; queued: (final: boolean) => void }

const newborn = (ctx: Ctx) => live(newLearnerId(sessionStorage), ctx)

async function live(id: string, ctx: Ctx): Promise<void> {
  const { ui, queue, knowledge, queued } = ctx
  const release = await lockLearner(id)
  if (release === null) return newborn(ctx) // another tab is living this learner's life
  const latest = await queue.latest(id)
  if (latest?.final) { release(); return newborn(ctx) } // this tab's learner already finished its life
  // Every log file is written with its checkpoint, so logs without one (e.g. the old flailer's) cannot be resumed.
  if (latest && !(await hasCheckpoint(indexedDB, id))) {
    release()
    ui.notice(`Learner ${id} has logs but no checkpoint, so its life cannot be resumed. A newborn takes its place.`)
    return newborn(ctx)
  }
  const b = birth(latest, knowledge)
  const cfg: LifeConfig = {
    learnerId: id, codeSha: __CODE_SHA__, ...b, hostSig: await hostSig(), seed: seedsOf(id).learner, lifetimeCU,
    ...(test.stepsPerSecond !== undefined ? { maxStepsPerSec: test.stepsPerSecond } : {}),
    ...(test.sleepInterval !== undefined ? { sleepInterval: test.sleepInterval } : {}),
  }
  ui.learner({ id, sessionsBefore: cfg.sessionsSeenAtBirth, lifetimeCU })

  const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" })
  let over = false
  /** The learner always runs: whether its life completed or failed, a newborn takes its place. */
  const end = (show: () => void) => {
    if (over) return
    over = true
    worker.terminate()
    release()
    show()
    setTimeout(() => void newborn(ctx).catch((e: unknown) => ui.error(e instanceof Error ? e.message : String(e))), NEWBORN_DELAY_MS)
  }
  worker.onmessage = (e: MessageEvent<FromWorker>) => {
    const m = e.data
    if (m.type === "frame") ui.frame(m)
    else if (m.type === "backend") ui.backend(m.label)
    else if (m.type === "learner") ui.settings(m.settings)
    else if (m.type === "curve") ui.curve(m.points, m.reset)
    else if (m.type === "status") ui.status(m.message)
    else if (m.type === "queued") { ui.progress(m.steps, m.cu); queued(m.final) }
    else if (m.type === "error") end(() => ui.error(`${m.message}. A newborn takes its place.`))
    else if (m.type === "final") end(() => ui.lifeComplete(id))
  }
  worker.onerror = e => end(() => ui.error(`${e.message || "worker failed"}. A newborn takes its place.`))
  worker.postMessage({ type: "start", cfg, ...(test.backend ? { backend: test.backend } : {}), ...(test.widths ? { widths: test.widths } : {}) } satisfies ToWorker)
}

async function main(ui: Ui): Promise<void> {
  answerPresence()
  const showTokenState = () => {
    const stored = getToken(localStorage) !== null
    ui.tokenPanel(!stored)
    ui.changeToken(stored)
  }
  showTokenState()
  const [knowledge, queue] = await Promise.all([loadKnowledge(import.meta.env.BASE_URL, fetchFn), openQueue(indexedDB)])
  ui.knowledge(knowledge)

  const contributor = createContributor({
    queue, ls: localStorage, fetchFn, now: () => new Date(),
    onStatus: s => {
      ui.contribution(s)
      // Anything the token itself is the cause of: offer to paste another one.
      if (s.state === "no-token" || s.message === REJECTED_MESSAGE || s.message === PERMISSION_MESSAGE) ui.tokenPanel(true)
    },
  })
  const flush = () => void contributor.flush().catch((e: unknown) => {
    ui.contribution({ state: "error", message: `Couldn't read the saved logs (${e instanceof Error ? e.message : String(e)}); will try again.`, pending: 0 })
  })
  const policy = flushPolicy(flush, undefined, () => void contributor.report())
  ui.onTokenSubmit(async token => {
    const r = await checkToken(token, fetchFn) // check first: a token is stored only once GitHub accepts it
    if (!r.ok) return r.reason
    setToken(localStorage, token)
    showTokenState()
    policy.tokenSaved()
    return null
  })
  ui.onChangeToken(() => {
    clearToken(localStorage)
    showTokenState()
    flush() // reports "add a token" with the number of logs waiting
  })
  flush() // logs left by earlier tabs or lives

  await live(getOrCreateLearnerId(sessionStorage), { ui, queue, knowledge, queued: policy.queued })
}

const ui = createUi()
void main(ui).catch((e: unknown) => {
  const message = e instanceof Error ? e.message : String(e)
  // An older tab without a versionchange handler holds the database open, so this page cannot upgrade it.
  if (message === DB_BLOCKED_MESSAGE) ui.notice(`${DB_BLOCKED_MESSAGE}: an older learner tab is holding this browser's saved logs open. Close it, then reload this page.`)
  else ui.error(message)
})
