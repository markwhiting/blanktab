// The page's main thread: knowledge, identity, the GitHub contributor and the token live here.
// The learner itself runs in ./worker.ts; this thread only starts it and shows what it reports.
import {
  DB_BLOCKED_MESSAGE, PERMISSION_MESSAGE, REJECTED_MESSAGE, adoptLearnerId, checkToken, chooseSettings, clearToken,
  createContributor, getToken, hasCheckpoint, loadKnowledge, loadManifest, newLearnerId, openQueue, setToken, storedLearnerId, sweepCheckpoints, waitLock, webLock, MAX_UNSENT,
  type Knowledge, type LogFile, type Queue, type SweepResult,
} from "@blanktab/harness"
import { LEARNER_STATE_VERSION } from "@blanktab/learner/version"
import { seedsOf, type LifeConfig } from "./life"
import type { FromWorker, ToWorker } from "./protocol"
import { readTestParams, type TestParams } from "./params"
import { flushPolicy } from "./schedule"
import { birthSettings } from "./settings"
import { buffersOf } from "./transfer"
import { createUi, type Ui } from "./ui"
import { startUpdates } from "./updates"

const NEWBORN_DELAY_MS = 2_000
const PRESENCE_CHANNEL = "blanktab:presence"
const PRESENCE_WAIT_MS = 250
/** A forced checkpoint not answered in this long counts as not saved (the reload is retried later). */
const CHECKPOINT_WAIT_MS = 60_000

// Test-only overrides. In the production build MODE is the literal "production" and this is dropped.
const test: TestParams = import.meta.env.MODE !== "production" ? readTestParams(location.search) : {}
const lifetimeCU = test.lifetimeCU // undefined: the learner lives as long as its tab

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
 * Birth facts are fixed for a learner's whole life. A resumed learner's seed, widths, learner settings, knowledgeSha
 * and sessionsSeenAtBirth come from its checkpoint (the session prefers them to these); its speed cap from its latest log.
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
 * learnerId; the Web Lock (blanktab:learner:<id>, held until the learner's life ends) tells them apart. Resolves to a
 * release function, or null if another tab holds it. Without the Locks API (very old browsers) every tab simply proceeds.
 */
function lockLearner(id: string): Promise<(() => void) | null> {
  if (!("locks" in navigator)) return Promise.resolve(() => {})
  return webLock(navigator.locks)(id).then(h => h && h.release)
}

/**
 * This tab's own learner's lock (a reload, or a tab the browser restored): waited for, up to OWN_LOCK_WAIT_MS, since
 * another tab's sweep may be looking at it for a moment (and, seeing this request waiting, lets it go). Null if
 * another tab lives it (a duplicated tab).
 */
function lockOwn(id: string): Promise<(() => void) | null> {
  if (!("locks" in navigator)) return Promise.resolve(() => {})
  return waitLock(navigator.locks, id).then(h => h && h.release)
}

/** The life running in this tab, as the updates see it. */
interface Life {
  /** Hands refreshed knowledge to the learner. */
  knowledge(k: Knowledge): void
  /** A forced checkpoint: true once saved; false if not (skips outstanding, no answer in CHECKPOINT_WAIT_MS, the life ended). */
  checkpoint(): Promise<boolean>
  /** Stops the learner where it is (before a reload onto the checkpoint just saved). */
  halt(): void
}

/** `knowledge`: the latest the page has (refreshed while it lives); newborns are born from it. `life`: the running one, if any. */
interface Ctx { ui: Ui; queue: Queue; knowledge: Knowledge; queued: (final: boolean) => void; life: Life | null }

const newborn = (ctx: Ctx) => live(newLearnerId(sessionStorage), ctx)

/**
 * `held`: this learner's lock, already taken (by the startup sweep). `record`: its checkpoint record, already read by
 * the sweep that adopted it (handed to the worker, so it is not read twice).
 */
async function live(id: string, ctx: Ctx, held?: () => void, record?: unknown): Promise<void> {
  const { ui, queue, queued } = ctx
  const knowledge = ctx.knowledge
  const release = held ?? (await lockLearner(id))
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
    learnerId: id, codeSha: __CODE_SHA__, ...b, hostSig: await hostSig(), seed: seedsOf(id).learner,
    ...(lifetimeCU !== undefined ? { lifetimeCU } : {}),
    ...(test.stepsPerSecond !== undefined ? { maxStepsPerSec: test.stepsPerSecond } : {}),
    ...(test.sleepInterval !== undefined ? { sleepInterval: test.sleepInterval } : {}),
  }
  ui.learner({ id, sessionsBefore: cfg.sessionsSeenAtBirth, ...(lifetimeCU !== undefined ? { lifetimeCU } : {}) })

  const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" })
  // A hidden page draws nothing: the worker stops posting frames until it is shown again.
  const visibility = () => worker.postMessage({ type: "visibility", hidden: document.hidden } satisfies ToWorker)
  document.addEventListener("visibilitychange", visibility)
  let over = false
  let nextCheckpoint = 0
  const waiting = new Map<number, (saved: boolean) => void>()
  const life: Life = {
    knowledge: k => worker.postMessage({ type: "knowledge", knowledge: k } satisfies ToWorker),
    checkpoint: () => new Promise<boolean>(resolve => {
      if (over) return resolve(false)
      const id = nextCheckpoint++
      const timer = setTimeout(() => { waiting.delete(id); resolve(false) }, CHECKPOINT_WAIT_MS)
      waiting.set(id, saved => { clearTimeout(timer); waiting.delete(id); resolve(saved) })
      worker.postMessage({ type: "checkpointNow", id } satisfies ToWorker)
    }),
    halt: () => worker.terminate(),
  }
  ctx.life = life
  /** The learner always runs: whether its life completed or failed, a newborn takes its place. */
  const end = (show: () => void) => {
    if (over) return
    over = true
    if (ctx.life === life) ctx.life = null
    for (const answer of [...waiting.values()]) answer(false)
    document.removeEventListener("visibilitychange", visibility)
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
    else if (m.type === "queued") {
      ui.progress(m.steps, m.cu)
      // Without a token nothing is ever sent: keep the outbox bounded (never this learner's files).
      void queue.capUnsent(MAX_UNSENT, [id]).catch(() => {}).then(() => queued(m.final))
    }
    else if (m.type === "checkpointed") waiting.get(m.id)?.(m.saved)
    else if (m.type === "error") end(() => ui.error(`${m.message}. A newborn takes its place.`))
    else if (m.type === "final") end(() => ui.lifeComplete(id))
  }
  worker.onerror = e => end(() => ui.error(`${e.message || "worker failed"}. A newborn takes its place.`))
  const start: ToWorker = {
    type: "start", cfg, hidden: document.hidden, ...(record !== undefined ? { checkpointRecord: record } : {}),
    ...(test.backend ? { backend: test.backend } : {}), ...(test.widths ? { widths: test.widths } : {}),
  }
  worker.postMessage(start, buffersOf(record)) // moved, not copied
  // The knowledge in force from now on, for a resumed learner too (it was saved with what it had then).
  if (!knowledge.offline) life.knowledge(knowledge)
}

async function main(ui: Ui): Promise<void> {
  // Before anything else (anything awaited): this tab's own learner, if it has one, is asked for at once, so no other
  // tab's startup sweep takes its checkpoint for an orphan while this page loads.
  const own = storedLearnerId(sessionStorage)
  const ownLock = own === null ? Promise.resolve(null) : lockOwn(own)
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

  const ctx: Ctx = { ui, queue, knowledge, queued: policy.queued, life: null }
  watchForUpdates(ctx)
  const release = await ownLock
  if (own !== null && release !== null) {
    await sweep(false, own)
    return live(own, ctx, release)
  }
  // A new tab (or a duplicated one, whose learner lives in the original): adopt a closed tab's learner, if any.
  // The sweep never touches the learner in this tab's sessionStorage.
  const adopted = own === null ? await sweep(true) : (await sweep(false, own), null)
  if (adopted) {
    adoptLearnerId(sessionStorage, adopted.learnerId)
    ui.notice(`Adopted learner ${adopted.learnerId}: its tab was closed, so this tab lives the rest of its life.`)
    return live(adopted.learnerId, ctx, adopted.release, adopted.record)
  }
  return newborn(ctx)
}

/**
 * Decision 0014: refreshed knowledge goes to the running learner (and to newborns); new code is reloaded onto if it can
 * resume this learner's checkpoint (the learner is saved first), and otherwise the learner lives on here and the page
 * says so. Nothing polls while the page is hidden.
 */
function watchForUpdates(ctx: Ctx): void {
  const visibility = {
    hidden: () => document.hidden,
    onChange: (fn: () => void) => {
      document.addEventListener("visibilitychange", fn)
      return () => document.removeEventListener("visibilitychange", fn)
    },
  }
  const poll = test.pollMs !== undefined ? { everyMs: test.pollMs, jitterMs: 0 } : undefined
  startUpdates({
    env: { timers: { setTimeout: (fn, ms) => setTimeout(fn, ms), clearTimeout: h => clearTimeout(h as ReturnType<typeof setTimeout>) }, visibility, random: Math.random },
    initial: ctx.knowledge, codeSha: __CODE_SHA__, learnerStateVersion: LEARNER_STATE_VERSION, storage: sessionStorage,
    loadKnowledge: () => loadKnowledge(import.meta.env.BASE_URL, fetchFn),
    loadManifest: () => loadManifest(import.meta.env.BASE_URL, fetchFn),
    onKnowledge: k => {
      ctx.knowledge = k
      ctx.ui.knowledge(k)
      ctx.life?.knowledge(k)
    },
    checkpoint: () => ctx.life?.checkpoint() ?? Promise.resolve(true), // no learner running: nothing to save
    reload: () => {
      ctx.life?.halt()
      location.reload()
    },
    notice: m => ctx.ui.update(m),
    ...(poll ? { knowledgePoll: poll, manifestPoll: poll, retryMs: poll.everyMs } : {}),
    ...(test.reloadJitterMs !== undefined ? { reloadJitterMs: test.reloadJitterMs } : {}),
  })
}

/**
 * Orphaned checkpoints (spec §10): adopts the newest one this tab may if `adopt`, and deletes those nobody will
 * resume, so they do not fill the disk (sweepCheckpoints). Needs Web Locks to tell orphans from live learners; without
 * them, and on any error, it does nothing.
 */
async function sweep(adopt: boolean, exclude?: string): Promise<SweepResult["adopted"]> {
  if (!("locks" in navigator)) return null
  try {
    const r = await sweepCheckpoints({ idb: indexedDB, tryLock: webLock(navigator.locks), now: () => new Date(), adopt, ...(exclude ? { exclude: [exclude] } : {}) })
    return r.adopted
  } catch (e) {
    console.warn("blanktab: the orphaned-checkpoint sweep failed", e)
    return null
  }
}

const ui = createUi()
void main(ui).catch((e: unknown) => {
  const message = e instanceof Error ? e.message : String(e)
  // An older tab without a versionchange handler holds the database open, so this page cannot upgrade it.
  if (message === DB_BLOCKED_MESSAGE) ui.notice(`${DB_BLOCKED_MESSAGE}: an older learner tab is holding this browser's saved logs open. Close it, then reload this page.`)
  else ui.error(message)
})
