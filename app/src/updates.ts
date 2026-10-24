// Live updates for a learner that lives as long as its tab (decision 0014). The page re-reads the shared knowledge
// and hands it to the running learner, and watches the deployed manifest for new code: compatible new code is
// reloaded onto after a forced checkpoint (the learner resumes on it); incompatible code never ends a life.
// Kept apart from main.ts (timers, visibility and storage injected) so it is testable.
import type { Knowledge, Manifest } from "@blanktab/harness"

/** A polling period: each delay is drawn uniformly from [everyMs - jitterMs, everyMs + jitterMs]. */
export interface Period { everyMs: number; jitterMs: number }

export const KNOWLEDGE_POLL: Period = Object.freeze({ everyMs: 15 * 60_000, jitterMs: 3 * 60_000 })
export const MANIFEST_POLL: Period = Object.freeze({ everyMs: 10 * 60_000, jitterMs: 2 * 60_000 })
/** Before a code reload, a wait drawn uniformly from [0, RELOAD_JITTER_MS): tabs do not all reload at once. */
export const RELOAD_JITTER_MS = 5 * 60_000
/** A forced checkpoint that was not saved (skipped updates outstanding, or no answer) is asked for again this much later. */
export const CHECKPOINT_RETRY_MS = 30_000
/** Shown when new code cannot resume this learner's checkpoint (another learnerStateVersion): the life goes on here. */
export const INCOMPATIBLE_MESSAGE = "A newer version is available; this learner continues on the version it was born with."
/** Shown when reloads for new code keep bringing back the old code (a stale cache): the tab stops trying. */
export const STALE_MESSAGE = "A newer version is published but this browser keeps loading the old one; reload later."
/**
 * sessionStorage: `{ codeSha, tries }`, the code this tab last reloaded for and how many reloads it took. A reload
 * that still brought the old code is retried once, after STALE_BACKOFF_POLLS manifest polls, then given up.
 */
export const RELOAD_TARGET_KEY = "blanktab:reloadTarget"
export const STALE_BACKOFF_POLLS = 2
const MAX_RELOADS_PER_CODE = 2

/** The reloads this tab already made for `codeSha` (RELOAD_TARGET_KEY) and still came back on the old code. */
function reloadsFor(storage: Storage, codeSha: string): number {
  try {
    const t = JSON.parse(storage.getItem(RELOAD_TARGET_KEY) ?? "null") as { codeSha?: unknown; tries?: unknown } | null
    return t?.codeSha === codeSha && Number.isInteger(t.tries) ? (t.tries as number) : 0
  } catch {
    return 0
  }
}

/** Shown when the learner's code cannot load and reloading for the newer code did not help: the tab stops trying. */
export const WORKER_LOAD_MESSAGE = "The learner could not load: this page's code was replaced by a newer version. Reload later."

/**
 * The learner's worker failed to load. After a deploy this page's hashed assets are gone, so every newborn would fail
 * the same way: if the deployed manifest names other code, reload onto it (no learner is running, so there is nothing
 * to save; its checkpoint resumes there), under the same guard as code updates (MAX_RELOADS_PER_CODE per code, in
 * RELOAD_TARGET_KEY). "gave-up": the guard is spent (the page stops trying); "same-code": not a deploy (or no manifest).
 */
export async function onWorkerLoadFailure(d: Pick<UpdatesDeps, "codeSha" | "storage" | "loadManifest" | "reload">): Promise<"reloaded" | "gave-up" | "same-code"> {
  const m = await d.loadManifest().catch(() => null)
  if (!m || m.codeSha === d.codeSha) return "same-code"
  const tries = reloadsFor(d.storage, m.codeSha)
  if (tries >= MAX_RELOADS_PER_CODE) return "gave-up"
  d.storage.setItem(RELOAD_TARGET_KEY, JSON.stringify({ codeSha: m.codeSha, tries: tries + 1 }))
  d.reload()
  return "reloaded"
}

export interface Timers { setTimeout(fn: () => void, ms: number): unknown; clearTimeout(handle: unknown): void }
export interface Visibility { hidden(): boolean; onChange(fn: () => void): () => void }
/** `now`: milliseconds, on the same clock as `timers`. */
export interface PollEnv { timers: Timers; visibility: Visibility; random: () => number; now: () => number }

export const jittered = (p: Period, random: () => number): number => p.everyMs + (2 * random() - 1) * p.jitterMs

/**
 * Calls `check` once per jittered period, never at start, and never two at once. While the tab is hidden nothing is
 * scheduled; when it becomes visible `check` runs at once (unless the last check, or the start, was under half a
 * period ago) and the period starts again from there. A check that throws is ignored (the next one comes as usual).
 */
export function createPoller(check: () => unknown, period: Period, env: PollEnv): { stop(): void } {
  const { timers, visibility, random, now } = env
  let timer: unknown = null
  let running = false
  let stopped = false
  /** When the last check started; the page's own load (just before the poller starts) counts as one. */
  let lastAt = now()
  const clear = () => {
    if (timer !== null) timers.clearTimeout(timer)
    timer = null
  }
  const schedule = () => {
    clear()
    if (stopped || visibility.hidden()) return
    timer = timers.setTimeout(() => { timer = null; void run() }, jittered(period, random))
  }
  const run = async () => {
    if (running || stopped) return
    if (visibility.hidden()) return // paused; checked when shown
    running = true
    lastAt = now()
    try {
      await check()
    } catch {
      // a failed check is just skipped
    } finally {
      running = false
    }
    schedule()
  }
  const unsubscribe = visibility.onChange(() => {
    if (visibility.hidden()) clear()
    else if (now() - lastAt < period.everyMs / 2) { if (!running) schedule() } // checked just now: wait a period
    else { clear(); void run() }
  })
  schedule()
  return {
    stop() {
      stopped = true
      clear()
      unsubscribe()
    },
  }
}

export interface UpdatesDeps {
  env: PollEnv
  /** The knowledge the page started with: refreshed knowledge that equals it is not handed on. */
  initial: Knowledge
  /** The running code (__CODE_SHA__) and the checkpoint format it reads and writes (LEARNER_STATE_VERSION). */
  codeSha: string
  learnerStateVersion: number
  /** sessionStorage (RELOAD_TARGET_KEY). */
  storage: Storage
  loadKnowledge(): Promise<Knowledge>
  loadManifest(): Promise<Manifest | null>
  /** Knowledge different from the last handed on (never offline). */
  onKnowledge(k: Knowledge): void
  /** A forced checkpoint of the running learner: true once saved (or if no learner is running), false if not. */
  checkpoint(): Promise<boolean>
  /** Stops the learner and reloads the page (it resumes from the checkpoint just saved). */
  reload(): void
  notice(message: string): void
  /** Test knobs: shorter periods. */
  knowledgePoll?: Period
  manifestPoll?: Period
  reloadJitterMs?: number
  retryMs?: number
}

/** What makes knowledge new: its sha and its content (a new summary or posterior is new knowledge even with the same priors). */
const fingerprint = (k: Knowledge) => JSON.stringify([k.manifest.knowledgeSha, k.summary, k.meta])

/**
 * Starts both pollers (knowledge, manifest); see UpdatesDeps. `halted()`: the learner stopped at a forced save (the
 * worker's `halted` message), possibly after this page had given up waiting for its answer: the page reloads onto that
 * save now (any retry still pending is dropped).
 */
export function startUpdates(d: UpdatesDeps): { stop(): void; halted(): void } {
  const { timers, random } = d.env
  let handed = fingerprint(d.initial)
  let pendingReload: unknown = null
  let stopped = false
  /** The reload being prepared (the code it is for and the how-many-th try), once one is decided. */
  let target: { codeSha: string; tries: number } | null = null
  let reloading = false
  const reloadNow = () => {
    if (reloading || stopped) return
    reloading = true
    if (pendingReload !== null) timers.clearTimeout(pendingReload)
    pendingReload = null
    if (target) d.storage.setItem(RELOAD_TARGET_KEY, JSON.stringify(target))
    d.reload()
  }

  const knowledge = createPoller(async () => {
    const k = await d.loadKnowledge()
    if (k.offline) return
    const f = fingerprint(k)
    if (f === handed) return
    handed = f
    d.onKnowledge(k)
  }, d.knowledgePoll ?? KNOWLEDGE_POLL, d.env)

  /**
   * Saves the learner, then reloads; if the save was refused, asks again later rather than reload onto a stale one.
   * A reload already scheduled goes ahead while the tab is hidden (only the polls pause).
   */
  const reloadFor = async (codeSha: string, tries: number): Promise<void> => {
    pendingReload = null
    if (stopped || reloading) return
    target = { codeSha, tries }
    const saved = await d.checkpoint().catch(() => false)
    if (stopped || reloading) return
    if (!saved) {
      pendingReload = timers.setTimeout(() => void reloadFor(codeSha, tries), d.retryMs ?? CHECKPOINT_RETRY_MS)
      return
    }
    reloadNow()
  }
  let staleSightings = 0

  const manifest = createPoller(async () => {
    const m = await d.loadManifest()
    if (!m || m.codeSha === d.codeSha) return
    // A manifest without learnerStateVersion says nothing about compatibility: skip it (the next poll may).
    if (m.learnerStateVersion === undefined) return
    if (m.learnerStateVersion !== d.learnerStateVersion) {
      manifest.stop() // decided: this learner stays on this code
      d.notice(INCOMPATIBLE_MESSAGE)
      return
    }
    const tries = reloadsFor(d.storage, m.codeSha)
    if (tries >= MAX_RELOADS_PER_CODE) {
      manifest.stop()
      d.notice(STALE_MESSAGE)
      return
    }
    if (tries > 0 && ++staleSightings <= STALE_BACKOFF_POLLS) return // reloaded for it already: back off first
    manifest.stop() // decided: a reload is coming
    pendingReload = timers.setTimeout(() => void reloadFor(m.codeSha, tries + 1), random() * (d.reloadJitterMs ?? RELOAD_JITTER_MS))
  }, d.manifestPoll ?? MANIFEST_POLL, d.env)

  return {
    halted: reloadNow,
    stop() {
      stopped = true
      knowledge.stop()
      manifest.stop()
      if (pendingReload !== null) timers.clearTimeout(pendingReload)
      pendingReload = null
    },
  }
}
