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
/** sessionStorage: the codeSha this tab last reloaded for. A reload that still brought the old code is not repeated for it. */
export const RELOAD_TARGET_KEY = "blanktab:reloadTarget"

export interface Timers { setTimeout(fn: () => void, ms: number): unknown; clearTimeout(handle: unknown): void }
export interface Visibility { hidden(): boolean; onChange(fn: () => void): () => void }
export interface PollEnv { timers: Timers; visibility: Visibility; random: () => number }

export const jittered = (p: Period, random: () => number): number => p.everyMs + (2 * random() - 1) * p.jitterMs

/**
 * Calls `check` once per jittered period, never at start, and never two at once. While the tab is hidden nothing is
 * scheduled; when it becomes visible `check` runs at once and the period starts again from there. A check that
 * throws is ignored (the next one comes as usual).
 */
export function createPoller(check: () => unknown, period: Period, env: PollEnv): { stop(): void } {
  const { timers, visibility, random } = env
  let timer: unknown = null
  let running = false
  let stopped = false
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

/** What makes knowledge new: its sha and its content (a new summary is new knowledge even with the same priors). */
const fingerprint = (k: Knowledge) => JSON.stringify([k.manifest.knowledgeSha, k.summary, k.settingsSpace])

/** Starts both pollers (knowledge, manifest); see UpdatesDeps. */
export function startUpdates(d: UpdatesDeps): { stop(): void } {
  const { timers, random } = d.env
  let handed = fingerprint(d.initial)
  let pendingReload: unknown = null
  let stopped = false

  const knowledge = createPoller(async () => {
    const k = await d.loadKnowledge()
    if (k.offline) return
    const f = fingerprint(k)
    if (f === handed) return
    handed = f
    d.onKnowledge(k)
  }, d.knowledgePoll ?? KNOWLEDGE_POLL, d.env)

  /** Saves the learner, then reloads; if the save was refused, asks again later rather than reload onto a stale one. */
  const reloadFor = async (codeSha: string): Promise<void> => {
    pendingReload = null
    if (stopped) return
    const saved = await d.checkpoint().catch(() => false)
    if (stopped) return
    if (!saved) {
      pendingReload = timers.setTimeout(() => void reloadFor(codeSha), d.retryMs ?? CHECKPOINT_RETRY_MS)
      return
    }
    d.storage.setItem(RELOAD_TARGET_KEY, codeSha)
    d.reload()
  }

  const manifest = createPoller(async () => {
    const m = await d.loadManifest()
    if (!m || m.codeSha === d.codeSha) return
    if (d.storage.getItem(RELOAD_TARGET_KEY) === m.codeSha) return // reloaded for it already and still on this code
    manifest.stop() // decided once: a reload is coming, or this learner stays on this code
    if (m.learnerStateVersion !== d.learnerStateVersion) {
      d.notice(INCOMPATIBLE_MESSAGE)
      return
    }
    pendingReload = timers.setTimeout(() => void reloadFor(m.codeSha), random() * (d.reloadJitterMs ?? RELOAD_JITTER_MS))
  }, d.manifestPoll ?? MANIFEST_POLL, d.env)

  return {
    stop() {
      stopped = true
      knowledge.stop()
      manifest.stop()
      if (pendingReload !== null) timers.clearTimeout(pendingReload)
      pendingReload = null
    },
  }
}
