// A learner's life: steps in the world, measured in compute units (CU), with sleeps, checkpoints and log files (v2).
import type { Observation, WorldView } from "@blanktab/interface"
import { clampSettings } from "@blanktab/learner"
import type { LearnerSettings, LearnerState, LearnerStats, TrainableLearner, Widths } from "@blanktab/learner"
import { deleteCheckpoint, fromRecord, loadCheckpoint, saveCheckpoint, type Checkpoint } from "./checkpoint"
import { OFFLINE_KNOWLEDGE_SHA, type Manifest } from "./knowledge"
import { MAX_CURVE_POINTS, MAX_RESTORES, type CurvePoint, type LogFileV2 } from "./log"
import { openQueue } from "./queue"

export const DEFAULT_SLEEP_INTERVAL = 250
export const DEFAULT_MAX_STEPS_PER_SEC = 30
/** This many skipped updates in a row (a learner gone non-finite) restore the last checkpoint. */
export const RESTORE_AFTER_SKIPS = 20
/** Restores from one checkpoint before the life is given up as failing (no new checkpoint in between). */
export const MAX_RESTORES_PER_CHECKPOINT = 3
/** Five minutes: a GPU checkpoint is tens of MB, too much to write at every sleep. */
export const DEFAULT_CHECKPOINT_EVERY_MS = 300_000
/** The first log crossing, 2^3 CU: a young life writes no log files (and their checkpoints) in a burst. */
export const FIRST_LOG_CU = 8
/** Doubling stops here; from then on a log is written at every multiple of this many CU, forever (about 8 an hour at the speed cap). */
export const LOG_STEP_CU = 1024
/** The last doubling threshold: 2^9 CU. */
export const LAST_DOUBLING_CU = 512

export interface SessionConfig {
  learnerId: string
  codeSha: string
  knowledgeSha: string
  sessionsSeenAtBirth: number
  hostSig: LogFileV2["hostSig"]
  /** Birth facts for a newborn. A resumed learner keeps its own, from its checkpoint's learnerState. */
  seed: number
  widths: Widths
  /** Clamped (clampSettings) before use; the log records the clamped values. */
  learnerSettings: LearnerSettings
  /** Undefined (the default): the life never ends. Test knobs set one; its end is a final log. */
  lifetimeCU?: number
  /** Steps between sleeps. Default DEFAULT_SLEEP_INTERVAL. */
  sleepInterval?: number
  /** Default DEFAULT_MAX_STEPS_PER_SEC. */
  maxStepsPerSec?: number
  /**
   * At a sleep, a checkpoint is saved only if this long (ms, on the injected clock) has passed since the last save.
   * Default DEFAULT_CHECKPOINT_EVERY_MS; 0 = at every clean sleep. So a reload or restore can lose up to this much
   * learning (log-file crossings always save, so it never loses a logged one).
   */
  checkpointEveryMs?: number
}

/**
 * What the session needs of a learner (a TrainableLearner has it). `onKnowledge`, if the learner has it, is handed
 * the manifest of each knowledge refresh (and, after a restore, the latest one); the 04a learner has none.
 */
export type SessionLearner = Pick<TrainableLearner, "step" | "stats" | "sleep" | "save" | "dispose"> & { onKnowledge?(m: Manifest): void }
/** How the session asks for a learner: newborn, or from a checkpoint's state (createLearner's options minus the backend). */
export type LearnerInit = { seed: number; widths: Widths; settings: LearnerSettings } | { state: LearnerState }

/** Milliseconds. */
export interface Clock { now(): number; sleep(ms: number): Promise<void> }
const realClock: Clock = { now: () => performance.now(), sleep: ms => new Promise(r => setTimeout(r, ms)) }

export type SessionEvent =
  /** Resumed from the checkpoint at `steps` and `cu`; `curveTail` is its curve since the latest log file. */
  | { type: "resumed"; steps: number; cu: number; curveTail: CurvePoint[] }
  /** After RESTORE_AFTER_SKIPS skipped updates in a row at `fromSteps`, the learner went back to its checkpoint at `toSteps`. */
  | { type: "restored"; fromSteps: number; toSteps: number; restores: number; lostCU: number }
  /** A log file was queued (with its checkpoint). A failed life's last file is final too (see runSession). */
  | { type: "logged"; seq: number; steps: number; cu: number; final: boolean }

/**
 * The page's requests to a running life (decision 0014), handled by runSession between steps, so never during a step
 * or a save. One inbox may outlive a run (the worker keeps it across GPU restores): requests wait for the next run.
 */
export class SessionInbox {
  #knowledge: Manifest | undefined
  #waiting: { resolve: (saved: boolean) => void; halt: boolean }[] = []

  /**
   * The manifest of refreshed knowledge, in force from now on: every later log file (and checkpoint) records its
   * knowledgeSha. The offline fallback's (OFFLINE_KNOWLEDGE_SHA) is ignored.
   */
  setKnowledge(m: Manifest): void {
    if (m.knowledgeSha !== OFFLINE_KNOWLEDGE_SHA) this.#knowledge = m
  }
  /**
   * A checkpoint of the learner as it is now, outside the throttle (in its own transaction, with no log file). True
   * once it is saved; false if it was not: skipped updates were outstanding (a save then would be of a learner going
   * non-finite), or the life ended first. The caller asks again later. With `halt`, a life that saves for it stops
   * right after the save, at that step (runSession returns `halted`): the page reloads onto exactly that checkpoint.
   */
  checkpointNow(opts: { halt?: boolean } = {}): Promise<boolean> {
    return new Promise(resolve => this.#waiting.push({ resolve, halt: opts.halt ?? false }))
  }

  /** runSession: the latest knowledge given (the same object until a newer one comes). */
  get knowledge(): Manifest | undefined { return this.#knowledge }
  /** runSession: the requests waiting now, taken (before the save that answers them: later ones wait for the next). */
  take(): { resolve: (saved: boolean) => void; halt: boolean }[] {
    return this.#waiting.splice(0)
  }
  /** runSession: answers every waiting request. */
  answer(saved: boolean): void {
    for (const r of this.take()) r.resolve(saved)
  }
}

export interface SessionDeps {
  /** drainTrace, if the world has one, is called after every action (nothing reads the trace yet). */
  world: WorldView & { drainTrace?(): unknown }
  createLearner(init: LearnerInit): Promise<SessionLearner>
  /** Holds the outbox and the checkpoints (one database, so a log and its checkpoint commit together). */
  idb: IDBFactory
  now: () => Date
  /** For the speed cap. Default: performance.now and setTimeout. */
  clock?: Clock
  /** Awaited after every step. */
  onTick?: (s: { steps: number; cu: number; obs: Observation }) => void | Promise<void>
  onEvent?: (e: SessionEvent) => void
  /**
   * Errors the caller recovers from by running the life again from its checkpoint (the page: a rejected GPU read on
   * WebGPU). They are rethrown as they are. Any other error ends the life: its final log is written first (runSession).
   */
  recoverable?: (e: unknown) => boolean
  /**
   * This learner's checkpoint record as stored, already read (by the startup sweep that adopted it): used instead of
   * reading it from `idb` again. Restores and later runs read the database as usual.
   */
  checkpointRecord?: unknown
  /** Refreshed knowledge and forced checkpoints from the page (SessionInbox). */
  inbox?: SessionInbox
}

export interface SessionResult {
  steps: number
  cu: number
  /** The next seq (= the number of log files this learner has written). */
  seq: number
  /** The checkpoint's step count when the life was resumed, else null. */
  resumedFrom: number | null
  /** The learner's last log file was already final: nothing ran (and any leftover checkpoint was removed). */
  alreadyFinal: boolean
  /** The life stopped after a forced save asked to halt (SessionInbox.checkpointNow): it is not over; its checkpoint is that save. */
  halted: boolean
}

/**
 * The speed cap: awaited before every step, it holds consecutive steps at least 1000 / maxStepsPerSec ms apart
 * (never more than maxStepsPerSec a second, even after a stall). It always yields to the event loop.
 */
export function createPacer(maxStepsPerSec: number, clock: Clock = realClock): () => Promise<void> {
  const interval = 1000 / maxStepsPerSec
  let last = Number.NEGATIVE_INFINITY
  return async () => {
    await clock.sleep(Math.max(0, last + interval - clock.now()))
    last = clock.now()
  }
}

/** The log schedule, forever: 8, 16, ..., 512, then 1024, 2048, 3072, ... (every LOG_STEP_CU). */
export function* logThresholds(): Iterable<number> {
  let t = FIRST_LOG_CU
  for (; t <= LAST_DOUBLING_CU; t *= 2) yield t
  for (; ; t += LOG_STEP_CU) yield t
}

/** The first threshold of logThresholds() above `cu` (the next log crossing). */
export function nextCrossing(cu: number): number {
  if (cu >= LAST_DOUBLING_CU) return (Math.floor(cu / LOG_STEP_CU) + 1) * LOG_STEP_CU
  let t = FIRST_LOG_CU
  while (t <= cu) t *= 2
  return t
}

/** An error thrown by the caller's own callback (onTick, onEvent): the caller stopping the life, not the life failing. */
class Interrupted {
  constructor(readonly error: unknown) {}
}

/** Keeps every other point, always the newest, until at most MAX_CURVE_POINTS remain. */
function thin(curve: CurvePoint[]): CurvePoint[] {
  let c = curve
  while (c.length > MAX_CURVE_POINTS) c = c.filter((_, i) => (c.length - 1 - i) % 2 === 0)
  return c
}

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v)
const WIDTH_KEYS = ["conv1", "conv2", "enc", "hidden"] as const

/** A checkpointed learner's birth facts, from its state. RangeError if they are missing or malformed. */
type Birth = { seed: number; widths: Widths; settings: LearnerSettings }
function birthOf(state: unknown, id: string): Birth {
  const fail = (what: string): never => { throw new RangeError(`learner ${id}: the checkpoint's learnerState ${what}`) }
  if (!isRecord(state)) return fail("is not an object")
  const { seed, widths, settings } = state
  if (!(Number.isInteger(seed) && (seed as number) >= 0 && (seed as number) < 2 ** 32)) fail("has no uint32 seed")
  if (!isRecord(widths) || !WIDTH_KEYS.every(k => Number.isInteger(widths[k]) && (widths[k] as number) >= 1)) fail("has malformed widths")
  if (!isRecord(settings)) fail("has no settings")
  let clamped: LearnerSettings
  try {
    clamped = clampSettings(settings as unknown as LearnerSettings)
  } catch (e) {
    return fail(`has bad settings (${e instanceof Error ? e.message : String(e)})`)
  }
  const w = widths as Record<string, number>
  return { seed: seed as number, widths: { conv1: w.conv1!, conv2: w.conv2!, enc: w.enc!, hidden: w.hidden! }, settings: clamped }
}

/**
 * Lives one learner's life (or the rest of it): for as long as it is run, or until it has used `lifetimeCU` compute units if there is one (a test knob).
 *
 * - Every `sleepInterval` steps the learner sleeps and a curve point (predLoss vs the copy-last baseline) is added if
 *   both are finite. A checkpoint is saved then if `checkpointEveryMs` have passed since the last save (on the
 *   injected clock); one that falls due during skips waits for the next clean sleep. A reload or restore therefore
 *   loses up to `checkpointEveryMs` of learning.
 * - A log file (v2) is queued each time CU crosses the next logThresholds() value (8, 16, ..., 512, then every 1024) and at the end of a test lifetime, carrying the
 *   curve points since the previous file. Each one is queued in one transaction with the checkpoint of that moment,
 *   so the checkpoint is never behind the latest log file: resuming never reuses a seq or goes back in steps or CU.
 * - No checkpoint is saved (and so no non-final log is written) while the learner has skipped updates outstanding;
 *   after RESTORE_AFTER_SKIPS in a row it is restored from the last checkpoint. Restores stay on the record: the log
 *   counts them (`restores`), the CU they threw away (`lostCU`) and their skipped updates (in `skippedUpdates`).
 * - The final log file removes the checkpoint in the same transaction.
 * - `inbox` (SessionInbox), checked after every step: knowledge given there is in force from then on (each log file's
 *   knowledgeSha is the knowledge in force when it was written, and the checkpoint keeps it across a reload) and is
 *   handed to `learner.onKnowledge`. A forced checkpoint is saved then, whatever the throttle, under the same rule as
 *   every other save: never while skipped updates are outstanding (it is answered false then, and when the life ends).
 *   One asked with `halt` stops the life right after its save (the result says `halted`; no final log is written).
 * - A life that fails (any error but the caller's recoverable ones and those its own onTick / onEvent throw: the
 *   restore limits, a checkpoint that will not load or resume, a full disk, the learner throwing) is ended on the
 *   record before the error is rethrown: a final log file built from its last checkpoint (steps, CU, seq, curve tail,
 *   restores; lostCU adds the CU lived since that checkpoint) goes into the outbox in the transaction that removes the
 *   checkpoint. Without a usable checkpoint it is built from the latest v2 log file, or from the life's counters if it
 *   has no log file yet; a life that took no step and wrote no log gets none (and its checkpoint is simply removed).
 *   Writing it is best effort: a failure there leaves the original error to be thrown.
 * - With a checkpoint for this learnerId the life resumes from it (its steps, CU, seq, curve and birth facts: seed,
 *   widths and settings from its learnerState, knowledgeSha and sessionsSeenAtBirth from the checkpoint; the config's
 *   are for newborns only); a learner with log files but
 *   no checkpoint cannot be resumed and throws, as does a checkpoint whose seq does not follow the latest log file.
 */
export async function runSession(cfg: SessionConfig, deps: SessionDeps): Promise<SessionResult> {
  const lifetimeCU = cfg.lifetimeCU ?? Number.POSITIVE_INFINITY // no lifetime: the life ends only by failure or the caller
  const sleepInterval = cfg.sleepInterval ?? DEFAULT_SLEEP_INTERVAL
  const maxStepsPerSec = cfg.maxStepsPerSec ?? DEFAULT_MAX_STEPS_PER_SEC
  if (cfg.lifetimeCU !== undefined && !(Number.isFinite(cfg.lifetimeCU) && cfg.lifetimeCU > 0)) throw new RangeError(`lifetimeCU must be finite and > 0, got ${cfg.lifetimeCU}`)
  if (!(Number.isInteger(sleepInterval) && sleepInterval >= 1)) throw new RangeError(`sleepInterval must be an integer >= 1, got ${sleepInterval}`)
  if (!(Number.isFinite(maxStepsPerSec) && maxStepsPerSec > 0)) throw new RangeError(`maxStepsPerSec must be finite and > 0, got ${maxStepsPerSec}`)
  const checkpointEveryMs = cfg.checkpointEveryMs ?? DEFAULT_CHECKPOINT_EVERY_MS
  if (!(Number.isFinite(checkpointEveryMs) && checkpointEveryMs >= 0)) throw new RangeError(`checkpointEveryMs must be finite and >= 0, got ${checkpointEveryMs}`)
  const { learnerId: id } = cfg
  const { world, idb, now, onTick, onEvent } = deps
  const clock = deps.clock ?? realClock
  const queue = await openQueue(idb)
  const emit = (e: SessionEvent) => {
    try {
      onEvent?.(e)
    } catch (err) {
      throw new Interrupted(err)
    }
  }

  const latest = await queue.latest(id)
  if (latest?.final) {
    await deleteCheckpoint(idb, id)
    return { steps: latest.stepsTotal, cu: latest.schemaVersion === 2 ? latest.cu : 0, seq: latest.seq + 1, resumedFrom: null, alreadyFinal: true, halted: false }
  }
  // What a failed life's final log is built from, as far as the life got: its latest checkpoint, its birth facts and
  // its counters (live, once there is a learner).
  let cp: Checkpoint | undefined
  let birth: Birth | undefined
  let knowledgeSha = cfg.knowledgeSha
  let sessionsSeenAtBirth = cfg.sessionsSeenAtBirth
  let live: (() => { steps: number; seq: number; curve: CurvePoint[]; restores: number; lostCU: number; lostSkips: number; stats: Pick<LearnerStats, "cu" | "skippedUpdates"> }) | undefined

  /** The final log file of a life that failed (see runSession), or null if there is nothing to put on the record. */
  const failedLog = async (): Promise<LogFileV2 | null> => {
    const last = await queue.latest(id)
    const nextSeq = (last?.seq ?? -1) + 1
    const prior = last?.schemaVersion === 2 ? last : undefined
    let lived: ReturnType<NonNullable<typeof live>> | undefined
    try {
      lived = live?.()
    } catch {
      lived = undefined // a learner too broken to report: its counters are not known
    }
    // A life that took no step and wrote no log file has nothing to put on the record.
    if (!last && !(Math.max(cp?.steps ?? 0, lived?.steps ?? 0) > 0)) return null
    const base = {
      schemaVersion: 2 as const, learnerId: id, seq: nextSeq, codeSha: cfg.codeSha, writtenAt: now().toISOString(), hostSig: { ...cfg.hostSig },
      final: true, condition: "A" as const,
    }
    const birthFields = (b: Birth) => ({ settings: { ...b.settings, maxStepsPerSec, seed: b.seed }, widths: { ...b.widths }, knowledgeSha, sessionsSeenAtBirth })
    // Never below the latest file's counters (they never decrease across seq).
    const atLeast = (f: LogFileV2): LogFileV2 => !prior ? f : {
      ...f, stepsTotal: Math.max(f.stepsTotal, prior.stepsTotal), cu: Math.max(f.cu, prior.cu), restores: Math.max(f.restores, prior.restores),
      lostCU: Math.max(f.lostCU, prior.lostCU), skippedUpdates: Math.max(f.skippedUpdates, prior.skippedUpdates),
    }
    if (cp && birth && cp.seq === nextSeq) {
      const st = (cp.learnerState as { stats?: Partial<LearnerStats> } | null)?.stats
      const savedSkips = typeof st?.skippedUpdates === "number" ? st.skippedUpdates : 0
      return atLeast({
        ...base, ...birthFields(birth), stepsTotal: cp.steps, cu: cp.cu, curve: thin([...cp.curveTail]), restores: cp.restores,
        // The CU lived since the checkpoint is thrown away with the life: on the record, as a restore's is.
        lostCU: cp.lostCU + Math.max(0, (lived?.stats.cu ?? cp.cu) - cp.cu),
        skippedUpdates: cp.lostSkips + Math.max(savedSkips, lived?.stats.skippedUpdates ?? 0),
      })
    }
    if (prior) {
      return { ...prior, ...base, knowledgeSha: prior.knowledgeSha, sessionsSeenAtBirth: prior.sessionsSeenAtBirth, curve: [] }
    }
    if (!last && birth && lived && lived.steps > 0) {
      return {
        ...base, ...birthFields(birth), stepsTotal: lived.steps, cu: lived.stats.cu, curve: thin([...lived.curve]), restores: lived.restores,
        lostCU: lived.lostCU, skippedUpdates: lived.stats.skippedUpdates + lived.lostSkips,
      }
    }
    return null
  }

  try {
    return await liveTheLife()
  } catch (e) {
    deps.inbox?.answer(false)
    if (e instanceof Interrupted) throw e.error
    if (deps.recoverable?.(e)) throw e
    try {
      const f = await failedLog()
      if (f) {
        await queue.put(f, { remove: id })
        onEvent?.({ type: "logged", seq: f.seq, steps: f.stepsTotal, cu: f.cu, final: true })
      } else {
        await deleteCheckpoint(idb, id)
      }
    } catch {
      // Best effort (the disk may be full): the life's own error is the one to report.
    }
    throw e
  }

  async function liveTheLife(): Promise<SessionResult> {
    cp = deps.checkpointRecord !== undefined ? await fromRecord(deps.checkpointRecord, id) : await loadCheckpoint(idb, id)
    if (!cp && latest) throw new Error(`learner ${id} has log files up to seq ${latest.seq} but no checkpoint: its life cannot be resumed`)
    // Every log file is queued with its checkpoint, so the checkpoint's seq is exactly the one after the latest file.
    if (cp && cp.seq !== (latest?.seq ?? -1) + 1) {
      throw new Error(`learner ${id}: the checkpoint's seq ${cp.seq} does not follow its latest log file (seq ${latest?.seq ?? "none"})`)
    }
    birth = cp ? birthOf(cp.learnerState, id) : { seed: cfg.seed, widths: cfg.widths, settings: clampSettings(cfg.learnerSettings) }
    const { seed, widths, settings } = birth
    // sessionsSeenAtBirth is a birth fact; knowledgeSha is the knowledge in force at the checkpoint (until the inbox
    // brings newer). A resumed life keeps the ones its checkpoint carries (older checkpoints do not; then the config's).
    knowledgeSha = cp?.knowledgeSha ?? cfg.knowledgeSha
    sessionsSeenAtBirth = cp?.sessionsSeenAtBirth ?? cfg.sessionsSeenAtBirth

    const inbox = deps.inbox
    /** The inbox's knowledge this run has put in force (none yet: the checkpoint's or the config's sha is). */
    let applied: Manifest | undefined
    const takeKnowledge = () => {
      const k = inbox?.knowledge
      if (!k || k === applied) return
      applied = k
      knowledgeSha = k.knowledgeSha
      learner.onKnowledge?.(k)
    }
    let learner = await deps.createLearner(cp ? { state: cp.learnerState } : { seed, widths, settings })
    try {
      let steps = cp?.steps ?? 0
      let seq = cp?.seq ?? 0
      let curve: CurvePoint[] = cp ? [...cp.curveTail] : []
      let crossing = nextCrossing(learner.stats().cu)
      let restoresSinceSave = 0
      // Restores stay on the record across the whole life (and reloads, through the checkpoint).
      let restores = cp?.restores ?? 0
      let lostCU = cp?.lostCU ?? 0
      let lostSkips = cp?.lostSkips ?? 0
      let lastSavedAt = clock.now()
      const resumedFrom = cp ? cp.steps : null
      live = () => ({ steps, seq, curve, restores, lostCU, lostSkips, stats: learner.stats() })
      if (cp) emit({ type: "resumed", steps: cp.steps, cu: cp.cu, curveTail: [...cp.curveTail] })

      const checkpointNow = async (): Promise<Checkpoint> => ({
        learnerId: id, savedAt: now().toISOString(), steps, cu: learner.stats().cu, seq, curveTail: [...curve],
        restores, lostCU, lostSkips, knowledgeSha, sessionsSeenAtBirth, learnerState: await learner.save(),
      })
      const logFile = (final: boolean): LogFileV2 => {
        const st = learner.stats()
        return {
          schemaVersion: 2, learnerId: id, seq, codeSha: cfg.codeSha, knowledgeSha, writtenAt: now().toISOString(),
          stepsTotal: steps, settings: { ...settings, maxStepsPerSec, seed }, hostSig: { ...cfg.hostSig },
          sessionsSeenAtBirth, final, cu: st.cu, condition: "A", widths: { ...widths }, curve: [...curve],
          skippedUpdates: st.skippedUpdates + lostSkips, restores, lostCU,
        }
      }
      const addPoint = () => {
        const { cu, predLoss, baselineLoss } = learner.stats()
        if (!Number.isFinite(predLoss) || !Number.isFinite(baselineLoss)) return // never NaN in a log
        if (curve.at(-1)?.steps === steps) return
        curve = thin([...curve, { cu, steps, predLoss, baselineLoss }])
      }
      /** After any save of the learner's current state. */
      const saved = () => {
        restoresSinceSave = 0
        lastSavedAt = clock.now()
      }

      // A newborn's first checkpoint, so there is always one to restore.
      if (!cp) {
        const first = await checkpointNow()
        await saveCheckpoint(idb, first)
        cp = first
        saved()
      }
      let restorePoint: Checkpoint = cp
      const pace = createPacer(maxStepsPerSec, clock)
      let halted = false

      while (learner.stats().cu < lifetimeCU) {
        await pace()
        const obs = world.observe()
        world.act(await learner.step(obs))
        world.drainTrace?.()
        steps++
        try {
          await onTick?.({ steps, cu: learner.stats().cu, obs })
        } catch (err) {
          throw new Interrupted(err)
        }
        takeKnowledge()
        let checkpointDue = false
        if (steps % sleepInterval === 0) {
          await learner.sleep()
          addPoint()
          checkpointDue = true
        }
        const st = learner.stats()
        if (st.consecutiveSkips >= RESTORE_AFTER_SKIPS) {
          if (restores >= MAX_RESTORES) throw new Error(`learner ${id}: ${MAX_RESTORES} restores, the most a log file may record`)
          if (++restoresSinceSave > MAX_RESTORES_PER_CHECKPOINT) {
            throw new Error(`learner ${id}: still skipping updates after ${MAX_RESTORES_PER_CHECKPOINT} restores from the checkpoint at step ${restorePoint.steps}`)
          }
          const back = await loadCheckpoint(idb, id)
          if (!back) throw new Error(`learner ${id}: no checkpoint to restore`)
          // Every log file carries its checkpoint, so nothing was logged after this one: seq must not move.
          if (back.seq !== seq) throw new Error(`learner ${id}: the checkpoint's seq ${back.seq} is not the next seq ${seq}`)
          learner.dispose()
          learner = await deps.createLearner({ state: back.learnerState })
          if (applied) learner.onKnowledge?.(applied)
          const now_ = learner.stats()
          restores++
          lostCU += st.cu - now_.cu
          lostSkips += st.skippedUpdates - now_.skippedUpdates
          // On the record at once, so a reload before the next save does not forget the restore.
          cp = restorePoint = { ...back, restores, lostCU, lostSkips }
          await saveCheckpoint(idb, cp)
          lastSavedAt = clock.now()
          emit({ type: "restored", fromSteps: steps, toSteps: back.steps, restores, lostCU })
          steps = back.steps
          curve = [...back.curveTail]
          crossing = nextCrossing(learner.stats().cu)
          continue
        }
        if (st.cu >= lifetimeCU) break
        if (st.consecutiveSkips > 0) {
          inbox?.answer(false) // not now: the page asks again later
          continue // no checkpoint now; a crossing waits until updates apply again
        }
        // Taken before the save: a request made while it is written waits for the next step's.
        const forced = inbox?.take() ?? []
        if (st.cu >= crossing) {
          const f = logFile(false)
          seq++
          curve = []
          crossing = nextCrossing(st.cu)
          const next = await checkpointNow()
          await queue.put(f, { save: next })
          cp = restorePoint = next
          saved()
          emit({ type: "logged", seq: f.seq, steps: f.stepsTotal, cu: f.cu, final: false })
        } else if (forced.length > 0 || (checkpointDue && clock.now() - lastSavedAt >= checkpointEveryMs)) {
          const next = await checkpointNow()
          await saveCheckpoint(idb, next)
          cp = restorePoint = next
          saved()
        }
        for (const r of forced) r.resolve(true)
        if (forced.some(r => r.halt)) {
          halted = true
          break
        }
      }

      if (halted) return { steps, cu: learner.stats().cu, seq, resumedFrom, alreadyFinal: false, halted: true }

      addPoint()
      const end = logFile(true)
      await queue.put(end, { remove: id })
      cp = undefined // the life is over: nothing left to fail
      live = undefined
      seq++
      inbox?.answer(false) // the life is over: there is no checkpoint now
      emit({ type: "logged", seq: end.seq, steps: end.stepsTotal, cu: end.cu, final: true })
      return { steps, cu: learner.stats().cu, seq, resumedFrom, alreadyFinal: false, halted: false }
    } finally {
      learner.dispose()
    }
  }
}
